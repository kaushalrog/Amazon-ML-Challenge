import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
from blocking import generate_candidate_pairs
from features import extract_features
from normalize import normalize_df

def load_test_data():
    s1_df = pl.read_csv("dataset/test/test_source1.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8}).to_pandas()
    s2_df = pl.read_csv("dataset/test/test_source2.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8}).to_pandas()
    s3_df = pl.read_csv("dataset/test/test_source3.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8}).to_pandas()
    
    cand_df = pd.concat([s2_df, s3_df])
    return s1_df, cand_df

def generate_test_candidates(s1_df, cand_df):
    from blocking import generate_candidate_pairs
    return generate_candidate_pairs(s1_df, cand_df)

def extract_test_features(s1_df, cand_df, cand_pairs):
    records = []
    cand_df_idx = cand_df.set_index('entity_id')
    
    for _, s1_row in s1_df.iterrows():
        s1 = s1_row['entity_id']
        cands = cand_pairs.get(s1, set())
        
        for c in cands:
            c_row = cand_df_idx.loc[c]
            features = extract_features(s1_row, c_row)
            rec = {'s1': s1, 'c': c}
            rec.update(features)
            records.append(rec)
            
    return pd.DataFrame(records)

def main():
    print("Loading test data...")
    s1_df, cand_df = load_test_data()
    
    print("Normalizing...")
    s1_df = normalize_df(s1_df)
    cand_df = normalize_df(cand_df)
    
    print("Generating candidates...")
    cand_pairs = generate_test_candidates(s1_df, cand_df)
    
    # Save candidate pairs
    print("Saving candidate pairs...")
    with open("output/candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_df['entity_id']:
            cands = list(cand_pairs.get(s1, []))
            f.write(f"{s1}\t{','.join(cands)}\n")
            
    print("Extracting features...")
    test_df = extract_test_features(s1_df, cand_df, cand_pairs)
    
    if len(test_df) > 0:
        print("Predicting...")
        model = lgb.Booster(model_file='models/lightgbm_model.txt')
        features = [c for c in test_df.columns if c not in ['s1', 'c']]
        
        # Load best threshold
        with open("reports/decision_analysis.md", "r") as f:
            for line in f:
                if "Best threshold:" in line:
                    best_th = float(line.strip().split()[-1])
                    break
        print(f"Using threshold: {best_th}")
        
        preds = model.predict(test_df[features])
        test_df['pred_prob'] = preds
        
        # Apply threshold
        pred_matches = {}
        for s1, group in test_df.groupby('s1'):
            matches = group[group['pred_prob'] >= best_th]['c'].tolist()
            pred_matches[s1] = matches
    else:
        pred_matches = {}
        
    print("Saving matching results...")
    with open("output/matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_df['entity_id']:
            matches = pred_matches.get(s1, [])
            f.write(f"{s1}\t{','.join(matches)}\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
