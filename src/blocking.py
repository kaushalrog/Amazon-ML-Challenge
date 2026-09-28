import polars as pl
import pandas as pd
import numpy as np
from normalize import normalize_df, normalize_whitespace, remove_punctuation, unicode_normalize, extract_numeric
import gc
import json

def generate_blocks(df):
    # generate multiple blocking keys
    df['key_norm_name'] = df['norm_name']
    df['key_norm_addr'] = df['norm_address']
    df['key_compact_name'] = df['norm_name'].str.replace(' ', '', regex=False)
    
    df.loc[df['key_norm_name'].str.len() < 4, 'key_norm_name'] = ""
    df.loc[df['key_compact_name'].str.len() < 5, 'key_compact_name'] = ""
    
    df['key_prefix_name'] = df['norm_name'].str[:6]
    df['key_prefix_addr'] = df['norm_address'].str[:6]
    
    df['key_name5_addr5'] = df['key_prefix_name'] + '_' + df['key_prefix_addr']
    df.loc[(df['norm_name'].str.len() < 3) | (df['norm_address'].str.len() < 3), 'key_name5_addr5'] = ""
    
    df['num_addr'] = df['business_address'].apply(lambda x: extract_numeric(str(x)))
    
    df['key_num_country'] = df['num_addr'] + '_' + df['country'].astype(str)
    df.loc[df['num_addr'] == "", 'key_num_country'] = ""
    
    # First token of name
    df['name_token1'] = df['norm_name'].apply(lambda x: x.split()[0] if str(x).split() else "")
    df['key_token1_num'] = df['name_token1'] + '_' + df['num_addr']
    df.loc[(df['num_addr'] == "") | (df['name_token1'] == ""), 'key_token1_num'] = ""
    
    return df

def generate_candidate_pairs(s1_df, cand_df):
    s1_df = generate_blocks(s1_df)
    cand_df = generate_blocks(cand_df)
    
    cand_pairs = {s1: set() for s1 in s1_df['entity_id']}
    
    keys_to_use = [
        'key_norm_name',
        'key_norm_addr',
        'key_compact_name',
        'key_name5_addr5',
        'key_num_country',
        'key_token1_num'
    ]
    
    for key in keys_to_use:
        print(f"Blocking on {key}...")
        # drop empty keys
        s1_valid = s1_df[s1_df[key].astype(bool)]
        cand_valid = cand_df[cand_df[key].astype(bool)]
        
        merged = pd.merge(s1_valid[['entity_id', key]], cand_valid[['entity_id', key]], on=key, suffixes=('_s1', '_cand'))
        
        # group by s1
        grouped = merged.groupby('entity_id_s1')['entity_id_cand'].apply(set).to_dict()
        for s1, cands in grouped.items():
            cand_pairs[s1] |= cands
            
    return cand_pairs

def load_split_data():
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t', schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}).to_pandas()
    gt_df = gt_df.dropna(subset=['source1_entity_id'])
    val_gt = gt_df.sample(n=min(len(gt_df), 20000), random_state=42)
    val_s1_ids = set(val_gt['source1_entity_id'])
    
    y_true = {}
    for _, row in val_gt.iterrows():
        s1 = row["source1_entity_id"]
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()] if not pd.isna(row["matched_entity_ids"]) else []
        y_true[s1] = set(matches)
        
    s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    s1_val = s1_df[s1_df['entity_id'].isin(val_s1_ids)].copy()
    
    s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    
    cand_df = pd.concat([s2_df, s3_df])
    return s1_val, cand_df, y_true

def evaluate_blocking(cand_pairs, y_true):
    candidate_counts = []
    true_positive_pairs = 0
    total_true_pairs = 0
    
    for s1, true_matches in y_true.items():
        cands = cand_pairs.get(s1, set())
        candidate_counts.append(len(cands))
        
        true_positive_pairs += len(true_matches & cands)
        total_true_pairs += len(true_matches)
        
    avg_cand = np.mean(candidate_counts)
    max_cand = np.max(candidate_counts)
    recall = true_positive_pairs / total_true_pairs if total_true_pairs > 0 else 0
    
    return {
        "candidate_count": sum(candidate_counts),
        "average_candidates_per_s1": avg_cand,
        "max_candidates_per_s1": max_cand,
        "true_positive_pairs": true_positive_pairs,
        "total_true_pairs": total_true_pairs,
        "candidate_recall": recall
    }

def main():
    print("Loading data...")
    s1_val, cand_df, y_true = load_split_data()
    print("Normalizing...")
    s1_val = normalize_df(s1_val)
    cand_df = normalize_df(cand_df)
    
    print("Generating candidate pairs...")
    cand_pairs = generate_candidate_pairs(s1_val, cand_df)
    
    print("Evaluating...")
    res = evaluate_blocking(cand_pairs, y_true)
    print(res)
    
    df_ab = pd.DataFrame([res])
    df_ab.to_csv("reports/blocking_ablation.csv", index=False)
    
    with open("reports/blocking_report.md", "w") as f:
        f.write("# Blocking Strategy Evaluation\n\n")
        f.write("We evaluate deterministic hashing logic for candidate generation.\n\n")
        f.write(df_ab.to_html(index=False))
        
if __name__ == "__main__":
    main()
