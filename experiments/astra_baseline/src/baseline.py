import polars as pl
import pandas as pd
from validation import evaluate_macro_f05
from normalize import normalize_whitespace, remove_punctuation, unicode_normalize
import gc
import json

def load_data():
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t', schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}).to_pandas()
    # sample for fast baseline evaluation
    # Let's take 20,000 S1 entities for validation
    gt_df = gt_df.dropna(subset=['source1_entity_id'])
    val_gt = gt_df.sample(n=min(len(gt_df), 20000), random_state=42)
    val_s1_ids = set(val_gt['source1_entity_id'])
    
    y_true = {}
    for _, row in val_gt.iterrows():
        s1 = row["source1_entity_id"]
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()] if not pd.isna(row["matched_entity_ids"]) else []
        y_true[s1] = set(matches)
        
    print("Loading S1...")
    s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    s1_df = s1_df[s1_df['entity_id'].isin(val_s1_ids)]
    
    print("Loading S2 and S3...")
    s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    
    return s1_df, pd.concat([s2_df, s3_df]), y_true

def normalize_df(df):
    def norm(x):
        if pd.isna(x): return ""
        return normalize_whitespace(remove_punctuation(unicode_normalize(str(x).lower())))
    
    df['norm_name'] = df['business_name'].apply(norm)
    df['norm_address'] = df['business_address'].apply(norm)
    return df

def run_baseline_exact_match(s1_df, candidates_df, y_true, field):
    """
    field: 'norm_name', 'norm_address', or 'both'
    """
    print(f"Running baseline on {field}...")
    
    cand_dict = {}
    
    if field == 'both':
        # Group by name and address
        grouped = candidates_df.groupby(['norm_name', 'norm_address'])['entity_id'].apply(list).to_dict()
        for _, row in s1_df.iterrows():
            n, a = row['norm_name'], row['norm_address']
            if n and a:
                cand_dict[row['entity_id']] = set(grouped.get((n, a), []))
            else:
                cand_dict[row['entity_id']] = set()
    else:
        grouped = candidates_df.groupby(field)['entity_id'].apply(list).to_dict()
        for _, row in s1_df.iterrows():
            val = row[field]
            if val:
                cand_dict[row['entity_id']] = set(grouped.get(val, []))
            else:
                cand_dict[row['entity_id']] = set()
                
    return evaluate_macro_f05(y_true, cand_dict)

def main():
    s1_df, cand_df, y_true = load_data()
    print("Normalizing S1...")
    s1_df = normalize_df(s1_df)
    print("Normalizing Candidates...")
    cand_df = normalize_df(cand_df)
    
    results = {}
    
    res_name = run_baseline_exact_match(s1_df, cand_df, y_true, 'norm_name')
    results['Baseline_A_Norm_Name'] = res_name
    
    res_addr = run_baseline_exact_match(s1_df, cand_df, y_true, 'norm_address')
    results['Baseline_B_Norm_Address'] = res_addr
    
    res_both = run_baseline_exact_match(s1_df, cand_df, y_true, 'both')
    results['Baseline_C_Norm_Both'] = res_both
    
    print(results)
    
    # Save to CSV
    res_list = []
    for k, v in results.items():
        v['baseline'] = k
        res_list.append(v)
    
    df_res = pd.DataFrame(res_list)
    df_res.to_csv("reports/baseline_results.csv", index=False)
    
if __name__ == "__main__":
    main()
