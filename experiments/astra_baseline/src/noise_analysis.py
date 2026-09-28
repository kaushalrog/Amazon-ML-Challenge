import polars as pl
import pandas as pd
import json
import os
import re

def normalize_text(text):
    if pd.isna(text) or text is None or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def exact_match(a, b):
    if pd.isna(a) or pd.isna(b): return False
    return a == b

def lower_match(a, b):
    if pd.isna(a) or pd.isna(b): return False
    return str(a).lower() == str(b).lower()

def norm_match(a, b):
    if pd.isna(a) or pd.isna(b): return False
    return normalize_text(str(a)) == normalize_text(str(b))

def main():
    print("Loading ground truth...")
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t', schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}).to_pandas()
    
    # We will sample 50,000 positive matches to avoid out-of-memory
    pairs = []
    
    print("Extracting pairs from ground truth...")
    gt_df = gt_df.dropna(subset=['matched_entity_ids'])
    for _, row in gt_df.sample(min(len(gt_df), 100000), random_state=42).iterrows():
        s1 = row["source1_entity_id"]
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        for m in matches:
            pairs.append({"s1": s1, "m": m})
            
    pairs_df = pd.DataFrame(pairs).sample(n=min(len(pairs), 50000), random_state=42)
    
    # split into S2 and S3 pairs
    s2_pairs = pairs_df[pairs_df['m'].str.startswith('S2')]
    s3_pairs = pairs_df[pairs_df['m'].str.startswith('S3')]
    
    print("Loading source 1...")
    s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    s1_df = s1_df.set_index("entity_id")
    
    print("Loading source 2...")
    s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    s2_df = s2_df.set_index("entity_id")
    
    print("Loading source 3...")
    s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator='\t', schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8}).to_pandas()
    s3_df = s3_df.set_index("entity_id")
    
    results = {}
    
    for name, pairs_subset, s_df in [("S1-S2", s2_pairs, s2_df), ("S1-S3", s3_pairs, s3_df)]:
        print(f"Analyzing {name} matches...")
        valid_pairs = []
        for _, row in pairs_subset.iterrows():
            s1_id = row['s1']
            m_id = row['m']
            if s1_id in s1_df.index and m_id in s_df.index:
                valid_pairs.append({
                    's1_name': s1_df.loc[s1_id, 'business_name'],
                    's1_address': s1_df.loc[s1_id, 'business_address'],
                    'm_name': s_df.loc[m_id, 'business_name'],
                    'm_address': s_df.loc[m_id, 'business_address']
                })
        
        v_df = pd.DataFrame(valid_pairs)
        if len(v_df) == 0:
            continue
            
        stats = {
            "total_pairs": len(v_df),
            "name_exact_match": v_df.apply(lambda r: exact_match(r['s1_name'], r['m_name']), axis=1).mean(),
            "name_lower_match": v_df.apply(lambda r: lower_match(r['s1_name'], r['m_name']), axis=1).mean(),
            "name_norm_match": v_df.apply(lambda r: norm_match(r['s1_name'], r['m_name']), axis=1).mean(),
            "address_exact_match": v_df.apply(lambda r: exact_match(r['s1_address'], r['m_address']), axis=1).mean(),
            "address_lower_match": v_df.apply(lambda r: lower_match(r['s1_address'], r['m_address']), axis=1).mean(),
            "address_norm_match": v_df.apply(lambda r: norm_match(r['s1_address'], r['m_address']), axis=1).mean()
        }
        results[name] = stats
        print(stats)
        
    with open("reports/noise_analysis.md", "w") as f:
        f.write("# Noise Analysis on Positive Pairs\n\n")
        f.write("Analyzing a sample of ground-truth matching pairs.\n\n")
        for k, v in results.items():
            f.write(f"## {k} Matches\n")
            f.write(f"- Total sampled pairs: {v['total_pairs']}\n")
            f.write("### Business Name\n")
            f.write(f"- Exact match rate: {v['name_exact_match']:.4f}\n")
            f.write(f"- Lowercase match rate: {v['name_lower_match']:.4f}\n")
            f.write(f"- Normalized match rate: {v['name_norm_match']:.4f}\n")
            f.write("### Business Address\n")
            f.write(f"- Exact match rate: {v['address_exact_match']:.4f}\n")
            f.write(f"- Lowercase match rate: {v['address_lower_match']:.4f}\n")
            f.write(f"- Normalized match rate: {v['address_norm_match']:.4f}\n\n")

if __name__ == "__main__":
    main()
