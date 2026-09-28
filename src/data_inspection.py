import pandas as pd
import polars as pl
import json
import gc
import os

def analyze_file(path):
    print(f"Analyzing {path}...")
    df = pl.read_csv(path, separator='\t', truncate_ragged_lines=True, ignore_errors=True, schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8})
    
    row_count = df.height
    col_count = df.width
    col_names = df.columns
    
    missing_values = {col: df[col].is_null().sum() for col in df.columns}
    empty_strings = {col: (df[col] == "").sum() for col in df.columns if df[col].dtype == pl.Utf8}
    
    duplicate_entities = df.height - df["entity_id"].n_unique()
    
    if "business_name" in df.columns:
        duplicate_names = df.height - df["business_name"].n_unique()
        name_len = df["business_name"].str.len_chars().drop_nulls()
        name_len_dist = {"mean": name_len.mean(), "median": name_len.median()}
    else:
        duplicate_names = 0
        name_len_dist = {}
        
    if "business_address" in df.columns:
        duplicate_address = df.height - df["business_address"].n_unique()
        addr_len = df["business_address"].str.len_chars().drop_nulls()
        addr_len_dist = {"mean": addr_len.mean(), "median": addr_len.median()}
    else:
        duplicate_address = 0
        addr_len_dist = {}
        
    if "country" in df.columns:
        country_counts = {row[0]: row[1] for row in df["country"].value_counts().iter_rows()}
    else:
        country_counts = {}
        
    stats = {
        "row_count": row_count,
        "col_count": col_count,
        "col_names": col_names,
        "missing_values": missing_values,
        "empty_strings": empty_strings,
        "duplicate_entities": duplicate_entities,
        "duplicate_names": duplicate_names,
        "duplicate_address": duplicate_address,
        "name_length_distribution": name_len_dist,
        "address_length_distribution": addr_len_dist,
        "country_distribution": country_counts
    }
    
    del df
    gc.collect()
    return stats

def main():
    base_dir = "dataset"
    results = {}
    
    files = [
        "train/train_source1.tsv",
        "train/train_source2.tsv",
        "train/train_source3.tsv",
        "test/test_source1.tsv",
        "test/test_source2.tsv",
        "test/test_source3.tsv"
    ]
    
    for f in files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            results[f] = analyze_file(path)
            
    # Ground truth analysis
    gt_path = os.path.join(base_dir, "train/train_ground_truth.tsv")
    print(f"Analyzing {gt_path}...")
    gt_df = pl.read_csv(gt_path, separator='\t', truncate_ragged_lines=True, ignore_errors=True, schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    
    # Process matched_entity_ids
    # it's a comma separated string
    def count_matches(s):
        if s is None or s == "":
            return 0
        return len(s.split(","))
        
    # use pandas for this part for easier map if polars apply is slow
    gt_pd = gt_df.to_pandas()
    gt_pd["match_count"] = gt_pd["matched_entity_ids"].fillna("").apply(lambda x: len([i for i in x.split(",") if i.strip()]) if x else 0)
    
    s2_matches = gt_pd["matched_entity_ids"].fillna("").apply(lambda x: len([i for i in x.split(",") if i.strip().startswith("S2")]) if x else 0)
    s3_matches = gt_pd["matched_entity_ids"].fillna("").apply(lambda x: len([i for i in x.split(",") if i.strip().startswith("S3")]) if x else 0)
    
    gt_stats = {
        "num_source1_entities": len(gt_pd),
        "zero_match_count": int((gt_pd["match_count"] == 0).sum()),
        "one_match_count": int((gt_pd["match_count"] == 1).sum()),
        "multiple_match_count": int((gt_pd["match_count"] > 1).sum()),
        "average_matches": float(gt_pd["match_count"].mean()),
        "median_matches": float(gt_pd["match_count"].median()),
        "max_matches": float(gt_pd["match_count"].max()),
        "percentage_s2": float((s2_matches > 0).mean()),
        "percentage_s3": float((s3_matches > 0).mean()),
        "percentage_both": float(((s2_matches > 0) & (s3_matches > 0)).mean()),
    }
    
    results["train_ground_truth"] = gt_stats
    
    with open("reports/dataset_stats.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Write markdown
    with open("reports/dataset_forensics.md", "w") as f:
        f.write("# Dataset Forensics\n\n")
        
        for k, v in results.items():
            if k == "train_ground_truth":
                continue
            f.write(f"## {k}\n")
            f.write(f"- Row Count: {v['row_count']}\n")
            f.write(f"- Duplicate Entities: {v['duplicate_entities']}\n")
            f.write(f"- Countries: {', '.join(map(str, v['country_distribution'].keys()))}\n")
            f.write("\n")
            
        f.write("## Ground Truth\n")
        f.write(f"- S1 Entities: {gt_stats['num_source1_entities']}\n")
        f.write(f"- Zero matches: {gt_stats['zero_match_count']}\n")
        f.write(f"- One match: {gt_stats['one_match_count']}\n")
        f.write(f"- Multiple matches: {gt_stats['multiple_match_count']}\n")
        f.write(f"- Average matches: {gt_stats['average_matches']:.2f}\n")
        
if __name__ == "__main__":
    main()
