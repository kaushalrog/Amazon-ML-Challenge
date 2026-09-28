# Dataset forensics — ASTRA verified raw TSV scan

All figures below are measured from the supplied TSV files, with explicit tab separation. No external data used.

| Split | Source | Rows | Unique names | Unique addresses | Missing names | Missing addresses |
|---|---|---:|---:|---:|---:|---:|
| train | S1 | 2,206,821 | 1,539,229 | 2,130,606 | 0 | 0 |
| train | S2 | 5,034,616 | 4,402,009 | 4,337,262 | 0 | 168,967 |
| train | S3 | 5,285,603 | 4,651,609 | 4,632,765 | 0 | 175,916 |
| test | S1 | 1,732,544 | 1,238,867 | 1,677,483 | 0 | 0 |
| test | S2 | 4,887,273 | 4,311,041 | 4,224,784 | 0 | 129,408 |
| test | S3 | 5,082,316 | 4,521,929 | 4,456,436 | 0 | 136,098 |

## Ground truth

- s1_entities: 2206821
- singletons: 123247
- one_match: 119157
- multi_match: 1964417
- mean_matches: 3.4612526344456573
- max_matches: 11
- s2_match_rate: 0.8696110830919227
- s3_match_rate: 0.8793395567651386
- both_source_rate: 0.8047988486605846

## Data integrity

Every raw entity ID is unique within its file. Leading-zero counts are recorded in astra_dataset_stats.json; do not reconstruct IDs without validating this property. Training countries and test countries are read as open strings. Test includes France, absent from labeled training.

Further per-file country counts, exact repeated name/address/country combinations, numeric-address rates and mean text lengths are in astra_dataset_stats.json. Repetition counts include missing values; empty strings must never become positive identity evidence.
