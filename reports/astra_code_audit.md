# ASTRA code audit — supplied local baseline

99.08 is another team's reported public score. It is not associated with these artifacts. The supplied LightGBM baseline is reproduced exactly: development macro F0.5 0.918449400834203, existing evaluation 0.9186750311113763. Both sets contain 20,000 S1 entities. Saved probabilities and fresh model predictions differ by exactly zero.

## Confirmed protections and validation

`experiments/astra_baseline` contains a separate copy of code, model, decision, reports, validation feature caches and scores, with SHA256 manifest. No existing model or output was overwritten. There are other processes developing this project; ASTRA experiments use separate directories.

Actual model-training, development and evaluation S1 sets are disjoint. No exact normalized name/address/country signature overlaps between the 60,000 training S1 sample and 20,000 development S1 sample. There are zero duplicate candidate triples in either scored holdout. Development positive labels were independently checked against the ground-truth pairs. Evaluation uses full truth denominators, including blocking misses, and correctly gives empty true singletons full credit.

This does not establish absence of all near-duplicate leakage. Shared names, addresses, and S2/S3 records require separate scrutiny. Corpus frequencies in blocking use the unlabeled retrieval pool; this is transductive retrieval, not supervised target leakage. Future fitted feature statistics will be trained on training data only. The existing `experiments.py` checks evaluation scores for every experiment; that set is consequently a reused evaluation set, not a pristine final holdout. New policy selection uses development data only.

## Actionable findings

1. **Blocking discards every record in oversized blocks.** `candidates_for_key` filters whole keys above cap=200; it does not retain the best 200. This also removes exact-name blocks. The rare-token blocker takes the rarest two tokens on both sides; a shared informative token can fall outside one side's top two. These mechanisms explain why retrieval needs independent fuzzy passes.
2. **Candidate recall is 90.7069%, not near 100%.** 6,422 of 69,105 development true pairs never enter the matcher; 5,280 additional true pairs are scored below 0.62. Blocking accounts for 54.88% of false-negative pairs. The candidate recall is a pair-recall ceiling, not numerically a macro-F0.5 ceiling.
3. **Current normalization is lossy.** Handwritten Indic romanization preserves more than the legacy ASCII deletion, but leaves substantial spelling differences. `build_cache.py` and `normalize_v2.basic_norm` implement different normalization paths. Unsupported scripts/punctuation can disappear. Preserve originals in future representations; avoid replacing caches in place.
4. **Missing-name flag is asymmetric.** `features_v2.compute` tests only the S1 name. Both-empty strings also produce exact-match and empty-token overlap evidence. Explicit both-side missingness should replace this only after retraining and ablation.
5. **Numeric conflict is overly coarse.** Any shared digit token suppresses the conflict flag, even if the building number differs; all numeric tokens are treated alike. Numeric agreement is evidence, not proof of identity. Postal extraction must remain data-derived rather than relying on external geography.
6. **Token frequency is occurrence frequency.** `blocking_rare.token_df` and `features_idf.build_idf` count repeated tokens within a record. They are not document frequencies despite their names. IDF also refits on the test pool, changing feature scale. Frozen training IDF should be compared empirically.
7. **Caches have no content/version keys.** `prepare_fold` names caches by sample size/cap only; changes to normalization, blocking, feature code, dataset or split can silently reuse stale features. ASTRA uses isolated versioned experiment paths and records input hashes.
8. **Existing inference overwrites output.** `inference_v2` opens files with mode `w`. It also recomputes full-pool token statistics per S1 batch and materializes joined features before chunking. New inference must write isolated versioned files with resumable bounded batches.
9. **Model handoff assumes fixed features.** `inference_v2` imports the constant 28-feature list instead of validating the feature schema against the model decision manifest; an IDF model cannot safely be dropped into it.
10. **Integer ID reconstruction is an assumption.** String IDs are stripped to integers and later reconstructed. Leading-zero IDs would not round-trip. Final output must use exact raw-ID mappings or verify every ID round-trips before inference.
11. **Official validator PASS is insufficient for the user's ten invariants.** ID existence is OFF unless `--check-ids` is passed; missing candidate files and matches outside candidates only produce warnings. A strict supplementary check is required. `check_outputs.read_map` hides duplicate S1 rows by dictionary overwrite.
12. **Top-1 policies are structurally wrong for most records.** Multiple true matches dominate. Experiments confirm no material gain from threshold/margin/top-1 changes alone. The best small contradiction filter improves development F0.5 by only 0.00002246; this is not enough evidence to promote it.

## Scope

Reviewed active cache, normalization, split, blocking, feature, training, evaluation, decision, inference, packaging and output-check code. Legacy implementation was inspected as historical context; its prior report is not treated as evidence about the current model. No business was looked up, and no external business data or labels were used.
