# Claude Audit — Amazon ML Challenge 2026, Business Entity Resolution

Audit of the pre-existing implementation (authored by a prior AI engineer).
Every number below is measured on this machine, not estimated from reading code.
Machine: 16 GB RAM, 10 cores, 101 GB free disk. Python 3.13, polars 1.44.2,
lightgbm 4.7.0.

## 0. Headline

The repository contains a *skeleton*, not a working pipeline. Two of five stages
have ever been executed. The three that have not (`model.py`, `inference.py`, and
therefore any submission) **cannot** execute at the data's real scale, and the
normalizer at the base of everything silently destroys ~6.7% of all true matches.

| Stage | File | Ever run? | Evidence |
|---|---|---|---|
| Data forensics | `src/data_inspection.py` | yes | `reports/dataset_stats.json` |
| Deterministic baseline | `src/baseline.py` | yes | `reports/baseline_results.csv` |
| Blocking eval | `src/blocking.py` | yes | `reports/blocking_ablation.csv` |
| Model training | `src/model.py` | **no** | `models/` empty; no `feature_importance.csv`, no `threshold_search.csv`, no `decision_analysis.md` |
| Test inference | `src/inference.py` | **no** | `output/` empty |
| Submission | — | **no** | `submissions/` empty |

## 1. Current architecture

```
TSV -> pandas (via polars read_csv -> to_pandas)
    -> normalize_df()           row-wise .apply, pure Python
    -> generate_candidate_pairs()  6 blocking keys, pandas merge + dict-of-sets
    -> extract_features()       per-pair Python dict, difflib.SequenceMatcher
    -> LightGBM GroupKFold(5)   OOF probabilities
    -> optimize_threshold()     grid 0.10..0.90 on the same OOF predictions
    -> inference.py             same path over the full test set
```

There is no caching layer, no intermediate materialization, and no parallelism.
Every stage re-reads and re-normalizes ~2.4 GB of TSV from scratch.

## 2. Current data flow and scale

| File | Rows |
|---|---|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |

Ground truth: 2,206,821 S1 entities — 123,247 with zero matches (5.6%),
119,157 with exactly one, 1,964,417 with multiple; mean 3.46 matches.

Countries: train = {US, India}; test = {US, India, **France**}.

## 3. Current blocking strategy

Six keys, unioned (`src/blocking.py:39`): `key_norm_name`, `key_norm_addr`,
`key_compact_name`, `key_name5_addr5` (6-char name prefix + 6-char address
prefix), `key_num_country` (**all digits in the address** + country), and
`key_token1_num` (first name token + all address digits).

Measured (`reports/blocking_ablation.csv`, 20k sampled S1 against the full
10.3M-record S2+S3 pool):

| metric | value |
|---|---|
| candidate recall | **0.7402** |
| candidates | 43,016,195 |
| avg candidates / S1 | 2,150.8 |
| max candidates / S1 | 39,312 |

This is bad on *both* axes simultaneously: a quarter of all true matches are
already unrecoverable after blocking, while the survivors are buried in ~2,151
candidates each. `key_num_country` is the main volume driver — the digit string
of an address is extremely low-entropy (an address containing only "12" blocks
against every other address containing only "12" in the same country).

## 4. Current feature engineering

Nine features (`src/features.py`): name exact/`difflib` ratio/token-Jaccard/length
difference, the same four for address, country equality, a `both_exact` product,
and two missingness flags.

Gaps: no character n-gram similarity, no numeric-token agreement, no postal/PIN
comparison, no house-number conflict signal, no token-sort or token-set variants,
no prefix/suffix similarity, no source indicator. Given that the *only* fuzzy
signals are `difflib` ratios, the model has very little to separate
"same name, different address" from a true match — exactly the hard-negative
class that F0.5 punishes hardest.

## 5. Current model

LightGBM, 500 trees, lr 0.05, `class_weight` set to the full negative/positive
ratio, `GroupKFold(n_splits=5)` grouped on the S1 id. Grouping is correct in
principle. It has never been fit.

## 6. Current validation methodology — and why no score from it is trustworthy

`src/validation.py:evaluate_macro_f05` implements the official metric correctly:
per-S1 precision/recall, `F0.5 = 1.25PR/(0.25P+R)`, empty-truth-and-empty-
prediction scores 1.0, and the mean is taken across S1 entities. That part is sound.

The *protocol* around it is not:

- **No held-out set exists.** `create_grouped_split()` (`src/validation.py:80`) is
  written but never called by any other module — it is dead code. Instead every
  stage calls `load_split_data()`, which samples the *same* 20,000 S1 entities
  with `random_state=42` and uses them as training data, cross-validation data,
  and threshold-tuning data simultaneously.
- **Threshold-tuning leakage.** `optimize_threshold()` picks the best of 17
  thresholds on exactly the OOF predictions it just produced, and that best value
  is reported as the system's score. With ~43M pairs the selection noise is small,
  but the reported figure is still a maximum-over-thresholds on the tuning set, not
  a held-out estimate.
- **Sampling bias.** 20,000 of 2.2M S1 entities (0.9%) is a thin but adequate
  sample; the problem is that it is never *partitioned*.

Verdict: the metric function can be kept; the surrounding protocol must be
replaced before any score is believed.

## 7. Current F0.5

No model score exists. The only measured end-to-end numbers are the three
deterministic baselines (`reports/baseline_results.csv`), reproduced here:

| baseline | macro F0.5 | precision | recall | singleton acc | false merges | missed |
|---|---|---|---|---|---|---|
| A: exact normalized **name** | 0.3560 | 0.4517 | 0.2776 | 0.5987 | 204,549 | 51,492 |
| B: exact normalized **address** | 0.2032 | 0.2696 | 0.1313 | 0.9643 | 1,320 | 63,721 |
| C: exact name **and** address | 0.0880 | 0.1059 | 0.0705 | 1.0000 | 0 | 68,418 |

The shape here is informative: A has 155× the false merges of B, and C — which
never produces a wrong answer at all — scores worst, because it almost never
produces an answer. The operating point must sit between A and B.

## 8. Current candidate recall

0.7402 (see §3). **This is the binding constraint on the whole system**: no
matcher, threshold or reranker can score above it.

## 9. Current runtime, and the two stages that cannot finish

Measured: `extract_features` runs at **24,107 pairs/sec** single-core.

- Training (`model.py`): 43.0M candidate pairs ≈ **0.5 core-hours** of feature
  work. Tolerable on its own.
- Inference (`inference.py`): 1,732,544 test S1 × 2,151 candidates ≈
  **3.73 billion pairs ≈ 43 core-hours** of `difflib` alone, single-threaded,
  before any I/O. Not viable.

Both are made far worse by `cand_df_idx.loc[c]` being executed *once per pair*
(`src/model.py:31`, `src/inference.py:38`) — a pandas label lookup against a
10.3M-row string index, returning a fresh Series each time.

## 10. Memory concerns

On this 16 GB machine, several steps are expected to fail or thrash:

- `pd.concat([s2_df, s3_df])` materializes 10.3M rows of Python string objects.
- `generate_candidate_pairs` builds a Python `dict` of `set`s holding all 43M
  candidate ids as interned strings — on the order of 10 GB for the training
  sample, and hundreds of GB at test scale.
- `records.append(rec)` accumulates one Python dict per pair, then
  `pd.DataFrame(records)` doubles it.

## 11. Bugs and suspicious logic

1. **FATAL — `unicode_normalize()` deletes non-Latin scripts.** It applies
   `NFKD` then `.encode('ascii','ignore')`. Measured effect:

   | file | rows whose name normalizes to empty |
   |---|---|
   | `train_source2` | 455,760 (9.05%) |
   | `train_source3` | 243,630 (4.61%) |
   | `test_source2` | 526,050 (10.76%) |
   | `test_source3` | 280,779 (5.52%) |

   Source 1 is 100% Latin, so these records lose their only strong join key.
   Among *true matches* specifically: 8.92% of matched S2 rows and 4.60% of
   matched S3 rows — **~6.7% of all true match pairs**. Nine Indic scripts are
   involved (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam,
   Oriya, Gurmukhi).

   I measured the information actually destroyed, on 4,000 true pairs whose S2
   name is non-Latin, using character-3-gram Jaccard against the Latin S1 name:

   | normalizer | true-pair mean | random-pair mean | true pairs > 0.2 |
   |---|---|---|---|
   | current (ascii-strip) | **0.000** | 0.000 | **0.0%** |
   | romanized (`src/normalize_v2.py`) | 0.175 | 0.064 | 32.2% |

   The current code has literally zero signal on these pairs. Romanization is
   not a refinement here, it is the difference between a usable and an unusable
   feature.

2. **`normalize_legal_suffixes()` is dead.** It is computed into the
   `legal_suffix_normalized` representation, which nothing consumes; the only
   normalizer actually used anywhere is `normalize_df()`, which does not call it.
   Its `" co "` -> `" company "` rule would also corrupt "CO" as the state
   Colorado.
3. **`get_name_representations` / `get_address_representations` are dead.**
   Ten carefully-built representations each; no caller. All matching uses the
   single `norm_name` / `norm_address` pair.
4. **Threshold loaded by scraping markdown.** `inference.py:64` parses
   `reports/decision_analysis.md` line-by-line for `"Best threshold:"`. If the
   file is absent or reworded, `best_th` is never bound and the run dies with
   `NameError` *after* the expensive stages.
5. **Only fold 0's model is saved** (`model.py:137`) while the reported score
   comes from the 5-fold OOF ensemble — the saved artifact does not correspond
   to the measured score.
6. `float(line.strip().split()[-1])` on a threshold written by `np.arange` can
   reintroduce float noise; minor, but the whole handoff should be JSON.
7. `evaluate_macro_f05` appends precision = recall = 1.0 for correctly-predicted
   singletons. The F0.5 figure is right, but the reported macro *precision* and
   *recall* are inflated by 5.6% of rows scoring a free 1.0. Reporting quirk, not
   a scoring bug — but it must not be compared against pair-level precision.
8. `key_num_country` and `key_token1_num` reuse `extract_numeric` over the *raw*
   address, so they include ZIP codes, unit numbers and street numbers
   indiscriminately in one unordered digit string.

## 12. Data leakage risks

- **Threshold leakage: present** (§6).
- **Model-selection leakage: present** — no split is held out at all.
- **Pair-level leakage: absent** — grouping by S1 in `GroupKFold` is correct, and
  candidate pairs are never split randomly.
- **Normalization leakage: absent** — normalization is stateless and row-local,
  with no corpus-level fitted statistics. (This will need re-checking once
  TF-IDF is introduced: an IDF fitted on train+test is a real risk there.)
- **Target leakage: absent** — no feature reads the label.
- **Duplicate leakage: low** — 0 duplicate entity ids in any source, though
  667,592 duplicate *names* in S1 mean near-duplicate S1 entities can straddle a
  split. Grouping is on the id, so this is a mild, acceptable residual.

## 13. Thresholding weaknesses

- The grid `np.arange(0.1, 0.95, 0.05)` is applied to probabilities produced with
  `class_weight={0:1, 1:pos_weight}`. At a ~0.1% positive rate `pos_weight` is
  ~1000×, which pushes calibrated probability mass towards 1 and makes a uniform
  grid a poor parameterization. No calibration step exists.
- A single global threshold is used. Nothing conditions on country, on source
  (S2 vs S3), or on the candidate-set size — all of which shift the operating
  point, and France is unseen at training time.
- No top-1 / margin / conservative-singleton rule is implemented, despite the
  metric rewarding empty predictions heavily (5.6% of S1 entities score a free
  1.0 for predicting nothing, and 0.0 for one wrong guess).
- `optimize_threshold` re-runs a Python `groupby` loop over every S1 for each of
  17 thresholds — 17 full passes where one sort would do.

## 14. False-positive risks (the metric's dominant term)

F0.5 weights precision 4:1, and baseline A already shows 204,549 false merges.
The structural sources visible in the data:

- 667,592 duplicate business names in S1 alone — chains and generic names
  ("Prime Money") make exact-name blocking actively dangerous.
- `key_num_country` merges businesses sharing only a digit string and a country.
- Shared-building addresses: 76,215 duplicate addresses in S1.
- Romanization collisions (a benefit of my fix that must be watched): retroflex
  and dental consonants both fold to `t`/`d`, so distinct names can converge.
- Missing addresses (168,967 in S2, 175,916 in S3) fall back to name-only
  evidence, where `missing_addr` is the only flag distinguishing the two cases.

## 15. False-negative risks

Currently dominated by **type A (blocked out)**: 25.98% of true matches never
reach the matcher. Within that, the erased non-Latin names (§11.1) are the
single largest identifiable cause. Type B cannot be measured yet because no
matcher has been fit.

## 16. Highest-value improvements, in priority order

1. **Fix normalization (largest single win).** Romanize the nine Indic scripts
   instead of deleting them; keep multiple representations. Directly addresses
   ~6.7% of all true matches, which currently carry zero signal.
2. **Rebuild blocking around character n-gram TF-IDF top-K retrieval** on name
   and address, replacing the low-entropy digit blocks. Target: recall well above
   0.74 at *far* fewer than 2,151 candidates/S1.
3. **Re-engineer the execution layer.** Materialize normalized sources to
   parquet once; do candidate generation and feature computation as vectorized
   polars/scipy-sparse operations over integer ids, never per-pair Python. This
   is what makes test inference finish at all.
4. **Establish a leakage-safe protocol**: a grouped S1 split into train / dev
   (threshold and model selection) / held-out eval, fixed seed, with the
   held-out set touched exactly once per reported number.
5. **Expand features** to character n-gram cosines, numeric/postal agreement, and
   explicit number-conflict flags — the signals that separate hard negatives.
6. **Hard-negative mining** targeted at same-name/different-address pairs.
7. **Decision logic beyond a global threshold**: per-source thresholds and an
   explicit abstention rule, tuned on dev only.

## 17. What to keep

- `evaluate_macro_f05` — a correct implementation of the official metric.
- The `GroupKFold`-on-S1 instinct in `model.py`.
- The deterministic baselines as a genuine floor to beat.
- The dataset forensics in `reports/`.
- LightGBM as the matcher; there is no evidence yet that anything heavier is needed.
