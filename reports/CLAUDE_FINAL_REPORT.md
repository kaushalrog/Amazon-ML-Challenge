# Claude Final Report — Amazon ML Challenge 2026, Business Entity Resolution

All figures measured on this machine (16 GB RAM, 10 cores). No metric in this
document is estimated, extrapolated, or carried over from the prior codebase.

Validation protocol: Source-1 entities are hash-partitioned into three disjoint
folds (60% train / 20% dev / 20% eval, seed 20260927). The matcher is fit on
train, the threshold and every decision rule are chosen on dev, and the eval fold
is scored exactly once per experiment with everything frozen. Candidate recall is
always measured against the *full* 10.3M-record Source-2+3 pool, and the recall
denominator is each entity's full ground-truth match count, including true
matches that blocking never generated.

## Current Gemini baseline

`model.py` and `inference.py` had never been executed (`models/`, `output/` and
`submissions/` were all empty), and could not have been: test inference required
~3.7 billion candidate pairs through `difflib` at a measured 24,107 pairs/sec,
i.e. ~43 core-hours single-threaded. So there was no model score to inherit. The
only measured end-to-end numbers were three deterministic baselines:

| baseline | macro F0.5 | precision | recall | singleton acc | candidate recall |
|---|---|---|---|---|---|
| A: exact normalized name | **0.3560** | 0.4517 | 0.2776 | 0.5987 | — |
| B: exact normalized address | 0.2032 | 0.2696 | 0.1313 | 0.9643 | — |
| C: exact name **and** address | 0.0880 | 0.1059 | 0.0705 | 1.0000 | — |
| Gemini blocking stage | — | — | — | — | **0.7402** |

Baseline A is the incumbent to beat. Note its shape: 204,549 false merges, and
baseline C — which never makes a wrong prediction at all — scores *worst*,
because it almost never predicts. The operating point has to sit between them.

## Claude improvements

**1. Fixed the normalizer (largest single win).** `unicode_normalize()` applied
`NFKD` then `encode('ascii','ignore')`, which deletes non-Latin scripts outright:
455,760 Source-2 names (9.05%) and 243,630 Source-3 names (4.61%) normalized to
the empty string, across nine Indic scripts. Source 1 is 100% Latin, so those
records lost their only join key — **~6.7% of all true match pairs**. Measured on
4,000 true pairs with a non-Latin Source-2 name, character-3-gram Jaccard against
the Latin Source-1 name:

| normalizer | true pairs | random pairs | true pairs > 0.2 |
|---|---|---|---|
| Gemini (ascii-strip) | **0.000** | 0.000 | **0.0%** |
| Claude (romanized) | 0.175 | 0.064 | 32.2% |

Zero signal, not weak signal. The fix exploits the ISCII-aligned Unicode layout:
Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam sit at
fixed offsets parallel to Devanagari, so one Devanagari table plus a block offset
romanizes all nine. The whole 2.4 GB corpus re-normalizes in 80 seconds with
**zero empty names**.

**2. Rebuilt blocking.** Replaced unbounded low-entropy digit blocks with ten
capped deterministic keys plus rare-token blocking on name and address. Every key
is capped at 200 pool records per value, so no single generic block can flood one
entity (Gemini's worst case was 39,312 candidates for one S1).

| | Gemini | Claude | change |
|---|---|---|---|
| candidate recall | 0.7402 | **0.9071** | **+16.7pp** |
| avg candidates / S1 | 2,150.8 | **121.4** | **17.7× fewer** |
| max candidates / S1 | 39,312 | 638 | 62× fewer |

Higher recall and far less work simultaneously. `rare_addr_k2` was the single
most valuable addition (+6.8pp for 36 extra candidates/S1). A notable inversion:
exact-address keys are the *weakest* signals (recall < 0.12) while rare address
*tokens* are the strongest — an address's discriminative content is a specific
token, not the reformatted whole string.

**3. Made the pipeline executable.** Normalized sources materialized once to
parquet with compact dtypes (ids stored as uint8 source + int64, not 10M Python
strings); `difflib` replaced with `rapidfuzz.process.cpdist` (parallel C++,
measured 2.1M–34M pairs/sec vs 24k); all features computed column-at-a-time.
End-to-end feature throughput 390,370 pairs/sec, a 16× improvement including
joins and set operations. Test inference runs in minutes rather than core-days.

**4. Fixed the validation protocol.** `create_grouped_split()` existed but was
never called — every stage used the *same* 20,000 entities to train,
cross-validate and tune the threshold, so no reported number was held out.
Replaced with the disjoint three-fold hash partition described above. Evidence it
works: dev 0.9184 vs held-out eval 0.9187, a 0.0003 gap.

## Experiments that did NOT earn their place

Recorded because negative results are results, and the brief asked for the
simpler system when complexity does not pay.

| experiment | dev macro F0.5 | verdict |
|---|---|---|
| global threshold @ 0.62 | **0.9184** | **kept** |
| relative floor (frac 0.5) | 0.9184 | degenerate — identical to global |
| top-8 cap | 0.9184 | degenerate — identical to global |
| relative floor (frac 0.7) | 0.9183 | rejected |
| top-1 only | worse | rejected |
| margin rules (0.1 / 0.2 / 0.3) | worse | rejected |
| blocking cap=400 | 0.9200 (eval 0.9202) | **rejected** — +0.0015 eval for 2× candidates and 2× inference cost |

Despite macro F0.5 rewarding abstention heavily (5.6% of Source-1 entities are
true singletons, scoring a free 1.0 for predicting nothing), **no adaptive
decision rule beat a single global threshold.** The final system uses the simple
threshold.

Not attempted, and honestly so: embedding rerankers and graph consistency. The
error decomposition showed the binding constraint is candidate generation
(type-A false negatives), which a reranker cannot address — it can only reorder
candidates that blocking already produced.

## Final system

| component | choice |
|---|---|
| normalization | NFKC + Latin diacritic folding + nine-script Indic romanization; alnum tokenization preserving digits |
| blocking | union of 10 capped deterministic keys + rare-name-k2 + rare-address-k2, cap = 200 pool records per key value |
| features | 28 vectorized pairwise features (name/address exact, Jaro-Winkler, token-sort, token-set, QRatio, token and numeric Jaccard, length ratios, number-conflict and missingness flags, cross-field evidence, candidate-set size) |
| model | LightGBM binary, 127 leaves, lr 0.05, early-stopped at iteration 212 on dev average-precision; **no class weighting** (positive rate 2.59%; reweighting distorts the probabilities the threshold reads) |
| decision logic | single global threshold, score ≥ 0.62, chosen on dev; empty prediction permitted and common |
| country handling | open-set throughout — country enters only as an equality feature and as a blocking-key component, never as an enumerated value, so France and any further unseen label pass through unchanged |

## Final validation (held-out eval fold, 20,000 entities, single evaluation)

| metric | value |
|---|---|
| **macro F0.5** | **0.9187** |
| macro precision | 0.9591 |
| macro recall | 0.8371 |
| singleton accuracy | 0.9289 |
| candidate recall | 0.9071 |
| false positives | 1,138 |
| false negatives | 11,554 |
| predicted matches | 58,503 |
| true pairs | 68,919 |
| empty predictions | 1,525 |

Versus Gemini's best measured baseline (0.3560), macro F0.5 improves by
**+0.5627 absolute (2.58×)**.

### Where the remaining loss is

| loss type | share of all true pairs | fix belongs in |
|---|---|---|
| type-A false negative (never blocked) | **9.29%** | candidate generation |
| type-B false negative (blocked, then rejected) | 7.64% | normalization / features / model / threshold |

Within type B, 39.6% are missing-address cases — where one side has no address at
all and name evidence alone must carry the decision. False positives are already
a minor term (1,156 on dev against 58,559 predictions), which is why further
effort belongs in recall, not precision.

### Ceiling analysis — why 0.986 is not reachable from here

Measured with an oracle matcher that accepts exactly the true pairs among the
generated candidates and never makes an error:

| candidate set | blocking recall | **oracle** macro F0.5 |
|---|---|---|
| cap = 200 | 0.9071 | **0.9626** |
| cap = 400 | 0.9294 | **0.9726** |

A *perfect* matcher on these candidate sets cannot exceed 0.9726. Any target
above that is bounded by candidate generation, not by the model — reaching ~0.986
would require candidate recall around 0.98, which means a different retrieval
stage (character n-gram TF-IDF top-K over the full 10.3M pool) rather than any
change to the matcher, features or threshold. The current system realizes 95.4%
of its achievable oracle score (0.9187 / 0.9626).

## Reproduction

```bash
venv/bin/python src/build_cache.py      # normalize all six sources -> parquet (~80s)
venv/bin/python src/split.py            # hash-partition S1 folds + explode ground truth
venv/bin/python src/prepare_fold.py     # candidates + features + labels per fold
venv/bin/python src/train_matcher.py    # fit, tune threshold on dev, score eval once
venv/bin/python src/decision.py         # decision-rule comparison (dev only)
venv/bin/python src/error_analysis.py   # FP/FN categorization
venv/bin/python src/inference_v2.py     # test inference -> output/
venv/bin/python src/check_outputs.py    # independent output re-derivation
```
