# Blocking Review

Protocol: 20,000 Source-1 entities sampled from the **dev** fold (disjoint from
train and eval by S1-id hash), matched against the **full** 10,320,219-record
train S2+S3 pool. 69,105 true pairs in the sample. All figures measured.

## Baseline being replaced

Gemini's blocker (`src/blocking.py`, `reports/blocking_ablation.csv`):

| candidate recall | candidates | avg/S1 | max/S1 |
|---|---|---|---|
| 0.7402 | 43,016,195 | 2,150.8 | 39,312 |

Weak on both axes at once. The dominant volume driver was `key_num_country`
(every digit in the address + country), a very low-entropy key, and no key had a
block-size cap, so one generic block could dump tens of thousands of candidates
onto a single S1.

## Individual keys (cap = 200 records per key value)

| key | recall | candidates | avg/S1 | max/S1 |
|---|---|---|---|---|
| name_pfx12_ctry | 0.5022 | 486,777 | 24.3 | 200 |
| name_tok1_addrnum | 0.4911 | 203,317 | 10.2 | 199 |
| name4_addr4 | 0.3327 | 51,248 | 2.6 | 199 |
| name_sorted | 0.3141 | 193,834 | 9.7 | 200 |
| name_compact | 0.2678 | 176,867 | 8.8 | 200 |
| addrnum_ctry | 0.2578 | 331,691 | 16.6 | 200 |
| name_full | 0.2571 | 173,981 | 8.7 | 200 |
| addr_sorted | 0.1174 | 9,952 | 0.5 | 17 |
| addr_compact | 0.0837 | 7,088 | 0.4 | 14 |
| addr_full | 0.0833 | 7,031 | 0.4 | 14 |
| **union of the above** | **0.8188** | 1,120,888 | 56.0 | 486 |

Exact-address keys are nearly worthless alone (recall < 0.12): addresses are
reformatted, reordered and abbreviated far more aggressively than names. They
are cheap enough (0.4 candidates/S1) to keep, but they are not the answer.

## Rare-token blocking

Prefix and full-string keys break on exactly the noise this dataset contains: a
junk prefix token (`<< Team Ecole`), a dropped article, a reordered name. A key
built from a record's *rarest* tokens is invariant to all three, since the
discriminative token survives wherever it appears in the string.

Token document frequency is computed from the candidate pool of the split being
processed — train pool for validation, test pool for inference — never pooled
across the two, which would be a corpus-statistics leak.

| variant | solo recall | solo avg/S1 | union with deterministic | avg/S1 | max/S1 |
|---|---|---|---|---|---|
| rare_name_k1 | 0.2889 | 28.8 | 0.8448 | 82.7 | 486 |
| rare_name_k2 | 0.2889 | 31.7 | 0.8504 | 85.7 | 486 |
| rare_name_k3 | 0.2864 | 31.3 | 0.8502 | 85.3 | 508 |
| **rare_addr_k2** | 0.4223 | 41.9 | **0.8871** | 91.9 | 601 |

`rare_addr_k2` is the single most valuable addition: +6.8pp over the
deterministic union for 36 extra candidates per S1. Note the inversion — exact
address keys are the *weakest* signals while rare address *tokens* are the
strongest. An address's discriminative content is a specific token (a street
name, a locality, a building number), not the whole reformatted string.
`k=3` does not beat `k=2`, so `k=2` is kept.

## Final blocker and cap selection

Final composition: the ten deterministic keys, plus `rare_name_k2`, plus
`rare_addr_k2`, unioned and deduplicated (`src/blocking_final.py`).

| cap | recall | candidates | avg/S1 | max/S1 | runtime |
|---|---|---|---|---|---|
| 100 | 0.8779 | 1,046,571 | 52.3 | 340 | 63s |
| **200** | **0.9071** | 2,428,761 | 121.4 | 638 | 47s |
| 400 | 0.9283 | 4,927,357 | 246.4 | 1,328 | 46s |

**cap = 200 selected.** Going to 400 buys +2.1pp recall for 2.0× the candidate
volume; at test scale that is the difference between ~210M and ~426M pairs to
featurize. cap=200 sits at the knee.

## Net change

| | Gemini | Claude | change |
|---|---|---|---|
| candidate recall | 0.7402 | **0.9071** | **+16.7pp** |
| avg candidates / S1 | 2,150.8 | **121.4** | **17.7× fewer** |
| max candidates / S1 | 39,312 | 638 | 62× fewer |

Better recall and far less work, simultaneously. Two changes account for it:
romanizing the nine Indic scripts instead of deleting them (see
`reports/claude_audit.md` §11.1), and replacing unbounded low-entropy digit
blocks with capped rare-token blocks.

## Remaining recall loss

9.3% of true pairs are still not generated at cap=200, and no matcher or
reranker can recover them. This is the ceiling on the whole system and the first
place to spend further effort (character n-gram TF-IDF top-K retrieval is the
natural next candidate, evaluated only if the matcher stops being the bottleneck).
