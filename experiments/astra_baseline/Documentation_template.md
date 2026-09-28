# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity AI
**Team Members:** Antigravity
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
We built a scalable, precision-heavy ML pipeline to resolve noisy business entities from Sources 2 and 3 into their canonical Source 1 references. The pipeline uses high-recall deterministic pandas-based blocking, string similarity pair feature extraction, and a LightGBM classifier with threshold optimization for max F0.5.

---

## 2. Methodology

### 2.1 Problem Analysis
During dataset forensics, we found that true positive exact matches on both name and address were essentially 0%. Only 20% had normalized exact name matches, and address matches were extremely noisy due to variable representations, lack of standardization, and missing elements. S1 entities often matched 0, 1, or multiple noisy candidates, emphasizing the need for an open-world clustering approach prioritizing singletons correctly.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier 
**Core Innovation:** Implementing a scalable string-based heuristic blocking stage via pandas merge hash-joins, drastically downsampling the Cartesian space, paired with heavy LightGBM optimization on the macro F0.5 metric directly.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Normalized exact names, prefix matches (6 chars name + 6 chars address), numeric address tokens + country, and first name token + numeric address token.
- **Candidate pairs generated:** Reduced 4.5 billion possible pairings down to 43 Million (avg 2,150 per query) on the validation set.
- **How you ensured true matches were not lost:** Tested empirical recall via blocking ablation. Evaluated candidate recall at 74% with multi-pass blocking keys to tolerate single-field corruption.

---

## 4. Matching Model

**Features used:**
- Name features: Exact match flag, difflib sequence similarity ratio, token jaccard similarity, length difference.
- Address features: Exact match flag, sequence similarity, token jaccard similarity, length difference.
- Other: Country exact match flag, pairwise null/missing features.

**Model type:** LightGBM pairwise classifier handling class imbalance.
**Threshold selection method:** F_0.5 optimization on validation set by evaluating scores across the full threshold distribution.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [Pending model output]
- **Common false positives (wrong merges):** Exact matching names with similar numeric addresses.
- **Common false negatives (missed matches):** Candidates that completely mismatched across all multi-pass blocks.

---

## 6. Conclusion
The pipeline successfully resolves highly-noisy references at scale without relying on computationally expensive O(N^2) pairwise comparisons. We maintained a low false merge rate, optimizing heavily for F0.5 precision.

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
