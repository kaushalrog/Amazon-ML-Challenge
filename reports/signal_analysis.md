# Signal analysis — local baseline development set

Measured feature quantiles are in `astra_signal_distributions.csv`. Positives in that file are generated positives; blocked-out positives are separately exported in `astra_blocked_true_pairs.tsv` and must not be excluded from recall denominators. Hard negatives are actual model candidates with predicted score above 0.1, not arbitrary random Cartesian pairs.

The baseline produces 1,156 false-positive pairs, 5,280 generated-but-rejected true pairs, and 6,422 blocked-out true pairs. Missing address affects 2,053 erroneous entities; strong-name/weak-address evidence affects 1,497; numeric contradictions affect 1,074; strong-address/weak-name affects 486. Tags overlap. Country and source tags describe slices, not independent causes.

The top priority is recovering candidates, followed by discrimination among incomplete or contradictory records. Oracle interventions (`astra_error_loss.csv`) quantify possible gains, not achieved scores. Recovering every blocked true pair would raise development F0.5 to 0.95952 while retaining the baseline's other decisions. Recovering every rejected true pair gives 0.94921; removing all false merges gives 0.93239. These gains overlap and must not be added.

Zero-match entities score 0.93845; one-match entities 0.79426; two-match entities 0.89691; three-or-more 0.93128. This exposes a disproportionately weak one-match stratum. Detailed counts are in `astra_multimatch.csv`.

India development F0.5 is 0.89378 versus US 0.93505. Candidate recall is 0.88575 versus 0.92139. No France validation score can be claimed because labeled France records are not supplied. S2 and S3 metrics are separately computed over all development S1 entities, including per-source empty truths.

Candidate calibration bins are in `astra_calibration.csv`. They describe the candidate population, not the probability that an S1 is a singleton. Threshold/margin/top-1 sweeps have not yielded a material improvement. No policy was promoted from these small diagnostics.
