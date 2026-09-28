"""Fast, vectorized macro F0.5 and threshold sweep.

The metric matches ``src/validation.py`` (which was correct) but is computed
group-wise in polars instead of a Python ``groupby`` loop per threshold -- the
prior code re-looped over every S1 for each of 17 thresholds.

One subtlety that must not be lost: the recall denominator is the S1 entity's
*full* ground-truth match count, including true matches that blocking never
produced. Scoring only against candidates would silently credit the matcher for
recall that blocking already threw away.
"""

import numpy as np
import polars as pl


def macro_f05_from_counts(tp, npred, ntrue):
    """Per-entity F0.5 under the official rules, as numpy arrays."""
    tp = tp.astype(np.float64)
    npred = npred.astype(np.float64)
    ntrue = ntrue.astype(np.float64)

    precision = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    recall = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0)

    denom = 0.25 * precision + recall
    f05 = np.where(denom > 0, (1.25 * precision * recall) / np.maximum(denom, 1e-12), 0.0)

    # A true singleton scores 1.0 for predicting nothing, 0.0 for any prediction.
    singleton = ntrue == 0
    f05 = np.where(singleton, np.where(npred == 0, 1.0, 0.0), f05)
    precision = np.where(singleton, np.where(npred == 0, 1.0, 0.0), precision)
    recall = np.where(singleton, np.where(npred == 0, 1.0, 0.0), recall)
    return f05, precision, recall, singleton


def evaluate(scored, ntrue_df, threshold):
    """Evaluate one threshold. ``scored`` needs columns s1, label, score."""
    sel = scored.filter(pl.col("score") >= threshold)
    agg = sel.group_by("s1").agg(
        pl.col("label").sum().alias("tp"), pl.len().alias("npred")
    )
    j = ntrue_df.join(agg, on="s1", how="left").with_columns(
        pl.col("tp").fill_null(0), pl.col("npred").fill_null(0)
    )
    f05, p, r, singleton = macro_f05_from_counts(
        j["tp"].to_numpy(), j["npred"].to_numpy(), j["ntrue"].to_numpy()
    )
    tp_tot = int(j["tp"].sum())
    return dict(
        threshold=threshold,
        macro_f05=float(f05.mean()),
        macro_precision=float(p.mean()),
        macro_recall=float(r.mean()),
        singleton_accuracy=float(f05[singleton].mean()) if singleton.any() else float("nan"),
        n_entities=int(len(f05)),
        predicted=int(j["npred"].sum()),
        true_pairs=int(j["ntrue"].sum()),
        tp=tp_tot,
        false_positives=int(j["npred"].sum() - tp_tot),
        false_negatives=int(j["ntrue"].sum() - tp_tot),
        empty_predictions=int((j["npred"] == 0).sum()),
    )


def sweep(scored, ntrue_df, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.99, 0.05), 3)
    return pl.DataFrame([evaluate(scored, ntrue_df, float(t)) for t in thresholds])
