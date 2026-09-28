"""Decision rules beyond a single global threshold, compared on the dev fold.

Macro F0.5 makes abstention valuable: a true singleton scores 1.0 for predicting
nothing and 0.0 for one wrong guess, and 5.6% of Source-1 entities are true
singletons. So rules that can decline to answer are worth testing against a flat
threshold -- but only worth *keeping* if dev says so.

Every rule here is fitted on dev and reported on dev. The held-out eval fold is
scored once, in ``train_matcher.py`` / ``final_eval.py``, with the winner frozen.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_v2 as E  # noqa: E402


def _with_rank(scored):
    return scored.with_columns(
        pl.col("score").rank("ordinal", descending=True).over("s1").alias("rk"),
        pl.col("score").max().over("s1").alias("top"),
    )


def rule_global(scored, ntrue, t):
    return E.evaluate(scored, ntrue, t) | dict(rule="global", t1=t, t2=None)


def rule_top1(scored, ntrue, t):
    """Predict at most one match: the best candidate, if it clears ``t``."""
    r = _with_rank(scored).filter(pl.col("rk") == 1)
    return E.evaluate(r, ntrue, t) | dict(rule="top1", t1=t, t2=None)


def rule_margin(scored, ntrue, t1, t2):
    """Accept candidates over ``t1``, but only if the entity's best candidate
    stands clear of the runner-up by ``t2`` -- i.e. the entity is unambiguous."""
    s = _with_rank(scored)
    second = (
        s.filter(pl.col("rk") == 2).select("s1", pl.col("score").alias("second"))
    )
    s = s.join(second, on="s1", how="left").with_columns(pl.col("second").fill_null(0.0))
    keep = s.filter((pl.col("top") - pl.col("second")) >= t2)
    dropped = s.join(keep.select("s1").unique(), on="s1", how="anti")
    # Entities failing the margin test predict nothing at all.
    combined = pl.concat([keep.select(scored.columns),
                          dropped.select(scored.columns).with_columns(
                              pl.lit(-1.0, dtype=pl.Float32).alias("score"))])
    return E.evaluate(combined, ntrue, t1) | dict(rule="margin", t1=t1, t2=t2)


def rule_relative(scored, ntrue, t1, frac):
    """Accept candidates over ``t1`` that also reach ``frac`` of the entity's best
    score -- a per-entity adaptive floor rather than one global cut."""
    s = _with_rank(scored).filter(pl.col("score") >= pl.col("top") * frac)
    return E.evaluate(s, ntrue, t1) | dict(rule="relative", t1=t1, t2=frac)


def rule_topk(scored, ntrue, t, k):
    """Threshold, then keep at most the ``k`` highest-scoring survivors."""
    s = _with_rank(scored).filter(pl.col("rk") <= k)
    return E.evaluate(s, ntrue, t) | dict(rule=f"top{k}", t1=t, t2=None)


def compare(scored, ntrue, base_t):
    rows = []
    grid = np.round(np.arange(max(0.05, base_t - 0.2), min(0.96, base_t + 0.25), 0.05), 3)

    for t in grid:
        rows.append(rule_global(scored, ntrue, float(t)))
        rows.append(rule_top1(scored, ntrue, float(t)))
        for k in (2, 3, 5, 8):
            rows.append(rule_topk(scored, ntrue, float(t), k))
        for t2 in (0.1, 0.2, 0.3):
            rows.append(rule_margin(scored, ntrue, float(t), t2))
        for frac in (0.5, 0.7, 0.9):
            rows.append(rule_relative(scored, ntrue, float(t), frac))

    return pl.DataFrame(rows)


if __name__ == "__main__":
    import json

    with open(ROOT / "models" / "decision.json") as f:
        base_t = json.load(f)["threshold"]
    scored = pl.read_parquet(ROOT / "work" / "cache" / "dev_scored.parquet")
    ntrue = pl.read_parquet(ROOT / "work" / "cache" / "ntrue_dev_20000_200.parquet")

    df = compare(scored, ntrue, base_t).sort("macro_f05", descending=True)
    df.write_csv(ROOT / "reports" / "claude_decision_rules.csv")
    print(df.select("rule", "t1", "t2", "macro_f05", "macro_precision", "macro_recall",
                    "singleton_accuracy", "empty_predictions").head(20))
