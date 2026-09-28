"""Fit the LightGBM matcher and select a decision rule on dev only.

Protocol (this is the part the prior code got wrong):

* ``train`` fold fits the model.
* ``dev`` fold selects the threshold and the decision rule.
* ``eval`` fold is scored exactly once, at the end, with everything frozen.

The three folds are disjoint sets of Source-1 entities (hash-partitioned in
``split.py``), so no S1 entity and none of its candidate pairs can appear on
both sides of any decision.
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_v2 as E  # noqa: E402
from features_v2 import FEATURES  # noqa: E402
from prepare_fold import prepare  # noqa: E402

PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=100,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_estimators=800,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)


def fit(train_X, dev_X):
    """Fit on train, early-stop on dev. No class weighting: the positive rate is
    ~3%, and reweighting distorts the probabilities the threshold search reads."""
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        train_X.select(FEATURES).to_numpy(), train_X["label"].to_numpy(),
        eval_set=[(dev_X.select(FEATURES).to_numpy(), dev_X["label"].to_numpy())],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model


def score_fold(model, X, feats=None):
    feats = feats or FEATURES
    p = model.predict_proba(X.select(feats).to_numpy())[:, 1]
    return X.select("s1", "src", "eid", "label").with_columns(
        pl.Series("score", p.astype(np.float32))
    )


def main():
    t0 = time.time()
    train_X, _ = prepare("train", 60000)
    dev_X, dev_nt = prepare("dev", 20000)
    eval_X, eval_nt = prepare("eval", 20000)
    print(f"train={train_X.height:,} dev={dev_X.height:,} eval={eval_X.height:,}")

    model = fit(train_X, dev_X)
    print(f"best_iteration={model.best_iteration_}")

    fi = pl.DataFrame({
        "feature": FEATURES,
        "gain": model.booster_.feature_importance("gain"),
    }).sort("gain", descending=True)
    fi.write_csv(ROOT / "reports" / "claude_feature_importance.csv")
    print(fi.head(12))

    dev_scored = score_fold(model, dev_X)
    sweep = E.sweep(dev_scored, dev_nt)
    sweep.write_csv(ROOT / "reports" / "claude_threshold_analysis.csv")
    print(sweep.select("threshold", "macro_f05", "macro_precision", "macro_recall",
                       "singleton_accuracy", "empty_predictions"))

    best = sweep.sort("macro_f05", descending=True).row(0, named=True)
    thr = float(best["threshold"])
    print(f"\nDEV best threshold={thr}  macro_f05={best['macro_f05']:.4f}")

    # Refine around the winner, still on dev only.
    fine = E.sweep(dev_scored, dev_nt, np.round(np.arange(max(0.02, thr - 0.05),
                                                          min(0.99, thr + 0.05), 0.01), 3))
    best_fine = fine.sort("macro_f05", descending=True).row(0, named=True)
    if best_fine["macro_f05"] > best["macro_f05"]:
        thr, best = float(best_fine["threshold"]), best_fine
    print(f"DEV refined threshold={thr}  macro_f05={best['macro_f05']:.4f}")

    eval_scored = score_fold(model, eval_X)
    held = E.evaluate(eval_scored, eval_nt, thr)
    print("\nHELD-OUT EVAL (single evaluation, threshold frozen from dev):")
    for k, v in held.items():
        print(f"  {k:22s} {v}")

    model.booster_.save_model(str(ROOT / "models" / "matcher_lgbm.txt"))
    dev_scored.write_parquet(ROOT / "work" / "cache" / "dev_scored.parquet")
    eval_scored.write_parquet(ROOT / "work" / "cache" / "eval_scored.parquet")
    with open(ROOT / "models" / "decision.json", "w") as f:
        json.dump({"threshold": thr, "features": FEATURES,
                   "best_iteration": int(model.best_iteration_ or PARAMS["n_estimators"]),
                   "dev": best, "eval": held}, f, indent=2, default=float)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
