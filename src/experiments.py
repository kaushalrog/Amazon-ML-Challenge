"""Ablation runner: one experiment = one row of reports/claude_ablation.csv.

Each experiment retrains from scratch, selects its threshold on dev, and is
reported on the held-out eval fold. Nothing is kept unless eval macro F0.5
improves on the incumbent.
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_v2 as E  # noqa: E402
from features_v2 import FEATURES  # noqa: E402
from prepare_fold import prepare  # noqa: E402
from train_matcher import PARAMS, score_fold  # noqa: E402

ABLATION = ROOT / "reports" / "claude_ablation.csv"


def run(name, cap=200, hard_negative_weight=None, n_train=60000, notes="", use_idf=False):
    t0 = time.time()
    train_X, _ = prepare("train", n_train, cap, use_idf=use_idf)
    dev_X, dev_nt = prepare("dev", 20000, cap, use_idf=use_idf)
    eval_X, eval_nt = prepare("eval", 20000, cap, use_idf=use_idf)

    feats = [c for c in train_X.columns if c not in ("s1", "src", "eid", "label")]

    blocking_recall = float(dev_X["label"].sum()) / float(dev_nt["ntrue"].sum())

    weights = None
    if hard_negative_weight is not None:
        # Mine hard negatives with a first-pass model, then refit with those
        # negatives upweighted. Mining uses train-fold predictions only.
        first = lgb.LGBMClassifier(**PARAMS)
        first.fit(train_X.select(feats).to_numpy(), train_X["label"].to_numpy(),
                  eval_set=[(dev_X.select(feats).to_numpy(), dev_X["label"].to_numpy())],
                  eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        p = first.predict_proba(train_X.select(feats).to_numpy())[:, 1]
        lab = train_X["label"].to_numpy()
        hard = (lab == 0) & (p > 0.5)
        weights = np.ones(len(lab), dtype=np.float32)
        weights[hard] = hard_negative_weight
        print(f"  hard negatives mined: {int(hard.sum()):,} "
              f"({hard.sum() / max((lab == 0).sum(), 1):.3%} of negatives)")

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(train_X.select(feats).to_numpy(), train_X["label"].to_numpy(),
              sample_weight=weights,
              eval_set=[(dev_X.select(feats).to_numpy(), dev_X["label"].to_numpy())],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])

    dev_scored = score_fold(model, dev_X, feats)
    sweep = E.sweep(dev_scored, dev_nt)
    best = sweep.sort("macro_f05", descending=True).row(0, named=True)
    thr = float(best["threshold"])
    fine = E.sweep(dev_scored, dev_nt,
                   np.round(np.arange(max(0.02, thr - 0.05), min(0.99, thr + 0.05), 0.01), 3))
    bf = fine.sort("macro_f05", descending=True).row(0, named=True)
    if bf["macro_f05"] > best["macro_f05"]:
        thr, best = float(bf["threshold"]), bf

    held = E.evaluate(score_fold(model, eval_X, feats), eval_nt, thr)
    row = dict(
        experiment=name, blocking=f"union+rare cap={cap}", features=f"{len(feats)} vectorized" + (" + idf" if use_idf else ""),
        model="LightGBM", hard_negative_mining=hard_negative_weight is not None,
        embedding=False, graph_consistency=False, threshold=thr,
        candidate_recall=round(blocking_recall, 4),
        dev_macro_f05=round(best["macro_f05"], 4),
        precision=round(held["macro_precision"], 4), recall=round(held["macro_recall"], 4),
        macro_f05=round(held["macro_f05"], 4),
        singleton_accuracy=round(held["singleton_accuracy"], 4),
        false_positives=held["false_positives"], false_negatives=held["false_negatives"],
        avg_cands_per_s1=round(dev_X.height / 20000, 1),
        runtime_s=round(time.time() - t0, 1), notes=notes,
    )
    print(json.dumps({k: row[k] for k in
                      ("experiment", "candidate_recall", "threshold", "dev_macro_f05",
                       "macro_f05", "precision", "recall")}, indent=2))

    prev = pl.read_csv(ABLATION) if ABLATION.exists() else None
    new = pl.DataFrame([row])
    out = pl.concat([prev, new], how="diagonal_relaxed") if prev is not None else new
    out.write_csv(ABLATION)
    return row, model, thr


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "cap":
        run(f"blocking_cap_{sys.argv[2]}", cap=int(sys.argv[2]),
            notes="blocking cap sweep, end-to-end")
    elif which == "idf":
        run("idf_features", cap=int(sys.argv[2]) if len(sys.argv) > 2 else 200,
            use_idf=True, notes="IDF-weighted name/address overlap added")
    elif which == "hardneg":
        run(f"hardneg_w{sys.argv[2]}", cap=int(sys.argv[3]) if len(sys.argv) > 3 else 200,
            use_idf=len(sys.argv) > 4 and sys.argv[4] == "idf",
            hard_negative_weight=float(sys.argv[2]),
            notes="false positives from a first-pass model upweighted on refit")
