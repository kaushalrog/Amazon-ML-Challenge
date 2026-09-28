"""Test-set inference, chunked over Source-1 so peak memory stays bounded.

The prior ``inference.py`` would have had to hold a Python dict-of-sets over
~3.7 billion candidate ids and then featurize them one pair at a time. Here the
1.73M test S1 entities are processed in chunks: each chunk is blocked,
featurized, scored and thresholded, then appended to both output files, so peak
memory is set by the chunk size and the candidate pool, not by the test set.

``candidate_pairs.tsv`` is written from the *same* candidate object that is
featurized and scored, so the final matches are a subset of the candidates by
construction rather than by a later reconciliation step.

Country is never special-cased: it enters only as the open-set equality feature
``country_eq`` and as a component of blocking keys, so France (absent from
training) and any further unseen label flow through unchanged.
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

import features_v2 as F  # noqa: E402
from blocking_final import generate_candidates  # noqa: E402
from blocking_v2 import load_pool  # noqa: E402
from features_v2 import FEATURES  # noqa: E402

CHUNK = 200_000


def fmt_ids(src, eid):
    return pl.concat_str([pl.lit("S"), pl.col(src).cast(pl.Utf8), pl.lit("-"),
                          pl.col(eid).cast(pl.Utf8)])


def main():
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else CHUNK
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    with open(ROOT / "models" / "decision.json") as f:
        decision = json.load(f)
    threshold = decision["threshold"]
    print(f"threshold={threshold}")

    model = lgb.Booster(model_file=str(ROOT / "models" / "matcher_lgbm.txt"))
    s1_all = pl.read_parquet(CACHE / "test_source1.parquet")
    pool = load_pool("test")
    print(f"test S1={s1_all.height:,}  pool={pool.height:,}")

    mpath, cpath = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    mf = open(mpath, "w", encoding="utf-8")
    cf = open(cpath, "w", encoding="utf-8")
    mf.write("source1_entity_id\tmatched_entity_ids\n")
    cf.write("source1_entity_id\tcandidate_entity_ids\n")

    stats = dict(cands=0, preds=0, empty=0, s2=0, s3=0, rows=0, max_cands=0)
    t_start = time.time()

    for start in range(0, s1_all.height, chunk_size):
        t0 = time.time()
        s1 = s1_all[start:start + chunk_size]
        cands = generate_candidates(s1, pool, cap=200, verbose=False)

        X = F.build(cands, s1, pool)
        scores = model.predict(X.select(FEATURES).to_numpy())
        X = X.select("s1", "src", "eid").with_columns(
            pl.Series("score", scores.astype(np.float32))
        )

        cand_lists = (
            X.with_columns(fmt_ids("src", "eid").alias("cid"))
            .group_by("s1").agg(pl.col("cid"))
        )
        match_lists = (
            X.filter(pl.col("score") >= threshold)
            .with_columns(fmt_ids("src", "eid").alias("cid"))
            .group_by("s1").agg(pl.col("cid"), pl.col("src"))
        )

        base = s1.select(pl.col("eid").alias("s1"))
        cj = base.join(cand_lists, on="s1", how="left")
        mj = base.join(match_lists, on="s1", how="left")

        for s1id, ids in zip(cj["s1"], cj["cid"]):
            lst = ids.to_list() if ids is not None else []
            stats["cands"] += len(lst)
            stats["max_cands"] = max(stats["max_cands"], len(lst))
            cf.write(f"S1-{s1id}\t{','.join(lst)}\n")

        for s1id, ids, srcs in zip(mj["s1"], mj["cid"], mj["src"]):
            lst = ids.to_list() if ids is not None else []
            if lst:
                sl = srcs.to_list()
                stats["s2"] += sum(1 for s in sl if s == 2)
                stats["s3"] += sum(1 for s in sl if s == 3)
            else:
                stats["empty"] += 1
            stats["preds"] += len(lst)
            stats["rows"] += 1
            mf.write(f"S1-{s1id}\t{','.join(lst)}\n")

        done = min(start + chunk_size, s1_all.height)
        print(
            f"  {done:>9,}/{s1_all.height:,}  cands={cands.height:>10,}  "
            f"preds={stats['preds']:>9,}  ({time.time() - t0:.0f}s)"
        )

    mf.close()
    cf.close()
    stats["runtime_s"] = round(time.time() - t_start, 1)
    stats["threshold"] = threshold
    stats["avg_cands_per_s1"] = stats["cands"] / max(stats["rows"], 1)
    with open(ROOT / "reports" / "test_inference_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
