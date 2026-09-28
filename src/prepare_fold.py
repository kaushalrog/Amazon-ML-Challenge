"""Materialize candidates + features + labels for one fold, cached to parquet."""

import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import features_v2 as F  # noqa: E402
from blocking_final import generate_candidates  # noqa: E402
from blocking_v2 import load_pool  # noqa: E402
from split import load_fold  # noqa: E402


def prepare(fold, n_s1, cap=200, force=False, use_idf=False):
    tag = f"{fold}_{n_s1}_{cap}" + ("_idf" if use_idf else "")
    fx = CACHE / f"feat_{tag}.parquet"
    nt = CACHE / f"ntrue_{tag}.parquet"
    if fx.exists() and nt.exists() and not force:
        return pl.read_parquet(fx), pl.read_parquet(nt)

    t0 = time.time()
    s1 = load_fold(fold, n=n_s1)
    pool = load_pool("train")
    cands = generate_candidates(s1, pool, cap, verbose=False)
    print(f"[{tag}] candidates={cands.height:,} ({time.time() - t0:.0f}s)")

    idf_n = idf_a = None
    if use_idf:
        from features_idf import build_idf
        idf_n, idf_a = build_idf(pool, "name"), build_idf(pool, "addr")
    X = F.build(cands, s1, pool, idf_name=idf_n, idf_addr=idf_a)

    gt = pl.read_parquet(CACHE / "gt_pairs.parquet").join(
        s1.select(pl.col("eid").alias("s1")), on="s1", how="semi"
    )
    X = X.join(
        gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("label")),
        on=["s1", "src", "eid"], how="left",
    ).with_columns(pl.col("label").fill_null(0))

    ntrue = s1.select(pl.col("eid").alias("s1")).join(
        gt.group_by("s1").len().rename({"len": "ntrue"}), on="s1", how="left"
    ).with_columns(pl.col("ntrue").fill_null(0))

    X.write_parquet(fx, compression="zstd")
    ntrue.write_parquet(nt, compression="zstd")
    hits = int(X["label"].sum())
    print(
        f"[{tag}] pairs={X.height:,} positives={hits:,} ({hits / X.height:.2%}) "
        f"blocking_recall={hits / int(ntrue['ntrue'].sum()):.4f} total={time.time() - t0:.0f}s"
    )
    return X, ntrue


if __name__ == "__main__":
    for fold, n in (("train", 60000), ("dev", 20000), ("eval", 20000)):
        prepare(fold, n)
