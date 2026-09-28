"""Precompute the pool side of every blocking key once per split.

``inference_v2.py`` called ``generate_candidates`` per Source-1 chunk, and each
call recomputed all twelve keys over the whole 10.3M-record candidate pool plus
two token-frequency passes. That pool-side work is identical for every chunk, so
it was being repeated nine times and dominated the runtime.

Here it is done once and written to parquet as (k, src, eid), already
validity-filtered and cap-filtered. Inference then only computes keys for its own
small Source-1 chunk and joins against these tables.
"""

import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking_rare import rarest_tokens, token_df  # noqa: E402
from blocking_v2 import KEYS, MIN_KEY_LEN, load_pool  # noqa: E402


def _cap_filter(right, cap):
    counts = right.group_by("k").len()
    return right.join(counts.filter(pl.col("len") <= cap).select("k"), on="k", how="semi")


def precompute(split, cap=200, force=False):
    outdir = CACHE / f"blocks_{split}_{cap}"
    outdir.mkdir(parents=True, exist_ok=True)
    done = outdir / "_COMPLETE"
    if done.exists() and not force:
        print(f"blocks for {split} already precomputed")
        return outdir

    pool = load_pool(split)
    print(f"pool={pool.height:,}")

    for key in KEYS:
        t0 = time.time()
        right = pool.select("src", "eid", KEYS[key]().alias("k")).filter(
            pl.col("k").is_not_null() & (pl.col("k").str.len_chars() >= MIN_KEY_LEN)
        )
        right = _cap_filter(right, cap)
        right.write_parquet(outdir / f"{key}.parquet", compression="zstd")
        print(f"  {key:20s} {right.height:>10,} rows ({time.time() - t0:.0f}s)")

    for label, col in (("rare_name", "name"), ("rare_addr", "addr")):
        t0 = time.time()
        dfreq = token_df(pool, col)
        dfreq.write_parquet(outdir / f"dfreq_{col}.parquet", compression="zstd")
        right = rarest_tokens(pool, dfreq, 2, col).select(
            "src", "eid", pl.col("t").alias("k")
        )
        right = _cap_filter(right, cap)
        right.write_parquet(outdir / f"{label}.parquet", compression="zstd")
        print(f"  {label:20s} {right.height:>10,} rows ({time.time() - t0:.0f}s)")

    done.write_text("ok")
    return outdir


if __name__ == "__main__":
    precompute(sys.argv[1] if len(sys.argv) > 1 else "test",
               int(sys.argv[2]) if len(sys.argv) > 2 else 200)
