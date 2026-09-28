"""Rare-token blocking, and its incremental value over the deterministic keys.

Prefix and full-string keys fail on exactly the noise this dataset is full of:
a leading junk token ("<< Team Ecole"), a dropped article, a reordered name.
A key built from a record's *rarest* tokens is invariant to all three, because
the discriminative token survives wherever it sits in the string.

Token document frequency is computed from the candidate pool of the split being
processed (train pool for validation, test pool for inference) -- never fitted
across the two, which would be a corpus-statistics leak.
"""

import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking_v2 import KEYS, candidates_for_key, load_pool, score  # noqa: E402


def token_df(pool, col="name"):
    """Document frequency of each token in the pool."""
    return (
        pool.select(pl.col(col).str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t").str.len_chars() >= 3)
        .group_by("t")
        .len()
        .rename({"len": "df"})
    )


def rarest_tokens(df_records, dfreq, k, col="name", id_cols=("src", "eid")):
    """For each record, its ``k`` least-frequent tokens, ordered by rarity."""
    exploded = (
        df_records.select(*id_cols, pl.col(col).str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t").str.len_chars() >= 3)
        .join(dfreq, on="t", how="inner")
        .sort("df", "t")
        .group_by(list(id_cols), maintain_order=True)
        .head(k)
    )
    return exploded


def rare_key_candidates(s1, pool, dfreq, cap, k=2, col="name", suffix=None):
    """Candidates linked by sharing one of each record's k rarest tokens."""
    left = rarest_tokens(s1, dfreq, k, col, id_cols=("eid",)).select(
        pl.col("eid").alias("s1"), pl.col("t").alias("k")
    )
    right = rarest_tokens(pool, dfreq, k, col).select("src", "eid", pl.col("t").alias("k"))

    if suffix is not None:
        left = left.join(s1.select(pl.col("eid").alias("s1"), suffix), on="s1").with_columns(
            (pl.col("k") + "|" + pl.col(suffix).cast(pl.Utf8)).alias("k")
        ).select("s1", "k")
        right = right.join(pool.select("src", "eid", suffix), on=["src", "eid"]).with_columns(
            (pl.col("k") + "|" + pl.col(suffix).cast(pl.Utf8)).alias("k")
        ).select("src", "eid", "k")

    counts = right.group_by("k").len()
    right = right.join(counts.filter(pl.col("len") <= cap).select("k"), on="k", how="semi")
    return left.join(right, on="k", how="inner").select("s1", "src", "eid").unique()


def main():
    from split import load_fold, load_truth

    n_s1 = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    s1 = load_fold("dev", n=n_s1)
    truth = load_truth(s1["eid"].to_list())
    pool = load_pool("train")
    print(f"dev S1={s1.height:,}  pool={pool.height:,}  true pairs={sum(len(v) for v in truth.values()):,}")

    # Deterministic union from blocking_v2, as the incremental baseline.
    base = None
    for key in KEYS:
        p = candidates_for_key(s1, pool, key, cap)
        base = p if base is None else pl.concat([base, p]).unique()
    r = score(base, truth, n_s1)
    print(f"  {'deterministic union':28s} recall={r['recall']:.4f} avg={r['avg_per_s1']:.1f} max={r['max_per_s1']:,}")

    rows = [dict(key="deterministic_union", **r)]

    dfreq_name = token_df(pool, "name")
    dfreq_addr = token_df(pool, "addr")

    variants = [
        ("rare_name_k1", dict(dfreq=dfreq_name, k=1, col="name", suffix=None)),
        ("rare_name_k2", dict(dfreq=dfreq_name, k=2, col="name", suffix=None)),
        ("rare_name_k3", dict(dfreq=dfreq_name, k=3, col="name", suffix=None)),
        ("rare_addr_k2", dict(dfreq=dfreq_addr, k=2, col="addr", suffix=None)),
    ]

    for name, kw in variants:
        t0 = time.time()
        p = rare_key_candidates(s1, pool, cap=cap, **kw)
        r_solo = score(p, truth, n_s1)
        merged = pl.concat([base, p]).unique()
        r_inc = score(merged, truth, n_s1)
        dt = round(time.time() - t0, 1)
        print(
            f"  {name:28s} solo={r_solo['recall']:.4f} avg={r_solo['avg_per_s1']:>6.1f} "
            f"|  +union={r_inc['recall']:.4f} avg={r_inc['avg_per_s1']:>6.1f} max={r_inc['max_per_s1']:,} ({dt}s)"
        )
        rows.append(dict(key=name + "_solo", **r_solo))
        rows.append(dict(key=name + "_plus_union", **r_inc))

    pl.DataFrame(rows).write_csv(ROOT / "reports" / "claude_blocking_rare.csv")


if __name__ == "__main__":
    main()
