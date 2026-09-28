"""Vectorized pairwise features.

The prior implementation built one Python dict per pair and scored strings with
``difflib.SequenceMatcher``: measured at 24,107 pairs/sec single-core, which puts
test-set inference at ~43 core-hours of string comparison alone before any I/O.

Everything here is computed column-at-a-time instead: set operations in polars,
and string similarity through ``rapidfuzz.process.cpdist``, which runs the
comparison elementwise in parallel C++ (measured on this machine: 2.1M pairs/sec
for token_sort_ratio, 34M/sec for Jaro-Winkler, versus difflib's 24k/sec).
"""

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURES = [
    "name_exact", "name_jw", "name_tsr", "name_tset", "name_qratio",
    "name_jac", "name_common_tok", "name_len_ratio", "name_len_diff",
    "addr_exact", "addr_jw", "addr_tsr", "addr_tset",
    "addr_jac", "addr_common_tok", "addr_len_ratio",
    "num_jac", "num_common", "num_conflict", "num_missing",
    "country_eq", "is_s3", "name_missing", "addr_missing",
    "name_x_addr", "name_strong_addr_weak", "addr_strong_name_weak",
    "cand_count",
]


def _cp(a, b, scorer):
    return np.asarray(process.cpdist(a, b, scorer=scorer, workers=-1), dtype=np.float32)


def attach_records(pairs, s1, pool):
    """Join candidate pairs to both sides' normalized fields."""
    left = s1.select(
        pl.col("eid").alias("s1"),
        pl.col("name").alias("n1"),
        pl.col("addr").alias("a1"),
        pl.col("country").cast(pl.Utf8).alias("c1"),
    )
    right = pool.select(
        "src", "eid",
        pl.col("name").alias("n2"),
        pl.col("addr").alias("a2"),
        pl.col("country").cast(pl.Utf8).alias("c2"),
    )
    return pairs.join(left, on="s1", how="inner").join(right, on=["src", "eid"], how="inner")


def compute(df, idf_name=None, idf_addr=None):
    """Add the feature columns to a joined candidate frame."""
    # --- set-based features, vectorized in polars
    df = df.with_columns(
        pl.col("n1").str.split(" ").alias("nt1"),
        pl.col("n2").str.split(" ").alias("nt2"),
        pl.col("a1").str.split(" ").alias("at1"),
        pl.col("a2").str.split(" ").alias("at2"),
        pl.col("a1").str.extract_all(r"[0-9]+").alias("nm1"),
        pl.col("a2").str.extract_all(r"[0-9]+").alias("nm2"),
    )

    def jac(x, y, out):
        inter = pl.col(x).list.set_intersection(pl.col(y)).list.len()
        union = pl.col(x).list.set_union(pl.col(y)).list.len()
        return [
            pl.when(union > 0).then(inter / union).otherwise(0.0).cast(pl.Float32).alias(out),
            inter.cast(pl.Float32).alias(out + "_n"),
        ]

    df = df.with_columns(
        *jac("nt1", "nt2", "name_jac"),
        *jac("at1", "at2", "addr_jac"),
        *jac("nm1", "nm2", "num_jac"),
    ).rename({"name_jac_n": "name_common_tok", "addr_jac_n": "addr_common_tok",
              "num_jac_n": "num_common"})

    l1n, l2n = pl.col("n1").str.len_chars(), pl.col("n2").str.len_chars()
    l1a, l2a = pl.col("a1").str.len_chars(), pl.col("a2").str.len_chars()

    df = df.with_columns(
        (pl.col("n1") == pl.col("n2")).cast(pl.Float32).alias("name_exact"),
        (pl.col("a1") == pl.col("a2")).cast(pl.Float32).alias("addr_exact"),
        (pl.min_horizontal(l1n, l2n) / pl.max_horizontal(l1n, l2n).clip(1))
            .cast(pl.Float32).alias("name_len_ratio"),
        (l1n - l2n).abs().cast(pl.Float32).alias("name_len_diff"),
        (pl.min_horizontal(l1a, l2a) / pl.max_horizontal(l1a, l2a).clip(1))
            .cast(pl.Float32).alias("addr_len_ratio"),
        (pl.col("c1") == pl.col("c2")).cast(pl.Float32).alias("country_eq"),
        (pl.col("src") == 3).cast(pl.Float32).alias("is_s3"),
        (l1n == 0).cast(pl.Float32).alias("name_missing"),
        ((l1a == 0) | (l2a == 0)).cast(pl.Float32).alias("addr_missing"),
        # Both sides carry street numbers but share none: strong evidence against.
        ((pl.col("nm1").list.len() > 0) & (pl.col("nm2").list.len() > 0)
         & (pl.col("num_common") == 0)).cast(pl.Float32).alias("num_conflict"),
        ((pl.col("nm1").list.len() == 0) | (pl.col("nm2").list.len() == 0))
            .cast(pl.Float32).alias("num_missing"),
    )

    # --- string similarity, parallel C++
    n1, n2 = df["n1"].to_list(), df["n2"].to_list()
    a1, a2 = df["a1"].to_list(), df["a2"].to_list()
    df = df.with_columns(
        pl.Series("name_jw", _cp(n1, n2, JaroWinkler.normalized_similarity)),
        pl.Series("name_tsr", _cp(n1, n2, fuzz.token_sort_ratio) / 100.0),
        pl.Series("name_tset", _cp(n1, n2, fuzz.token_set_ratio) / 100.0),
        pl.Series("name_qratio", _cp(n1, n2, fuzz.QRatio) / 100.0),
        pl.Series("addr_jw", _cp(a1, a2, JaroWinkler.normalized_similarity)),
        pl.Series("addr_tsr", _cp(a1, a2, fuzz.token_sort_ratio) / 100.0),
        pl.Series("addr_tset", _cp(a1, a2, fuzz.token_set_ratio) / 100.0),
    )

    # --- cross-field evidence: the combinations that separate hard negatives
    df = df.with_columns(
        (pl.col("name_tsr") * pl.col("addr_tsr")).cast(pl.Float32).alias("name_x_addr"),
        ((pl.col("name_tsr") > 0.9) & (pl.col("addr_tsr") < 0.5))
            .cast(pl.Float32).alias("name_strong_addr_weak"),
        ((pl.col("addr_tsr") > 0.9) & (pl.col("name_tsr") < 0.5))
            .cast(pl.Float32).alias("addr_strong_name_weak"),
        pl.len().over("s1").cast(pl.Float32).alias("cand_count"),
    )

    extra = []
    if idf_name is not None:
        from features_idf import IDF_FEATURES, add_idf_features
        df = add_idf_features(df, idf_name, idf_addr)
        extra = IDF_FEATURES

    return df.select("s1", "src", "eid", *FEATURES, *extra)


def build(pairs, s1, pool, chunk=4_000_000, idf_name=None, idf_addr=None):
    """Feature table for a candidate set, computed in memory-bounded chunks."""
    joined = attach_records(pairs, s1, pool)
    if joined.height <= chunk:
        return compute(joined, idf_name, idf_addr)
    # Chunk on s1 so that cand_count (a per-s1 aggregate) stays correct.
    joined = joined.sort("s1")
    bounds, out, start = joined["s1"].to_numpy(), [], 0
    while start < len(bounds):
        stop = min(start + chunk, len(bounds))
        while stop < len(bounds) and bounds[stop] == bounds[stop - 1]:
            stop += 1
        out.append(compute(joined[start:stop], idf_name, idf_addr))
        start = stop
    return pl.concat(out)
