"""IDF-weighted overlap features.

The false-positive analysis showed the model being misled by shared boilerplate:
"private limited" (and its romanized form "praaiveta limiteda"), "enterprises",
"services", street words like "road" and "no". Plain token Jaccard treats those
exactly like a distinctive company name, so two unrelated Indian firms can reach
high name similarity on legal suffixes alone.

Rather than hand-maintaining a stopword list -- which would not transfer to
France, an unseen country at training time -- overlap is weighted by inverse
document frequency measured on the pool being processed. Boilerplate earns a low
weight because it is common, distinctive tokens dominate, and the whole thing is
open-set by construction.

IDF is fitted on the split's own pool (train pool for validation, test pool for
inference), never across the two.
"""

import numpy as np
import polars as pl

IDF_FEATURES = ["name_idf_jac", "name_idf_cov", "addr_idf_jac", "addr_idf_cov",
                "name_max_shared_idf"]


def build_idf(pool, col):
    """Smoothed IDF per token over the candidate pool."""
    n = pool.height
    return (
        pool.select(pl.col(col).str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t").str.len_chars() > 0)
        .group_by("t")
        .len()
        .with_columns((np.log(n) - pl.col("len").log()).cast(pl.Float32).alias("idf"))
        .select("t", "idf")
    )


def _idf_sums(df, list_col, idf, out):
    """Sum of token IDF over a list column, per row."""
    return (
        df.select(pl.col("_i"), pl.col(list_col).alias("t"))
        .explode("t")
        .join(idf, on="t", how="left")
        .with_columns(pl.col("idf").fill_null(0.0))
        .group_by("_i")
        .agg(pl.col("idf").sum().alias(out), pl.col("idf").max().alias(out + "_max"))
    )


def add_idf_features(df, idf_name, idf_addr):
    """Attach IDF-weighted Jaccard and coverage for name and address."""
    df = df.with_row_index("_i").with_columns(
        pl.col("n1").str.split(" ").alias("_nt1"),
        pl.col("n2").str.split(" ").alias("_nt2"),
        pl.col("a1").str.split(" ").alias("_at1"),
        pl.col("a2").str.split(" ").alias("_at2"),
    ).with_columns(
        pl.col("_nt1").list.set_intersection(pl.col("_nt2")).alias("_ni"),
        pl.col("_nt1").list.set_union(pl.col("_nt2")).alias("_nu"),
        pl.col("_at1").list.set_intersection(pl.col("_at2")).alias("_ai"),
        pl.col("_at1").list.set_union(pl.col("_at2")).alias("_au"),
    )

    for lc, out, idf in (("_ni", "ni", idf_name), ("_nu", "nu", idf_name),
                         ("_nt1", "n1s", idf_name),
                         ("_ai", "ai", idf_addr), ("_au", "au", idf_addr),
                         ("_at1", "a1s", idf_addr)):
        df = df.join(_idf_sums(df, lc, idf, out), on="_i", how="left")

    df = df.with_columns(
        # Jaccard: shared distinctive mass over total distinctive mass.
        (pl.col("ni") / pl.col("nu").clip(1e-6)).cast(pl.Float32).alias("name_idf_jac"),
        (pl.col("ai") / pl.col("au").clip(1e-6)).cast(pl.Float32).alias("addr_idf_jac"),
        # Coverage: how much of the *Source-1* side's distinctive mass is matched.
        # Differs from Jaccard when the candidate carries extra tokens, which is
        # the common noise direction here (suffixes, appended locality words).
        (pl.col("ni") / pl.col("n1s").clip(1e-6)).cast(pl.Float32).alias("name_idf_cov"),
        (pl.col("ai") / pl.col("a1s").clip(1e-6)).cast(pl.Float32).alias("addr_idf_cov"),
        # The single rarest token the two records share: one highly distinctive
        # token in common is much stronger evidence than several common ones.
        pl.col("ni_max").fill_null(0.0).cast(pl.Float32).alias("name_max_shared_idf"),
    )

    drop = [c for c in df.columns if c.startswith("_")] + \
           ["ni", "nu", "ai", "au", "n1s", "a1s",
            "ni_max", "nu_max", "ai_max", "au_max", "n1s_max", "a1s_max"]
    return df.drop([c for c in drop if c in df.columns])
