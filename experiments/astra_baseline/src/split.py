"""Leakage-safe grouped split over Source-1 entities.

The prior code sampled the same 20,000 S1 entities with ``random_state=42`` and
used them to train, cross-validate and tune the decision threshold, so no number
it produced was held out. ``create_grouped_split`` existed but was never called.

Here the split is a deterministic function of the S1 id (hash-based, not a
shuffle), so it is stable across runs, across machines, and across changes to the
row order or to the sample size. Three disjoint folds:

* ``train`` -- matcher fitting
* ``dev``   -- threshold search, decision rules, model selection
* ``eval``  -- touched once per reported number, never optimized against

Ground truth is also materialized here in long form (one row per true pair),
which every downstream stage joins against instead of re-parsing a 127 MB TSV.
"""

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"
SEED = 20260927
FRACTIONS = {"train": 0.60, "dev": 0.20, "eval": 0.20}


def build_ground_truth():
    """Explode ``matched_entity_ids`` into one row per (s1, match) pair."""
    out = CACHE / "gt_pairs.parquet"
    if out.exists():
        return out
    gt = pl.read_csv(
        ROOT / "dataset" / "train" / "train_ground_truth.tsv",
        separator="\t",
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        quote_char=None,
    )
    pairs = (
        gt.select(
            pl.col("source1_entity_id").str.extract(r"-([0-9]+)$").cast(pl.Int64).alias("s1"),
            pl.col("matched_entity_ids").fill_null("").str.split(","),
        )
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
        .select(
            "s1",
            pl.col("matched_entity_ids").str.slice(1, 1).cast(pl.UInt8).alias("src"),
            pl.col("matched_entity_ids").str.extract(r"-([0-9]+)$").cast(pl.Int64).alias("eid"),
        )
    )
    pairs.write_parquet(out, compression="zstd")
    print(f"gt_pairs: {pairs.height:,} true pairs")
    return out


def assign_folds():
    """Assign every S1 entity (matched or not) to exactly one fold."""
    out = CACHE / "folds.parquet"
    if out.exists():
        return out
    s1 = pl.read_parquet(CACHE / "train_source1.parquet", columns=["eid"])
    # hash -> [0,1) bucket; deterministic and independent of row order.
    bucket = (pl.col("eid").hash(seed=SEED) % 10_000) / 10_000.0
    folds = s1.with_columns(
        pl.when(bucket < FRACTIONS["train"])
        .then(pl.lit("train"))
        .when(bucket < FRACTIONS["train"] + FRACTIONS["dev"])
        .then(pl.lit("dev"))
        .otherwise(pl.lit("eval"))
        .alias("fold")
    )
    folds.write_parquet(out, compression="zstd")
    print(folds.group_by("fold").len().sort("fold"))
    return out


def load_fold(fold, n=None, seed=SEED):
    """Return the S1 records of one fold, optionally a reproducible subsample."""
    folds = pl.read_parquet(CACHE / "folds.parquet").filter(pl.col("fold") == fold)
    s1 = pl.read_parquet(CACHE / "train_source1.parquet").join(
        folds.select("eid"), on="eid", how="semi"
    )
    if n is not None and n < s1.height:
        s1 = s1.sample(n=n, seed=seed)
    return s1


def load_truth(s1_ids):
    """Ground-truth matches restricted to the given S1 ids, as {s1: set[(src,eid)]}."""
    pairs = pl.read_parquet(CACHE / "gt_pairs.parquet").filter(pl.col("s1").is_in(s1_ids))
    truth = {int(i): set() for i in s1_ids}
    for s1, src, eid in zip(pairs["s1"], pairs["src"], pairs["eid"]):
        truth[int(s1)].add((int(src), int(eid)))
    return truth


if __name__ == "__main__":
    build_ground_truth()
    assign_folds()
