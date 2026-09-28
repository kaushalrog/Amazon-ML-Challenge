"""Materialize every source TSV once as a normalized parquet cache.

The original pipeline re-read and re-normalized ~2.4 GB of TSV in every stage,
with row-wise ``.apply``. Here each file is normalized exactly once and written
as parquet with compact dtypes, so downstream stages load in seconds.

Two representation choices matter for memory on a 16 GB machine:

* ``entity_id`` is split into ``src`` (uint8) + ``eid`` (int64). All ids match
  ``S[123]-<digits>``, verified across every file, so the string form is fully
  recoverable and we avoid holding ~10M Python strings.
* Normalization is vectorized in polars. Only the ~7% of rows containing Indic
  characters take the Python romanization path (``normalize_v2``); Latin rows
  are folded with a single Aho-Corasick ``replace_many``.
"""

import sys
import time
import unicodedata
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_v2 import indic_to_latin  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"

SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}


def _accent_map():
    """Map every precomposed Latin letter to its unaccented base."""
    pairs = {}
    for cp in range(0xC0, 0x2B0):
        ch = chr(cp)
        base = "".join(
            c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)
        )
        if base and base != ch and base.isascii():
            pairs[ch] = base.lower()
    return pairs


ACCENTS = _accent_map()
INDIC_RE = r"[ऀ-ൿ]"


def _normalize_col(name):
    """Vectorized lowercase + accent-fold + reduce-to-alnum-tokens expression."""
    return (
        pl.col(name)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_many(list(ACCENTS.keys()), list(ACCENTS.values()))
        .str.replace_all(r"[^0-9a-zऀ-ൿ]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def build(split, source):
    src_no = int(source[-1])
    src_path = ROOT / "dataset" / split / f"{split}_{source}.tsv"
    out = CACHE / f"{split}_{source}.parquet"
    if out.exists():
        print(f"  {out.name}: cached, skipping")
        return out

    t0 = time.time()
    df = pl.read_csv(src_path, separator="\t", schema_overrides=SCHEMA, quote_char=None)

    # Romanize only the rows that actually contain Indic script, then re-run the
    # vectorized pass so romanized output is folded and tokenized identically.
    for col in ("business_name", "business_address"):
        mask = pl.col(col).fill_null("").str.contains(INDIC_RE)
        df = df.with_columns(
            pl.when(mask)
            .then(
                pl.col(col)
                .fill_null("")
                .map_elements(indic_to_latin, return_dtype=pl.Utf8)
            )
            .otherwise(pl.col(col).fill_null(""))
            .alias(col)
        )

    df = df.select(
        pl.lit(src_no, dtype=pl.UInt8).alias("src"),
        pl.col("entity_id").str.extract(r"-([0-9]+)$").cast(pl.Int64).alias("eid"),
        _normalize_col("business_name").alias("name"),
        _normalize_col("business_address").alias("addr"),
        pl.col("country").fill_null("").cast(pl.Categorical),
    )

    CACHE.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    empty = df.select((pl.col("name") == "").sum()).item()
    print(
        f"  {out.name}: {df.height:,} rows in {time.time() - t0:.0f}s "
        f"({empty:,} empty names after normalization)"
    )
    return out


def main():
    for split, sources in (
        ("train", ["source1", "source2", "source3"]),
        ("test", ["source1", "source2", "source3"]),
    ):
        print(split)
        for s in sources:
            build(split, s)


if __name__ == "__main__":
    main()
