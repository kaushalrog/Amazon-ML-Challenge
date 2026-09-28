"""Blocking v2: measured, capped, key-at-a-time candidate generation.

The prior blocker unioned six keys and reached candidate recall 0.7402 at 2,151
candidates per S1 -- weak on both axes at once. Two structural problems:

* ``key_num_country`` blocked on *every digit in the address* plus the country,
  which is very low entropy and drove most of the 43M candidate volume.
* No key had a block-size cap, so a single generic block could contribute tens
  of thousands of candidates to one S1 (observed max: 39,312).

Every key here is evaluated in isolation *and* as a cumulative union, and every
key is capped: a key value occurring in more than ``cap`` pool records is
discarded rather than emitting a huge block. Recall lost to the cap is measured,
not assumed.
"""

import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_pool(split):
    """Source 2 and Source 3 stacked into one candidate pool."""
    return pl.concat(
        [
            pl.read_parquet(CACHE / f"{split}_source{n}.parquet")
            for n in (2, 3)
        ]
    )


# ---------------------------------------------------------------- key definitions

def _tokens(col):
    return pl.col(col).str.split(" ")


def _sorted_tokens(col):
    return _tokens(col).list.sort().list.join(" ")


def _digits(col):
    return pl.col(col).str.extract_all(r"[0-9]+")


KEYS = {
    # --- name only
    "name_full":      lambda: pl.col("name"),
    "name_sorted":    lambda: _sorted_tokens("name"),
    "name_compact":   lambda: pl.col("name").str.replace_all(" ", ""),
    "name_pfx12_ctry": lambda: pl.col("name").str.replace_all(" ", "").str.slice(0, 12)
                               + "|" + pl.col("country").cast(pl.Utf8),
    # --- address only
    "addr_full":      lambda: pl.col("addr"),
    "addr_sorted":    lambda: _sorted_tokens("addr"),
    "addr_compact":   lambda: pl.col("addr").str.replace_all(" ", ""),
    # --- combined / numeric
    "name_tok1_addrnum": lambda: _tokens("name").list.first()
                                 + "|" + _digits("addr").list.first(),
    "addrnum_ctry":   lambda: _digits("addr").list.join("-")
                              + "|" + pl.col("country").cast(pl.Utf8),
    "name4_addr4":    lambda: pl.col("name").str.replace_all(" ", "").str.slice(0, 6)
                              + "|" + pl.col("addr").str.replace_all(" ", "").str.slice(0, 6),
}

MIN_KEY_LEN = 4


def candidates_for_key(s1, pool, key_name, cap):
    """Return (s1, src, eid) candidate pairs produced by one capped key."""
    expr = KEYS[key_name]().alias("k")
    left = s1.select("eid", expr).rename({"eid": "s1"})
    right = pool.select("src", "eid", expr)

    valid = pl.col("k").is_not_null() & (pl.col("k").str.len_chars() >= MIN_KEY_LEN)
    left = left.filter(valid)
    right = right.filter(valid)

    # Cap: discard key values that are too generic to be informative.
    counts = right.group_by("k").len()
    keep = counts.filter(pl.col("len") <= cap).select("k")
    right = right.join(keep, on="k", how="semi")

    return left.join(right, on="k", how="inner").select("s1", "src", "eid").unique()


# ---------------------------------------------------------------- evaluation

def score(pairs, truth, n_s1):
    """Candidate recall and volume for a candidate set against ground truth."""
    total_true = sum(len(v) for v in truth.values())
    if pairs.height == 0:
        return dict(candidates=0, recall=0.0, avg_per_s1=0.0, max_per_s1=0, hits=0)

    got = pairs.group_by("s1").len()
    hits = 0
    by_s1 = {}
    for s1, src, eid in zip(pairs["s1"], pairs["src"], pairs["eid"]):
        by_s1.setdefault(int(s1), set()).add((int(src), int(eid)))
    for s1, true_set in truth.items():
        hits += len(true_set & by_s1.get(s1, set()))

    return dict(
        candidates=pairs.height,
        recall=hits / total_true if total_true else 0.0,
        avg_per_s1=pairs.height / n_s1,
        max_per_s1=int(got["len"].max()),
        hits=hits,
    )


def main():
    from split import load_fold, load_truth

    n_s1 = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    print(f"Loading dev sample of {n_s1:,} S1 vs full train S2+S3 pool (cap={cap})")
    s1 = load_fold("dev", n=n_s1)
    truth = load_truth(s1["eid"].to_list())
    pool = load_pool("train")
    total_true = sum(len(v) for v in truth.values())
    print(f"pool={pool.height:,}  true pairs in sample={total_true:,}")

    rows = []
    union = None
    for key in KEYS:
        t0 = time.time()
        pairs = candidates_for_key(s1, pool, key, cap)
        r = score(pairs, truth, n_s1)
        r.update(key=key, kind="single", runtime_s=round(time.time() - t0, 1), cap=cap)
        rows.append(r)
        print(
            f"  {key:20s} recall={r['recall']:.4f} cands={r['candidates']:>10,} "
            f"avg={r['avg_per_s1']:>8.1f} max={r['max_per_s1']:>6,} ({r['runtime_s']}s)"
        )
        union = pairs if union is None else pl.concat([union, pairs]).unique()

    r = score(union, truth, n_s1)
    r.update(key="UNION_ALL", kind="union", runtime_s=0, cap=cap)
    rows.append(r)
    print(
        f"  {'UNION_ALL':20s} recall={r['recall']:.4f} cands={r['candidates']:>10,} "
        f"avg={r['avg_per_s1']:>8.1f} max={r['max_per_s1']:>6,}"
    )

    out = ROOT / "reports" / "claude_blocking_review.csv"
    pl.DataFrame(rows).write_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
