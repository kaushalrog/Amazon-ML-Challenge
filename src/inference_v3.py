"""Test inference against precomputed pool-side blocking tables.

Identical candidate definition and identical model to the validated system; the
only change is that the pool side of each blocking key is read from parquet
(built once by ``precompute_blocks.py``) instead of being recomputed for every
Source-1 chunk. Same candidates, same scores, far less work.
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
from blocking_rare import rarest_tokens  # noqa: E402
from blocking_v2 import KEYS, MIN_KEY_LEN, load_pool  # noqa: E402
from features_v2 import FEATURES  # noqa: E402


def chunk_candidates(s1, blocks, dfreq):
    """Candidates for one Source-1 chunk, joining precomputed pool-side tables."""
    valid = pl.col("k").is_not_null() & (pl.col("k").str.len_chars() >= MIN_KEY_LEN)
    parts = []

    for key in KEYS:
        left = s1.select(pl.col("eid").alias("s1"), KEYS[key]().alias("k")).filter(valid)
        right = pl.read_parquet(blocks / f"{key}.parquet")
        parts.append(left.join(right, on="k", how="inner").select("s1", "src", "eid"))
        del right

    for label, col in (("rare_name", "name"), ("rare_addr", "addr")):
        left = rarest_tokens(s1, dfreq[col], 2, col, id_cols=("eid",)).select(
            pl.col("eid").alias("s1"), pl.col("t").alias("k")
        )
        right = pl.read_parquet(blocks / f"{label}.parquet")
        parts.append(left.join(right, on="k", how="inner").select("s1", "src", "eid"))
        del right

    return pl.concat(parts).unique()


def fmt(src, eid):
    return pl.concat_str([pl.lit("S"), pl.col(src).cast(pl.Utf8), pl.lit("-"),
                          pl.col(eid).cast(pl.Utf8)])


def main():
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    cap = 200
    blocks = CACHE / f"blocks_test_{cap}"

    decision = json.loads((ROOT / "models" / "decision.json").read_text())
    threshold = decision["threshold"]
    model = lgb.Booster(model_file=str(ROOT / "models" / "matcher_lgbm.txt"))

    s1_all = pl.read_parquet(CACHE / "test_source1.parquet")
    per = -(-s1_all.height // nparts)
    s1_all = s1_all[part * per:(part + 1) * per]
    pool = load_pool("test")
    dfreq = {c: pl.read_parquet(blocks / f"dfreq_{c}.parquet") for c in ("name", "addr")}
    print(f"threshold={threshold}  test S1={s1_all.height:,}  pool={pool.height:,}")

    out = ROOT / "work" / "parts"
    out.mkdir(parents=True, exist_ok=True)
    mf = open(out / f"match_{part}.tsv", "w", encoding="utf-8")
    cf = open(out / f"cand_{part}.tsv", "w", encoding="utf-8")

    st = dict(cands=0, preds=0, empty=0, s2=0, s3=0, rows=0, max_cands=0, max_matches=0)
    t_start = time.time()

    for start in range(0, s1_all.height, chunk_size):
        t0 = time.time()
        s1 = s1_all[start:start + chunk_size]
        cands = chunk_candidates(s1, blocks, dfreq)

        X = F.build(cands, s1, pool)
        sc = model.predict(X.select(FEATURES).to_numpy())
        X = X.select("s1", "src", "eid").with_columns(
            pl.Series("score", sc.astype(np.float32))
        )

        ids = X.with_columns(fmt("src", "eid").alias("cid"))
        cl = ids.group_by("s1").agg(pl.col("cid").sort().str.join(",").alias("c"),
                                    pl.len().alias("nc"))
        ml = (ids.filter(pl.col("score") >= threshold).group_by("s1")
                 .agg(pl.col("cid").sort().str.join(",").alias("m"), pl.len().alias("nm"),
                      (pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3")))
        base = s1.select(pl.col("eid").alias("s1"))
        j = (base.join(cl, on="s1", how="left").join(ml, on="s1", how="left")
                 .with_columns(pl.col("c").fill_null(""), pl.col("m").fill_null(""),
                               pl.col("nc").fill_null(0), pl.col("nm").fill_null(0),
                               pl.col("n2").fill_null(0), pl.col("n3").fill_null(0)))
        sid = pl.lit("S1-") + pl.col("s1").cast(pl.Utf8)
        cf.write("\n".join(j.select(sid + "\t" + pl.col("c")).to_series().to_list()) + "\n")
        mf.write("\n".join(j.select(sid + "\t" + pl.col("m")).to_series().to_list()) + "\n")
        st["cands"] += int(j["nc"].sum()); st["max_cands"] = max(st["max_cands"], int(j["nc"].max()))
        st["preds"] += int(j["nm"].sum()); st["max_matches"] = max(st["max_matches"], int(j["nm"].max()))
        st["empty"] += int((j["nm"] == 0).sum()); st["rows"] += j.height
        st["s2"] += int(j["n2"].sum()); st["s3"] += int(j["n3"].sum())
        mf.flush(); cf.flush()

        done = min(start + chunk_size, s1_all.height)
        print(f"  {done:>9,}/{s1_all.height:,} cands={cands.height:>9,} "
              f"preds={st['preds']:>9,} empty={st['empty']:>8,} ({time.time() - t0:.0f}s)",
              flush=True)

    mf.close()
    cf.close()
    st["runtime_s"] = round(time.time() - t_start, 1)
    st["threshold"] = threshold
    st["avg_cands_per_s1"] = round(st["cands"] / max(st["rows"], 1), 2)
    (out / f"stats_{part}.json").write_text(json.dumps(st, indent=2))
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
