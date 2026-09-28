"""Production candidate generator: the union of every key that earned its place.

Composition and the measured contribution of each part are recorded in
``reports/claude_blocking_review.md``. This module is the single source of truth
for candidate generation -- validation and test inference both call
``generate_candidates`` so that the candidate set fed to the matcher and the one
written to ``candidate_pairs.tsv`` are the same object by construction.
"""

import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking_v2 import KEYS, candidates_for_key, load_pool, score  # noqa: E402
from blocking_rare import rare_key_candidates, token_df  # noqa: E402

DEFAULT_CAP = 200


def generate_candidates(s1, pool, cap=DEFAULT_CAP, verbose=True):
    """Union of the deterministic keys with rare-name and rare-address blocking."""
    parts = []
    for key in KEYS:
        parts.append(candidates_for_key(s1, pool, key, cap))

    dfreq_name = token_df(pool, "name")
    dfreq_addr = token_df(pool, "addr")
    parts.append(rare_key_candidates(s1, pool, dfreq_name, cap, k=2, col="name"))
    parts.append(rare_key_candidates(s1, pool, dfreq_addr, cap, k=2, col="addr"))

    out = pl.concat(parts).unique()
    if verbose:
        print(f"  candidates: {out.height:,}")
    return out


def main():
    from split import load_fold, load_truth

    n_s1 = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    caps = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else [200]

    s1 = load_fold("dev", n=n_s1)
    truth = load_truth(s1["eid"].to_list())
    pool = load_pool("train")
    total_true = sum(len(v) for v in truth.values())
    print(f"dev S1={s1.height:,}  pool={pool.height:,}  true pairs={total_true:,}\n")

    rows = []
    for cap in caps:
        t0 = time.time()
        cands = generate_candidates(s1, pool, cap, verbose=False)
        r = score(cands, truth, n_s1)
        r.update(cap=cap, runtime_s=round(time.time() - t0, 1))
        rows.append(r)
        print(
            f"cap={cap:<5} recall={r['recall']:.4f} cands={r['candidates']:>10,} "
            f"avg={r['avg_per_s1']:>7.1f} max={r['max_per_s1']:>6,} ({r['runtime_s']}s)"
        )

    pl.DataFrame(rows).write_csv(ROOT / "reports" / "claude_blocking_final.csv")


if __name__ == "__main__":
    main()
