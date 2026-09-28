"""Version the current output/ into submissions/submission_NNN/ with metadata.

Leaderboard submissions are limited, so outputs are never overwritten in place:
each serious candidate gets its own numbered folder with the exact validation
numbers that justified it.
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(notes=""):
    subs = ROOT / "submissions"
    subs.mkdir(exist_ok=True)
    n = 1 + max(
        (int(p.name.split("_")[1]) for p in subs.glob("submission_*") if p.is_dir()),
        default=0,
    )
    dest = subs / f"submission_{n:03d}"
    dest.mkdir()

    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(ROOT / "output" / f, dest / f)

    decision = json.loads((ROOT / "models" / "decision.json").read_text())
    stats_path = ROOT / "reports" / "test_inference_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    meta = dict(
        experiment_id=f"submission_{n:03d}",
        model="LightGBM binary, 28 vectorized pairwise features",
        blocking="10 capped deterministic keys + rare-name-k2 + rare-address-k2, cap=200",
        features=decision.get("features"),
        threshold=decision.get("threshold"),
        validation_f05=decision.get("eval", {}).get("macro_f05"),
        precision=decision.get("eval", {}).get("macro_precision"),
        recall=decision.get("eval", {}).get("macro_recall"),
        candidate_recall=0.9071,
        singleton_accuracy=decision.get("eval", {}).get("singleton_accuracy"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        test_stats=stats,
        notes=notes,
    )
    (dest / "experiment_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {dest}")
    print(json.dumps(meta, indent=2)[:900])


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
