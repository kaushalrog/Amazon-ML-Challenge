"""Independent sanity checks on the generated submission files.

Deliberately does not reuse anything from the pipeline: it re-reads the TSVs and
the test sources from disk and re-derives every count, so a bug in the pipeline
cannot hide itself here. Complements ``utils/validate_submission.py`` rather
than replacing it.
"""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split("\t", 1)[0].strip() for line in f if line.strip()}


def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            m[s1] = rest.split(",") if rest else []
    return m


def main():
    out = ROOT / "output"
    test = ROOT / "dataset" / "test"
    required = read_ids(test / "test_source1.tsv")

    matching = read_map(out / "matching_results.tsv")
    candidates = read_map(out / "candidate_pairs.tsv")

    checks, stats = [], {}

    def check(label, ok, detail=""):
        checks.append((label, ok, detail))

    check("matching rows == test S1 count",
          len(matching) == len(required), f"{len(matching):,} vs {len(required):,}")
    check("candidate rows == test S1 count",
          len(candidates) == len(required), f"{len(candidates):,} vs {len(required):,}")
    check("no missing S1 in matching", not (required - set(matching)),
          f"{len(required - set(matching)):,} missing")
    check("no extra S1 in matching", not (set(matching) - required),
          f"{len(set(matching) - required):,} extra")

    bad_prefix = dup_within = self_match = not_in_cand = 0
    n_pred = n_empty = n_s2 = n_s3 = 0
    per_s1 = Counter()

    for s1, ids in matching.items():
        cset = set(candidates.get(s1, []))
        if not ids:
            n_empty += 1
            continue
        n_pred += len(ids)
        if len(ids) != len(set(ids)):
            dup_within += 1
        for i in ids:
            if i.startswith("S1-"):
                self_match += 1
            elif i.startswith("S2-"):
                n_s2 += 1
            elif i.startswith("S3-"):
                n_s3 += 1
            else:
                bad_prefix += 1
            if i not in cset:
                not_in_cand += 1
        per_s1[len(ids)] += 1

    check("no S1 self-matches", self_match == 0, str(self_match))
    check("no malformed prefixes", bad_prefix == 0, str(bad_prefix))
    check("no duplicate IDs within a row", dup_within == 0, str(dup_within))
    check("every match is in candidate_pairs", not_in_cand == 0, str(not_in_cand))

    # ID existence against the real test sources (streamed, one file at a time).
    for n, seen in (("2", n_s2), ("3", n_s3)):
        valid = read_ids(test / f"test_source{n}.tsv")
        bad = sum(
            1 for ids in matching.values() for i in ids
            if i.startswith(f"S{n}-") and i not in valid
        )
        check(f"all S{n} matched IDs exist in test_source{n}", bad == 0, str(bad))
        del valid

    n_cands = sum(len(v) for v in candidates.values())
    stats.update(
        test_s1_entities=len(required),
        predicted_matches=n_pred,
        empty_predictions=n_empty,
        s2_matches=n_s2,
        s3_matches=n_s3,
        candidate_pairs=n_cands,
        avg_candidates_per_s1=round(n_cands / max(len(candidates), 1), 2),
        max_candidates_per_s1=max((len(v) for v in candidates.values()), default=0),
        max_matches_per_s1=max(per_s1) if per_s1 else 0,
        mean_matches_per_nonempty=round(n_pred / max(len(matching) - n_empty, 1), 2),
    )

    ok = all(c[1] for c in checks)
    lines = ["# Final Output Validation", "",
             "Independent re-derivation from the written TSVs and the raw test",
             "sources; shares no code with the pipeline that produced them.", "",
             "## Checks", "", "| check | result | detail |", "|---|---|---|"]
    for label, passed, detail in checks:
        lines.append(f"| {label} | {'PASS' if passed else 'FAIL'} | {detail} |")
    lines += ["", "## Output statistics", "", "| metric | value |", "|---|---|"]
    for k, v in stats.items():
        lines.append(f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |")
    lines += ["", f"## Overall: {'PASS' if ok else 'FAIL'}", ""]

    (ROOT / "reports" / "final_validation.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
