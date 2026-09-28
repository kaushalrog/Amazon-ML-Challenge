"""False-positive and false-negative analysis on the dev fold.

The false-negative split matters most: a true match lost in *blocking* (type A)
can only be fixed by better candidate generation, while one that was generated
and then *rejected* (type B) is a normalization, feature, model or threshold
problem. Conflating the two sends effort to the wrong stage.

False positives are categorized by the evidence pattern that produced them,
because under F0.5 they cost four times what a miss does.
"""

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking_v2 import load_pool  # noqa: E402
from split import load_fold  # noqa: E402


def enrich(pairs, s1, pool):
    return pairs.join(
        s1.select(pl.col("eid").alias("s1"), pl.col("name").alias("n1"),
                  pl.col("addr").alias("a1"), pl.col("country").cast(pl.Utf8).alias("c1")),
        on="s1", how="inner",
    ).join(
        pool.select("src", "eid", pl.col("name").alias("n2"),
                    pl.col("addr").alias("a2"), pl.col("country").cast(pl.Utf8).alias("c2")),
        on=["src", "eid"], how="inner",
    )


def categorize_fp(df):
    """Assign each false positive to its dominant evidence pattern."""
    ntok = pl.col("n1").str.split(" ").list.set_intersection(pl.col("n2").str.split(" ")).list.len()
    nun = pl.col("n1").str.split(" ").list.set_union(pl.col("n2").str.split(" ")).list.len()
    atok = pl.col("a1").str.split(" ").list.set_intersection(pl.col("a2").str.split(" ")).list.len()
    aun = pl.col("a1").str.split(" ").list.set_union(pl.col("a2").str.split(" ")).list.len()
    njac = pl.when(nun > 0).then(ntok / nun).otherwise(0.0)
    ajac = pl.when(aun > 0).then(atok / aun).otherwise(0.0)
    num1 = pl.col("a1").str.extract_all(r"[0-9]+")
    num2 = pl.col("a2").str.extract_all(r"[0-9]+")
    numshare = num1.list.set_intersection(num2).list.len()

    return df.with_columns(
        pl.when(pl.col("c1") != pl.col("c2")).then(pl.lit("country_conflict"))
        .when((pl.col("a1").str.len_chars() == 0) | (pl.col("a2").str.len_chars() == 0))
            .then(pl.lit("missing_address"))
        .when((pl.col("n1") == pl.col("n2")) & (pl.col("a1") == pl.col("a2")))
            .then(pl.lit("identical_name_and_address"))
        .when(pl.col("n1") == pl.col("n2")).then(pl.lit("same_name_diff_address"))
        .when(pl.col("a1") == pl.col("a2")).then(pl.lit("same_address_diff_name"))
        .when((njac > 0.7) & (ajac < 0.3)).then(pl.lit("similar_name_diff_address"))
        .when((ajac > 0.7) & (njac < 0.3)).then(pl.lit("similar_address_diff_name"))
        .when((numshare == 0) & (num1.list.len() > 0) & (num2.list.len() > 0))
            .then(pl.lit("number_conflict"))
        .when((njac > 0.5) & (ajac > 0.5)).then(pl.lit("both_similar_overconfident"))
        .otherwise(pl.lit("other"))
        .alias("category")
    )


def main():
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else None
    if threshold is None:
        import json
        with open(ROOT / "models" / "decision.json") as f:
            threshold = json.load(f)["threshold"]
    print(f"threshold={threshold}")

    scored = pl.read_parquet(CACHE / "dev_scored.parquet")
    s1 = load_fold("dev", n=20000)
    pool = load_pool("train")
    gt = pl.read_parquet(CACHE / "gt_pairs.parquet").join(
        s1.select(pl.col("eid").alias("s1")), on="s1", how="semi"
    )

    pred = scored.filter(pl.col("score") >= threshold)
    fp = pred.filter(pl.col("label") == 0)
    fn_rejected = scored.filter((pl.col("label") == 1) & (pl.col("score") < threshold))
    fn_blocked = gt.join(scored.select("s1", "src", "eid"), on=["s1", "src", "eid"], how="anti")

    total_true = gt.height
    print(f"\ntrue pairs={total_true:,}  predicted={pred.height:,}  "
          f"TP={pred.height - fp.height:,}  FP={fp.height:,}")
    print(f"FN type A (never blocked)   : {fn_blocked.height:,} "
          f"({fn_blocked.height / total_true:.2%} of all true pairs)")
    print(f"FN type B (blocked, rejected): {fn_rejected.height:,} "
          f"({fn_rejected.height / total_true:.2%} of all true pairs)")

    fp_cat = categorize_fp(enrich(fp, s1, pool)).group_by("category").len().sort("len", descending=True)
    print("\nFalse-positive categories:")
    print(fp_cat)

    fn_cat_a = categorize_fp(enrich(fn_blocked, s1, pool)).group_by("category").len().sort("len", descending=True)
    fn_cat_b = categorize_fp(enrich(fn_rejected, s1, pool)).group_by("category").len().sort("len", descending=True)

    # --- reports
    with open(ROOT / "reports" / "false_positive_patterns.md", "w") as f:
        f.write("# False-Positive Patterns (dev fold)\n\n")
        f.write(f"Threshold {threshold}. {fp.height:,} false positives out of "
                f"{pred.height:,} predictions.\n\n")
        f.write("Under macro F0.5 a false merge costs roughly four times what a miss\n"
                "does, and a false merge onto a true singleton takes that entity's\n"
                "score from 1.0 to 0.0 outright.\n\n")
        f.write("| category | count | share |\n|---|---|---|\n")
        for c, n in zip(fp_cat["category"], fp_cat["len"]):
            f.write(f"| {c} | {n:,} | {n / fp.height:.1%} |\n")
        f.write("\n## Examples\n\n")
        ex = categorize_fp(enrich(fp, s1, pool))
        for cat in fp_cat["category"][:6]:
            f.write(f"\n### {cat}\n\n| S1 name | S1 address | candidate name | candidate address | score |\n|---|---|---|---|---|\n")
            for r in ex.filter(pl.col("category") == cat).join(
                    scored, on=["s1", "src", "eid"]).head(5).iter_rows(named=True):
                f.write(f"| {r['n1'][:40]} | {r['a1'][:40]} | {r['n2'][:40]} | {r['a2'][:40]} | {r['score']:.3f} |\n")

    with open(ROOT / "reports" / "false_negative_patterns.md", "w") as f:
        f.write("# False-Negative Patterns (dev fold)\n\n")
        f.write(f"Threshold {threshold}. {total_true:,} true pairs in the dev sample.\n\n")
        f.write("| type | meaning | count | share of all true pairs | fix belongs in |\n|---|---|---|---|---|\n")
        f.write(f"| A | never generated by blocking | {fn_blocked.height:,} | "
                f"{fn_blocked.height / total_true:.2%} | candidate generation |\n")
        f.write(f"| B | generated, then scored below threshold | {fn_rejected.height:,} | "
                f"{fn_rejected.height / total_true:.2%} | normalization / features / model / threshold |\n")
        for title, tbl, tot in (("Type A (blocked out)", fn_cat_a, fn_blocked.height),
                                ("Type B (rejected)", fn_cat_b, fn_rejected.height)):
            f.write(f"\n## {title}\n\n| pattern | count | share |\n|---|---|---|\n")
            for c, n in zip(tbl["category"], tbl["len"]):
                f.write(f"| {c} | {n:,} | {n / max(tot,1):.1%} |\n")

    fp_cat.write_csv(ROOT / "reports" / "false_positive_categories.csv")
    print("\nwrote reports/false_positive_patterns.md, reports/false_negative_patterns.md")


if __name__ == "__main__":
    main()
