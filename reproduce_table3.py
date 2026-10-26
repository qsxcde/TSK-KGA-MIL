"""Print the rows of Table 3 (Section 4.2.5) from the saved explanation records.

The case study is a representative positive bag of Musk1: fold 1, bag 11
(8 instances, 0-based indices). The records were produced by one forward pass
of the trained model and are stored in:
    results/benchmark_2026_07_01/artifacts/explanations/musk1+/musk1_all_instances_report.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = (
    ROOT
    / "results"
    / "benchmark_2026_07_01"
    / "artifacts"
    / "explanations"
    / "musk1+"
    / "musk1_all_instances_report.csv"
)

FOLD = "1"
BAG_INDEX = "11"


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["fold"] == FOLD and r["bag_index"] == BAG_INDEX
        ]

    if not rows:
        raise SystemExit(f"No records found for fold={FOLD}, bag={BAG_INDEX} in {CSV_PATH}")

    rows.sort(key=lambda r: int(r["instance_index"]))

    print(f"Table 3 case study: fold {FOLD}, bag {BAG_INDEX}, {len(rows)} instances")
    print(f"source: {CSV_PATH.relative_to(ROOT)}")
    print()
    print(f"{'ID':>2}  {'Attention':>9}  {'Keyness':>9}  {'Rule':>6}  {'Membership':>10}")
    for r in rows:
        print(
            f"{int(r['instance_index']):>2}  "
            f"{float(r['attention_weight']):>9.4f}  "
            f"{float(r['instance_keyness']):>9.4f}  "
            f"{'Rule ' + r['top_rule']:>6}  "
            f"{float(r['top_rule_prob']):>10.4f}"
        )


if __name__ == "__main__":
    main()
