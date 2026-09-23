#!/usr/bin/env python3
"""Average the per-composition MiniCPM scores into Table 2."""

import argparse
import csv
import re
from pathlib import Path


METHODS = (
    ("W-Switch", "weighted_switch_adaptive_tailed_ablated"),
    ("W-Composite", "weighted_composite_adaptive_triggered"),
)
SIZES = (2, 3, 4, 5)
METRICS = (
    ("integration", "Element Integration"),
    ("consistency", "Spatial Consistency"),
    ("accuracy", "Semantic Accuracy"),
    ("appeal", "Aesthetic Appeal"),
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", type=Path, default=root / "outputs" / "llm_evals")
    parser.add_argument(
        "--output_csv", type=Path, default=root / "outputs" / "table2.csv"
    )
    return parser.parse_args()


def parse_file(path: Path) -> dict:
    scores = {}
    current = None
    for line in path.read_text().splitlines():
        method_match = re.match(r"Method:\s*(.+)", line)
        if method_match:
            current = method_match.group(1).strip()
            scores[current] = {}
            continue
        metric_match = re.match(r"\s*(\w+):\s*([0-9.]+)", line)
        if metric_match and current is not None:
            scores[current][metric_match.group(1)] = float(metric_match.group(2))
    return scores


def main():
    args = parse_args()
    rows = [["Method", "Metric", "Score"]]
    printable = []

    for display_name, method in METHODS:
        collected = {key: [] for key, _ in METRICS}
        for size in SIZES:
            path = args.eval_dir / f"{method}_{size}.txt"
            if not path.is_file():
                print(f"Missing {path}")
                collected = None
                break
            parsed = parse_file(path)
            block = parsed.get(method)
            if block is None:
                print(f"No scores for {method} in {path}")
                collected = None
                break
            for key, _ in METRICS:
                collected[key].append(block[key])
        if collected is None:
            continue

        metric_avgs = []
        for key, label in METRICS:
            average = sum(collected[key]) / len(collected[key])
            metric_avgs.append(average)
            rows.append([display_name, label, f"{average:.3f}"])
            printable.append((display_name, label, average))
        overall = sum(metric_avgs) / len(metric_avgs)
        rows.append([display_name, "Avg", f"{overall:.3f}"])
        printable.append((display_name, "Avg", overall))

    if len(rows) == 1:
        raise SystemExit("No MiniCPM logs found. Run scripts/evaluate_table2.sh first.")

    current = None
    for display_name, label, average in printable:
        if display_name != current:
            print(f"\n{display_name}")
            current = display_name
        print(f"  {label}: {average:.3f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
