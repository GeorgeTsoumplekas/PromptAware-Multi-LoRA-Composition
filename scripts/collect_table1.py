#!/usr/bin/env python3
"""Print Table 1 for W-Switch and W-Composite from the evaluation CSVs."""

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


METHODS = (
    ("W-Switch", "weighted_switch_adaptive_tailed_ablated"),
    ("W-Composite", "weighted_composite_adaptive_triggered"),
)
SIZES = (2, 3, 4, 5)
METRICS = (
    ("ICLIP", "image_alignment_clip_results_max.csv", "cropped"),
    ("IDINO", "image_alignment_dino_results_max.csv", "cropped"),
    ("IArcFace", "identity_alignment_arcface_composlora_results_max.csv", "cropped"),
    ("TCLIP", "text_alignment_clip_composlora_results.csv", "generated"),
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated_dir", type=Path, default=root / "outputs" / "generated"
    )
    parser.add_argument(
        "--cropped_dir", type=Path, default=root / "outputs" / "cropped"
    )
    parser.add_argument(
        "--output_csv", type=Path, default=root / "outputs" / "table1.csv"
    )
    return parser.parse_args()


def round_half_up(value: float, ndigits: int = 2) -> Decimal:
    quant = Decimal("1").scaleb(-ndigits)
    return Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)


def read_score(path: Path) -> float:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    header = rows[0]
    if header and header[0] == "folder_path":
        for row in rows:
            if row and row[0] == "OVERALL":
                return float(row[3])
        raise ValueError(f"No OVERALL row in {path}")

    if header and header[0] == "Metric":
        for row in rows[1:]:
            if row and row[0] == "overall_average_similarity":
                return float(row[1])
        raise ValueError(f"No overall similarity in {path}")

    for row in reversed(rows):
        if row and row[0] == "Average" and row[-1] != "":
            return float(row[-1])
    raise ValueError(f"No Average row in {path}")


def main():
    args = parse_args()
    lines = []
    header = ["Method", "Metric"] + [f"N={size}" for size in SIZES] + ["Avg"]
    lines.append(header)

    for display_name, method in METHODS:
        for metric_name, filename, source in METRICS:
            values = []
            missing = False
            for size in SIZES:
                folder_name = f"{method}_{size}"
                if source == "cropped":
                    folder_name = f"{folder_name}_cropped"
                    base = args.cropped_dir
                else:
                    base = args.generated_dir
                csv_path = base / folder_name / filename
                if not csv_path.is_file():
                    print(f"Missing {csv_path}")
                    missing = True
                    break
                values.append(read_score(csv_path) * 100.0)
            if missing:
                continue
            cells = [round_half_up(value) for value in values]
            average = round_half_up(sum(cells) / Decimal(len(cells)))
            lines.append(
                [
                    display_name,
                    metric_name,
                    *[f"{cell:.2f}" for cell in cells],
                    f"{average:.2f}",
                ]
            )

    if len(lines) == 1:
        raise SystemExit("No Table 1 CSVs found. Run scripts/evaluate_table1.sh first.")

    widths = [max(len(row[i]) for row in lines) for i in range(len(header))]
    for row in lines:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        csv.writer(handle).writerows(lines)
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
