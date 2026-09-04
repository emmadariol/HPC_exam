#!/usr/bin/env python3
"""Add estimated GFLOP/s columns to N-body benchmark CSV files.

The solver reports pair-interaction throughput because it is the most stable
algorithmic rate for direct N-body. This helper converts that rate to an
estimated floating-point rate using an explicit FLOP-per-pair assumption.
"""

from __future__ import annotations

import argparse
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append estimated GFLOP/s columns to a benchmark CSV."
    )
    parser.add_argument("input_csv", help="CSV containing gpairs or gpairs_median")
    parser.add_argument("output_csv", help="CSV to write with estimated GFLOP/s")
    parser.add_argument(
        "--flops-per-pair",
        type=float,
        default=20.0,
        help="operation-count model for one pair interaction (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input_csv, newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        rows = list(reader)
        if reader.fieldnames is None:
            raise SystemExit("empty CSV")
        fieldnames = list(reader.fieldnames)

    rate_columns = [name for name in ("gpairs_median", "gpairs") if name in fieldnames]
    if not rate_columns:
        raise SystemExit("CSV must contain either gpairs_median or gpairs")

    for rate_column in rate_columns:
        out_column = f"estimated_gflops_from_{rate_column}"
        if out_column not in fieldnames:
            fieldnames.append(out_column)
        for row in rows:
            try:
                gpairs = float(row[rate_column])
                row[out_column] = f"{gpairs * args.flops_per_pair:.9g}"
            except (TypeError, ValueError):
                row[out_column] = "nan"

    model_column = "flops_per_pair_model"
    if model_column not in fieldnames:
        fieldnames.append(model_column)
    for row in rows:
        row[model_column] = f"{args.flops_per_pair:g}"

    with open(args.output_csv, "w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
