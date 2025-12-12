#!/usr/bin/env python3
"""
One-time helper to mirror all result CSVs to JSONL.

Scans exps_performance results (default path) for *.csv, writes
siblings with a .jsonl extension (one JSON object per line), and
leaves the original CSV files untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def _default_results_root() -> Path:
    # scripts/ -> exps_performance/ -> results/
    return Path(__file__).resolve().parents[1] / "results"


def iter_csvs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def convert_csv_to_jsonl(path: Path) -> Tuple[Path, int]:
    df = pd.read_csv(path)
    out_path = path.with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", lines=True)
    return out_path, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert result CSVs to JSONL (CSV files remain).")
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_results_root(),
        help="Root directory containing result CSVs (default: exps_performance/results)",
    )
    args = parser.parse_args()

    csv_files = list(iter_csvs(args.root))
    if not csv_files:
        print(f"No CSV files found under {args.root}")
        return

    total_rows = 0
    for csv_file in csv_files:
        out, rows = convert_csv_to_jsonl(csv_file)
        total_rows += rows
        print(f"Converted {csv_file} -> {out} ({rows} rows)")

    print(f"Done. Converted {len(csv_files)} files, {total_rows} rows total.")


if __name__ == "__main__":
    main()
