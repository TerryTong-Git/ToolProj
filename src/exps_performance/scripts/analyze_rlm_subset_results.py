#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence


DEFAULT_OUTCOMES = Path("src/exps_performance/results/analysis/rlm_subset25_outcomes.csv")
DEFAULT_REPORT = Path("results/rlm_results.md")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_raw_runs(items: Sequence[str]) -> list[dict[str, str]]:
    outcomes: list[dict[str, str]] = []
    for item in items:
        label, raw_path = item.split("=", 1)
        for row in load_jsonl(Path(raw_path)):
            outcomes.append(
                {
                    "model": label,
                    "kind": str(row.get("kind", "")),
                    "digit": str(row.get("digit", "")),
                    "index_in_kind": str(row.get("index_in_kind", "")),
                    "request_id": str(row.get("request_id", "")),
                    "rlm_code_correct": str(bool(row.get("rlmcode_correct", False))),
                    "rlm_nl_correct": str(bool(row.get("rlmnl_correct", False))),
                }
            )
    return outcomes


def read_outcomes(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_outcomes(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "kind", "digit", "index_in_kind", "request_id", "rlm_code_correct", "rlm_nl_correct"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: str) -> bool:
    return value.lower() == "true"


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%"


def summarize(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    summary: list[dict[str, str]] = []
    for model, model_rows in grouped.items():
        total = len(model_rows)
        code_correct = sum(is_true(row["rlm_code_correct"]) for row in model_rows)
        nl_correct = sum(is_true(row["rlm_nl_correct"]) for row in model_rows)
        if code_correct > nl_correct:
            better = "Code"
        elif nl_correct > code_correct:
            better = "NL"
        else:
            better = "Tie"
        summary.append(
            {
                "model": model,
                "total": str(total),
                "rlm_code_correct": str(code_correct),
                "rlm_code_acc": pct(code_correct, total),
                "rlm_nl_correct": str(nl_correct),
                "rlm_nl_acc": pct(nl_correct, total),
                "better_arm": better,
            }
        )
    return summary


def write_report(path: Path, summary: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RLM Results",
        "",
        "25% subset, seed 0.",
        "",
        "| Model | RLM Code Correct | RLM Code Acc | RLM NL Correct | RLM NL Acc | Better Arm |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        total = row["total"]
        lines.append(
            f"| {row['model']} | {row['rlm_code_correct']} / {total} | {row['rlm_code_acc']} | "
            f"{row['rlm_nl_correct']} / {total} | {row['rlm_nl_acc']} | {row['better_arm']} |"
        )
    lines.extend(
        [
            "",
            "Companion outcome CSV:",
            "- `src/exps_performance/results/analysis/rlm_subset25_outcomes.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-run", action="append", default=[], help="Label=path/to/res.jsonl")
    parser.add_argument("--outcomes-csv", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if args.raw_run:
        rows = parse_raw_runs(args.raw_run)
        write_outcomes(args.outcomes_csv, rows)
    else:
        rows = read_outcomes(args.outcomes_csv)

    summary = summarize(rows)
    write_report(args.report_path, summary)


if __name__ == "__main__":
    main()
