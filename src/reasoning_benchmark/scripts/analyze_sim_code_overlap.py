#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class AccuracySummary:
    label: str
    rows: int
    raw_nl_acc: float
    raw_sim_acc: float
    raw_code_acc: float
    parse_norm_nl_acc: float
    parse_norm_sim_acc: float
    parse_norm_code_acc: float
    parse_ok_nl_rows: int
    parse_ok_sim_rows: int
    parse_ok_code_rows: int
    eligible_rows: int


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(load_jsonl(path))
    return rows


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def pct(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def parse_normalized_accuracy(rows: Sequence[dict[str, Any]], arm: str) -> tuple[float, int]:
    if arm == "code":
        scoped = [row for row in rows if row.get("code_err_msg") == "ok,ok"]
    else:
        scoped = [row for row in rows if not bool(row.get(f"{arm}_parse_err"))]
    correct = sum(bool(row.get(f"{arm}_correct")) for row in scoped)
    return safe_div(correct, len(scoped)), len(scoped)


def summarize_rows(label: str, rows: Sequence[dict[str, Any]]) -> AccuracySummary:
    row_count = len(rows)
    parse_norm_nl_acc, parse_ok_nl_rows = parse_normalized_accuracy(rows, "nl")
    parse_norm_sim_acc, parse_ok_sim_rows = parse_normalized_accuracy(rows, "sim")
    parse_norm_code_acc, parse_ok_code_rows = parse_normalized_accuracy(rows, "code")
    eligible_rows = sum(1 for row in rows if (not bool(row.get("sim_parse_err"))) and row.get("code_err_msg") == "ok,ok")

    return AccuracySummary(
        label=label,
        rows=row_count,
        raw_nl_acc=safe_div(sum(bool(row.get("nl_correct")) for row in rows), row_count),
        raw_sim_acc=safe_div(sum(bool(row.get("sim_correct")) for row in rows), row_count),
        raw_code_acc=safe_div(sum(bool(row.get("code_correct")) for row in rows), row_count),
        parse_norm_nl_acc=parse_norm_nl_acc,
        parse_norm_sim_acc=parse_norm_sim_acc,
        parse_norm_code_acc=parse_norm_code_acc,
        parse_ok_nl_rows=parse_ok_nl_rows,
        parse_ok_sim_rows=parse_ok_sim_rows,
        parse_ok_code_rows=parse_ok_code_rows,
        eligible_rows=eligible_rows,
    )


def table_lines(summaries: Sequence[AccuracySummary]) -> list[str]:
    lines = [
        "# Structured Frontier Accuracy Report",
        "",
        "| Model | Rows | Raw NL | Raw Sim | Raw Code | Parse-ok NL | Parse-ok Sim | Parse-ok Code | Eligible rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.label} | {summary.rows} | {pct(summary.raw_nl_acc)} | {pct(summary.raw_sim_acc)} | "
            f"{pct(summary.raw_code_acc)} | {pct(summary.parse_norm_nl_acc)} | {pct(summary.parse_norm_sim_acc)} | "
            f"{pct(summary.parse_norm_code_acc)} | {summary.eligible_rows} |"
        )
    return lines


def write_csv(path: Path, summaries: Sequence[AccuracySummary]) -> None:
    fieldnames = list(AccuracySummary.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def parse_labeled_path(item: str) -> tuple[str, Path]:
    label, sep, raw_path = item.partition("=")
    if not sep or not label.strip() or not raw_path.strip():
        raise ValueError(f"Expected LABEL=path/to/res.jsonl, got: {item}")
    return label.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize structured frontier route accuracies from cached res.jsonl files.")
    parser.add_argument("--results-root", type=Path, required=True, help="Accepted for compatibility with earlier reproduction commands.")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--patching-run", action="append", default=[], help="Label=path/to/res.jsonl")
    args = parser.parse_args()

    if not args.patching_run:
        raise SystemExit("At least one --patching-run is required")

    summaries = []
    for item in args.patching_run:
        label, path = parse_labeled_path(item)
        summaries.append(summarize_rows(label, load_run_rows([path])))

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_path, summaries)
    args.report_path.write_text("\n".join(table_lines(summaries)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
