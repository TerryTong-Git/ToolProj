#!/usr/bin/env python3
"""Summarize code-arm failure distributions from experiment result JSONL files.

Bucket definitions:
- wrong_answer: execution completed (`gen_err == "ok"`) but the row is still incorrect.
  This also includes answer-format/type-check failures when code ran to completion.
- syntax_error: compile-time / generation failures, plus `no_code`.
- runtime_error: execution failed for a non-syntax, non-timeout reason.
- time_limit: executor returned `timeout`.

Use `--exclude-parse-only` to drop rows where execution succeeded (`gen_err == "ok"`)
but the returned value failed downstream parsing/type checks.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_MODELS: dict[str, str] = {
    "Haiku": "claude-haiku-4.5",
    "Codestral": "codestral-2508",
    "Gemini 2.0": "gemini-2.0-flash-001",
    "Gemini 2.5": "gemini-2.5-flash",
    "GPT-4o mini": "gpt-4o-mini",
    "Mixtral": "mixtral-8x22b-instruct",
}

SYNTAX_MARKERS = (
    "invalid syntax",
    "unterminated string literal",
    "was never closed",
    "unexpected indent",
    "expected ",
    "'return' outside function",
    "import * only allowed at module level",
    "invalid decimal literal",
    "unexpected character after line continuation character",
    "cannot assign to",
    "eol while scanning string literal",
    "unmatched",
    "non-default argument follows default argument",
    "positional argument follows keyword argument",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("src/exps_performance/results"),
        help="Root directory containing model_seed/tb/run_*/res.jsonl outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/exps_performance/results/analysis/code_failure_distribution.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--exclude-parse-only",
        action="store_true",
        help="Exclude rows where execution completed but the returned value failed parsing/type checks.",
    )
    return parser.parse_args()


def iter_rows(results_dir: Path, model_name: str) -> Iterable[dict]:
    for path in sorted(results_dir.glob(f"{model_name}_seed*/tb/run_*/res.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)


def split_error_message(err_msg: str) -> tuple[str, str]:
    if err_msg == "no_code":
        return "type_check_failed", "no_code"
    if "," in err_msg:
        return err_msg.split(",", 1)
    return "", err_msg


def classify_failure(row: dict, *, exclude_parse_only: bool) -> str:
    if bool(row.get("code_correct", False)):
        return "success"

    parse_err, gen_err = split_error_message(str(row.get("code_err_msg", "")))
    lowered = gen_err.lower()

    if exclude_parse_only and gen_err == "ok" and parse_err != "ok":
        return "excluded_parse_only"
    if gen_err == "ok":
        return "wrong_answer"
    if gen_err == "timeout":
        return "time_limit"
    if gen_err == "no_code":
        return "syntax_error"
    if any(marker in lowered for marker in SYNTAX_MARKERS):
        return "syntax_error"
    return "runtime_error"


def build_rows(results_dir: Path, *, exclude_parse_only: bool) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for label, model_name in DEFAULT_MODELS.items():
        counts: Counter[str] = Counter()
        total = 0
        for row in iter_rows(results_dir, model_name):
            total += 1
            counts[classify_failure(row, exclude_parse_only=exclude_parse_only)] += 1

        failures = total - counts["success"]
        if exclude_parse_only:
            failures -= counts["excluded_parse_only"]
        record: dict[str, object] = {
            "model": label,
            "model_dir": model_name,
            "total": total,
            "success": counts["success"],
            "failures": failures,
        }
        if exclude_parse_only:
            record["excluded_parse_only"] = counts["excluded_parse_only"]
        for category in ("wrong_answer", "syntax_error", "runtime_error", "time_limit"):
            count = counts[category]
            pct = (100.0 * count / failures) if failures else 0.0
            record[f"{category}_count"] = count
            record[f"{category}_pct"] = round(pct, 2)
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args.results_dir, exclude_parse_only=args.exclude_parse_only)
    write_csv(args.output, rows)
    for row in rows:
        extra = ""
        if args.exclude_parse_only:
            extra = f" | excluded_parse_only={row['excluded_parse_only']}"
        print(
            f"{row['model']}: failures={row['failures']} | "
            f"wrong={row['wrong_answer_pct']:.2f}% | "
            f"syntax={row['syntax_error_pct']:.2f}% | "
            f"runtime={row['runtime_error_pct']:.2f}% | "
            f"timeout={row['time_limit_pct']:.2f}%"
            f"{extra}"
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
