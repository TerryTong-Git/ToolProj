#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_ANALYSIS_DIR

DEFAULT_INPUT = LEGACY_BENCHMARK_ANALYSIS_DIR / "coding_model_outcomes.csv"
DEFAULT_OUTPUT = Path("results/coding_model_table.md")


MODEL_ORDER = [
    "x-ai/grok-code-fast-1 (25% data)",
    "qwen/qwen3-coder (25% data)",
    "codestral-2508 (original)",
]


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes"}


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}%"


def arm_rows(rows: list[dict[str, str]], arm: str) -> list[dict[str, str]]:
    if arm == "code":
        return [row for row in rows if row["code_err_msg"] == "ok,ok"]
    return [row for row in rows if not truthy(row[f"{arm}_parse_err"])]


def arm_accuracy(rows: list[dict[str, str]], arm: str) -> tuple[str, str]:
    scoped = arm_rows(rows, arm)
    correct = sum(truthy(row[f"{arm}_correct"]) for row in scoped)
    return f"{correct}/{len(scoped)}", pct(correct, len(scoped))


def build_table(input_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model_label"]].append(row)

    table_rows: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        scoped = by_model[model]
        nl_count, nl_acc = arm_accuracy(scoped, "nl")
        sim_count, sim_acc = arm_accuracy(scoped, "sim")
        code_count, code_acc = arm_accuracy(scoped, "code")
        sources = sorted({row["source_run"] for row in scoped})
        table_rows.append(
            {
                "Model": model,
                "Rows": str(len(scoped)),
                "NL": nl_acc,
                "Sim": sim_acc,
                "Code Exec": code_acc,
                "NL Count": nl_count,
                "Sim Count": sim_count,
                "Code Count": code_count,
                "Sources": "; ".join(sources),
            }
        )

    lines = [
        "# Coding-Model Rebuttal Table",
        "",
        "Accuracies use per-arm parse-normalized denominators: NL and Sim drop rows where that arm failed to parse; Code Exec keeps rows with `code_err_msg == ok,ok`.",
        "",
        "| Model | NL | Sim | Code Exec |",
        "|---|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(f"| {row['Model']} | {row['NL']} | {row['Sim']} | {row['Code Exec']} |")

    lines.extend(
        [
            "",
            "## Denominators",
            "",
            "| Model | Rows | NL correct/parse-ok | Sim correct/parse-ok | Code correct/executed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in table_rows:
        lines.append(f"| {row['Model']} | {row['Rows']} | {row['NL Count']} | {row['Sim Count']} | {row['Code Count']} |")

    lines.extend(["", "## Source Runs", ""])
    for row in table_rows:
        lines.append(f"- {row['Model']}: `{row['Sources']}`")

    return lines, table_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lines, _table_rows = build_table(args.input_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
