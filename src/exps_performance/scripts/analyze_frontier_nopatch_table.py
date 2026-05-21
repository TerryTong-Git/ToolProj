#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("src/exps_performance/results/analysis/frontier_nopatch_outcomes.csv")
DEFAULT_OUTPUT = Path("results/frontier_nopatch_table.md")


REBUTTAL_CELLS = [
    ("GPT-5.4", "Route 1", "run_20260406_nopatch_gpt54_seed1_subset350_py310", "nl"),
    ("GPT-5.4", "Route 2", "run_20260406_nopatch_gpt54_seed1_subset350_py310", "sim"),
    ("GPT-5.4", "Route 3", "run_20260406_nopatch_gpt54_seed1_subset350_py310", "code"),
    ("Claude Opus 4.6", "Route 1", "run_20260406_nopatch_opus46_seed1_subset350_py310", "nl"),
    ("Claude Opus 4.6", "Route 2", "run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310", "sim"),
    ("Claude Opus 4.6", "Route 3", "run_20260406_nopatch_opus46_seed1_subset350_py310", "code"),
]


RUN_ORDER = [
    "run_20260406_nopatch_gpt54_seed1_subset350_py310",
    "run_20260406_nopatch_opus46_seed1_subset350_py310",
    "run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310",
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


def build_tables(input_csv: Path) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["source_run"]].append(row)

    rebuttal: dict[str, dict[str, str]] = defaultdict(dict)
    rebuttal_counts: dict[tuple[str, str], str] = {}
    cell_sources: dict[tuple[str, str], str] = {}
    for model, route, source_run, arm in REBUTTAL_CELLS:
        count, acc = arm_accuracy(by_run[source_run], arm)
        rebuttal[model][route] = acc
        rebuttal_counts[(model, route)] = count
        cell_sources[(model, route)] = source_run

    single_run_rows: list[dict[str, str]] = []
    for run in RUN_ORDER:
        scoped = by_run[run]
        nl_count, nl_acc = arm_accuracy(scoped, "nl")
        sim_count, sim_acc = arm_accuracy(scoped, "sim")
        code_count, code_acc = arm_accuracy(scoped, "code")
        single_run_rows.append(
            {
                "Run": run,
                "Model": scoped[0]["model_label"],
                "Rows": str(len(scoped)),
                "Route 1": nl_acc,
                "Route 2": sim_acc,
                "Route 3": code_acc,
                "Route 1 Count": nl_count,
                "Route 2 Count": sim_count,
                "Route 3 Count": code_count,
            }
        )

    lines = [
        "# Frontier No-Patching Rebuttal Table",
        "",
        "Accuracies use per-arm parse-normalized denominators: Route 1/NL and Route 2/Sim drop rows where that arm failed to parse; Route 3/Code keeps rows with `code_err_msg == ok,ok`.",
        "",
        "| Model | Route 1 | Route 2 | Route 3 |",
        "|---|---:|---:|---:|",
    ]
    for model in ["GPT-5.4", "Claude Opus 4.6"]:
        lines.append(f"| {model} | {rebuttal[model]['Route 1']} | {rebuttal[model]['Route 2']} | {rebuttal[model]['Route 3']} |")

    lines.extend(
        [
            "",
            "## Cell Provenance",
            "",
            "| Model | Route | Correct / Denominator | Source run |",
            "|---|---|---:|---|",
        ]
    )
    for model, route, source_run, _arm in REBUTTAL_CELLS:
        lines.append(f"| {model} | {route} | {rebuttal_counts[(model, route)]} | `{source_run}` |")

    lines.extend(
        [
            "",
            "## Single-Run Diagnostics",
            "",
            "| Run | Model | Rows | Route 1 | Route 2 | Route 3 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in single_run_rows:
        lines.append(f"| `{row['Run']}` | {row['Model']} | {row['Rows']} | {row['Route 1']} | {row['Route 2']} | {row['Route 3']} |")

    return lines, [], single_run_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lines, _rebuttal_rows, _single_run_rows = build_tables(args.input_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
