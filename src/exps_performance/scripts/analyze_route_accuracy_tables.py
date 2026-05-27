#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

MODEL_TYPE = {
    "anthropic/claude-haiku-4.5": "closed",
    "google/gemini-2.0-flash-001": "closed",
    "google/gemini-2.5-flash": "closed",
    "openai/gpt-4o-mini": "closed",
    "mistralai/codestral-2508": "open",
    "mistralai/mixtral-8x22b-instruct": "open",
}

MODEL_ORDER = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "mistralai/codestral-2508",
    "mistralai/mixtral-8x22b-instruct",
]

DISPLAY_MODEL = {
    "anthropic/claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "google/gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash": "google/gemini-2.5-flash",
    "openai/gpt-4o-mini": "openai/gpt-4o-mini",
    "mistralai/codestral-2508": "mistralai/codestral-2508",
    "mistralai/mixtral-8x22b-instruct": "mistralai/mixtral-8x22b-instruct",
}

COMPLEXITY_TASKS = {
    "O(1)": ["add", "sub", "mul", "segments_intersect"],
    "O(log n)": ["binary_search"],
    "O(n)": ["minimum", "quickselect", "find_maximum_subarray_kadane", "kmp_matcher"],
    "O(n log n)": ["activity_selector", "task_scheduling", "heapsort", "quicksort", "graham_scan"],
    "O(n^2)": [
        "lcs",
        "rod",
        "knap",
        "bubble_sort",
        "insertion_sort",
        "lcs_length",
        "naive_string_matcher",
        "jarvis_march",
        "dfs",
        "bfs",
        "topological_sort",
        "strongly_connected_components",
        "articulation_points",
        "bridges",
        "mst_prim",
        "dijkstra",
        "dag_shortest_paths",
    ],
    "O(n^2 log n)": ["mst_kruskal"],
    "O(n^3)": ["matrix_chain_order", "optimal_bst", "floyd_warshall", "bellman_ford"],
    "NP-hard": ["ilp_assign", "ilp_prod", "ilp_partition", "edp", "gcp", "ksp", "spp", "tsp"],
}

TASK_TO_COMPLEXITY = {task: klass for klass, tasks in COMPLEXITY_TASKS.items() for task in tasks}
INCLUDED_TASKS = set(TASK_TO_COMPLEXITY)


@dataclass
class RouteSummary:
    group: str
    tasks: int
    instances: int
    rows: int
    route1: float
    route2: float
    route3: float
    delta_21: float
    p_21: float
    delta_32: float
    p_32: float
    n01_21: int
    n10_21: int
    n01_32: int
    n10_32: int


def load_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_root.glob("*/tb/run_*/res.jsonl")):
        if "/unused/" in path.as_posix():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") not in INCLUDED_TASKS:
                    continue
                model = row.get("model", "")
                if model not in MODEL_TYPE:
                    continue
                rows.append(row)
    return rows


def paired_counts(rows: Sequence[dict], a: str, b: str) -> tuple[int, int]:
    n01 = 0
    n10 = 0
    for row in rows:
        av = bool(row.get(a))
        bv = bool(row.get(b))
        if (not av) and bv:
            n01 += 1
        elif av and (not bv):
            n10 += 1
    return n01, n10


def mcnemar_p(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail_k = min(n01, n10)
    log_probs = [
        math.lgamma(discordant + 1) - math.lgamma(k + 1) - math.lgamma(discordant - k + 1) - discordant * math.log(2) for k in range(tail_k + 1)
    ]
    max_log = max(log_probs)
    tail = math.exp(max_log) * sum(math.exp(value - max_log) for value in log_probs)
    return min(1.0, 2.0 * tail)


def instance_count(rows: Sequence[dict]) -> int:
    return len({(row.get("kind"), row.get("digit"), row.get("index_in_kind")) for row in rows})


def summarize(group: str, rows: Sequence[dict]) -> RouteSummary:
    if not rows:
        raise ValueError(f"No rows for {group}")
    route1 = mean(bool(row.get("nl_correct")) for row in rows)
    route2 = mean(bool(row.get("sim_correct")) for row in rows)
    route3 = mean(bool(row.get("code_correct")) for row in rows)
    n01_21, n10_21 = paired_counts(rows, "nl_correct", "sim_correct")
    n01_32, n10_32 = paired_counts(rows, "sim_correct", "code_correct")
    return RouteSummary(
        group=group,
        tasks=len({row.get("kind") for row in rows}),
        instances=instance_count(rows),
        rows=len(rows),
        route1=route1,
        route2=route2,
        route3=route3,
        delta_21=route2 - route1,
        p_21=mcnemar_p(n01_21, n10_21),
        delta_32=route3 - route2,
        p_32=mcnemar_p(n01_32, n10_32),
        n01_21=n01_21,
        n10_21=n10_21,
        n01_32=n01_32,
        n10_32=n10_32,
    )


def p_value(value: float) -> str:
    if value == 0:
        return "<1e-300"
    if value < 1e-300:
        return "<1e-300"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3g}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}"


def delta(value: float) -> str:
    return f"{value * 100:.1f}"


def write_csv(path: Path, summaries: Iterable[RouteSummary], *, include_model_type: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "group",
        "tasks",
        "instances",
        "rows",
        "route1",
        "route2",
        "route3",
        "delta_route2_route1",
        "mcnemar_p_route2_route1",
        "delta_route3_route2",
        "mcnemar_p_route3_route2",
        "n01_route2_route1",
        "n10_route2_route1",
        "n01_route3_route2",
        "n10_route3_route2",
    ]
    fieldnames = ["type", *base_fields] if include_model_type else base_fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for s in summaries:
            row = {
                "group": s.group,
                "tasks": s.tasks,
                "instances": s.instances,
                "rows": s.rows,
                "route1": f"{s.route1:.6f}",
                "route2": f"{s.route2:.6f}",
                "route3": f"{s.route3:.6f}",
                "delta_route2_route1": f"{s.delta_21:.6f}",
                "mcnemar_p_route2_route1": f"{s.p_21:.12g}",
                "delta_route3_route2": f"{s.delta_32:.6f}",
                "mcnemar_p_route3_route2": f"{s.p_32:.12g}",
                "n01_route2_route1": s.n01_21,
                "n10_route2_route1": s.n10_21,
                "n01_route3_route2": s.n01_32,
                "n10_route3_route2": s.n10_32,
            }
            if include_model_type:
                row = {"type": MODEL_TYPE[s.group], **row}
            writer.writerow(row)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    align = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_markdown(path: Path, complexity: Sequence[RouteSummary], by_model: Sequence[RouteSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complexity_rows = [
        [
            s.group,
            str(s.tasks),
            str(s.instances),
            pct(s.route1),
            pct(s.route2),
            pct(s.route3),
            delta(s.delta_21),
            p_value(s.p_21),
            delta(s.delta_32),
            p_value(s.p_32),
        ]
        for s in complexity
    ]
    model_rows = [
        [
            DISPLAY_MODEL[s.group],
            MODEL_TYPE[s.group],
            str(s.instances),
            pct(s.route1),
            pct(s.route2),
            pct(s.route3),
            delta(s.delta_21),
            p_value(s.p_21),
            delta(s.delta_32),
            p_value(s.p_32),
        ]
        for s in by_model
    ]

    lines = [
        "# Route Accuracy Tables",
        "",
        "Source data: checked-in main-run JSONL files matched by `src/exps_performance/results/*/tb/run_*/res.jsonl`.",
        "",
        "Filters applied:",
        "- includes the 44 tasks in the rebuttal complexity mapping",
        "- excludes `bsp`, `msp`, `gcp_d`, and `tsp_d`, which are present in raw runs but not in the rebuttal table",
        "- includes the six full-coverage models used in the main aggregate",
        "- excludes `unused/` result directories and subset/debug runs",
        "",
        "Conventions:",
        "- `Route 1`, `Route 2`, and `Route 3` correspond to `nl_correct`, `sim_correct`, and `code_correct`",
        "- accuracies and deltas are percentages / percentage-point differences",
        "- `Instances` counts unique `(kind, digit, index_in_kind)` problems",
        "- McNemar p-values are exact two-sided tests on paired outcomes",
        "",
        "## Results Grouped By Asymptotic Complexity Class",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Complexity Class",
                "Tasks",
                "Instances",
                "Route 1",
                "Route 2",
                "Route 3",
                "Route 2 - Route 1",
                "McNemar p (Route 2 vs Route 1)",
                "Route 3 - Route 2",
                "McNemar p (Route 3 vs Route 2)",
            ],
            complexity_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Results Grouped By Model",
            "",
            "Only full-coverage models are shown; each row has `1340` problems and `4020` paired evaluations.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Model",
                "Type",
                "Instances",
                "Route 1",
                "Route 2",
                "Route 3",
                "Route 2 - Route 1",
                "McNemar p (Route 2 vs Route 1)",
                "Route 3 - Route 2",
                "McNemar p (Route 3 vs Route 2)",
            ],
            model_rows,
        )
    )
    lines.extend(["", "Task mapping:"])
    for klass, tasks in COMPLEXITY_TASKS.items():
        lines.append(f"- `{klass}`: " + ", ".join(f"`{task}`" for task in tasks))
    lines.extend(
        [
            "",
            "Companion CSVs:",
            "- `src/exps_performance/results/analysis/accuracy_by_asymptotic_class.csv`",
            "- `src/exps_performance/results/analysis/accuracy_by_model.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("src/exps_performance/results"))
    parser.add_argument("--report-path", type=Path, default=Path("results/route_accuracy_tables.md"))
    parser.add_argument(
        "--complexity-csv",
        type=Path,
        default=Path("src/exps_performance/results/analysis/accuracy_by_asymptotic_class.csv"),
    )
    parser.add_argument("--model-csv", type=Path, default=Path("src/exps_performance/results/analysis/accuracy_by_model.csv"))
    args = parser.parse_args()

    rows = load_rows(args.results_root)
    by_complexity: dict[str, list[dict]] = defaultdict(list)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_complexity[TASK_TO_COMPLEXITY[row["kind"]]].append(row)
        by_model[row["model"]].append(row)

    complexity_summaries = [summarize(klass, by_complexity[klass]) for klass in COMPLEXITY_TASKS]
    model_summaries = [summarize(model, by_model[model]) for model in MODEL_ORDER]

    observed = Counter(row["kind"] for row in rows)
    missing = sorted(INCLUDED_TASKS - set(observed))
    if missing:
        raise ValueError(f"Missing mapped tasks from results: {missing}")

    write_csv(args.complexity_csv, complexity_summaries)
    write_csv(args.model_csv, model_summaries, include_model_type=True)
    write_markdown(args.report_path, complexity_summaries, model_summaries)


if __name__ == "__main__":
    main()
