#!/usr/bin/env python3
"""Summarize translation-additivity shot ablation trial files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

CONDITIONS = ("x", "x_nl_native", "x_nl_translated")
DEFAULT_PATTERN = "translation_anthropic_claude-haiku-4.5_source_claude-haiku-4.5_shots*_subset25_*_trials.jsonl"
DEFAULT_LEGACY = Path("src/translation_additivity/results/translation_claude-haiku-4.5_20260127_081757_trials.jsonl")


@dataclass
class ShotRow:
    shots: str
    x: float
    native: float
    translated: float

    @property
    def delta_native(self) -> float:
        return self.native - self.x

    @property
    def delta_translated(self) -> float:
        return self.translated - self.x

    @property
    def gap(self) -> float:
        return self.translated - self.native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize functional shot ablation trials")
    parser.add_argument("--results-dir", type=Path, default=Path("src/translation_additivity/results"))
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--legacy-trials", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--model-label", default="Claude Haiku 4.5")
    parser.add_argument("--report-path", type=Path, default=Path("results/functional_shot_ablation_summary.md"))
    parser.add_argument("--csv-path", type=Path, default=Path("results/functional_shot_ablation_summary.csv"))
    return parser.parse_args()


def _accuracy_by_condition(path: Path) -> dict[str, float]:
    counts = {condition: [0, 0] for condition in CONDITIONS}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            condition = row.get("condition")
            if condition not in counts:
                continue
            counts[condition][1] += 1
            counts[condition][0] += int(bool(row.get("correct")))
    return {condition: (correct / total if total else 0.0) for condition, (correct, total) in counts.items()}


def _shot_number(path: Path) -> int:
    match = re.search(r"_shots(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not parse shot count from {path}")
    return int(match.group(1))


def build_rows(results_dir: Path, pattern: str, legacy_trials: Path | None) -> list[ShotRow]:
    latest_by_shot: dict[int, Path] = {}
    for path in sorted(results_dir.glob(pattern)):
        latest_by_shot[_shot_number(path)] = path

    rows: list[ShotRow] = []
    for shots, path in sorted(latest_by_shot.items()):
        acc = _accuracy_by_condition(path)
        rows.append(
            ShotRow(
                shots=str(shots),
                x=acc["x"],
                native=acc["x_nl_native"],
                translated=acc["x_nl_translated"],
            )
        )

    if legacy_trials and legacy_trials.exists():
        acc = _accuracy_by_condition(legacy_trials)
        rows.append(
            ShotRow(
                shots="10 (paper legacy)",
                x=acc["x"],
                native=acc["x_nl_native"],
                translated=acc["x_nl_translated"],
            )
        )

    return rows


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def write_csv(path: Path, rows: list[ShotRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["shots", "x", "x_native_nl", "x_translated_nl", "delta_native", "delta_translated", "gap"])
        for row in rows:
            writer.writerow(
                [
                    row.shots,
                    _pct(row.x),
                    _pct(row.native),
                    _pct(row.translated),
                    _pp(row.delta_native),
                    _pp(row.delta_translated),
                    _pp(row.gap),
                ]
            )


def render_markdown(model_label: str, rows: list[ShotRow]) -> str:
    lines = [
        "# Functional Shot Ablation Summary",
        "",
        "This compact rebuttal summary reports the verified translation-additivity shot ablation.",
        "",
        "Metric definitions:",
        "- `x`: question only",
        "- `x + native NL`: question plus native NL reasoning",
        "- `x + translated NL`: question plus translated natural-language reasoning produced from code",
        "- `Delta native`: `(x + native NL) - x`",
        "- `Delta translated`: `(x + translated NL) - x`",
        "- `Gap`: `(x + translated NL) - (x + native NL)`",
        "",
        f"## {model_label}",
        "",
        "| Shots | x | x + native NL | x + translated NL | Delta native | Delta translated | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.shots} | {_pct(row.x)} | {_pct(row.native)} | {_pct(row.translated)} | "
            f"{_pp(row.delta_native)} | {_pp(row.delta_translated)} | {_pp(row.gap)} |"
        )
    lines.extend(
        [
            "",
            "The 10-shot row is the older paper result and is not from the same `subset_fraction=0.25` sweep as the 0-5 shot rows.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: Path, model_label: str, rows: list[ShotRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(model_label, rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args.results_dir, args.pattern, args.legacy_trials)
    write_csv(args.csv_path, rows)
    write_markdown(args.report_path, args.model_label, rows)
    print(f"Wrote {args.csv_path}")
    print(f"Wrote {args.report_path}")


if __name__ == "__main__":
    main()
