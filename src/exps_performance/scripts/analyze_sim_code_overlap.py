#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


BASELINE_MODELS: dict[str, list[str]] = {
    "Haiku": ["claude-haiku-4.5_seed0", "claude-haiku-4.5_seed1", "claude-haiku-4.5_seed2"],
    "Codestral": ["codestral-2508_seed0", "codestral-2508_seed1", "codestral-2508_seed2"],
    "Gemini 2.0": ["gemini-2.0-flash-001_seed0", "gemini-2.0-flash-001_seed1", "gemini-2.0-flash-001_seed2"],
    "Gemini 2.5": ["gemini-2.5-flash_seed0", "gemini-2.5-flash_seed1", "gemini-2.5-flash_seed2"],
    "GPT-4o mini": ["gpt-4o-mini_seed0", "gpt-4o-mini_seed1", "gpt-4o-mini_seed2"],
    "Mixtral": ["mixtral-8x22b-instruct_seed0", "mixtral-8x22b-instruct_seed1", "mixtral-8x22b-instruct_seed2"],
}


@dataclass
class OverlapSummary:
    label: str
    rows: int
    both_fail: int
    sim_only_fail: int
    code_only_fail: int
    both_succeed: int
    sim_fail_rate: float
    code_fail_rate: float
    both_fail_rate: float
    p_code_fail_given_sim_fail: float
    p_sim_fail_given_code_fail: float
    expected_both_if_independent: float
    lift_vs_independence: float
    phi: float
    eligible_rows: int
    eligible_both_fail: int
    eligible_sim_only_fail: int
    eligible_code_only_fail: int
    eligible_both_succeed: int
    eligible_sim_fail_rate: float
    eligible_code_fail_rate: float
    eligible_both_fail_rate: float
    eligible_p_code_fail_given_sim_fail: float
    eligible_p_sim_fail_given_code_fail: float
    eligible_expected_both_if_independent: float
    eligible_lift_vs_independence: float
    eligible_phi: float
    raw_nl_acc: float
    raw_sim_acc: float
    raw_code_acc: float
    parse_norm_nl_acc: float
    parse_norm_sim_acc: float
    parse_norm_code_acc: float
    parse_ok_nl_rows: int
    parse_ok_sim_rows: int
    parse_ok_code_rows: int


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_model_rows(results_root: Path, seeds: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in seeds:
        tb_root = results_root / seed_dir / "tb"
        if not tb_root.exists():
            continue
        for run_dir in sorted(tb_root.iterdir()):
            res_path = run_dir / "res.jsonl"
            if res_path.exists():
                rows.extend(load_jsonl(res_path))
    return rows


def load_run_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            rows.extend(load_jsonl(path))
    return rows


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def phi_from_counts(both_fail: int, sim_only_fail: int, code_only_fail: int, both_succeed: int) -> float:
    a = both_fail
    b = sim_only_fail
    c = code_only_fail
    d = both_succeed
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return float("nan")
    return ((a * d) - (b * c)) / denom


def overlap_counts(rows: Sequence[dict[str, Any]], *, parse_normalized: bool) -> tuple[int, int, int, int, int]:
    scoped = rows
    if parse_normalized:
        scoped = [r for r in rows if (not bool(r.get("sim_parse_err"))) and (r.get("code_err_msg") == "ok,ok")]
    both_fail = 0
    sim_only_fail = 0
    code_only_fail = 0
    both_succeed = 0
    for row in scoped:
        sim_fail = not bool(row.get("sim_correct"))
        code_fail = not bool(row.get("code_correct"))
        if sim_fail and code_fail:
            both_fail += 1
        elif sim_fail and not code_fail:
            sim_only_fail += 1
        elif not sim_fail and code_fail:
            code_only_fail += 1
        else:
            both_succeed += 1
    return len(scoped), both_fail, sim_only_fail, code_only_fail, both_succeed


def parse_normalized_accuracy(rows: Sequence[dict[str, Any]], arm: str) -> tuple[float, int]:
    if arm == "code":
        scoped = [r for r in rows if r.get("code_err_msg") == "ok,ok"]
    else:
        scoped = [r for r in rows if not bool(r.get(f"{arm}_parse_err"))]
    if not scoped:
        return float("nan"), 0
    good = sum(bool(r.get(f"{arm}_correct")) for r in scoped)
    return good / len(scoped), len(scoped)


def summarize_rows(label: str, rows: Sequence[dict[str, Any]]) -> OverlapSummary:
    row_count, both_fail, sim_only_fail, code_only_fail, both_succeed = overlap_counts(rows, parse_normalized=False)
    sim_fail_rate = safe_div(both_fail + sim_only_fail, row_count)
    code_fail_rate = safe_div(both_fail + code_only_fail, row_count)
    both_fail_rate = safe_div(both_fail, row_count)
    expected = sim_fail_rate * code_fail_rate if not math.isnan(sim_fail_rate) and not math.isnan(code_fail_rate) else float("nan")
    lift = safe_div(both_fail_rate, expected) if expected and not math.isnan(expected) else float("nan")

    eligible_rows, eligible_both_fail, eligible_sim_only_fail, eligible_code_only_fail, eligible_both_succeed = overlap_counts(
        rows, parse_normalized=True
    )
    eligible_sim_fail_rate = safe_div(eligible_both_fail + eligible_sim_only_fail, eligible_rows)
    eligible_code_fail_rate = safe_div(eligible_both_fail + eligible_code_only_fail, eligible_rows)
    eligible_both_fail_rate = safe_div(eligible_both_fail, eligible_rows)
    eligible_expected = (
        eligible_sim_fail_rate * eligible_code_fail_rate
        if not math.isnan(eligible_sim_fail_rate) and not math.isnan(eligible_code_fail_rate)
        else float("nan")
    )
    eligible_lift = safe_div(eligible_both_fail_rate, eligible_expected) if eligible_expected and not math.isnan(eligible_expected) else float("nan")

    parse_norm_nl_acc, parse_ok_nl_rows = parse_normalized_accuracy(rows, "nl")
    parse_norm_sim_acc, parse_ok_sim_rows = parse_normalized_accuracy(rows, "sim")
    parse_norm_code_acc, parse_ok_code_rows = parse_normalized_accuracy(rows, "code")

    return OverlapSummary(
        label=label,
        rows=row_count,
        both_fail=both_fail,
        sim_only_fail=sim_only_fail,
        code_only_fail=code_only_fail,
        both_succeed=both_succeed,
        sim_fail_rate=sim_fail_rate,
        code_fail_rate=code_fail_rate,
        both_fail_rate=both_fail_rate,
        p_code_fail_given_sim_fail=safe_div(both_fail, both_fail + sim_only_fail),
        p_sim_fail_given_code_fail=safe_div(both_fail, both_fail + code_only_fail),
        expected_both_if_independent=expected,
        lift_vs_independence=lift,
        phi=phi_from_counts(both_fail, sim_only_fail, code_only_fail, both_succeed),
        eligible_rows=eligible_rows,
        eligible_both_fail=eligible_both_fail,
        eligible_sim_only_fail=eligible_sim_only_fail,
        eligible_code_only_fail=eligible_code_only_fail,
        eligible_both_succeed=eligible_both_succeed,
        eligible_sim_fail_rate=eligible_sim_fail_rate,
        eligible_code_fail_rate=eligible_code_fail_rate,
        eligible_both_fail_rate=eligible_both_fail_rate,
        eligible_p_code_fail_given_sim_fail=safe_div(eligible_both_fail, eligible_both_fail + eligible_sim_only_fail),
        eligible_p_sim_fail_given_code_fail=safe_div(eligible_both_fail, eligible_both_fail + eligible_code_only_fail),
        eligible_expected_both_if_independent=eligible_expected,
        eligible_lift_vs_independence=eligible_lift,
        eligible_phi=phi_from_counts(eligible_both_fail, eligible_sim_only_fail, eligible_code_only_fail, eligible_both_succeed),
        raw_nl_acc=safe_div(sum(bool(r.get("nl_correct")) for r in rows), row_count),
        raw_sim_acc=safe_div(sum(bool(r.get("sim_correct")) for r in rows), row_count),
        raw_code_acc=safe_div(sum(bool(r.get("code_correct")) for r in rows), row_count),
        parse_norm_nl_acc=parse_norm_nl_acc,
        parse_norm_sim_acc=parse_norm_sim_acc,
        parse_norm_code_acc=parse_norm_code_acc,
        parse_ok_nl_rows=parse_ok_nl_rows,
        parse_ok_sim_rows=parse_ok_sim_rows,
        parse_ok_code_rows=parse_ok_code_rows,
    )


def pct(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def scalar(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def baseline_markdown(summaries: Sequence[OverlapSummary]) -> list[str]:
    lines = [
        "## Baseline 6-model overlap",
        "",
        "Raw overlap over all rows:",
        "",
        "| Model | Rows | Sim fail | Code fail | Both fail | P(code fail | sim fail) | P(sim fail | code fail) | Expected both if independent | Lift | Phi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.label} | {s.rows} | {pct(s.sim_fail_rate)} | {pct(s.code_fail_rate)} | {pct(s.both_fail_rate)} | "
            f"{pct(s.p_code_fail_given_sim_fail)} | {pct(s.p_sim_fail_given_code_fail)} | {pct(s.expected_both_if_independent)} | "
            f"{scalar(s.lift_vs_independence)} | {scalar(s.phi)} |"
        )
    lines.extend(
        [
            "",
            "Parse-normalized overlap, restricting to rows where `sim` parsed and `code` reached `code_err_msg == ok,ok`:",
            "",
            "| Model | Eligible rows | Sim fail | Code fail | Both fail | P(code fail | sim fail) | P(sim fail | code fail) | Expected both if independent | Lift | Phi |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for s in summaries:
        lines.append(
            f"| {s.label} | {s.eligible_rows} | {pct(s.eligible_sim_fail_rate)} | {pct(s.eligible_code_fail_rate)} | {pct(s.eligible_both_fail_rate)} | "
            f"{pct(s.eligible_p_code_fail_given_sim_fail)} | {pct(s.eligible_p_sim_fail_given_code_fail)} | {pct(s.eligible_expected_both_if_independent)} | "
            f"{scalar(s.eligible_lift_vs_independence)} | {scalar(s.eligible_phi)} |"
        )
    return lines


def patch_block(title: str, summaries: Sequence[OverlapSummary]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Model | Rows | Raw NL | Raw Sim | Raw Code | Parse-ok NL | Parse-ok Sim | Parse-ok Code | Eligible rows | Both fail (parse-ok) | P(code fail | sim fail) | P(sim fail | code fail) | Lift | Phi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.label} | {s.rows} | {pct(s.raw_nl_acc)} | {pct(s.raw_sim_acc)} | {pct(s.raw_code_acc)} | "
            f"{pct(s.parse_norm_nl_acc)} | {pct(s.parse_norm_sim_acc)} | {pct(s.parse_norm_code_acc)} | {s.eligible_rows} | "
            f"{pct(s.eligible_both_fail_rate)} | {pct(s.eligible_p_code_fail_given_sim_fail)} | {pct(s.eligible_p_sim_fail_given_code_fail)} | "
            f"{scalar(s.eligible_lift_vs_independence)} | {scalar(s.eligible_phi)} |"
        )
    return lines


def write_csv(path: Path, summaries: Sequence[OverlapSummary]) -> None:
    fieldnames = list(OverlapSummary.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--patching-run", action="append", default=[], help="Label=path/to/res.jsonl")
    parser.add_argument("--no-patching-run", action="append", default=[], help="Label=path/to/res.jsonl")
    args = parser.parse_args()

    baseline = [summarize_rows(label, iter_model_rows(args.results_root, seeds)) for label, seeds in BASELINE_MODELS.items()]

    patching: list[OverlapSummary] = []
    for item in args.patching_run:
        label, raw_path = item.split("=", 1)
        patching.append(summarize_rows(label, load_run_rows([Path(raw_path)])))

    no_patching: list[OverlapSummary] = []
    for item in args.no_patching_run:
        label, raw_path = item.split("=", 1)
        no_patching.append(summarize_rows(label, load_run_rows([Path(raw_path)])))

    all_rows = baseline + patching + no_patching
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_path, all_rows)

    lines: list[str] = [
        "# Sim/Code Failure Overlap Report",
        "",
        "This report measures whether `sim` and `code` failures are entangled.",
        "",
        "- `Raw` uses all rows.",
        "- `Parse-normalized` restricts to rows where `sim` parsed successfully and `code` reached `code_err_msg == ok,ok`.",
        "- `Lift` compares observed joint-failure rate against the independent baseline `P(sim fail) * P(code fail)`.",
        "- `Phi` is the binary correlation between sim-fail and code-fail indicators.",
        "",
    ]
    lines.extend(baseline_markdown(baseline))
    lines.append("")
    if patching:
        lines.extend(patch_block("Patching", patching))
        lines.append("")
    if no_patching:
        lines.extend(patch_block("No patching", no_patching))
        lines.append("")

    args.report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
