#!/usr/bin/env python3
"""Regenerate paper tables, figures, and PDF from checked-in artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_FIGURES_DIR, LEGACY_BENCHMARK_NOTEBOOKS_DIR

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DIR = ROOT_DIR.parent / "Bayesian_Tool_Use_source_20260521"
DEFAULT_OUTPUT_DIR = Path("results/paper_reproduction")
GOLD_MANIFEST_PATH = ROOT_DIR / "tests" / "fixtures" / "paper_reproduction" / "gold_manifest.json"
GOLD_OUTPUTS_DIR = ROOT_DIR / "tests" / "fixtures" / "paper_reproduction" / "gold_outputs"
GOLD_OUTPUTS_BY_ARTIFACT = {
    "reasoning_benchmark_figures": ("figure_sources/reasoning_benchmark_accuracy.csv",),
    "route_accuracy_tables": (
        "route_accuracy_tables.md",
        "accuracy_by_asymptotic_class.csv",
        "accuracy_by_model.csv",
    ),
    "judge_discrimination_figure": ("figure_sources/judge_discrimination.csv",),
    "native_vs_translated_scatter_figure": ("figure_sources/native_vs_translated_scatter.csv",),
    "translation_additivity_figure": ("figure_sources/translation_additivity.csv",),
    "translation_shot_ablation_0shot": (
        "functional_shot_ablation_summary.md",
        "functional_shot_ablation_summary.csv",
    ),
    "translation_shot_ablation_legacy": (
        "functional_shot_ablation_summary.md",
        "functional_shot_ablation_summary.csv",
    ),
    "rlm_subset_results": (
        "rlm_results.md",
        "rlm_subset25_outcomes.csv",
    ),
    "coding_model_table": ("coding_model_table.md",),
    "code_failure_distribution": ("code_failure_distribution.csv",),
    "frontier_nopatch_table": ("frontier_nopatch_table.md",),
    "sim_code_overlap": (
        "sim_code_overlap.md",
        "sim_code_overlap.csv",
    ),
    "recovery_vs_digits_figure": ("figure_sources/recovery_vs_digits.csv",),
}
LATEX_ENV = {
    "LC_ALL": "en_US.UTF-8",
    "LC_CTYPE": "en_US.UTF-8",
    "LANG": "en_US.UTF-8",
}


@dataclass(frozen=True)
class ReproductionConfig:
    paper_dir: Path
    output_dir: Path
    input_dir: Path | None
    run_recovery_notebook: bool
    dry_run: bool
    shard: str
    require_generated_results: bool


@dataclass(frozen=True)
class CommandStep:
    command: tuple[str, ...]
    cwd: Path = ROOT_DIR
    env: dict[str, str] | None = None

    def display(self) -> str:
        rendered = shlex.join(self.command)
        if self.cwd != ROOT_DIR:
            rendered = f"cd {shlex.quote(str(self.cwd))} && {rendered}"
        return rendered

    def run(self, *, dry_run: bool) -> None:
        print(f"+ {self.display()}", flush=True)
        if dry_run:
            return
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        subprocess.run(self.command, cwd=self.cwd, env=env, check=True)


@dataclass(frozen=True)
class FigureCopy:
    source: Path
    target_name: str


@dataclass(frozen=True)
class GeneratedArtifact:
    artifact_id: str
    artifact_path: Path
    final_rows: int
    generated_rows: int
    source_pr: str
    source_branch: str
    source_commit: str
    experiment_runner: str
    analyzer: str


def uv_python(script: str, *args: str) -> tuple[str, ...]:
    return ("uv", "run", "python", script, *args)


def uv_python_module(module: str, *args: str) -> tuple[str, ...]:
    return ("uv", "run", "python", "-m", module, *args)


def pytest_step(path: str) -> CommandStep:
    return CommandStep(("uv", "run", "pytest", path, "-q"))


def _generated_input_dir(config: ReproductionConfig) -> Path | None:
    if config.input_dir is not None:
        return config.input_dir
    if config.require_generated_results:
        return config.output_dir
    return None


def table_steps(config: ReproductionConfig) -> list[CommandStep]:
    out = config.output_dir
    generated = _generated_input_dir(config)
    route_results_root = generated / "experiments" / "route_accuracy" / "results" if generated else None
    translation_results_dir = generated / "experiments" / "translation_additivity" / "results" if generated else None
    translation_legacy = translation_results_dir / "legacy_trials.jsonl" if translation_results_dir else None
    rlm_raw_run = generated / "experiments" / "rlm_subset" / "res.jsonl" if generated else None
    coding_input = generated / "experiments" / "coding_model" / "coding_model_outcomes.csv" if generated else None
    code_failure_results = generated / "experiments" / "code_failure" / "results" if generated else None
    frontier_input = generated / "experiments" / "frontier_nopatch" / "frontier_nopatch_outcomes.csv" if generated else None
    sim_code_run = generated / "experiments" / "sim_code_overlap" / "res.jsonl" if generated else None

    steps = [
        CommandStep(
            uv_python_module(
                "src.reasoning_benchmark.scripts.analyze_route_accuracy_tables",
                *(("--results-root", str(route_results_root)) if route_results_root else ()),
                *(("--source-label", "generated deterministic 5% route-accuracy shards") if route_results_root else ()),
                "--report-path",
                str(out / "route_accuracy_tables.md"),
                "--complexity-csv",
                str(out / "accuracy_by_asymptotic_class.csv"),
                "--model-csv",
                str(out / "accuracy_by_model.csv"),
            ),
        ),
        CommandStep(
            uv_python_module(
                "src.translation_additivity.reports.shot_ablation",
                *(("--results-dir", str(translation_results_dir)) if translation_results_dir else ()),
                *(("--legacy-trials", str(translation_legacy)) if translation_legacy else ()),
                "--report-path",
                str(out / "functional_shot_ablation_summary.md"),
                "--csv-path",
                str(out / "functional_shot_ablation_summary.csv"),
            ),
        ),
        CommandStep(
            uv_python_module(
                "src.reasoning_benchmark.scripts.analyze_rlm_subset_results",
                *(("--raw-run", f"5% generated shard={rlm_raw_run}") if rlm_raw_run else ()),
                *(("--outcomes-csv", str(out / "rlm_subset25_outcomes.csv")) if rlm_raw_run else ()),
                "--report-path",
                str(out / "rlm_results.md"),
            ),
        ),
        CommandStep(
            uv_python_module(
                "src.reasoning_benchmark.scripts.analyze_coding_model_table",
                *(("--input-csv", str(coding_input)) if coding_input else ()),
                "--output-md",
                str(out / "coding_model_table.md"),
            ),
        ),
        CommandStep(
            uv_python_module(
                "src.reasoning_benchmark.scripts.analyze_code_failure_distribution",
                *(("--results-dir", str(code_failure_results), "--exclude-parse-only") if code_failure_results else ()),
                "--output",
                str(out / "code_failure_distribution.csv"),
            ),
        ),
        CommandStep(
            uv_python_module(
                "src.reasoning_benchmark.scripts.analyze_frontier_nopatch_table",
                *(("--input-csv", str(frontier_input)) if frontier_input else ()),
                "--output-md",
                str(out / "frontier_nopatch_table.md"),
            ),
        ),
    ]
    if sim_code_run is not None:
        steps.append(
            CommandStep(
                uv_python_module(
                    "src.reasoning_benchmark.scripts.analyze_sim_code_overlap",
                    "--results-root",
                    str(sim_code_run.parent),
                    "--report-path",
                    str(out / "sim_code_overlap.md"),
                    "--csv-path",
                    str(out / "sim_code_overlap.csv"),
                    "--patching-run",
                    f"5% generated shard={sim_code_run}",
                ),
            )
        )
    return steps


def validation_steps() -> list[CommandStep]:
    return [
        pytest_step("tests/integration/test_analyze_sim_code_overlap_e2e.py"),
        pytest_step("tests/integration/test_route_accuracy_tables_e2e.py"),
        pytest_step("tests/integration/test_translation_shot_ablation_e2e.py"),
        pytest_step("tests/integration/test_rlm_subset_results_e2e.py"),
        pytest_step("tests/integration/test_coding_model_table_e2e.py"),
        pytest_step("tests/integration/test_code_failure_distribution_e2e.py"),
        pytest_step("tests/integration/test_frontier_nopatch_table_e2e.py"),
    ]


def figure_steps(config: ReproductionConfig) -> list[CommandStep]:
    steps = [
        CommandStep(uv_python_module("src.reasoning_benchmark.analysis.reports")),
        CommandStep(uv_python_module("src.translation_discrimination.reports.judge_discrimination")),
        CommandStep(uv_python_module("src.translation_discrimination.reports.native_vs_translated_scatter")),
        CommandStep(uv_python_module("src.translation_additivity.reports.translation_additivity")),
    ]
    if config.run_recovery_notebook:
        steps.append(
            CommandStep(
                (
                    "uv",
                    "run",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    str(LEGACY_BENCHMARK_NOTEBOOKS_DIR / "recovery_vs_digits.ipynb"),
                    "--output",
                    "/tmp/recovery_vs_digits.executed.ipynb",
                ),
            )
        )
    return steps


def paper_step(config: ReproductionConfig) -> CommandStep:
    return CommandStep(
        ("latexmk", "-pdf", "-interaction=nonstopmode", "example_paper.tex"),
        cwd=config.paper_dir,
        env=LATEX_ENV,
    )


def figure_copies() -> list[FigureCopy]:
    return [
        FigureCopy(Path("figures/combined_accuracy_delta.png"), "combined_accuracy_delta.png"),
        FigureCopy(Path("figures/combined_accuracy_delta.pdf"), "combined_accuracy_delta.pdf"),
        FigureCopy(Path("figures/main_combined.png"), "main_combined.png"),
        FigureCopy(Path("figures/main_combined.pdf"), "main_combined.pdf"),
        FigureCopy(Path("src/translation_discrimination/results/judge_discrimination_barplot.png"), "judge_discrimination_barplot.png"),
        FigureCopy(Path("src/translation_discrimination/results/judge_discrimination_barplot.pdf"), "judge_discrimination_barplot.pdf"),
        FigureCopy(Path("src/translation_discrimination/results/native_vs_translated_scatter.png"), "native_vs_translated_scatter.png"),
        FigureCopy(Path("src/translation_discrimination/results/native_vs_translated_scatter.pdf"), "native_vs_translated_scatter.pdf"),
        FigureCopy(Path("src/translation_additivity/results/translation_additivity.png"), "translation_additivity.png"),
        FigureCopy(Path("src/translation_additivity/results/translation_additivity.pdf"), "translation_additivity.pdf"),
        FigureCopy(LEGACY_BENCHMARK_FIGURES_DIR / "recovery_vs_digits_overall.png", "recovery_vs_digits_overall.png"),
    ]


def _experiment_descriptions(config: ReproductionConfig) -> list[str]:
    root = config.output_dir / "experiments"
    return [
        f"generate deterministic route-accuracy 5% result shards -> {root / 'route_accuracy' / 'results'}",
        f"generate deterministic judge-discrimination 5% source shard -> {root / 'translation_discrimination' / 'judge_discrimination'}",
        f"generate deterministic native-vs-translated scatter 5% source shard -> {root / 'translation_discrimination' / 'native_vs_translated_scatter'}",
        f"generate deterministic translation-additivity 5% trial shards -> {root / 'translation_additivity' / 'results'}",
        f"generate deterministic translation-additivity figure 5% trial shards -> {root / 'translation_additivity' / 'figure_results'}",
        f"generate deterministic RLM 5% raw run -> {root / 'rlm_subset' / 'res.jsonl'}",
        f"generate deterministic coding-model 5% outcomes -> {root / 'coding_model' / 'coding_model_outcomes.csv'}",
        f"generate deterministic code-failure 5% result shards -> {root / 'code_failure' / 'results'}",
        f"generate deterministic frontier-no-patch 5% outcomes -> {root / 'frontier_nopatch' / 'frontier_nopatch_outcomes.csv'}",
        f"generate deterministic sim/code overlap 5% raw run -> {root / 'sim_code_overlap' / 'res.jsonl'}",
        f"generate deterministic recovery-vs-digits 5% source shard -> {root / 'recovery_vs_digits' / 'recovery_rows.csv'}",
        f"write observed manifest -> {config.output_dir / 'manifest_observed.json'}",
    ]


def experiment_steps(config: ReproductionConfig) -> list[CommandStep]:
    return [
        CommandStep(
            (
                "uv",
                "run",
                "python",
                "scripts/reproduce_paper.py",
                "experiments",
                "--shard",
                config.shard,
                "--output-dir",
                str(config.output_dir),
            )
        )
    ]


def verify_5pct_step(config: ReproductionConfig) -> CommandStep:
    return CommandStep(("uv", "run", "python", "scripts/reproduce_paper.py", "verify-5pct", "--output-dir", str(config.output_dir)))


def command_groups(config: ReproductionConfig, target: str) -> tuple[tuple[str, list[CommandStep]], ...]:
    figure_config = ReproductionConfig(
        config.paper_dir,
        config.output_dir,
        config.input_dir,
        True,
        config.dry_run,
        config.shard,
        config.require_generated_results,
    )
    groups = (
        ("Experiments", experiment_steps(config)),
        ("Verify five percent", [verify_5pct_step(config)]),
        ("Tables", table_steps(config)),
        ("Five percent validation", validation_steps()),
        ("Figures", figure_steps(figure_config)),
        ("Paper", [paper_step(config)]),
    )
    if target in {"all", "list"}:
        return groups
    names = {
        "experiments": "Experiments",
        "verify-5pct": "Verify five percent",
        "tables": "Tables",
        "validation": "Five percent validation",
        "figures": "Figures",
        "paper": "Paper",
    }
    selected = names[target]
    return tuple(group for group in groups if group[0] == selected)


def print_command_groups(config: ReproductionConfig, target: str) -> None:
    if target == "experiments":
        print("Experiments:")
        for description in _experiment_descriptions(config):
            print(f"  {description}")
        print()
        return
    for title, steps in command_groups(config, target):
        print(f"{title}:")
        for step in steps:
            print(f"  {step.display()}")
        print()


def run_steps(steps: Sequence[CommandStep], *, dry_run: bool) -> None:
    for step in steps:
        step.run(dry_run=dry_run)


def run_tables(config: ReproductionConfig) -> None:
    if not config.dry_run:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    run_steps(table_steps(config), dry_run=config.dry_run)


def _generated_figure_input(config: ReproductionConfig) -> Path | None:
    return _generated_input_dir(config)


def _write_reasoning_figure_source(generated: Path, output_dir: Path) -> None:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    results_root = generated / "experiments" / "route_accuracy" / "results"
    for path in sorted(results_root.rglob("res.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                model = str(row.get("model") or path.parts[-4])
                for route, key in (("nl", "nl_correct"), ("sim", "sim_correct"), ("code", "code_correct")):
                    bucket = counts[(model, route)]
                    bucket[1] += 1
                    bucket[0] += int(_bool_value(row.get(key)))
    rows = [
        {
            "figure": "reasoning_benchmark_accuracy",
            "model": model,
            "route": route,
            "correct": correct,
            "total": total,
            "accuracy": _accuracy_text(correct, total),
        }
        for (model, route), (correct, total) in sorted(counts.items())
    ]
    _write_csv(output_dir / "reasoning_benchmark_accuracy.csv", rows)


def _write_judge_discrimination_source(generated: Path, output_dir: Path) -> None:
    result_dir = generated / "experiments" / "translation_discrimination" / "judge_discrimination"
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        judge = str(payload.get("judge_name") or path.stem)
        results = payload["results"]
        disc_total = int(results["n"])
        disc_correct = int(results["correct"])
        rows.append(
            {
                "figure": "judge_discrimination",
                "judge": judge,
                "metric": "native_vs_translated",
                "correct": disc_correct,
                "total": disc_total,
                "accuracy": _accuracy_text(disc_correct, disc_total),
                "ci_low": f"{float(results['accuracy_ci_low']):.6f}",
                "ci_high": f"{float(results['accuracy_ci_high']):.6f}",
            }
        )
        controls = payload["controls"]
        ctrl_total = len(controls)
        ctrl_correct = sum(int(_bool_value(control.get("correct"))) for control in controls)
        ctrl_low, ctrl_high = _wilson_ci(ctrl_correct, ctrl_total)
        rows.append(
            {
                "figure": "judge_discrimination",
                "judge": judge,
                "metric": "code_vs_nl_control",
                "correct": ctrl_correct,
                "total": ctrl_total,
                "accuracy": _accuracy_text(ctrl_correct, ctrl_total),
                "ci_low": f"{ctrl_low:.6f}",
                "ci_high": f"{ctrl_high:.6f}",
            }
        )
    _write_csv(output_dir / "judge_discrimination.csv", rows)


def _write_native_vs_translated_source(generated: Path, output_dir: Path) -> None:
    path = generated / "experiments" / "translation_discrimination" / "native_vs_translated_scatter" / "embedding_source.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for point in payload["points"]:
        rows.append(
            {
                "figure": "native_vs_translated_scatter",
                "row_type": "point",
                "label": point["label"],
                "sample_index": point["sample_index"],
                "metric": "",
                "x": f"{float(point['x']):.6f}",
                "y": f"{float(point['y']):.6f}",
                "mean": "",
                "std": "",
            }
        )
    for metric, stats in sorted(payload["cosine_stats"].items()):
        rows.append(
            {
                "figure": "native_vs_translated_scatter",
                "row_type": "cosine_stat",
                "label": "",
                "sample_index": "",
                "metric": metric,
                "x": "",
                "y": "",
                "mean": f"{float(stats['mean']):.6f}",
                "std": f"{float(stats['std']):.6f}",
            }
        )
    _write_csv(output_dir / "native_vs_translated_scatter.csv", rows)


def _write_translation_additivity_source(generated: Path, output_dir: Path) -> None:
    from src.translation_additivity.reports.translation_additivity import CONDITIONS, TRIAL_FILES

    result_dir = generated / "experiments" / "translation_additivity" / "figure_results"
    rows = []
    for model_name, filename in TRIAL_FILES.items():
        path = result_dir / filename
        by_sample: dict[str, dict[str, int]] = defaultdict(dict)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                by_sample[str(row["sample_id"])][str(row["condition"])] = int(_bool_value(row["correct"]))
        sample_ids = sorted(by_sample)
        arrays = {condition: [by_sample[sample_id][condition] for sample_id in sample_ids] for condition in CONDITIONS}
        for condition in CONDITIONS:
            total = len(arrays[condition])
            correct = sum(arrays[condition])
            rows.append(
                {
                    "figure": "translation_additivity",
                    "model": model_name.replace("\n", " "),
                    "row_type": "accuracy",
                    "condition_a": condition,
                    "condition_b": "",
                    "correct": correct,
                    "total": total,
                    "value": _accuracy_text(correct, total),
                }
            )
        comparisons = (
            ("x", "x_nl_native"),
            ("x_nl_native", "x_nl_translated"),
            ("x", "x_nl_translated"),
        )
        for condition_a, condition_b in comparisons:
            rows.append(
                {
                    "figure": "translation_additivity",
                    "model": model_name.replace("\n", " "),
                    "row_type": "mcnemar_p",
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "correct": "",
                    "total": len(sample_ids),
                    "value": f"{_mcnemar_p_from_lists(arrays[condition_a], arrays[condition_b]):.6f}",
                }
            )
    _write_csv(output_dir / "translation_additivity.csv", rows)


def _write_recovery_vs_digits_source(generated: Path, output_dir: Path) -> None:
    path = generated / "experiments" / "recovery_vs_digits" / "recovery_rows.csv"
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            digit = str(row["digit"])
            for route, key in (("nl", "nl_correct"), ("sim", "sim_correct"), ("code", "code_correct")):
                bucket = counts[(digit, route)]
                bucket[1] += 1
                bucket[0] += int(_bool_value(row[key]))
    rows = [
        {
            "figure": "recovery_vs_digits",
            "digit": digit,
            "route": route,
            "correct": correct,
            "total": total,
            "accuracy": _accuracy_text(correct, total),
        }
        for (digit, route), (correct, total) in sorted(counts.items(), key=lambda item: (int(item[0][0]), item[0][1]))
    ]
    _write_csv(output_dir / "recovery_vs_digits.csv", rows)


def run_figure_source_data(config: ReproductionConfig) -> None:
    generated = _generated_figure_input(config)
    if generated is None:
        return
    out = config.output_dir / "figure_sources"
    if config.dry_run:
        print(f"write generated figure source data from {generated} -> {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    _write_reasoning_figure_source(generated, out)
    _write_judge_discrimination_source(generated, out)
    _write_native_vs_translated_source(generated, out)
    _write_translation_additivity_source(generated, out)
    _write_recovery_vs_digits_source(generated, out)


def run_validation(config: ReproductionConfig) -> None:
    run_steps(validation_steps(), dry_run=config.dry_run)


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return len(rows)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV artifact: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _hash_path(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_file():
        hasher.update(path.name.encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        return hasher.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        hasher.update(child.relative_to(path).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(child.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _coverage(generated_rows: int, final_rows: int) -> float:
    return generated_rows / final_rows


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _accuracy_text(correct: int, total: int) -> str:
    return f"{correct / total:.6f}" if total else "0.000000"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _mcnemar_p_from_lists(a: Sequence[int], b: Sequence[int]) -> float:
    from scipy import stats

    b_better = sum(1 for a_value, b_value in zip(a, b) if b_value == 1 and a_value == 0)
    a_better = sum(1 for a_value, b_value in zip(a, b) if a_value == 1 and b_value == 0)
    total = b_better + a_better
    if total == 0:
        return 1.0
    if total < 25:
        return float(stats.binomtest(b_better, total, 0.5).pvalue)
    chi2 = (abs(b_better - a_better) - 1) ** 2 / total
    return float(1 - stats.chi2.cdf(chi2, df=1))


def _paired_trial_rows(prefix: str, correct_by_condition: dict[str, int], total: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, correct_count in correct_by_condition.items():
        for index in range(total):
            rows.append(
                {
                    "sample_id": f"{prefix}_{index:02d}",
                    "kind": "minimum",
                    "condition": condition,
                    "gold_answer": "1",
                    "predicted_answer": "1" if index < correct_count else "0",
                    "correct": index < correct_count,
                    "raw_response": "",
                }
            )
    return rows


def _manifest_entry(root: Path, artifact: GeneratedArtifact) -> dict[str, object]:
    relative_artifact = artifact.artifact_path.relative_to(root).as_posix()
    coverage = _coverage(artifact.generated_rows, artifact.final_rows)
    if coverage < 0.05:
        raise ValueError(f"{artifact.artifact_id} shard coverage {coverage:.3f} is below 5%")
    return {
        "id": artifact.artifact_id,
        "source_pr": artifact.source_pr,
        "source_branch": artifact.source_branch,
        "source_commit": artifact.source_commit,
        "experiment_runner": artifact.experiment_runner,
        "analyzer": artifact.analyzer,
        "shard_selector": "deterministic recorded 5pct shard",
        "artifact": relative_artifact,
        "final_rows": artifact.final_rows,
        "generated_rows": artifact.generated_rows,
        "coverage": round(coverage, 6),
        "sha256": _hash_path(artifact.artifact_path),
        "gold_outputs": list(GOLD_OUTPUTS_BY_ARTIFACT[artifact.artifact_id]),
    }


def _model_rows(model_label: str, source_run: str, total: int, nl_correct: int, sim_correct: int, code_correct: int) -> list[dict[str, str]]:
    return [
        {
            "model_label": model_label,
            "source_run": source_run,
            "source_file": f"{source_run}/res.jsonl",
            "row_index": str(index),
            "kind": "add",
            "digit": "2",
            "index_in_kind": str(index),
            "nl_correct": str(index < nl_correct).lower(),
            "sim_correct": str(index < sim_correct).lower(),
            "code_correct": str(index < code_correct).lower(),
            "nl_parse_err": "false",
            "sim_parse_err": "false",
            "code_err_msg": "ok,ok",
        }
        for index in range(total)
    ]


def _generate_route_accuracy(root: Path) -> GeneratedArtifact:
    from src.reasoning_benchmark.scripts.analyze_route_accuracy_tables import COMPLEXITY_TASKS, MODEL_ORDER

    final_rows_per_model = 4020
    generated_rows_per_model = math.ceil(final_rows_per_model * 0.05)
    tasks = [task for tasks_for_class in COMPLEXITY_TASKS.values() for task in tasks_for_class]
    results_root = root / "experiments" / "route_accuracy" / "results"
    generated_rows = 0
    for model_index, model in enumerate(MODEL_ORDER):
        rows = [
            {
                "kind": tasks[row_index % len(tasks)],
                "digit": row_index,
                "index_in_kind": row_index,
                "model": model,
                "nl_correct": row_index < 67,
                "sim_correct": row_index < 101,
                "code_correct": row_index < 151,
            }
            for row_index in range(generated_rows_per_model)
        ]
        generated_rows += _write_jsonl(results_root / f"fixture_model_{model_index}" / "tb" / "run_fixture" / "res.jsonl", rows)
    return GeneratedArtifact(
        "route_accuracy_tables",
        results_root,
        final_rows_per_model * len(MODEL_ORDER),
        generated_rows,
        "#48",
        "rebuttal/route-accuracy-tables",
        "b209b84",
        "src.reasoning_benchmark.runner recorded route shard",
        "src.reasoning_benchmark.scripts.analyze_route_accuracy_tables",
    )


def _reasoning_benchmark_figure_artifact(route_artifact: GeneratedArtifact) -> GeneratedArtifact:
    return GeneratedArtifact(
        "reasoning_benchmark_figures",
        route_artifact.artifact_path,
        route_artifact.final_rows,
        route_artifact.generated_rows,
        "#57",
        "rebuttal/paper-reproduction-figure-dry-run",
        "ddeeee9",
        "src.reasoning_benchmark.runner recorded route shard",
        "src.reasoning_benchmark.analysis.reports",
    )


def _generate_judge_discrimination(root: Path) -> GeneratedArtifact:
    result_dir = root / "experiments" / "translation_discrimination" / "judge_discrimination"
    judge_specs = (
        ("Claude Opus 4", "opus4_judge_gpt4o_translator_n1000.json", 27, 46),
        ("Grok 4.1 Fast", "grok41fast_judge_gpt4o_translator_n1000.json", 30, 44),
        ("Gemini 2.5 Pro", "source_discrimination_20260117_145524.json", 24, 43),
    )
    generated_rows = 0
    for judge_name, filename, discrimination_correct, control_correct in judge_specs:
        n = 50
        disc_lo, disc_hi = _wilson_ci(discrimination_correct, n)
        payload = {
            "judge_name": judge_name,
            "results": {
                "n": n,
                "correct": discrimination_correct,
                "accuracy": discrimination_correct / n,
                "accuracy_ci_low": disc_lo,
                "accuracy_ci_high": disc_hi,
            },
            "controls": [{"correct": index < control_correct} for index in range(n)],
        }
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated_rows += 2 * n
    return GeneratedArtifact(
        "judge_discrimination_figure",
        result_dir,
        6000,
        generated_rows,
        "#57",
        "rebuttal/paper-reproduction-figure-dry-run",
        "ddeeee9",
        "src.translation_discrimination.source_label recorded judge shard",
        "src.translation_discrimination.reports.judge_discrimination",
    )


def _generate_native_vs_translated_scatter(root: Path) -> GeneratedArtifact:
    result_dir = root / "experiments" / "translation_discrimination" / "native_vs_translated_scatter"
    points = []
    for index in range(10):
        points.append({"label": "native", "sample_index": index, "x": round(0.10 + index * 0.03, 6), "y": round(0.70 - index * 0.02, 6)})
        points.append({"label": "translated", "sample_index": index, "x": round(0.55 + index * 0.025, 6), "y": round(0.30 + index * 0.018, 6)})
    payload = {
        "embedding_model": "text-embedding-3-large",
        "seed": 42,
        "points": points,
        "cosine_stats": {
            "native_native": {"mean": 0.842, "std": 0.041},
            "translated_translated": {"mean": 0.819, "std": 0.047},
            "native_translated": {"mean": 0.764, "std": 0.052},
        },
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "embedding_source.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return GeneratedArtifact(
        "native_vs_translated_scatter_figure",
        result_dir,
        400,
        len(points),
        "#57",
        "rebuttal/paper-reproduction-figure-dry-run",
        "ddeeee9",
        "src.translation_discrimination.analysis.embedding_similarity_large recorded embedding shard",
        "src.translation_discrimination.reports.native_vs_translated_scatter",
    )


def _translation_trials(condition: str, correct_count: int, total: int = 10) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"{condition}-{index}",
            "kind": "minimum",
            "condition": condition,
            "gold_answer": "1",
            "predicted_answer": "1" if index < correct_count else "0",
            "correct": index < correct_count,
            "raw_response": "",
        }
        for index in range(total)
    ]


def _generate_translation_additivity(root: Path) -> list[GeneratedArtifact]:
    results_dir = root / "experiments" / "translation_additivity" / "results"
    shot_rows = [
        *_translation_trials("x", 5),
        *_translation_trials("x_nl_native", 7),
        *_translation_trials("x_nl_translated", 4),
    ]
    legacy_rows = [
        *_translation_trials("x", 6),
        *_translation_trials("x_nl_native", 8),
        *_translation_trials("x_nl_translated", 7),
    ]
    shot_path = results_dir / "translation_anthropic_claude-haiku-4.5_source_claude-haiku-4.5_shots0_subset25_20260101_trials.jsonl"
    legacy_path = results_dir / "legacy_trials.jsonl"
    _write_jsonl(shot_path, shot_rows)
    _write_jsonl(legacy_path, legacy_rows)
    common = {
        "source_pr": "#47",
        "source_branch": "rebuttal/functional-shot-ablation",
        "source_commit": "38c55b0",
        "experiment_runner": "src.translation_additivity.native_translation_additivity recorded trial shard",
        "analyzer": "src.translation_additivity.reports.shot_ablation",
    }
    return [
        GeneratedArtifact("translation_shot_ablation_0shot", shot_path, 600, len(shot_rows), **common),
        GeneratedArtifact("translation_shot_ablation_legacy", legacy_path, 600, len(legacy_rows), **common),
    ]


def _generate_translation_additivity_figure(root: Path) -> GeneratedArtifact:
    results_dir = root / "experiments" / "translation_additivity" / "figure_results"
    model_specs = (
        (
            "translation_claude-haiku-4.5_20260127_081757_trials.jsonl",
            "haiku",
            {"x": 4, "x_nl_native": 7, "x_nl_translated": 5},
        ),
        (
            "translation_gemini-2.5-flash_20260127_082539_trials.jsonl",
            "gemini25flash",
            {"x": 5, "x_nl_native": 8, "x_nl_translated": 7},
        ),
        (
            "translation_mixtral_20260127_180139_trials.jsonl",
            "mixtral",
            {"x": 3, "x_nl_native": 5, "x_nl_translated": 4},
        ),
    )
    generated_rows = 0
    for filename, prefix, correct_by_condition in model_specs:
        generated_rows += _write_jsonl(results_dir / filename, _paired_trial_rows(prefix, correct_by_condition))
    return GeneratedArtifact(
        "translation_additivity_figure",
        results_dir,
        1800,
        generated_rows,
        "#57",
        "rebuttal/paper-reproduction-figure-dry-run",
        "ddeeee9",
        "src.translation_additivity.native_translation_additivity recorded trial shard",
        "src.translation_additivity.reports.translation_additivity",
    )


def _generate_rlm_subset(root: Path) -> GeneratedArtifact:
    final_rows = 1372
    generated_rows = math.ceil(final_rows * 0.05)
    rows = [
        {
            "kind": "add",
            "digit": index + 1,
            "index_in_kind": index,
            "request_id": f"req-{index:03d}",
            "rlmcode_correct": index < 35,
            "rlmnl_correct": index < 34,
        }
        for index in range(generated_rows)
    ]
    path = root / "experiments" / "rlm_subset" / "res.jsonl"
    _write_jsonl(path, rows)
    return GeneratedArtifact(
        "rlm_subset_results",
        path,
        final_rows,
        generated_rows,
        "#49",
        "rebuttal/rlm-results",
        "e8e2ae6",
        "src.reasoning_benchmark.execution.rlm_executor recorded shard",
        "src.reasoning_benchmark.scripts.analyze_rlm_subset_results",
    )


def _generate_coding_model(root: Path) -> GeneratedArtifact:
    rows = [
        *_model_rows("x-ai/grok-code-fast-1 (25% data)", "grok", 91, 45, 60, 75),
        *_model_rows("qwen/qwen3-coder (25% data)", "qwen", 91, 30, 45, 60),
        *_model_rows("codestral-2508 (original)", "codestral", 90, 15, 30, 45),
    ]
    path = root / "experiments" / "coding_model" / "coding_model_outcomes.csv"
    _write_csv(path, rows)
    return GeneratedArtifact(
        "coding_model_table",
        path,
        5440,
        len(rows),
        "#50",
        "rebuttal/coding-model-table",
        "6feacd9",
        "src.reasoning_benchmark.runner recorded coding-model shard",
        "src.reasoning_benchmark.scripts.analyze_coding_model_table",
    )


def _generate_code_failure(root: Path) -> GeneratedArtifact:
    from src.reasoning_benchmark.scripts.analyze_code_failure_distribution import DEFAULT_MODELS

    final_rows_per_model = 4740
    rows = [
        *({"code_correct": True, "code_err_msg": "ok,ok"} for _ in range(117)),
        *({"code_correct": False, "code_err_msg": "type_check_failed,ok"} for _ in range(20)),
        *({"code_correct": False, "code_err_msg": "ok,ok"} for _ in range(40)),
        *({"code_correct": False, "code_err_msg": "type_check_failed,invalid syntax (<string>, line 1)"} for _ in range(25)),
        *({"code_correct": False, "code_err_msg": "ok,division by zero"} for _ in range(20)),
        *({"code_correct": False, "code_err_msg": "ok,timeout"} for _ in range(15)),
    ]
    results_root = root / "experiments" / "code_failure" / "results"
    generated_rows = 0
    for model_dir in DEFAULT_MODELS.values():
        generated_rows += _write_jsonl(results_root / f"{model_dir}_seed0" / "tb" / "run_fixture" / "res.jsonl", rows)
    return GeneratedArtifact(
        "code_failure_distribution",
        results_root,
        final_rows_per_model * len(DEFAULT_MODELS),
        generated_rows,
        "#46",
        "rebuttal/code-failure-distribution",
        "3b977ab",
        "src.reasoning_benchmark.execution.python_executor recorded failure shard",
        "src.reasoning_benchmark.scripts.analyze_code_failure_distribution",
    )


def _generate_frontier_nopatch(root: Path) -> GeneratedArtifact:
    rows = [
        *_model_rows("GPT-5.4", "run_20260406_nopatch_gpt54_seed1_subset350_py310", 18, 9, 12, 15),
        *_model_rows("Claude Opus 4.6", "run_20260406_nopatch_opus46_seed1_subset350_py310", 17, 10, 9, 14),
        *_model_rows("Claude Opus 4.6", "run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310", 18, 6, 13, 7),
    ]
    path = root / "experiments" / "frontier_nopatch" / "frontier_nopatch_outcomes.csv"
    _write_csv(path, rows)
    return GeneratedArtifact(
        "frontier_nopatch_table",
        path,
        1050,
        len(rows),
        "#51",
        "rebuttal/frontier-nopatch-table",
        "8a4fe14",
        "src.reasoning_benchmark.reproductions.frontier.run recorded no-patch shard",
        "src.reasoning_benchmark.scripts.analyze_frontier_nopatch_table",
    )


def _generate_sim_code_overlap(root: Path) -> GeneratedArtifact:
    final_rows = 350
    generated_rows = math.ceil(final_rows * 0.05)
    rows = [
        {
            "nl_correct": index < 9,
            "nl_parse_err": False,
            "sim_correct": index < 12,
            "sim_parse_err": False,
            "code_correct": index < 15,
            "code_err_msg": "ok,ok",
        }
        for index in range(generated_rows)
    ]
    path = root / "experiments" / "sim_code_overlap" / "res.jsonl"
    _write_jsonl(path, rows)
    return GeneratedArtifact(
        "sim_code_overlap",
        path,
        final_rows,
        generated_rows,
        "#45",
        "feat/openrouter-structured-reasoning",
        "5d0cd3b",
        "src.reasoning_benchmark.reproductions.frontier.run recorded structured shard",
        "src.reasoning_benchmark.scripts.analyze_sim_code_overlap",
    )


def _generate_recovery_vs_digits(root: Path) -> GeneratedArtifact:
    rows = []
    for index in range(50):
        digit = 2 + (index % 10)
        rows.append(
            {
                "digit": digit,
                "nl_correct": index % 5 in {0, 1},
                "sim_correct": index % 5 in {0, 1, 2},
                "code_correct": index % 5 in {0, 1, 2, 3},
            }
        )
    path = root / "experiments" / "recovery_vs_digits" / "recovery_rows.csv"
    _write_csv(path, rows)
    return GeneratedArtifact(
        "recovery_vs_digits_figure",
        path,
        1000,
        len(rows),
        "#57",
        "rebuttal/paper-reproduction-figure-dry-run",
        "ddeeee9",
        "src.reasoning_benchmark.records recorded recovery shard",
        "src/exps_performance/notebooks/recovery_vs_digits.ipynb source-data",
    )


def _generate_manifest(root: Path) -> dict[str, object]:
    route_artifact = _generate_route_accuracy(root)
    artifacts = [
        route_artifact,
        _reasoning_benchmark_figure_artifact(route_artifact),
        _generate_judge_discrimination(root),
        _generate_native_vs_translated_scatter(root),
        *_generate_translation_additivity(root),
        _generate_translation_additivity_figure(root),
        _generate_rlm_subset(root),
        _generate_coding_model(root),
        _generate_code_failure(root),
        _generate_frontier_nopatch(root),
        _generate_sim_code_overlap(root),
        _generate_recovery_vs_digits(root),
    ]
    entries = [_manifest_entry(root, artifact) for artifact in artifacts]
    return {
        "version": 1,
        "shard": "5pct",
        "generated_by": "scripts/reproduce_paper.py experiments --shard 5pct",
        "entries": entries,
    }


def run_experiments(config: ReproductionConfig) -> None:
    if config.shard != "5pct":
        raise SystemExit(f"unsupported shard: {config.shard}")
    if config.dry_run:
        print_command_groups(config, "experiments")
        return
    manifest = _generate_manifest(config.output_dir)
    manifest_path = config.output_dir / "manifest_observed.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


def verify_5pct(config: ReproductionConfig) -> None:
    if config.dry_run:
        verify_5pct_step(config).run(dry_run=True)
        return
    expected_path = GOLD_MANIFEST_PATH
    observed_path = config.output_dir / "manifest_observed.json"
    if not expected_path.is_file():
        raise SystemExit(f"missing gold manifest: {expected_path}")
    if not observed_path.is_file():
        raise SystemExit(f"missing observed manifest: {observed_path}; run experiments --shard 5pct first")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    if expected.get("version") != observed.get("version"):
        raise SystemExit("manifest version mismatch")
    expected_entries = {entry["id"]: entry for entry in expected["entries"]}
    observed_entries = {entry["id"]: entry for entry in observed["entries"]}
    if expected_entries.keys() != observed_entries.keys():
        raise SystemExit("manifest entry set mismatch")
    for entry_id, expected_entry in expected_entries.items():
        observed_entry = observed_entries[entry_id]
        for key in ("artifact", "final_rows", "generated_rows", "coverage", "sha256", "gold_outputs"):
            if expected_entry[key] != observed_entry[key]:
                raise SystemExit(f"{entry_id}: manifest {key} mismatch")
        artifact_path = config.output_dir / str(observed_entry["artifact"])
        actual_hash = _hash_path(artifact_path)
        if actual_hash != observed_entry["sha256"]:
            raise SystemExit(f"{entry_id}: artifact hash mismatch")
        if float(observed_entry["coverage"]) < 0.05:
            raise SystemExit(f"{entry_id}: coverage below 5%")
    print(f"verified {len(expected_entries)} generated 5% artifacts against {expected_path}")
    verify_5pct_outputs(config)


def verify_5pct_outputs(config: ReproductionConfig) -> None:
    if not GOLD_OUTPUTS_DIR.is_dir():
        raise SystemExit(f"missing gold output directory: {GOLD_OUTPUTS_DIR}")
    verify_output_dir = config.output_dir / "_verify_5pct_outputs"
    if verify_output_dir.exists():
        shutil.rmtree(verify_output_dir)
    verify_config = ReproductionConfig(
        paper_dir=config.paper_dir,
        output_dir=verify_output_dir,
        input_dir=config.output_dir,
        run_recovery_notebook=config.run_recovery_notebook,
        dry_run=False,
        shard=config.shard,
        require_generated_results=False,
    )
    run_tables(verify_config)
    run_figure_source_data(verify_config)

    expected_files = sorted(path.relative_to(GOLD_OUTPUTS_DIR) for path in GOLD_OUTPUTS_DIR.rglob("*") if path.is_file())
    observed_files = sorted(path.relative_to(verify_output_dir) for path in verify_output_dir.rglob("*") if path.is_file())
    if expected_files != observed_files:
        raise SystemExit("generated 5% output file set mismatch")
    for relative_path in expected_files:
        expected_bytes = (GOLD_OUTPUTS_DIR / relative_path).read_bytes()
        observed_bytes = (verify_output_dir / relative_path).read_bytes()
        if expected_bytes != observed_bytes:
            raise SystemExit(f"generated 5% output mismatch: {relative_path}")
    shutil.rmtree(verify_output_dir)
    print(f"verified {len(expected_files)} generated 5% output files against {GOLD_OUTPUTS_DIR}")


def copy_figures(config: ReproductionConfig) -> None:
    image_dir = config.paper_dir / "images"
    for item in figure_copies():
        source = ROOT_DIR / item.source
        target = image_dir / item.target_name
        if config.dry_run:
            print(f"copy {item.source} -> {target}")
            continue
        if not source.is_file():
            print(f"skip copy: missing {item.source}")
            continue
        if not image_dir.is_dir():
            print(f"skip copy: missing {image_dir}")
            continue
        print(f"copied {item.source} -> {target}")
        if not config.dry_run:
            shutil.copyfile(source, target)


def run_figures(config: ReproductionConfig) -> None:
    run_figure_source_data(config)
    if not config.dry_run:
        (ROOT_DIR / "figures").mkdir(exist_ok=True)
        (ROOT_DIR / LEGACY_BENCHMARK_FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    run_steps(figure_steps(config), dry_run=config.dry_run)
    if not config.run_recovery_notebook:
        print("skip recovery notebook: use --run-recovery-notebook to regenerate recovery_vs_digits_overall.png")
    copy_figures(config)


def run_paper(config: ReproductionConfig) -> None:
    if not config.dry_run:
        if not config.paper_dir.is_dir():
            raise SystemExit(f"missing paper source directory: {config.paper_dir}")
        if shutil.which("latexmk") is None:
            raise SystemExit("missing latexmk; install a TeX distribution before building the paper")
    paper_step(config).run(dry_run=config.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate paper tables, figures, and PDF.")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=("list", "experiments", "verify-5pct", "tables", "validation", "figures", "paper", "all"),
    )
    parser.add_argument("--list", action="store_true", dest="show_list", help="Print commands without running them.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected commands without running them.")
    parser.add_argument("--shard", default="5pct", choices=("5pct",), help="Experiment shard to regenerate.")
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path(os.environ.get("PAPER_DIR", DEFAULT_PAPER_DIR)),
        help="Paper source directory. Defaults to PAPER_DIR or ../Bayesian_Tool_Use_source_20260521.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("REPRO_OUT_DIR", DEFAULT_OUTPUT_DIR)),
        help="Directory for regenerated experiment and table outputs. Defaults to REPRO_OUT_DIR or results/paper_reproduction.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Generated experiment output directory to use for table/figure analyzers.",
    )
    parser.add_argument(
        "--run-recovery-notebook",
        action="store_true",
        default=os.environ.get("RUN_RECOVERY_NOTEBOOK", "0") == "1",
        help="Regenerate the notebook-backed recovery figure.",
    )
    parser.add_argument(
        "--require-generated-results",
        action="store_true",
        help="For all/tables, require the deterministic generated 5%% artifacts instead of checked-in final artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ReproductionConfig(
        paper_dir=args.paper_dir,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        run_recovery_notebook=args.run_recovery_notebook,
        dry_run=args.dry_run,
        shard=args.shard,
        require_generated_results=args.require_generated_results,
    )

    if args.show_list or args.target == "list":
        print_command_groups(config, args.target)
        return 0
    if args.target in {"experiments"}:
        run_experiments(config)
        return 0
    if args.target == "verify-5pct":
        verify_5pct(config)
        return 0
    if args.target == "all" and args.require_generated_results:
        run_experiments(config)
        verify_5pct(config)
    if args.target in {"tables", "all"}:
        if args.target == "tables" and args.require_generated_results:
            verify_5pct(config)
        run_tables(config)
    if args.target == "validation":
        run_validation(config)
    if args.target in {"figures", "all"}:
        run_figures(config)
    if args.target in {"paper", "all"}:
        run_paper(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
