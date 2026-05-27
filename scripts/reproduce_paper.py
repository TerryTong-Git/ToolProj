#!/usr/bin/env python3
"""Regenerate paper tables, figures, and PDF from checked-in artifacts."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DIR = ROOT_DIR.parent / "Bayesian_Tool_Use_source_20260521"
DEFAULT_OUTPUT_DIR = Path("results/paper_reproduction")
LATEX_ENV = {
    "LC_ALL": "en_US.UTF-8",
    "LC_CTYPE": "en_US.UTF-8",
    "LANG": "en_US.UTF-8",
}


@dataclass(frozen=True)
class ReproductionConfig:
    paper_dir: Path
    output_dir: Path
    run_recovery_notebook: bool
    dry_run: bool


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


def uv_python(script: str, *args: str) -> tuple[str, ...]:
    return ("uv", "run", "python", script, *args)


def pytest_step(path: str) -> CommandStep:
    return CommandStep(("uv", "run", "pytest", path, "-q"))


def table_steps(config: ReproductionConfig) -> list[CommandStep]:
    out = config.output_dir
    return [
        CommandStep(
            uv_python(
                "src/exps_performance/scripts/analyze_route_accuracy_tables.py",
                "--report-path",
                str(out / "route_accuracy_tables.md"),
                "--complexity-csv",
                str(out / "accuracy_by_asymptotic_class.csv"),
                "--model-csv",
                str(out / "accuracy_by_model.csv"),
            ),
        ),
        CommandStep(
            uv_python(
                "src/exps_functional/scripts/analyze_translation_shot_ablation.py",
                "--report-path",
                str(out / "functional_shot_ablation_summary.md"),
                "--csv-path",
                str(out / "functional_shot_ablation_summary.csv"),
            ),
        ),
        CommandStep(
            uv_python(
                "src/exps_performance/scripts/analyze_rlm_subset_results.py",
                "--report-path",
                str(out / "rlm_results.md"),
            ),
        ),
        CommandStep(
            uv_python(
                "src/exps_performance/scripts/analyze_coding_model_table.py",
                "--output-md",
                str(out / "coding_model_table.md"),
            ),
        ),
        CommandStep(
            uv_python(
                "src/exps_performance/scripts/analyze_code_failure_distribution.py",
                "--output",
                str(out / "code_failure_distribution.csv"),
            ),
        ),
        CommandStep(
            uv_python(
                "src/exps_performance/scripts/analyze_frontier_nopatch_table.py",
                "--output-md",
                str(out / "frontier_nopatch_table.md"),
            ),
        ),
    ]


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
        CommandStep(uv_python("src/exps_performance/analysis.py")),
        CommandStep(uv_python("src/exps_control_again/scripts/plot_judge_discrimination.py")),
        CommandStep(uv_python("src/exps_control_again/scripts/native_vs_translated_scatter.py")),
        CommandStep(uv_python("src/exps_functional/scripts/plot_translation_additivity.py")),
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
                    "src/exps_performance/notebooks/recovery_vs_digits.ipynb",
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
        FigureCopy(Path("src/exps_control_again/results/judge_discrimination_barplot.png"), "judge_discrimination_barplot.png"),
        FigureCopy(Path("src/exps_control_again/results/judge_discrimination_barplot.pdf"), "judge_discrimination_barplot.pdf"),
        FigureCopy(Path("src/exps_control_again/results/native_vs_translated_scatter.png"), "native_vs_translated_scatter.png"),
        FigureCopy(Path("src/exps_control_again/results/native_vs_translated_scatter.pdf"), "native_vs_translated_scatter.pdf"),
        FigureCopy(Path("src/exps_functional/results/translation_additivity.png"), "translation_additivity.png"),
        FigureCopy(Path("src/exps_functional/results/translation_additivity.pdf"), "translation_additivity.pdf"),
        FigureCopy(Path("src/exps_performance/figures/recovery_vs_digits_overall.png"), "recovery_vs_digits_overall.png"),
    ]


def command_groups(config: ReproductionConfig, target: str) -> tuple[tuple[str, list[CommandStep]], ...]:
    figure_config = ReproductionConfig(config.paper_dir, config.output_dir, True, config.dry_run)
    groups = (
        ("Tables", table_steps(config)),
        ("Five percent validation", validation_steps()),
        ("Figures", figure_steps(figure_config)),
        ("Paper", [paper_step(config)]),
    )
    if target in {"all", "list"}:
        return groups
    names = {"tables": "Tables", "validation": "Five percent validation", "figures": "Figures", "paper": "Paper"}
    selected = names[target]
    return tuple(group for group in groups if group[0] == selected)


def print_command_groups(config: ReproductionConfig, target: str) -> None:
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


def run_validation(config: ReproductionConfig) -> None:
    run_steps(validation_steps(), dry_run=config.dry_run)


def copy_figures(config: ReproductionConfig) -> None:
    image_dir = config.paper_dir / "images"
    for item in figure_copies():
        source = ROOT_DIR / item.source
        target = image_dir / item.target_name
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
    if not config.dry_run:
        (ROOT_DIR / "figures").mkdir(exist_ok=True)
        (ROOT_DIR / "src/exps_performance/figures").mkdir(parents=True, exist_ok=True)
    run_steps(figure_steps(config), dry_run=config.dry_run)
    if not config.run_recovery_notebook:
        print("skip recovery notebook: set RUN_RECOVERY_NOTEBOOK=1 to regenerate recovery_vs_digits_overall.png")
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
    parser.add_argument("target", nargs="?", default="all", choices=("list", "tables", "validation", "figures", "paper", "all"))
    parser.add_argument("--list", action="store_true", dest="show_list", help="Print commands without running them.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected commands without running them.")
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
        help="Directory for regenerated table outputs. Defaults to REPRO_OUT_DIR or results/paper_reproduction.",
    )
    parser.add_argument(
        "--run-recovery-notebook",
        action="store_true",
        default=os.environ.get("RUN_RECOVERY_NOTEBOOK", "0") == "1",
        help="Regenerate the notebook-backed recovery figure.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ReproductionConfig(
        paper_dir=args.paper_dir,
        output_dir=args.output_dir,
        run_recovery_notebook=args.run_recovery_notebook,
        dry_run=args.dry_run,
    )

    if args.show_list or args.target == "list":
        print_command_groups(config, args.target)
        return 0
    if args.target in {"tables", "all"}:
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
