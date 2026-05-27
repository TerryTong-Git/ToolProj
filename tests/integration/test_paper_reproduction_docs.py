from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from shlex import quote

ROOT = Path(__file__).resolve().parents[2]


FIVE_PERCENT_TESTS = (
    "test_analyze_sim_code_overlap_e2e.py",
    "test_route_accuracy_tables_e2e.py",
    "test_translation_shot_ablation_e2e.py",
    "test_rlm_subset_results_e2e.py",
    "test_coding_model_table_e2e.py",
    "test_code_failure_distribution_e2e.py",
    "test_frontier_nopatch_table_e2e.py",
)


def test_paper_reproduction_script_lists_core_commands() -> None:
    result = subprocess.run(
        ["bash", "scripts/reproduce_paper_results.sh", "--list"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "analyze_route_accuracy_tables.py" in result.stdout
    assert "analyze_translation_shot_ablation.py" in result.stdout
    assert "analyze_rlm_subset_results.py" in result.stdout
    assert "analyze_coding_model_table.py" in result.stdout
    assert "analyze_code_failure_distribution.py" in result.stdout
    assert "analyze_frontier_nopatch_table.py" in result.stdout
    assert "Five percent validation:" in result.stdout
    assert all(test_name in result.stdout for test_name in FIVE_PERCENT_TESTS)
    assert "plot_judge_discrimination.py" in result.stdout
    assert "plot_translation_additivity.py" in result.stdout
    assert "latexmk -pdf" in result.stdout


def test_python_reproduction_cli_lists_same_core_commands() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py", "--list"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "analyze_route_accuracy_tables.py" in result.stdout
    assert "Five percent validation:" in result.stdout
    assert "test_analyze_sim_code_overlap_e2e.py" in result.stdout
    assert "plot_judge_discrimination.py" in result.stdout
    assert "latexmk -pdf" in result.stdout


def test_python_reproduction_cli_lists_only_selected_target() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py", "--list", "tables"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "analyze_route_accuracy_tables.py" in result.stdout
    assert "plot_judge_discrimination.py" not in result.stdout
    assert "latexmk -pdf" not in result.stdout


def test_python_reproduction_cli_lists_only_validation_target() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py", "--list", "validation"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Five percent validation:" in result.stdout
    assert all(test_name in result.stdout for test_name in FIVE_PERCENT_TESTS)
    assert "analyze_route_accuracy_tables.py" not in result.stdout
    assert "latexmk -pdf" not in result.stdout


def test_python_reproduction_cli_dry_run_tables_uses_dedicated_output_dir() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py", "--dry-run", "tables"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "results/paper_reproduction/route_accuracy_tables.md" in result.stdout
    assert "results/paper_reproduction/code_failure_distribution.csv" in result.stdout
    assert "results/route_accuracy_tables.md" not in result.stdout


def test_python_reproduction_cli_dry_run_validation_lists_five_percent_tests() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py", "--dry-run", "validation"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "+ uv run pytest tests/integration/test_analyze_sim_code_overlap_e2e.py -q" in result.stdout
    assert all(test_name in result.stdout for test_name in FIVE_PERCENT_TESTS)


def test_python_reproduction_cli_dry_run_paper_does_not_need_local_latex(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper source"
    env = os.environ.copy()
    env["PAPER_DIR"] = str(paper_dir)

    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py", "--dry-run", "paper"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert f"cd {quote(str(paper_dir))} && latexmk -pdf" in result.stdout


def test_python_reproduction_cli_accepts_paper_dir_flag(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper source"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper.py",
            "--dry-run",
            "--paper-dir",
            str(paper_dir),
            "paper",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"cd {quote(str(paper_dir))} && latexmk -pdf" in result.stdout


def test_python_reproduction_cli_accepts_output_dir_flag(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper tables"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper.py",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "tables",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert str(output_dir / "route_accuracy_tables.md") in result.stdout
    assert "results/paper_reproduction/route_accuracy_tables.md" not in result.stdout


def test_python_reproduction_cli_figures_hint_uses_recovery_flag() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py", "--dry-run", "figures"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--run-recovery-notebook" in result.stdout
    assert "RUN_RECOVERY_NOTEBOOK=1" not in result.stdout


def test_paper_reproduction_readme_names_source_of_truth() -> None:
    text = (ROOT / "PAPER_REPRODUCTION.md").read_text()

    assert "../Bayesian_Tool_Use_source_20260521" in text
    assert "../Bayesian_Tool_Use` is not used" in text
    assert "Appendix/part_1.tex" in text
    assert "--list validation" in text
    assert "uv run python scripts/reproduce_paper.py validation" in text
