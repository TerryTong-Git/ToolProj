from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
    assert "plot_judge_discrimination.py" in result.stdout
    assert "latexmk -pdf" in result.stdout


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


def test_paper_reproduction_readme_names_source_of_truth() -> None:
    text = (ROOT / "PAPER_REPRODUCTION.md").read_text()

    assert "../Bayesian_Tool_Use_source_20260521" in text
    assert "../Bayesian_Tool_Use` is not used" in text
    assert "Appendix/part_1.tex" in text
