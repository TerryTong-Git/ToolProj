from __future__ import annotations

import subprocess
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
    assert "test_analyze_sim_code_overlap_e2e.py" in result.stdout
    assert "test_route_accuracy_tables_e2e.py" in result.stdout
    assert "test_translation_shot_ablation_e2e.py" in result.stdout
    assert "test_rlm_subset_results_e2e.py" in result.stdout
    assert "test_coding_model_table_e2e.py" in result.stdout
    assert "test_code_failure_distribution_e2e.py" in result.stdout
    assert "test_frontier_nopatch_table_e2e.py" in result.stdout
    assert "plot_judge_discrimination.py" in result.stdout
    assert "plot_translation_additivity.py" in result.stdout
    assert "latexmk -pdf" in result.stdout


def test_paper_reproduction_script_lists_five_percent_validation_target() -> None:
    result = subprocess.run(
        ["bash", "scripts/reproduce_paper_results.sh", "--list"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Five percent validation:" in result.stdout
    assert "uv run pytest tests/integration/test_analyze_sim_code_overlap_e2e.py -q" in result.stdout


def test_paper_reproduction_readme_names_source_of_truth() -> None:
    text = (ROOT / "PAPER_REPRODUCTION.md").read_text()

    assert "../Bayesian_Tool_Use_source_20260521" in text
    assert "../Bayesian_Tool_Use` is not used" in text
    assert "Appendix/part_1.tex" in text
    assert "bash scripts/reproduce_paper_results.sh validation" in text
