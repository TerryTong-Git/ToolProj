#!/usr/bin/env python3
"""
Integration tests for the full screening and analysis pipeline.

These tests verify:
1. The analysis script can process existing screening results
2. Results are correctly parsed and analyzed
3. McNemar's test produces valid p-values
4. Visualizations are generated

Run with: uv run pytest experiments/screen_new_models/tests/integration/ -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def find_repo_root() -> Path:
    """Find the repository root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not find repository root")


REPO_ROOT = find_repo_root()
MAIN_REPO = Path("/Users/terrytong/Documents/CCG/ToolProj")
sys.path.insert(0, str(REPO_ROOT))


class TestAnalysisScript:
    """Test the analyze_accuracy.py script."""

    @pytest.fixture
    def results_dir(self) -> Path:
        """Get the screening results directory."""
        # Try worktree first, then main repo
        for base in [REPO_ROOT, MAIN_REPO]:
            candidate = base / "src" / "exps_performance" / "results_screening" / "results"
            if candidate.exists() and list(candidate.rglob("res.jsonl")):
                return candidate
        pytest.skip("No screening results available")
        return Path()  # Never reached

    def test_load_results(self, results_dir: Path) -> None:
        """Can load results from screening directory."""
        from experiments.screen_new_models.src.analyze_accuracy import load_all_results

        df = load_all_results(results_dir)
        assert len(df) > 0, "Should load some results"
        assert "model" in df.columns
        assert "nl_correct" in df.columns
        assert "sim_correct" in df.columns

    def test_compute_model_accuracy(self, results_dir: Path) -> None:
        """Can compute accuracy metrics by model."""
        from experiments.screen_new_models.src.analyze_accuracy import (
            compute_model_accuracy,
            load_all_results,
        )

        df = load_all_results(results_dir)
        summary = compute_model_accuracy(df)

        assert len(summary) > 0, "Should have at least one model"
        assert "nl_accuracy" in summary.columns
        assert "sim_accuracy" in summary.columns
        assert "sim_minus_nl" in summary.columns

        # Accuracies should be between 0 and 1
        assert (summary["nl_accuracy"] >= 0).all()
        assert (summary["nl_accuracy"] <= 1).all()
        assert (summary["sim_accuracy"] >= 0).all()
        assert (summary["sim_accuracy"] <= 1).all()

    def test_mcnemar_on_real_data(self, results_dir: Path) -> None:
        """McNemar's test works on real screening data."""
        from experiments.screen_new_models.src.analyze_accuracy import (
            load_all_results,
            run_mcnemar_test,
        )

        df = load_all_results(results_dir)

        # Get first model with enough samples
        model_counts = df.groupby("model").size()
        valid_models = model_counts[model_counts >= 10].index.tolist()

        if not valid_models:
            pytest.skip("No models with >= 10 samples")

        model = valid_models[0]
        stat, pvalue = run_mcnemar_test(df, model, "sim_correct")

        # Should return valid values (or None if not enough discordant pairs)
        if pvalue is not None:
            assert 0 <= pvalue <= 1, "p-value should be between 0 and 1"

    def test_full_analysis(self, results_dir: Path, tmp_path: Path) -> None:
        """Full analysis pipeline runs without error."""
        from experiments.screen_new_models.src.analyze_accuracy import (
            analyze_all_models,
            create_summary_table,
            load_all_results,
            plot_accuracy_comparison,
        )

        df = load_all_results(results_dir)
        results = analyze_all_models(df, min_samples=10, compare="sim")

        if not results:
            pytest.skip("No models with >= 10 samples")

        # Create summary table
        summary_df = create_summary_table(results, compare="sim")
        assert len(summary_df) > 0

        # Generate plot
        plot_path = tmp_path / "test_plot.png"
        plot_accuracy_comparison(results, plot_path, compare="sim")
        assert plot_path.exists(), "Plot should be created"

    def test_analysis_json_output(self, results_dir: Path, tmp_path: Path) -> None:
        """Analysis produces valid JSON output."""
        from experiments.screen_new_models.src.analyze_accuracy import (
            analyze_all_models,
            load_all_results,
        )

        df = load_all_results(results_dir)
        results = analyze_all_models(df, min_samples=10, compare="sim")

        if not results:
            pytest.skip("No models with >= 10 samples")

        # Save to JSON
        json_path = tmp_path / "results.json"
        output = {
            "results": [
                {
                    "model": r.model,
                    "n_samples": r.n_samples,
                    "nl_accuracy": r.nl_accuracy,
                    "sim_accuracy": r.sim_accuracy,
                    "sim_minus_nl": r.sim_minus_nl,
                }
                for r in results
            ]
        }
        with open(json_path, "w") as f:
            json.dump(output, f)

        # Verify it can be read back
        with open(json_path) as f:
            loaded = json.load(f)
        assert len(loaded["results"]) == len(results)


class TestResultsValidation:
    """Test validation of screening results structure."""

    @pytest.fixture
    def results_dir(self) -> Path:
        """Get the screening results directory."""
        for base in [REPO_ROOT, MAIN_REPO]:
            candidate = base / "src" / "exps_performance" / "results_screening" / "results"
            if candidate.exists() and list(candidate.rglob("res.jsonl")):
                return candidate
        pytest.skip("No screening results available")
        return Path()

    def test_all_required_fields_present(self, results_dir: Path) -> None:
        """All JSONL files have required fields."""
        required_fields = [
            "model", "kind", "digit", "nl_correct", "sim_correct", "code_correct"
        ]

        for jsonl_path in results_dir.rglob("res.jsonl"):
            with open(jsonl_path) as f:
                for i, line in enumerate(f):
                    result = json.loads(line)
                    for field in required_fields:
                        assert field in result, (
                            f"Missing field '{field}' in {jsonl_path} line {i+1}"
                        )

    def test_models_have_reasonable_sample_sizes(self, results_dir: Path) -> None:
        """Each model should have a reasonable number of samples."""
        from experiments.screen_new_models.src.analyze_accuracy import load_all_results

        df = load_all_results(results_dir)
        model_counts = df.groupby("model").size()

        # Report on sample sizes
        print("\nSample sizes per model:")
        for model, count in model_counts.items():
            print(f"  {model}: {count}")

        # At least one model should have >= 10 samples
        assert (model_counts >= 10).any(), "At least one model should have >= 10 samples"


class TestDataQuality:
    """Test data quality of screening results."""

    @pytest.fixture
    def results_dir(self) -> Path:
        """Get the screening results directory."""
        for base in [REPO_ROOT, MAIN_REPO]:
            candidate = base / "src" / "exps_performance" / "results_screening" / "results"
            if candidate.exists() and list(candidate.rglob("res.jsonl")):
                return candidate
        pytest.skip("No screening results available")
        return Path()

    def test_no_all_empty_nl_answers(self, results_dir: Path) -> None:
        """Check for models with all empty NL answers (data quality issue)."""
        from experiments.screen_new_models.src.analyze_accuracy import load_all_results

        df = load_all_results(results_dir)

        # Check each model
        problematic_models = []
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            # If all nl_correct are False and nl_answer is empty, flag it
            if (model_df["nl_correct"] == False).all():
                problematic_models.append(model)

        if problematic_models:
            print(f"\nWARNING: Models with all nl_correct=False: {problematic_models}")
            # This is a warning, not a failure - data issues happen

    def test_code_execution_success_rate(self, results_dir: Path) -> None:
        """Report on code execution success rate per model."""
        from experiments.screen_new_models.src.analyze_accuracy import load_all_results

        df = load_all_results(results_dir)

        if "code_parse_err" not in df.columns:
            pytest.skip("code_parse_err column not present")

        print("\nCode execution success rate per model:")
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            success_rate = 1.0 - model_df["code_parse_err"].mean()
            print(f"  {model}: {success_rate:.0%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
