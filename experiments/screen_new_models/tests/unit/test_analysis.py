#!/usr/bin/env python3
"""
Unit tests for accuracy analysis functions.

Tests cover:
1. Result parsing from JSONL files
2. Accuracy computation (nl_correct vs code_correct)
3. Statistical tests (McNemar's test for paired comparison)

Run with: uv run pytest experiments/screen_new_models/tests/unit/ -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


class TestResultParsing:
    """Test parsing of res.jsonl result files."""

    @pytest.fixture
    def sample_results(self) -> list[dict]:
        """Generate sample results matching the actual format."""
        return [
            {
                "model": "test/model-a",
                "seed": 0,
                "kind": "add",
                "digit": 2,
                "nl_correct": True,
                "code_correct": True,
                "sim_correct": True,
                "nl_parse_err": False,
                "code_parse_err": False,
            },
            {
                "model": "test/model-a",
                "seed": 0,
                "kind": "add",
                "digit": 4,
                "nl_correct": True,
                "code_correct": False,
                "sim_correct": True,
                "nl_parse_err": False,
                "code_parse_err": True,
            },
            {
                "model": "test/model-a",
                "seed": 0,
                "kind": "sub",
                "digit": 2,
                "nl_correct": False,
                "code_correct": True,
                "sim_correct": False,
                "nl_parse_err": False,
                "code_parse_err": False,
            },
            {
                "model": "test/model-a",
                "seed": 0,
                "kind": "sub",
                "digit": 4,
                "nl_correct": False,
                "code_correct": False,
                "sim_correct": False,
                "nl_parse_err": False,
                "code_parse_err": False,
            },
        ]

    @pytest.fixture
    def temp_results_file(self, sample_results: list[dict], tmp_path: Path) -> Path:
        """Create a temporary JSONL file with sample results."""
        results_file = tmp_path / "res.jsonl"
        with open(results_file, "w") as f:
            for result in sample_results:
                f.write(json.dumps(result) + "\n")
        return results_file

    def test_parse_jsonl_to_dataframe(self, temp_results_file: Path) -> None:
        """Can parse JSONL file into DataFrame with expected columns."""
        df = pd.read_json(temp_results_file, lines=True)

        assert len(df) == 4
        assert "model" in df.columns
        assert "nl_correct" in df.columns
        assert "code_correct" in df.columns
        assert "sim_correct" in df.columns
        assert "kind" in df.columns
        assert "digit" in df.columns

    def test_parse_preserves_boolean_types(self, temp_results_file: Path) -> None:
        """Boolean fields are preserved correctly."""
        df = pd.read_json(temp_results_file, lines=True)

        assert df["nl_correct"].dtype == bool
        assert df["code_correct"].dtype == bool
        assert df["sim_correct"].dtype == bool

    def test_filter_by_model(self, temp_results_file: Path) -> None:
        """Can filter results by model name."""
        df = pd.read_json(temp_results_file, lines=True)
        filtered = df[df["model"] == "test/model-a"]

        assert len(filtered) == 4

    def test_filter_by_kind(self, temp_results_file: Path) -> None:
        """Can filter results by problem kind."""
        df = pd.read_json(temp_results_file, lines=True)
        add_results = df[df["kind"] == "add"]
        sub_results = df[df["kind"] == "sub"]

        assert len(add_results) == 2
        assert len(sub_results) == 2


class TestAccuracyComputation:
    """Test accuracy computation functions."""

    @pytest.fixture
    def accuracy_df(self) -> pd.DataFrame:
        """DataFrame for accuracy computation tests."""
        return pd.DataFrame(
            {
                "model": ["m1"] * 10,
                "nl_correct": [True, True, True, False, False, True, False, True, False, False],
                "code_correct": [True, True, True, True, True, False, False, False, False, False],
                "sim_correct": [True, True, False, True, False, True, False, True, False, False],
            }
        )

    def test_compute_overall_accuracy(self, accuracy_df: pd.DataFrame) -> None:
        """Can compute overall accuracy for each metric."""
        nl_acc = accuracy_df["nl_correct"].mean()
        code_acc = accuracy_df["code_correct"].mean()
        sim_acc = accuracy_df["sim_correct"].mean()

        assert nl_acc == 0.5  # 5/10
        assert code_acc == 0.5  # 5/10
        assert sim_acc == 0.5  # 5/10

    def test_compute_accuracy_by_model(self) -> None:
        """Can compute accuracy grouped by model."""
        df = pd.DataFrame(
            {
                "model": ["m1", "m1", "m2", "m2"],
                "nl_correct": [True, True, False, False],
                "code_correct": [True, False, True, True],
            }
        )

        model_acc = df.groupby("model").agg(
            {"nl_correct": "mean", "code_correct": "mean"}
        )

        assert model_acc.loc["m1", "nl_correct"] == 1.0
        assert model_acc.loc["m1", "code_correct"] == 0.5
        assert model_acc.loc["m2", "nl_correct"] == 0.0
        assert model_acc.loc["m2", "code_correct"] == 1.0

    def test_compute_accuracy_with_parse_errors(self) -> None:
        """Accuracy computation handles parse errors correctly."""
        df = pd.DataFrame(
            {
                "nl_correct": [True, True, False],
                "code_correct": [True, False, False],
                "code_parse_err": [False, True, False],
            }
        )

        # Parse errors should still count as incorrect
        code_acc = df["code_correct"].mean()
        assert code_acc == pytest.approx(1 / 3)

    def test_code_vs_nl_difference(self, accuracy_df: pd.DataFrame) -> None:
        """Can compute the difference between code and NL accuracy."""
        nl_acc = accuracy_df["nl_correct"].mean()
        code_acc = accuracy_df["code_correct"].mean()
        diff = code_acc - nl_acc

        assert diff == 0.0  # Equal in this test case


class TestMcNemar:
    """Test McNemar's test for paired comparison."""

    def test_mcnemar_contingency_table(self) -> None:
        """Can build contingency table for McNemar's test."""
        df = pd.DataFrame(
            {
                "nl_correct": [True, True, False, False, True, False, True, True],
                "code_correct": [True, False, True, False, True, True, True, False],
            }
        )

        # Build contingency table:
        # Both correct, NL only correct, Code only correct, Both incorrect
        both_correct = ((df["nl_correct"]) & (df["code_correct"])).sum()
        nl_only = ((df["nl_correct"]) & (~df["code_correct"])).sum()
        code_only = ((~df["nl_correct"]) & (df["code_correct"])).sum()
        both_incorrect = ((~df["nl_correct"]) & (~df["code_correct"])).sum()

        assert both_correct == 3  # rows 0, 4, 6
        assert nl_only == 2  # rows 1, 7
        assert code_only == 2  # rows 2, 5
        assert both_incorrect == 1  # row 3

        # Verify totals
        assert both_correct + nl_only + code_only + both_incorrect == len(df)

    def test_mcnemar_test_import(self) -> None:
        """Can import McNemar test from scipy/statsmodels."""
        try:
            from statsmodels.stats.contingency_tables import mcnemar

            assert mcnemar is not None
        except ImportError:
            # Fallback to scipy
            from scipy.stats import chi2

            assert chi2 is not None

    def test_mcnemar_test_basic(self) -> None:
        """McNemar's test runs without error."""
        from statsmodels.stats.contingency_tables import mcnemar

        # Contingency table: [[both_correct, nl_only], [code_only, both_incorrect]]
        contingency = [[30, 5], [15, 10]]

        result = mcnemar(contingency, exact=True)
        assert hasattr(result, "pvalue")
        assert hasattr(result, "statistic")
        assert 0 <= result.pvalue <= 1

    def test_mcnemar_detects_significant_difference(self) -> None:
        """McNemar's test can detect significant difference."""
        from statsmodels.stats.contingency_tables import mcnemar

        # Strong difference: code wins more often
        # [[both_correct, nl_only], [code_only, both_incorrect]]
        contingency = [[50, 5], [30, 15]]

        result = mcnemar(contingency, exact=True)
        # With code winning 30 vs NL winning 5, should be significant
        assert result.pvalue < 0.05

    def test_mcnemar_no_difference(self) -> None:
        """McNemar's test shows no difference when balanced."""
        from statsmodels.stats.contingency_tables import mcnemar

        # Balanced: equal discordant pairs
        contingency = [[50, 15], [15, 20]]

        result = mcnemar(contingency, exact=True)
        # Equal discordant pairs, should not be significant
        assert result.pvalue > 0.05


class TestResultAggregation:
    """Test aggregation of results across multiple models."""

    @pytest.fixture
    def multi_model_df(self) -> pd.DataFrame:
        """DataFrame with results from multiple models."""
        return pd.DataFrame(
            {
                "model": ["m1"] * 4 + ["m2"] * 4 + ["m3"] * 4,
                "kind": ["add", "sub", "mul", "lcs"] * 3,
                "digit": [2, 4, 6, 2] * 3,
                "nl_correct": [
                    True, True, False, False,  # m1: 50%
                    True, True, True, False,  # m2: 75%
                    False, False, False, False,  # m3: 0%
                ],
                "code_correct": [
                    True, True, True, False,  # m1: 75%
                    True, True, True, True,  # m2: 100%
                    True, False, False, False,  # m3: 25%
                ],
            }
        )

    def test_summary_by_model(self, multi_model_df: pd.DataFrame) -> None:
        """Can create summary table by model."""
        summary = multi_model_df.groupby("model").agg(
            {
                "nl_correct": "mean",
                "code_correct": "mean",
            }
        )
        summary["code_minus_nl"] = summary["code_correct"] - summary["nl_correct"]

        assert summary.loc["m1", "nl_correct"] == 0.5
        assert summary.loc["m1", "code_correct"] == 0.75
        assert summary.loc["m1", "code_minus_nl"] == 0.25

        assert summary.loc["m2", "nl_correct"] == 0.75
        assert summary.loc["m2", "code_correct"] == 1.0
        assert summary.loc["m2", "code_minus_nl"] == 0.25

        assert summary.loc["m3", "nl_correct"] == 0.0
        assert summary.loc["m3", "code_correct"] == 0.25
        assert summary.loc["m3", "code_minus_nl"] == 0.25

    def test_overall_code_vs_nl_trend(self, multi_model_df: pd.DataFrame) -> None:
        """Can verify overall code > NL trend across all models."""
        summary = multi_model_df.groupby("model").agg(
            {
                "nl_correct": "mean",
                "code_correct": "mean",
            }
        )

        # All models should show code >= NL
        code_better_count = (summary["code_correct"] >= summary["nl_correct"]).sum()
        assert code_better_count == 3  # All 3 models

    def test_count_samples_per_model(self, multi_model_df: pd.DataFrame) -> None:
        """Can count samples per model."""
        counts = multi_model_df.groupby("model").size()

        assert counts["m1"] == 4
        assert counts["m2"] == 4
        assert counts["m3"] == 4


class TestVisualization:
    """Test data preparation for visualization."""

    def test_prepare_bar_chart_data(self) -> None:
        """Can prepare data for code vs NL bar chart."""
        summary = pd.DataFrame(
            {
                "model": ["model-a", "model-b", "model-c"],
                "nl_correct": [0.5, 0.6, 0.7],
                "code_correct": [0.7, 0.8, 0.85],
            }
        )

        # Reshape for grouped bar chart
        plot_data = summary.melt(
            id_vars=["model"],
            value_vars=["nl_correct", "code_correct"],
            var_name="method",
            value_name="accuracy",
        )

        assert len(plot_data) == 6  # 3 models x 2 methods
        assert "method" in plot_data.columns
        assert "accuracy" in plot_data.columns

    def test_sort_models_by_code_accuracy(self) -> None:
        """Can sort models by code accuracy for visualization."""
        summary = pd.DataFrame(
            {
                "model": ["m-low", "m-high", "m-mid"],
                "code_correct": [0.3, 0.9, 0.6],
            }
        )

        sorted_df = summary.sort_values("code_correct", ascending=False)
        models_sorted = sorted_df["model"].tolist()

        assert models_sorted == ["m-high", "m-mid", "m-low"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
