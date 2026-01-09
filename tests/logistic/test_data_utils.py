#!/usr/bin/env python3
"""Unit tests for data_utils module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.exps_logistic.data_utils import (
    _convert_results_df,
    create_theta_new_label,
    equal_width_bin,
    load_data,
    make_gamma_label,
    stratified_split_robust,
)


class TestEqualWidthBin:
    """Tests for equal-width binning."""

    def test_basic_binning(self) -> None:
        # 0-99 into 10 bins
        assert equal_width_bin(0, 0, 99, 10) == 0
        assert equal_width_bin(99, 0, 99, 10) == 9
        assert equal_width_bin(50, 0, 99, 10) == 5

    def test_single_bin(self) -> None:
        assert equal_width_bin(50, 0, 100, 1) == 0

    def test_clipping(self) -> None:
        # Values outside range should be clipped
        assert equal_width_bin(-10, 0, 100, 10) == 0
        assert equal_width_bin(200, 0, 100, 10) == 9

    def test_swapped_bounds(self) -> None:
        # lo > hi should be handled
        result = equal_width_bin(50, 100, 0, 10)
        assert 0 <= result <= 9

    def test_edge_cases(self) -> None:
        # Test boundary values
        assert equal_width_bin(10, 10, 19, 10) == 0
        assert equal_width_bin(19, 10, 19, 10) == 9
        assert equal_width_bin(15, 10, 19, 10) == 5


class TestLabelCreation:
    """Tests for label creation functions."""

    def test_theta_new_label(self) -> None:
        label = create_theta_new_label("add", 3)
        assert label == "add__d3"

        label = create_theta_new_label("lcs", 5)
        assert label == "lcs__d5"

    def test_gamma_label_add(self) -> None:
        text = "Compute: 150 + 250"
        label = make_gamma_label("add", 3, text, K_bins=8)
        assert label.startswith("add|d3|b")
        assert "bNA" not in label

    def test_gamma_label_sub(self) -> None:
        text = "Compute: 500 - 100"
        label = make_gamma_label("sub", 3, text, K_bins=8)
        assert label.startswith("sub|d3|b")
        assert "bNA" not in label

    def test_gamma_label_mul(self) -> None:
        text = "Compute: 12 * 34"
        label = make_gamma_label("mul", 2, text, K_bins=8)
        assert label.startswith("mul|d2|b")
        assert "bNA" not in label

    def test_gamma_label_no_match(self) -> None:
        text = "Random text without arithmetic"
        label = make_gamma_label("add", 3, text, K_bins=8)
        assert label == "add|d3|bNA"

    def test_gamma_label_lcs(self) -> None:
        text = 'S = "abc" T = "xy"'
        label = make_gamma_label("lcs", 3, text, K_bins=4)
        assert label.startswith("lcs|d3|b")

    def test_gamma_label_knap(self) -> None:
        text = "W = [10, 20, 30] V = [60, 100, 120] C = 50"
        label = make_gamma_label("knap", 3, text, K_bins=4)
        assert label.startswith("knap|d3|b")

    def test_gamma_label_rod(self) -> None:
        text = "N = 8"
        label = make_gamma_label("rod", 3, text, K_bins=4)
        assert label.startswith("rod|d3|b")

    def test_gamma_label_unknown_kind(self) -> None:
        text = "Some problem"
        label = make_gamma_label("unknown_kind", 5, text, K_bins=8)
        assert label == "unknown_kind|d5|bNA"


class TestStratifiedSplit:
    """Tests for robust stratified splitting."""

    def test_basic_split(self) -> None:
        df = pd.DataFrame({"label": ["A"] * 50 + ["B"] * 50, "value": range(100)})
        train, test = stratified_split_robust(df, test_size=0.2, verbose=False)

        assert len(train) + len(test) == 100
        # Check approximate stratification
        train_ratio = (train["label"] == "A").mean()
        assert 0.4 <= train_ratio <= 0.6

    def test_rare_class_handling(self) -> None:
        # Create data with some rare classes
        df = pd.DataFrame(
            {
                "label": ["A"] * 50 + ["B"] * 50 + ["C"],  # C only has 1 sample
                "value": range(101),
            }
        )
        train, test = stratified_split_robust(df, test_size=0.2, min_count=2, verbose=False)

        # C should be dropped
        assert "C" not in train["label"].values
        assert "C" not in test["label"].values

    def test_preserves_data(self) -> None:
        df = pd.DataFrame(
            {
                "label": ["A"] * 40 + ["B"] * 40 + ["C"] * 20,
                "value": range(100),
                "extra": [f"item_{i}" for i in range(100)],
            }
        )
        train, test = stratified_split_robust(df, test_size=0.2, verbose=False)

        # Check that all columns are preserved
        assert set(train.columns) == set(df.columns)
        assert set(test.columns) == set(df.columns)

    def test_reproducibility(self) -> None:
        df = pd.DataFrame({"label": ["A"] * 50 + ["B"] * 50, "value": range(100)})

        train1, test1 = stratified_split_robust(df, test_size=0.2, seed=42, verbose=False)
        train2, test2 = stratified_split_robust(df, test_size=0.2, seed=42, verbose=False)

        assert train1["value"].tolist() == train2["value"].tolist()
        assert test1["value"].tolist() == test2["value"].tolist()

    def test_different_seeds_different_splits(self) -> None:
        df = pd.DataFrame({"label": ["A"] * 50 + ["B"] * 50, "value": range(100)})

        train1, _ = stratified_split_robust(df, test_size=0.2, seed=42, verbose=False)
        train2, _ = stratified_split_robust(df, test_size=0.2, seed=123, verbose=False)

        # Different seeds should produce different splits
        assert train1["value"].tolist() != train2["value"].tolist()


class TestConvertResultsDf:
    """Tests for converting Record-schema results into balanced rationale rows."""

    def test_balanced_counts_and_reps(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "digit": 1,
                    "kind": "knap",
                    "question": "Q0",
                    "nl_reasoning": "nl 0",
                    "sim_code": "code 0",
                },
                {
                    "digit": 2,
                    "kind": "add",
                    "question": "Q1",
                    "nl_reasoning": "nl 1",
                    "sim_code": "code 1",
                },
            ]
        )

        out = _convert_results_df(df)

        assert len(out) == 4  # two records -> two reps each
        counts = out["rep"].value_counts()
        assert counts["nl"] == counts["code"] == 2
        assert set(out["rep"]) == {"nl", "code"}

    def test_filters_missing_modalities(self) -> None:
        df = pd.DataFrame(
            [
                {  # missing nl -> drop
                    "digit": 1,
                    "kind": "knap",
                    "question": "Q0",
                    "nl_reasoning": "",
                    "sim_code": "code 0",
                },
                {  # missing code -> drop
                    "digit": 2,
                    "kind": "add",
                    "question": "Q1",
                    "nl_reasoning": "nl 1",
                    "sim_code": "   ",
                },
                {  # both present -> keep
                    "digit": 3,
                    "kind": "mul",
                    "question": "Q2",
                    "nl_reasoning": "nl 2",
                    "sim_code": "code 2",
                },
            ]
        )

        out = _convert_results_df(df)

        assert len(out) == 2  # only the record with both modalities remains
        counts = out["rep"].value_counts()
        assert counts["nl"] == counts["code"] == 1
        assert set(out["rep"]) == {"nl", "code"}

    def test_all_filtered_raises(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "digit": 1,
                    "kind": "knap",
                    "question": "Q0",
                    "nl_reasoning": "",
                    "sim_code": "code 0",
                },
                {
                    "digit": 2,
                    "kind": "add",
                    "question": "Q1",
                    "nl_reasoning": "nl 1",
                    "sim_code": "",
                },
            ]
        )

        with pytest.raises(ValueError):
            _convert_results_df(df)


class TestLoadDataFiltering:
    """Tests for model/seed filtering in load_data."""

    def _write_jsonl(self, dir_path: Path, records: list[dict]) -> None:
        path = dir_path / "data.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec))
                f.write("\n")

    def test_filters_by_seed(self, tmp_path: Path) -> None:
        records = [
            {"rationale": "r0", "kind": "add", "digits": 2, "prompt": "p0", "seed": 1, "model": "m0"},
            {"rationale": "r1", "kind": "add", "digits": 2, "prompt": "p1", "seed": 2, "model": "m0"},
            {"rationale": "r2", "kind": "mul", "digits": 3, "prompt": "p2", "seed": 2, "model": "m1"},
        ]
        self._write_jsonl(tmp_path, records)

        df = load_data(str(tmp_path), models=None, seeds=[2])

        assert set(df["seed"].unique()) == {2}
        assert len(df) == 2
        assert set(df["model"].unique()) == {"m0", "m1"}

    def test_seed_filter_empty_raises(self, tmp_path: Path) -> None:
        records = [
            {"rationale": "r0", "kind": "add", "digits": 2, "prompt": "p0", "seed": 0, "model": "m0"},
        ]
        self._write_jsonl(tmp_path, records)

        with pytest.raises(ValueError):
            load_data(str(tmp_path), models=None, seeds=[123])
