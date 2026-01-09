#!/usr/bin/env python3
"""Integration tests for the logistic regression classification pipeline."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.exps_logistic import main as logistic_main
from src.exps_logistic.classifier import ConceptClassifier
from src.exps_logistic.config import ExperimentConfig
from src.exps_logistic.data_utils import (
    filter_by_rep,
    load_data,
    prepare_labels,
    stratified_split_robust,
)
from src.exps_logistic.featurizer import TfidfFeaturizer, build_featurizer
from src.exps_logistic.metrics import compute_metrics


class TestEndToEndPipeline:
    """Integration tests for the full classification pipeline."""

    @pytest.fixture
    def synthetic_results_dir(self, tmp_path: Any) -> str:
        """Create a synthetic JSONL dataset with separable classes."""
        np.random.seed(42)
        n_per_class = 50

        data = []
        # Class 1: "add" problems with specific vocabulary
        for i in range(n_per_class):
            a, b = np.random.randint(100, 999, size=2)
            data.append(
                {
                    "rationale": f"To add {a} and {b}, we compute step by step. First add ones, then tens, then hundreds. The sum is {a + b}.",
                    "kind": "add",
                    "digits": 3,
                    "prompt": f"Compute: {a} + {b}",
                }
            )

        # Class 2: "mul" problems with different vocabulary
        for i in range(n_per_class):
            a, b = np.random.randint(10, 99, size=2)
            data.append(
                {
                    "rationale": f"To multiply {a} by {b}, we use the multiplication algorithm. The product is {a * b}.",
                    "kind": "mul",
                    "digits": 2,
                    "prompt": f"Compute: {a} * {b}",
                }
            )

        # Class 3: "lcs" problems with distinct vocabulary
        for i in range(n_per_class):
            s1 = "".join(np.random.choice(list("abcd"), size=5))
            s2 = "".join(np.random.choice(list("abcd"), size=5))
            data.append(
                {
                    "rationale": f"To find LCS of '{s1}' and '{s2}', we build a dynamic programming table and trace back the longest common subsequence.",
                    "kind": "lcs",
                    "digits": 3,
                    "prompt": f'S = "{s1}" T = "{s2}"',
                }
            )

        df = pd.DataFrame(data)
        results_dir = Path(tmp_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = results_dir / "synthetic_data.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in df.to_dict("records"):
                f.write(json.dumps(rec))
                f.write("\n")
        return str(results_dir)

    @pytest.fixture
    def default_config(self, synthetic_results_dir: str) -> ExperimentConfig:
        """Create default experiment config for testing."""
        return ExperimentConfig(
            results_dir=synthetic_results_dir,
            models=None,
            rep="all",
            label="kind",  # Use simple kind label for testing
            value_bins=4,
            test_size=0.2,
            seed=42,
            feats="tfidf",
            C=1.0,
            max_iter=200,
            bits=False,
        )

    def test_full_pipeline_tfidf(self, synthetic_results_dir: str, default_config: ExperimentConfig) -> None:
        """Test the full pipeline with TF-IDF features."""
        config = default_config

        # Load and prepare data
        assert config.results_dir is not None
        df = load_data(config.results_dir, config.models)
        assert len(df) == 150  # 50 * 3 classes

        df = filter_by_rep(df, config.rep)
        df = prepare_labels(df, config.label, config.value_bins)

        # Verify labels were created
        assert "label" in df.columns
        assert "gamma" in df.columns
        assert "theta_new" in df.columns

        # Filter valid samples
        df = df[df["label"].astype(str).str.len() > 0].reset_index(drop=True)
        df = df[df["rationale"].astype(str).str.len() > 0].reset_index(drop=True)

        # Split data
        train_df, test_df = stratified_split_robust(df, y_col="label", test_size=config.test_size, seed=config.seed)

        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) == len(df)

        # Extract features
        texts_tr = train_df["rationale"].astype(str).tolist()
        texts_te = test_df["rationale"].astype(str).tolist()

        featurizer = build_featurizer(
            config.feats,
            config.embed_model,
            config.pool,
            config.strip_fences,
            config.device,
            config.batch,
        )
        featurizer.fit(texts_tr)

        X_train = featurizer.transform(texts_tr)
        X_test = featurizer.transform(texts_te)

        assert X_train.shape[0] == len(train_df)
        assert X_test.shape[0] == len(test_df)

        # Train classifier
        classifier = ConceptClassifier(C=config.C, max_iter=config.max_iter)
        classifier.fit(X_train, train_df["label"].astype(str).tolist())

        assert classifier.n_classes == 3  # add, mul, lcs

        # Evaluate
        result = classifier.evaluate(X_test, test_df["label"].astype(str).tolist())

        assert len(result.predictions) == len(test_df)
        assert result.probabilities.shape == (len(test_df), 3)
        assert np.allclose(result.probabilities.sum(axis=1), 1.0)

        # Compute metrics
        metrics = compute_metrics(
            y_true=result.true_labels,
            y_pred=result.predictions,
            probabilities=result.probabilities,
            classes_idx=np.arange(classifier.n_classes),
            n_train=len(train_df),
            n_test=len(test_df),
            test_labels=test_df["label"].astype(str).tolist(),
        )

        # With distinct vocabulary per class, accuracy should be high
        assert metrics.accuracy > 0.7, f"Accuracy {metrics.accuracy} too low for separable data"
        assert metrics.macro_f1 > 0.6, f"Macro F1 {metrics.macro_f1} too low"
        assert metrics.n_classes == 3

    def test_run_without_cv(self, synthetic_results_dir: str, monkeypatch: Any, tmp_path: Any) -> None:
        """Ensure hyperparameter tuning is skipped when CV is disabled."""
        cfg = ExperimentConfig(
            results_dir=synthetic_results_dir,
            models=None,
            rep="all",
            label="kind",
            value_bins=4,
            test_size=0.2,
            seed=42,
            feats="tfidf",
            C=0.5,
            max_iter=50,
            enable_cv=False,
            save_preds=str(tmp_path / "preds_no_cv.json"),
        )

        def fail_tune(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("tune_hyperparams should not run when enable_cv is False")

        monkeypatch.setattr(logistic_main.ConceptClassifier, "tune_hyperparams", staticmethod(fail_tune))

        logistic_main.run(cfg)

        assert cfg.save_preds is not None
        assert os.path.exists(cfg.save_preds)

    def test_run_with_cv_enabled(self, synthetic_results_dir: str, monkeypatch: Any, tmp_path: Any) -> None:
        """Ensure hyperparameter tuning executes when CV is enabled."""
        tune_calls = {"count": 0}

        def fake_tune(*args: Any, **kwargs: Any) -> tuple[float, int, float]:
            tune_calls["count"] += 1
            return 0.25, 30, 0.99

        monkeypatch.setattr(logistic_main.ConceptClassifier, "tune_hyperparams", staticmethod(fake_tune))

        cfg = ExperimentConfig(
            results_dir=synthetic_results_dir,
            models=None,
            rep="all",
            label="kind",
            value_bins=4,
            test_size=0.2,
            seed=42,
            feats="tfidf",
            enable_cv=True,
            save_preds=str(tmp_path / "preds_cv.json"),
        )

        logistic_main.run(cfg)

        assert tune_calls["count"] == 1
        assert cfg.save_preds is not None
        assert os.path.exists(cfg.save_preds)

    def test_pipeline_with_gamma_labels(self, synthetic_results_dir: str) -> None:
        """Test pipeline using gamma labels instead of kind."""
        config = ExperimentConfig(
            results_dir=synthetic_results_dir,
            models=None,
            rep="all",
            label="gamma",
            value_bins=4,
            test_size=0.2,
            seed=42,
            feats="tfidf",
            C=1.0,
            max_iter=200,
        )

        assert config.results_dir is not None
        df = load_data(config.results_dir, config.models)
        df = prepare_labels(df, config.label, config.value_bins)

        # Gamma labels should be more fine-grained
        unique_labels = df["label"].nunique()
        assert unique_labels >= 3  # At least as many as kinds

        # Verify gamma label format
        sample_gamma = df["gamma"].iloc[0]
        assert "|d" in sample_gamma
        assert "|b" in sample_gamma

    def test_pipeline_with_theta_new_labels(self, synthetic_results_dir: str) -> None:
        """Test pipeline using theta_new labels."""
        config = ExperimentConfig(
            results_dir=synthetic_results_dir,
            models=None,
            rep="all",
            label="theta_new",
            value_bins=4,
            test_size=0.2,
            seed=42,
            feats="tfidf",
            C=1.0,
            max_iter=200,
        )

        assert config.results_dir is not None
        df = load_data(config.results_dir, config.models)
        df = prepare_labels(df, config.label, config.value_bins)

        # Verify theta_new label format: kind__dX
        sample_theta = df["theta_new"].iloc[0]
        assert "__d" in sample_theta

        # Should have kind x digits combinations
        unique_labels = df["theta_new"].nunique()
        assert unique_labels == 3  # add__d3, mul__d2, lcs__d3

    def test_save_predictions(self, synthetic_results_dir: str, tmp_path: Any) -> None:
        """Test that predictions can be saved correctly."""
        config = ExperimentConfig(
            results_dir=synthetic_results_dir,
            models=None,
            rep="all",
            label="kind",
            value_bins=4,
            test_size=0.2,
            seed=42,
            feats="tfidf",
            C=1.0,
            max_iter=200,
            save_preds=str(tmp_path / "predictions.json"),
        )

        # Run pipeline
        assert config.results_dir is not None
        df = load_data(config.results_dir, config.models)
        df = prepare_labels(df, config.label, config.value_bins)
        df = df[df["label"].astype(str).str.len() > 0].reset_index(drop=True)
        df = df[df["rationale"].astype(str).str.len() > 0].reset_index(drop=True)

        train_df, test_df = stratified_split_robust(df, y_col="label", test_size=config.test_size, seed=config.seed)

        texts_tr = train_df["rationale"].astype(str).tolist()
        texts_te = test_df["rationale"].astype(str).tolist()

        featurizer = TfidfFeaturizer(strip_fences=config.strip_fences)
        featurizer.fit(texts_tr)
        X_train = featurizer.transform(texts_tr)
        X_test = featurizer.transform(texts_te)

        classifier = ConceptClassifier(C=config.C, max_iter=config.max_iter)
        classifier.fit(X_train, train_df["label"].astype(str).tolist())
        result = classifier.evaluate(X_test, test_df["label"].astype(str).tolist())

        # Save predictions
        out = test_df[["rationale", "kind", "digits", "prompt"]].copy()
        out["true_label"] = test_df["label"].values
        out["pred_label"] = result.label_encoder.inverse_transform(result.predictions)
        out["neglogp_true_nat"] = -np.log(result.probabilities[np.arange(len(result.probabilities)), result.true_labels] + 1e-15)
        out.to_json(config.save_preds, orient="records", date_format="iso")

        # Verify saved file
        assert config.save_preds is not None
        assert os.path.exists(config.save_preds)
        loaded = pd.read_json(config.save_preds)
        assert "true_label" in loaded.columns
        assert "pred_label" in loaded.columns
        assert "neglogp_true_nat" in loaded.columns
        assert len(loaded) == len(test_df)

    def test_save_predictions_json_format(self, synthetic_results_dir: str, tmp_path: Any) -> None:
        """Test that predictions are saved as JSON with correct filename format."""
        # Create test data with model and seed metadata
        import shutil
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
        temp_jsonl = temp_dir / "test_data.jsonl"

        # Load synthetic data and add model/seed metadata
        df = load_data(synthetic_results_dir, None)
        df["model"] = "test-model-name"
        df["seed"] = 123

        # Save to temp location
        with temp_jsonl.open("w", encoding="utf-8") as f:
            for rec in df.to_dict("records"):
                f.write(json.dumps(rec))
                f.write("\n")

        config = ExperimentConfig(
            results_dir=str(temp_dir),
            models=None,
            rep="nl",  # Test with specific rep value (synthetic data has rep column)
            label="kind",
            value_bins=4,
            test_size=0.2,
            seed=123,
            feats="tfidf",
            embed_model="test-embed-model",
            C=1.0,
            max_iter=200,
            save_preds=None,  # Let it use default path
        )

        # Run the full pipeline
        logistic_main.run(config)

        # Verify file was created
        assert config.save_preds is not None
        assert os.path.exists(config.save_preds)
        assert config.save_preds.endswith(".json")

        # Verify filename contains required components
        filename = Path(config.save_preds).name
        assert "test-model-name" in filename
        assert "seed123" in filename
        assert "nl" in filename
        assert "tfidf" in filename
        assert "test-embed-model" in filename or "test-embed-model".replace("/", "-") in filename

        # Verify JSON structure
        with open(config.save_preds, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Should be a list of objects
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify required fields in first record
        first_record = data[0]
        required_fields = [
            "rationale",
            "kind",
            "digits",
            "prompt",
            "true_label",
            "pred_label",
            "neglogp_true_nat",
            "model",
            "seed",
            "rep",
            "feats",
            "embed_model",
            "run_ts",
        ]
        for field in required_fields:
            assert field in first_record, f"Missing required field: {field}"

        # Verify metadata values
        assert first_record["rep"] == "nl"
        assert first_record["feats"] == "tfidf"
        assert str(first_record["seed"]) == "123"
        assert first_record["model"] == "test-model-name"

        # Verify data can be loaded as DataFrame
        loaded_df = pd.read_json(config.save_preds)
        assert len(loaded_df) == len(data)
        assert "true_label" in loaded_df.columns
        assert "pred_label" in loaded_df.columns

        # Cleanup
        shutil.rmtree(temp_dir)


class TestDataPipelineEdgeCases:
    """Test edge cases in the data pipeline."""

    def test_empty_prompt_handling(self, tmp_path: Any) -> None:
        """Test that empty prompts are handled correctly."""
        data = {
            "rationale": ["This is a test rationale"] * 20,
            "kind": ["add"] * 10 + ["mul"] * 10,
            "digits": [3] * 20,
            "prompt": [""] * 20,  # All empty
        }
        df = pd.DataFrame(data)
        results_dir = Path(tmp_path)
        jsonl_path = results_dir / "empty_prompt.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in df.to_dict("records"):
                f.write(json.dumps(rec))
                f.write("\n")

        loaded = load_data(str(results_dir))
        prepared = prepare_labels(loaded, "kind", 4)

        # Should still work with empty prompts
        assert len(prepared) == 20
        assert "label" in prepared.columns

    def test_single_class_warning(self, tmp_path: Any) -> None:
        """Test behavior with single class (should fail gracefully)."""
        data = {
            "rationale": ["Test rationale"] * 10,
            "kind": ["add"] * 10,
            "digits": [3] * 10,
            "prompt": ["Compute: 1 + 2"] * 10,
        }
        df = pd.DataFrame(data)
        results_dir = Path(tmp_path)
        jsonl_path = results_dir / "single_class.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in df.to_dict("records"):
                f.write(json.dumps(rec))
                f.write("\n")

        loaded = load_data(str(results_dir))
        prepared = prepare_labels(loaded, "kind", 4)

        # With only one class, stratified split should handle it
        train_df, test_df = stratified_split_robust(prepared, y_col="label", test_size=0.2, seed=42, verbose=False)

        assert len(train_df) + len(test_df) == 10

    def test_rep_filtering(self, tmp_path: Any) -> None:
        """Test filtering by representation type."""
        data = {
            "rationale": ["NL rationale"] * 10 + ["Code rationale"] * 10,
            "kind": ["add"] * 10 + ["mul"] * 10,
            "digits": [3] * 20,
            "prompt": [""] * 20,
            "rep": ["nl"] * 10 + ["code"] * 10,
        }
        df = pd.DataFrame(data)
        results_dir = Path(tmp_path)
        jsonl_path = results_dir / "with_rep.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in df.to_dict("records"):
                f.write(json.dumps(rec))
                f.write("\n")

        loaded = load_data(str(results_dir))

        # Filter to NL only
        nl_only = filter_by_rep(loaded, "nl")
        assert len(nl_only) == 10
        assert all(nl_only["rep"] == "nl")

        # Filter to code only
        code_only = filter_by_rep(loaded, "code")
        assert len(code_only) == 10
        assert all(code_only["rep"] == "code")

        # All reps
        all_reps = filter_by_rep(loaded, "all")
        assert len(all_reps) == 20


class TestFeaturizerIntegration:
    """Integration tests for featurizer with real text."""

    def test_tfidf_on_varied_text(self) -> None:
        """Test TF-IDF featurizer on varied text."""
        texts = [
            "Adding numbers step by step, we get the sum.",
            "Multiplying using the algorithm produces the product.",
            "Finding LCS requires dynamic programming approach.",
            "Addition is commutative, so order doesn't matter.",
            "Multiplication tables help with basic products.",
            "Subsequence problems use memoization techniques.",
        ]

        featurizer = TfidfFeaturizer(strip_fences=False)
        featurizer.fit(texts[:4])  # Fit on subset
        features = featurizer.transform(texts)

        # Should produce sparse matrix with reasonable dimensions
        assert features.shape[0] == 6
        assert features.shape[1] > 0

    def test_build_featurizer_invalid_type(self) -> None:
        """Test that invalid featurizer type raises error."""
        with pytest.raises(ValueError, match="Unknown"):
            build_featurizer("invalid_type")

    def test_hf_cls_requires_model(self) -> None:
        """Test that hf-cls requires embed_model."""
        with pytest.raises(ValueError, match="--embed-model"):
            build_featurizer("hf-cls", embed_model=None)

    def test_st_requires_model(self) -> None:
        """Test that st requires embed_model."""
        with pytest.raises(ValueError, match="--embed-model"):
            build_featurizer("st", embed_model=None)
