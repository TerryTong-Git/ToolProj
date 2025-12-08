#!/usr/bin/env python3
"""Unit tests for classifier module."""

from typing import List, Tuple

import numpy as np
import pytest

from src.exps_logistic.classifier import ClassifierResult, ConceptClassifier


class TestConceptClassifier:
    """Tests for the ConceptClassifier."""

    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, List[str]]:
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        # Make class 0 have higher feature 0, class 1 have higher feature 1
        X[:50, 0] += 2
        X[50:, 1] += 2

        y = ["class_A"] * 50 + ["class_B"] * 50

        return X, y

    def test_fit_predict(self, sample_data: Tuple[np.ndarray, List[str]]) -> None:
        X, y = sample_data

        clf = ConceptClassifier(C=1.0, max_iter=100)
        clf.fit(X, y)

        predictions, probs = clf.predict(X)

        assert len(predictions) == len(y)
        assert probs.shape == (len(y), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_evaluate(self, sample_data: Tuple[np.ndarray, List[str]]) -> None:
        X, y = sample_data

        clf = ConceptClassifier(C=1.0, max_iter=100)
        clf.fit(X, y)

        result = clf.evaluate(X, y)

        assert isinstance(result, ClassifierResult)
        assert len(result.predictions) == len(y)
        assert result.probabilities.shape == (len(y), 2)

    def test_classes_property(self, sample_data: Tuple[np.ndarray, List[str]]) -> None:
        X, y = sample_data

        clf = ConceptClassifier()
        clf.fit(X, y)

        assert clf.n_classes == 2
        assert set(clf.classes) == {"class_A", "class_B"}

    def test_not_fitted_error(self) -> None:
        clf = ConceptClassifier()
        X = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(X)

    def test_multiclass(self) -> None:
        """Test with more than 2 classes."""
        np.random.seed(42)
        n_per_class = 30
        n_features = 5

        X = np.random.randn(n_per_class * 3, n_features)
        X[:n_per_class, 0] += 3
        X[n_per_class : 2 * n_per_class, 1] += 3
        X[2 * n_per_class :, 2] += 3

        y = ["A"] * n_per_class + ["B"] * n_per_class + ["C"] * n_per_class

        clf = ConceptClassifier(C=1.0, max_iter=200)
        clf.fit(X, y)

        assert clf.n_classes == 3
        predictions, probs = clf.predict(X)
        assert probs.shape == (len(y), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
