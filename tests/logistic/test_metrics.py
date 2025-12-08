#!/usr/bin/env python3
"""Unit tests for metrics module."""

import numpy as np

from src.exps_logistic.metrics import (
    EvaluationMetrics,
    compute_metrics,
    empirical_entropy_bits,
)


class TestEmpiricalEntropy:
    """Tests for entropy computation."""

    def test_uniform_distribution(self) -> None:
        # 2 classes, uniform -> 1 bit
        labels = ["A"] * 50 + ["B"] * 50
        entropy = empirical_entropy_bits(labels)
        assert np.isclose(entropy, 1.0, atol=0.01)

    def test_single_class(self) -> None:
        # All same class -> 0 entropy
        labels = ["A"] * 100
        entropy = empirical_entropy_bits(labels)
        assert np.isclose(entropy, 0.0)

    def test_four_classes_uniform(self) -> None:
        # 4 classes, uniform -> 2 bits
        labels = ["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25
        entropy = empirical_entropy_bits(labels)
        assert np.isclose(entropy, 2.0, atol=0.01)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        probs = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.2, 0.8],
            ]
        )

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probs,
            classes_idx=np.array([0, 1]),
            n_train=80,
            n_test=4,
            test_labels=["A", "A", "B", "B"],
        )

        assert metrics.accuracy == 1.0
        assert metrics.macro_f1 == 1.0
        assert metrics.n_classes == 2

    def test_entropy_conversion(self) -> None:
        metrics = EvaluationMetrics(
            cross_entropy=0.693,  # ~log(2) nats
            accuracy=0.5,
            macro_f1=0.5,
            empirical_entropy=1.0,
            n_classes=2,
            n_train=100,
            n_test=20,
        )

        # log(2) nats ≈ 1 bit
        assert np.isclose(metrics.cross_entropy_bits, 1.0, atol=0.01)
