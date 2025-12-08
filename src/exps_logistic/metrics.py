#!/usr/bin/env python3
"""Evaluation metrics for concept classification."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    cross_entropy: float  # in nats by default
    accuracy: float
    macro_f1: float
    empirical_entropy: float  # in bits
    n_classes: int
    n_train: int
    n_test: int

    @property
    def cross_entropy_bits(self) -> float:
        """Cross-entropy in bits."""
        return self.cross_entropy / math.log(2.0)

    @property
    def mutual_info_lower_bound(self) -> float:
        """Variational lower bound on mutual information (in bits)."""
        return self.empirical_entropy - self.cross_entropy_bits


def empirical_entropy_bits(labels: List[str]) -> float:
    """Compute empirical entropy in bits."""
    cnt = Counter(labels)
    n = sum(cnt.values())
    ps = np.array([c / n for c in cnt.values() if c > 0], dtype=float)
    return float(-(ps * np.log2(ps)).sum())


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes_idx: np.ndarray,
    n_train: int,
    n_test: int,
    test_labels: List[str],
) -> EvaluationMetrics:
    """Compute all evaluation metrics."""
    ce_nat = log_loss(y_true, probabilities, labels=classes_idx)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    H_bits = empirical_entropy_bits(test_labels)

    return EvaluationMetrics(
        cross_entropy=ce_nat,
        accuracy=acc,
        macro_f1=f1m,
        empirical_entropy=H_bits,
        n_classes=len(classes_idx),
        n_train=n_train,
        n_test=n_test,
    )


def print_results(metrics: EvaluationMetrics, config: object) -> None:  # type: ignore[no-untyped-def]
    """Print evaluation results."""
    unit = "bits" if config.bits else "nats"  # type: ignore[attr-defined]
    ce_display = metrics.cross_entropy_bits if config.bits else metrics.cross_entropy  # type: ignore[attr-defined]

    print("\n=== Results (embedding features) ===")
    print(f"Target label:     {config.label}  (#classes={metrics.n_classes})")  # type: ignore[attr-defined]
    print(f"Feats:            {config.feats} | model={config.embed_model or 'n/a'} | pool={config.pool if config.feats == 'hf-cls' else '-'}")  # type: ignore[attr-defined]
    print(f"Rep filter:       {config.rep}")  # type: ignore[attr-defined]
    print(f"N_train / N_test: {metrics.n_train} / {metrics.n_test}")
    print(f"Cross-entropy:    {ce_display:.4f} {unit}")
    print(f"Accuracy:         {metrics.accuracy:.4f}")
    print(f"Macro F1:         {metrics.macro_f1:.4f}")
    print(f"H({config.label}):  {metrics.empirical_entropy:.4f} bits  (empirical on test)")  # type: ignore[attr-defined]
    print(f"I({config.label}; Z_r) ≥ {metrics.mutual_info_lower_bound:.4f} bits   (variational lower bound)")  # type: ignore[attr-defined]
