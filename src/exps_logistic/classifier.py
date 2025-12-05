#!/usr/bin/env python3
"""Logistic regression classifier for concept classification."""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class ClassifierResult:
    """Results from classifier training and prediction."""

    predictions: np.ndarray
    probabilities: np.ndarray
    true_labels: np.ndarray
    label_encoder: LabelEncoder


class ConceptClassifier:
    """Multinomial logistic regression classifier for concept labels."""

    def __init__(self, C: float = 2.0, max_iter: int = 400):
        self.C = C
        self.max_iter = max_iter
        self.label_encoder = LabelEncoder()

    def fit(self, X: np.ndarray, y_labels: List[str]) -> "ConceptClassifier":
        """Fit the classifier on training data."""
        y = self.label_encoder.fit_transform(y_labels)

        self.clf = LogisticRegression(
            penalty="l2",
            C=self.C,
            solver="saga",
            multi_class="multinomial",
            max_iter=self.max_iter,
            n_jobs=-1,
            verbose=1,
        )
        self.clf.fit(X, y)
        logger.info("Classifier training complete")

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and probabilities."""
        if self.clf is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        P = self.clf.predict_proba(X)
        yhat = P.argmax(1)

        return yhat, P

    def evaluate(self, X: np.ndarray, y_labels: List[str]) -> ClassifierResult:
        """Evaluate on test data and return results."""
        y_true = self.label_encoder.transform(y_labels)
        yhat, P = self.predict(X)

        return ClassifierResult(
            predictions=yhat,
            probabilities=P,
            true_labels=y_true,
            label_encoder=self.label_encoder,
        )

    @property
    def classes(self) -> np.ndarray:
        """Return the class labels."""
        return self.label_encoder.classes_

    @property
    def n_classes(self) -> int:
        """Return the number of classes."""
        return len(self.label_encoder.classes_)
