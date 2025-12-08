#!/usr/bin/env python3
"""Logistic regression classifier for concept classification."""

import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

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
        self.clf: Optional[LogisticRegression] = None

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

    @staticmethod
    def tune_hyperparams(
        X: np.ndarray,
        y_labels: List[str],
        c_grid: Iterable[float],
        max_iter_grid: Iterable[int],
        cv_folds: int = 5,
    ) -> Tuple[float, int, float]:
        """
        Grid-search over (C, max_iter) using stratified CV, returning best (C, max_iter, score).
        """
        le = LabelEncoder()
        y = le.fit_transform(y_labels)

        best_c, best_mi, best_score = None, None, -np.inf
        splitter = StratifiedKFold(n_splits=min(cv_folds, len(np.unique(y))), shuffle=True, random_state=0)

        combos = [(float(c), int(mi)) for c in c_grid for mi in max_iter_grid]
        for c, mi in tqdm(combos, desc="LogReg CV"):
            clf = LogisticRegression(
                penalty="l2",
                C=c,
                solver="saga",
                multi_class="multinomial",
                max_iter=mi,
                n_jobs=-1,
                verbose=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(clf, X, y, cv=splitter, scoring="accuracy", n_jobs=-1)
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_c = c
                best_mi = mi
        logger.info(f"Selected hyperparams via CV: C={best_c}, max_iter={best_mi}, acc={best_score:.4f}")
        return best_c if best_c is not None else 1.0, best_mi if best_mi is not None else 100, best_score

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
        return np.asarray(self.label_encoder.classes_)

    @property
    def n_classes(self) -> int:
        """Return the number of classes."""
        return len(self.label_encoder.classes_)
