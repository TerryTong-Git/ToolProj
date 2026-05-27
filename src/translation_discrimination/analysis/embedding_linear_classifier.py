#!/usr/bin/env python3
"""
Visualise native vs translated traces in embedding space and fit a linear
classifier to measure separability.

1. Loads cached text-embedding-3-large embeddings (from embedding_analysis_large.py).
2. Projects to 2-D via PCA.
3. Plots native vs translated in the PCA space.
4. Fits logistic regression on full-dimensional embeddings (with train/test split)
   and reports AUC, accuracy, precision, recall, F1.
5. Fits logistic regression on 2-D PCA and overlays the decision boundary.

Usage:
    uv run python -m src.translation_discrimination.analysis.embedding_linear_classifier
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR


def load_data():
    """Load cached embeddings and labels."""
    cache = np.load(OUTPUT_DIR / "_embedding_cache_large.npz")
    native_emb = cache["native"]
    translated_emb = cache["translated"]

    X = np.vstack([native_emb, translated_emb])
    y = np.array([0] * len(native_emb) + [1] * len(translated_emb))
    return X, y, len(native_emb), len(translated_emb)


def fit_and_report(X: np.ndarray, y: np.ndarray, label: str):
    """Fit logistic regression with stratified 5-fold CV, return metrics + model."""
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Native", "Translated"])

    print(f"\n{'=' * 60}")
    print(f"Logistic Regression — {label}")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"\n{report}")

    # Refit on full data for decision boundary
    clf.fit(X, y)
    return clf, {"accuracy": acc, "auc": auc, "confusion_matrix": cm.tolist(),
                 "report": report}


def plot_embedding_space(X_2d, y, clf_2d, n_native, n_translated, output_path):
    """Scatter plot with logistic regression decision boundary in PCA space."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Decision boundary mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    Z = clf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Shaded regions
    cmap_bg = ListedColormap(["#d5f5d5", "#f5d5d5"])
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=cmap_bg, alpha=0.35)

    # Decision boundary line
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2,
               linestyles="--")

    # Scatter points
    native_mask = y == 0
    translated_mask = y == 1
    ax.scatter(X_2d[native_mask, 0], X_2d[native_mask, 1],
               c="#2ecc71", alpha=0.5, s=30, label=f"Native NL (n={n_native})",
               edgecolors="none")
    ax.scatter(X_2d[translated_mask, 0], X_2d[translated_mask, 1],
               c="#e74c3c", alpha=0.5, s=30, label=f"Translated (n={n_translated})",
               edgecolors="none")

    ax.set_xlabel("PC 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("PC 2", fontsize=13, fontweight="bold")
    ax.set_title(
        "Embedding Space: Native NL vs GPT-4o Translated\n"
        "(PCA of text-embedding-3-large · logistic regression boundary)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved plot → {output_path.with_suffix('.png')} / .pdf")


def main():
    print("Loading cached embeddings...")
    X, y, n_native, n_translated = load_data()
    print(f"  {n_native} native, {n_translated} translated  (dim={X.shape[1]})")

    # --- Full-dimensional logistic regression ---
    clf_full, metrics_full = fit_and_report(X, y, f"Full-dimensional ({X.shape[1]}-d)")

    # --- PCA to 2-D ---
    print("\nRunning PCA → 2-D...")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
          f"total={sum(pca.explained_variance_ratio_):.3f}")

    # --- 2-D logistic regression ---
    clf_2d, metrics_2d = fit_and_report(X_2d, y, "2-D PCA")

    # --- Plot ---
    plot_embedding_space(X_2d, y, clf_2d, n_native, n_translated,
                         OUTPUT_DIR / "embedding_linear_classifier")

    # --- Save metrics ---
    out = {
        "full_dim": {
            "dimensions": X.shape[1],
            "accuracy": metrics_full["accuracy"],
            "auc": metrics_full["auc"],
            "confusion_matrix": metrics_full["confusion_matrix"],
        },
        "pca_2d": {
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "accuracy": metrics_2d["accuracy"],
            "auc": metrics_2d["auc"],
            "confusion_matrix": metrics_2d["confusion_matrix"],
        },
    }
    stats_path = OUTPUT_DIR / "embedding_classifier_metrics.json"
    with open(stats_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics → {stats_path}")


if __name__ == "__main__":
    main()
