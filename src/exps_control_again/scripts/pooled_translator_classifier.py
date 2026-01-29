#!/usr/bin/env python3
"""
Native vs Translated classification with ALL translators pooled.

Previous runs conditioned on a single translator (GPT-4o OR Gemini OR Claude).
This pools all translated traces into one class regardless of translator.

If the signal is about "translation vs native" in general → still high accuracy.
If it was about a single translator's style → accuracy drops with mixed translators.

Usage:
    uv run python src/exps_control_again/scripts/pooled_translator_classifier.py
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
SEED = 42

CACHE_FILES = [
    "_translator_GPT-4o_emb_cache.npz",
    "_translator_Gemini 2.0_Flash_emb_cache.npz",
    "_translator_Claude_Sonnet 4.5_emb_cache.npz",
]


def main():
    print("=" * 60)
    print("Pooled Translator: Native vs Translated (mixed)")
    print("=" * 60)

    # Load native (shared across translators — same 288 samples)
    native_emb = np.load(RESULTS_DIR / "_translator_native_emb_cache.npz")["emb"]
    n = len(native_emb)

    # Load all translated embeddings
    trans_embs = []
    for fname in CACHE_FILES:
        emb = np.load(RESULTS_DIR / fname)["emb"]
        trans_embs.append(emb)
        print(f"  Loaded {fname}: {emb.shape}")

    # Pool: repeat native 3× to match (each translator used same 288 questions)
    # Native side: 288 × 3 = 864, Translated side: 288 × 3 = 864
    X = np.vstack([
        np.tile(native_emb, (3, 1)),  # native repeated 3×
        *trans_embs,                   # all translated pooled
    ])
    y = np.array([0] * (n * 3) + [1] * (n * 3))
    print(f"\n  Native: {n*3}  Translated: {n*3}  dim={X.shape[1]}")

    # --- Full-dim logistic regression (5-fold CV) ---
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Native", "Translated (mixed)"])

    print(f"\n  Full-dim ({X.shape[1]}-d)")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"\n{report}")

    # --- PCA + 2-D classifier ---
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X)

    clf_2d = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    y_pred_2d = cross_val_predict(clf_2d, X_2d, y, cv=cv, method="predict")
    y_prob_2d = cross_val_predict(clf_2d, X_2d, y, cv=cv, method="predict_proba")[:, 1]
    acc_2d = accuracy_score(y, y_pred_2d)
    auc_2d = roc_auc_score(y, y_prob_2d)

    print(f"  2-D PCA (var explained: {sum(pca.explained_variance_ratio_):.3f})")
    print(f"  Accuracy: {acc_2d:.4f}")
    print(f"  AUC:      {auc_2d:.4f}")

    # Refit for boundary
    clf.fit(X, y)
    clf_2d.fit(X_2d, y)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Decision boundary
    pad = 1.0
    x_min, x_max = X_2d[:, 0].min() - pad, X_2d[:, 0].max() + pad
    y_min, y_max = X_2d[:, 1].min() - pad, X_2d[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    Z = clf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    cmap_bg = ListedColormap(["#d5f5d5", "#f5d5d5"])
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=cmap_bg, alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2, linestyles="--")

    # Scatter — colour translated by translator for visual insight
    native_mask = y == 0
    native_2d = X_2d[native_mask]
    ax.scatter(native_2d[:, 0], native_2d[:, 1],
               c="#2ecc71", alpha=0.35, s=18, label=f"Native NL (n={n*3})", edgecolors="none")

    trans_colors = ["#3498db", "#e74c3c", "#9b59b6"]
    trans_labels = ["GPT-4o", "Gemini 2.0 Flash", "Claude Sonnet 4.5"]
    offset = n * 3  # start of translated in X
    for i in range(3):
        start = offset + i * n
        end = start + n
        ax.scatter(X_2d[start:end, 0], X_2d[start:end, 1],
                   c=trans_colors[i], alpha=0.45, s=18, label=f"Trans: {trans_labels[i]}",
                   edgecolors="none")

    ax.set_xlabel("PC 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("PC 2", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Native NL vs Translated (3 translators pooled)\n"
        f"Full-dim: Acc={acc:.1%} AUC={auc:.3f}  |  2-D PCA: Acc={acc_2d:.1%}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = RESULTS_DIR / "pooled_translator_classifier"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\nSaved → {out.with_suffix('.png')} / .pdf")

    # Save
    metrics = {
        "full_dim": {"accuracy": acc, "auc": auc, "confusion_matrix": cm.tolist()},
        "pca_2d": {"accuracy": acc_2d, "auc": auc_2d},
        "n_native": n * 3,
        "n_translated": n * 3,
        "translators_pooled": trans_labels,
    }
    with open(RESULTS_DIR / "pooled_translator_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
