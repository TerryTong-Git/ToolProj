#!/usr/bin/env python3
"""
Can a linear classifier distinguish WHICH translator produced a trace?

Uses the same 288 samples translated by GPT-4o, Gemini 2.0 Flash, and
Claude Sonnet 4.5. Pools all translated embeddings and fits a 3-class
logistic regression to predict translator identity.

If the stylistic signal is translator-specific → high accuracy.
If the signal is a shared code-to-NL artefact → near chance (33%).

Usage:
    uv run python -m src.translation_discrimination.analysis.translator_identity_classifier
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

TRANSLATORS = ["GPT-4o", "Gemini 2.0\nFlash", "Claude\nSonnet 4.5"]
# Cache files were saved by replacing \n with _ in the translator name
SAFE_NAMES = ["GPT-4o", "Gemini 2.0_Flash", "Claude_Sonnet 4.5"]
COLORS = ["#3498db", "#e74c3c", "#2ecc71"]


def load_embeddings():
    """Load cached translated embeddings for all 3 translators."""
    embs = []
    for safe in SAFE_NAMES:
        path = RESULTS_DIR / f"_translator_{safe}_emb_cache.npz"
        embs.append(np.load(path)["emb"])
    # Also load native
    native = np.load(RESULTS_DIR / "_translator_native_emb_cache.npz")["emb"]
    return embs, native


def main():
    print("=" * 60)
    print("Translator Identity Classification")
    print("=" * 60)

    embs, native_emb = load_embeddings()
    n = len(embs[0])
    print(f"  {n} samples per translator, dim={embs[0].shape[1]}")

    # --- 3-class: which translator? ---
    X = np.vstack(embs)
    y = np.concatenate([np.full(n, i) for i in range(3)])

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, multi_class="multinomial")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
    cm = confusion_matrix(y, y_pred)
    short_names = ["GPT-4o", "Gemini Flash", "Claude Sonnet"]
    report = classification_report(y, y_pred, target_names=short_names)

    print(f"\n  3-class Accuracy: {acc:.3f}  (chance = 0.333)")
    print(f"  AUC (macro OvR):  {auc:.3f}")
    print("\nConfusion Matrix:")
    print(f"  {'':>16} {'GPT-4o':>10} {'Gemini':>10} {'Claude':>10}")
    for i, name in enumerate(short_names):
        print(f"  {name:>16} {cm[i,0]:>10} {cm[i,1]:>10} {cm[i,2]:>10}")
    print(f"\n{report}")

    # --- 4-class: native + 3 translators ---
    X4 = np.vstack([native_emb, *embs])
    y4 = np.concatenate([
        np.full(n, 0),  # native
        np.full(n, 1),  # gpt4o
        np.full(n, 2),  # gemini
        np.full(n, 3),  # claude
    ])

    clf4 = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, multi_class="multinomial")
    y4_pred = cross_val_predict(clf4, X4, y4, cv=cv, method="predict")
    y4_prob = cross_val_predict(clf4, X4, y4, cv=cv, method="predict_proba")

    acc4 = accuracy_score(y4, y4_pred)
    auc4 = roc_auc_score(y4, y4_prob, multi_class="ovr", average="macro")
    cm4 = confusion_matrix(y4, y4_pred)
    names4 = ["Native NL", "GPT-4o", "Gemini Flash", "Claude Sonnet"]
    report4 = classification_report(y4, y4_pred, target_names=names4)

    print("=" * 60)
    print("4-class: Native + 3 Translators")
    print("=" * 60)
    print(f"  Accuracy: {acc4:.3f}  (chance = 0.250)")
    print(f"  AUC:      {auc4:.3f}")
    print("\nConfusion Matrix:")
    print(f"  {'':>16} {'Native':>10} {'GPT-4o':>10} {'Gemini':>10} {'Claude':>10}")
    for i, name in enumerate(names4):
        print(f"  {name:>16} {cm4[i,0]:>10} {cm4[i,1]:>10} {cm4[i,2]:>10} {cm4[i,3]:>10}")
    print(f"\n{report4}")

    # --- PCA plot: 4-class ---
    clf4.fit(X4, y4)
    pca = PCA(n_components=2, random_state=SEED)
    X4_2d = pca.fit_transform(X4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: 3 translators only
    colors_3 = COLORS
    labels_3 = ["GPT-4o", "Gemini 2.0 Flash", "Claude Sonnet 4.5"]
    X3_2d = pca.transform(X)  # project 3-translator data using same PCA

    ax = axes[0]
    for i in range(3):
        mask = (y == i)
        ax.scatter(X3_2d[mask, 0], X3_2d[mask, 1],
                   c=colors_3[i], alpha=0.45, s=20, label=labels_3[i], edgecolors="none")
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title(f"Translator Identity (3-class)\nAcc={acc:.1%}  AUC={auc:.3f}  (chance=33%)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: native + 3 translators
    colors_4 = ["#9b59b6"] + COLORS
    labels_4 = ["Native NL", "GPT-4o", "Gemini 2.0 Flash", "Claude Sonnet 4.5"]

    ax = axes[1]
    for i in range(4):
        mask = (y4 == i)
        ax.scatter(X4_2d[mask, 0], X4_2d[mask, 1],
                   c=colors_4[i], alpha=0.45, s=20, label=labels_4[i], edgecolors="none")
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title(f"Native + 3 Translators (4-class)\nAcc={acc4:.1%}  AUC={auc4:.3f}  (chance=25%)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Can a Linear Classifier Identify the Translator?\n"
                 "(text-embedding-3-large · logistic regression · 5-fold CV)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "translator_identity_classifier"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\nSaved plot → {out.with_suffix('.png')} / .pdf")

    # Save metrics
    metrics = {
        "three_class": {
            "accuracy": acc, "auc": auc,
            "confusion_matrix": cm.tolist(),
            "chance": 1 / 3,
        },
        "four_class": {
            "accuracy": acc4, "auc": auc4,
            "confusion_matrix": cm4.tolist(),
            "chance": 0.25,
        },
    }
    with open(RESULTS_DIR / "translator_identity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics → translator_identity_metrics.json")


if __name__ == "__main__":
    main()
