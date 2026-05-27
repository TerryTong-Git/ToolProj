#!/usr/bin/env python3
"""
2-D t-SNE scatter of native NL vs GPT-4o translated traces.

Uses the cached embeddings from the pooled experiment, takes the first 200
samples, and projects with t-SNE.

Usage:
    uv run python -m src.translation_discrimination.reports.native_vs_translated_scatter
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED = 42
N = 200


def main():
    native_emb = np.load(RESULTS_DIR / "_translator_native_emb_cache.npz")["emb"][:N]
    trans_emb = np.load(RESULTS_DIR / "_translator_GPT-4o_emb_cache.npz")["emb"][:N]

    X = np.vstack([native_emb, trans_emb])
    print(f"{N} native + {N} translated, dim={X.shape[1]}")

    # Load pre-computed cosine similarities (conditioned on same task type)
    with open(RESULTS_DIR / "embedding_paired_stats_large.json") as f:
        cos_stats = json.load(f)
    nn = cos_stats["native_native"]
    tt = cos_stats["translated_translated"]
    nt = cos_stats["native_translated"]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, learning_rate="auto", init="pca")
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(X_tsne[:N, 0], X_tsne[:N, 1], c="#2ecc71", alpha=0.6, s=50, label="Native NL", edgecolors="white", linewidths=0.3)
    ax.scatter(X_tsne[N:, 0], X_tsne[N:, 1], c="#e74c3c", alpha=0.6, s=50, label="Translated (GPT-4o)", edgecolors="white", linewidths=0.3)

    ax.set_xlabel("t-SNE Dim 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("t-SNE Dim 2", fontsize=14, fontweight="bold")
    ax.set_title(
        "Native NL vs GPT-4o Translated in Embedding Space\n" f"(text-embedding-3-large \u00b7 t-SNE \u00b7 n={N} per class)",
        fontsize=15,
        fontweight="bold",
    )

    # Scatter legend — top-left, compact
    ax.legend(fontsize=13, loc="upper left", framealpha=0.9)

    # Cosine similarity box — bottom-left, separate from data
    cos_text = (
        "Cosine Similarity (same task)\n"
        f"  Nat \u2194 Nat:      {nn['mean']:.3f} \u00b1 {nn['std']:.3f}\n"
        f"  Trans \u2194 Trans:  {tt['mean']:.3f} \u00b1 {tt['std']:.3f}\n"
        f"  Nat \u2194 Trans:   {nt['mean']:.3f} \u00b1 {nt['std']:.3f}"
    )
    ax.text(
        0.02,
        0.02,
        cos_text,
        transform=ax.transAxes,
        fontsize=12,
        fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # Cluster detection on t-SNE coordinates and draw ellipses
    db = DBSCAN(eps=3.0, min_samples=3).fit(X_tsne)
    labels = db.labels_
    unique_labels = set(labels) - {-1}  # exclude noise

    for label in unique_labels:
        mask = labels == label
        pts = X_tsne[mask]
        if len(pts) < 3:
            continue

        # Fit ellipse via covariance
        cx, cy = pts.mean(axis=0)
        cov = np.cov(pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        # 2.0 std covers ~86% of points
        width = 2.0 * 2 * np.sqrt(eigenvalues[0])
        height = 2.0 * 2 * np.sqrt(eigenvalues[1])

        ellipse = Ellipse(
            (cx, cy),
            width,
            height,
            angle=angle,
            fill=False,
            edgecolor="#444444",
            linewidth=2.0,
            linestyle="--",
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(ellipse)

    print(f"Found {len(unique_labels)} clusters")

    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    out = RESULTS_DIR / "native_vs_translated_scatter"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out.with_suffix('.png')} / .pdf")


if __name__ == "__main__":
    main()
