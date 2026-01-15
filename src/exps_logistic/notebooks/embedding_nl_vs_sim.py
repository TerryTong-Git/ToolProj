#!/usr/bin/env python3
"""Compare NL reasoning vs sim_reasoning (code simulation) embeddings.

This script creates embedding visualizations comparing:
- nl_reasoning: Pure natural language CoT reasoning
- sim_reasoning: Model's NL simulation of its generated code

Use OpenAI embeddings for high-quality semantic representations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from src.exps_logistic.config import (
    ARITHMETIC_KINDS,
    CLRS_KINDS,
    EXTENDED_KINDS,
    FG_KINDS,
    ILP_KINDS,
    NPHARD_KINDS,
)

# Default paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "src/exps_performance/results"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "src/exps_logistic/notebooks"

# Colors for NL vs Sim
NL_COLOR = "#3498db"  # Blue
SIM_COLOR = "#9b59b6"  # Purple (different from code red)

KINDS_PRESETS = {
    "fg": FG_KINDS,
    "clrs": CLRS_KINDS,
    "nphard": NPHARD_KINDS,
    "extended": EXTENDED_KINDS,
    "arithmetic": ARITHMETIC_KINDS,
    "ilp": ILP_KINDS,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Compare NL reasoning vs sim_reasoning or code embeddings."
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Filter by model names",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Filter by seeds",
    )
    p.add_argument(
        "--embed-model",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model (default: text-embedding-3-large)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embeddings (cuda/cpu/None for auto)",
    )
    p.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30.0)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max total samples to use (default: 2000)",
    )
    p.add_argument(
        "--kinds",
        choices=["fg", "clrs", "nphard", "extended", "arithmetic", "ilp"],
        default="nphard",
        help="Algorithm kinds preset (default: nphard)",
    )
    p.add_argument(
        "--compare",
        choices=["sim", "code"],
        default="sim",
        help="Compare NL with: sim (sim_reasoning) or code (sim_code) (default: sim)",
    )
    p.add_argument(
        "--draw-ellipses",
        action="store_true",
        help="Draw confidence ellipses around clusters",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--by-task",
        action="store_true",
        help="Plot NL and comparison side-by-side, colored by task/algorithm kind",
    )
    return p.parse_args()


def load_paired_data(
    results_dir: str,
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    max_samples: int = 2000,
    kinds_preset: str = "nphard",
    random_seed: int = 42,
    compare_field: str = "sim_reasoning",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load paired NL and comparison rationales.

    Args:
        compare_field: Field to compare with NL - either "sim_reasoning" or "sim_code"

    Returns:
        nl_texts: List of NL reasoning texts
        compare_texts: List of comparison texts (sim_reasoning or sim_code)
        kinds: List of algorithm kinds for each pair
    """
    from src.exps_performance.logger import create_big_df

    root = Path(results_dir)
    jsonl_files = sorted(root.rglob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {results_dir}")

    print(f"Found {len(jsonl_files)} JSONL files")
    df = create_big_df(jsonl_files)
    print(f"Loaded {len(df)} total rows")

    # Filter by models/seeds
    if models:
        df = df[df["model"].isin(models)]
        print(f"Filtered to {len(df)} rows for models: {models}")
    if seeds:
        df = df[df["seed"].isin(seeds)]
        print(f"Filtered to {len(df)} rows for seeds: {seeds}")

    # Filter by kinds
    kinds_set = KINDS_PRESETS.get(kinds_preset, NPHARD_KINDS)
    df = df[df["kind"].isin(kinds_set)]
    print(f"Filtered to {len(df)} rows with {kinds_preset} kinds ({len(kinds_set)} types)")
    print(f"Kinds present: {sorted(df['kind'].unique())}")

    # Keep only rows with both NL and comparison field
    df = df.dropna(subset=["nl_reasoning", compare_field])
    df = df[df["nl_reasoning"].str.strip() != ""]
    df = df[df[compare_field].str.strip() != ""]
    print(f"Filtered to {len(df)} rows with both NL and {compare_field}")

    if len(df) == 0:
        raise ValueError(f"No rows with both nl_reasoning and {compare_field} found!")

    # Subsample if needed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_seed)
        print(f"Subsampled to {len(df)} rows")

    nl_texts = df["nl_reasoning"].tolist()
    compare_texts = df[compare_field].tolist()
    kinds = df["kind"].tolist()

    return nl_texts, compare_texts, kinds


def compute_embeddings_openai(
    texts: List[str],
    model_name: str = "text-embedding-3-large",
    batch_size: int = 100,
) -> np.ndarray:
    """Compute embeddings using OpenAI API."""
    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}...")
        response = client.embeddings.create(input=batch, model=model_name)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def compute_embeddings_sentence_transformers(
    texts: List[str],
    model_name: str,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute embeddings using sentence-transformers."""
    from src.exps_logistic.featurizer import SentenceTransformersFeaturizer

    featurizer = SentenceTransformersFeaturizer(model_name=model_name, device=device)
    embeddings = featurizer.transform(texts)
    return embeddings


def compute_embeddings(
    texts: List[str],
    model_name: str,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute embeddings using OpenAI or sentence-transformers."""
    if model_name.startswith("text-embedding"):
        return compute_embeddings_openai(texts, model_name)
    else:
        return compute_embeddings_sentence_transformers(texts, model_name, device)


def draw_confidence_ellipse(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    n_std: float = 2.0,
    alpha: float = 0.2,
    label: str = "",
) -> None:
    """Draw a confidence ellipse around a set of points."""
    if len(points) < 3:
        return

    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)

    if np.any(np.isnan(cov)) or np.linalg.det(cov) < 1e-10:
        return

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=2,
        label=label,
    )
    ax.add_patch(ellipse)


def build_task_colormap(kinds: List[str]) -> dict:
    """Build color mapping for algorithm kinds."""
    unique_kinds = sorted(set(kinds))
    n_kinds = len(unique_kinds)
    if n_kinds <= 10:
        colors = sns.color_palette("tab10", n_kinds)
    else:
        colors = sns.color_palette("husl", n_kinds)
    return {kind: colors[i] for i, kind in enumerate(unique_kinds)}


def generate_side_by_side_by_task(
    nl_texts: List[str],
    compare_texts: List[str],
    kinds: List[str],
    args,
    compare_label: str = "sim_reasoning",
) -> None:
    """Generate side-by-side plot with NL and comparison colored by task."""
    print("\n" + "=" * 60)
    print(f"Side-by-side NL vs {compare_label} (colored by task)")
    print("=" * 60)

    # Compute embeddings separately
    print("\nComputing NL embeddings...")
    nl_embeddings = compute_embeddings(nl_texts, args.embed_model, args.device)
    print(f"NL embedding shape: {nl_embeddings.shape}")

    print(f"Computing {compare_label} embeddings...")
    compare_embeddings = compute_embeddings(compare_texts, args.embed_model, args.device)
    print(f"{compare_label} embedding shape: {compare_embeddings.shape}")

    # Run t-SNE on combined embeddings for comparable coordinates
    print("\nRunning joint t-SNE...")
    all_embeddings = np.vstack([nl_embeddings, compare_embeddings])
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, len(all_embeddings) // 4),  # Adjust perplexity for small samples
        max_iter=1000,
        random_state=args.seed,
    )
    all_coords = tsne.fit_transform(all_embeddings)

    n_samples = len(nl_texts)
    nl_coords = all_coords[:n_samples]
    compare_coords = all_coords[n_samples:]

    # Compute silhouette scores by task
    le = LabelEncoder()
    kind_labels = le.fit_transform(kinds)

    sil_nl = silhouette_score(nl_coords, kind_labels) if len(set(kind_labels)) > 1 else float('nan')
    sil_compare = silhouette_score(compare_coords, kind_labels) if len(set(kind_labels)) > 1 else float('nan')

    print(f"Silhouette by task (NL):           {sil_nl:.4f}")
    print(f"Silhouette by task ({compare_label}): {sil_compare:.4f}")

    # Build colormap
    color_map = build_task_colormap(kinds)
    unique_kinds = sorted(set(kinds))

    # Determine display title for comparison
    compare_title = "Code (sim_code)" if compare_label == "code" else "sim_reasoning (Code Simulation)"

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, coords, title, sil in [
        (axes[0], nl_coords, "NL Reasoning", sil_nl),
        (axes[1], compare_coords, compare_title, sil_compare),
    ]:
        # Plot each kind
        for kind in unique_kinds:
            mask = [k == kind for k in kinds]
            kind_coords = coords[mask]
            ax.scatter(
                kind_coords[:, 0],
                kind_coords[:, 1],
                c=[color_map[kind]],
                label=kind,
                alpha=0.7,
                s=40,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_title(f"{title}\nSilhouette (by task): {sil:.3f}", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(unique_kinds), 9),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    short_label = "code" if compare_label == "code" else "sim"
    plt.suptitle(f"{args.kinds.upper()} Tasks: NL vs {compare_label} ({args.embed_model})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92)

    # Save figure
    output_path = Path(args.output_dir) / f"embedding_nl_vs_{short_label}_{args.kinds}_by_task.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")
    plt.close(fig)


def generate_combined_plot(
    nl_texts: List[str],
    sim_texts: List[str],
    kinds: List[str],
    args,
) -> None:
    """Generate combined plot showing NL vs sim_reasoning separation."""
    print("\n" + "=" * 60)
    print("Combined NL vs sim_reasoning embedding plot")
    print("=" * 60)

    n_samples = len(nl_texts)

    # Combine texts for joint embedding
    all_texts = nl_texts + sim_texts
    labels = ["NL"] * n_samples + ["sim_reasoning"] * n_samples

    print(f"\nComputing embeddings for {len(all_texts)} texts...")
    embeddings = compute_embeddings(all_texts, args.embed_model, args.device)
    print(f"Embedding shape: {embeddings.shape}")

    # Compute silhouette score in embedding space
    numeric_labels = np.array([0 if lab == "NL" else 1 for lab in labels])
    sil_embed = silhouette_score(embeddings, numeric_labels)
    print(f"Silhouette (embedding space): {sil_embed:.4f}")

    # Dimensionality reduction
    print("\nRunning PCA...")
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)
    sil_pca = silhouette_score(pca_coords, numeric_labels)
    print(f"Silhouette (PCA): {sil_pca:.4f}")
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    print("\nRunning t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, len(all_texts) // 4),
        max_iter=1000,
        random_state=args.seed,
    )
    tsne_coords = tsne.fit_transform(embeddings)
    sil_tsne = silhouette_score(tsne_coords, numeric_labels)
    print(f"Silhouette (t-SNE): {sil_tsne:.4f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coords, method, sil in [
        (axes[0], pca_coords, "PCA", sil_pca),
        (axes[1], tsne_coords, "t-SNE", sil_tsne),
    ]:
        nl_coords = coords[:n_samples]
        sim_coords = coords[n_samples:]

        # Draw ellipses first (behind points)
        if args.draw_ellipses:
            draw_confidence_ellipse(ax, nl_coords, NL_COLOR, n_std=2.0, alpha=0.15)
            draw_confidence_ellipse(ax, sim_coords, SIM_COLOR, n_std=2.0, alpha=0.15)

        # Scatter plot
        ax.scatter(
            nl_coords[:, 0],
            nl_coords[:, 1],
            c=NL_COLOR,
            label="NL Reasoning",
            alpha=0.6,
            s=30,
            edgecolors="white",
            linewidth=0.3,
        )
        ax.scatter(
            sim_coords[:, 0],
            sim_coords[:, 1],
            c=SIM_COLOR,
            label="sim_reasoning",
            alpha=0.6,
            s=30,
            edgecolors="white",
            linewidth=0.3,
        )

        # Compute centroid distance
        nl_centroid = np.mean(nl_coords, axis=0)
        sim_centroid = np.mean(sim_coords, axis=0)
        centroid_dist = np.linalg.norm(nl_centroid - sim_centroid)

        ax.set_title(
            f"{method}\nSilhouette: {sil:.3f}, Centroid dist: {centroid_dist:.2f}",
            fontsize=12,
        )
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        ax.legend(loc="upper right")

        for spine in ax.spines.values():
            spine.set_visible(False)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Silhouette (embedding): {sil_embed:.4f}")
    print(f"  Silhouette (PCA):       {sil_pca:.4f}")
    print(f"  Silhouette (t-SNE):     {sil_tsne:.4f}")
    print("=" * 60)

    # Compute paired similarity
    nl_embeddings = embeddings[:n_samples]
    sim_embeddings = embeddings[n_samples:]

    nl_norm = nl_embeddings / (np.linalg.norm(nl_embeddings, axis=1, keepdims=True) + 1e-10)
    sim_norm = sim_embeddings / (np.linalg.norm(sim_embeddings, axis=1, keepdims=True) + 1e-10)

    # Paired similarity (same problem, NL vs sim_reasoning)
    paired_sim = np.sum(nl_norm * sim_norm, axis=1)
    print("\nPaired NL-sim_reasoning similarity (same problem):")
    print(f"  Mean: {paired_sim.mean():.4f}")
    print(f"  Std:  {paired_sim.std():.4f}")

    # Random cross-similarity
    random_idx = np.random.permutation(n_samples)
    random_sim = np.sum(nl_norm * sim_norm[random_idx], axis=1)
    print("\nRandom NL-sim_reasoning similarity (different problems):")
    print(f"  Mean: {random_sim.mean():.4f}")
    print(f"  Std:  {random_sim.std():.4f}")

    # Save figure
    plt.suptitle(f"NPHardEval: NL vs sim_reasoning ({args.embed_model})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    output_path = Path(args.output_dir) / f"embedding_nl_vs_sim_{args.kinds}.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")
    plt.close(fig)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    np.random.seed(args.seed)

    # Determine comparison field and label
    compare_field = "sim_code" if args.compare == "code" else "sim_reasoning"
    compare_label = "code" if args.compare == "code" else "sim_reasoning"

    print("=" * 60)
    print(f"NL vs {compare_label} Embedding Comparison")
    print("=" * 60)
    print(f"Embedding model: {args.embed_model}")
    print(f"Max samples: {args.max_samples}")
    print(f"Kinds preset: {args.kinds}")
    print(f"Compare: NL vs {compare_label}")
    print(f"Perplexity: {args.perplexity}")
    print("=" * 60)

    # Load paired data
    nl_texts, compare_texts, kinds = load_paired_data(
        args.results_dir,
        args.models,
        args.seeds,
        args.max_samples,
        args.kinds,
        args.seed,
        compare_field=compare_field,
    )

    n_samples = len(nl_texts)
    print(f"\nLoaded {n_samples} paired samples")
    print(f"Unique tasks: {len(set(kinds))}")

    # Side-by-side by task mode
    if args.by_task:
        generate_side_by_side_by_task(nl_texts, compare_texts, kinds, args, compare_label=compare_label)
    else:
        generate_combined_plot(nl_texts, compare_texts, kinds, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
