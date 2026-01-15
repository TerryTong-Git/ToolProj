#!/usr/bin/env python3
"""CoT embedding visualization with PCA and t-SNE for extended algorithm kinds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.exps_logistic.config import CLRS_KINDS, EXTENDED_KINDS, FG_KINDS, NPHARD_KINDS
from src.exps_logistic.data_utils import _convert_results_df, filter_by_kinds, filter_by_rep
from src.exps_logistic.featurizer import SentenceTransformersFeaturizer
from src.exps_performance.logger import create_big_df

# Kind presets for filtering
KINDS_PRESETS = {
    "fg": FG_KINDS,
    "clrs": CLRS_KINDS,
    "nphard": NPHARD_KINDS,
    "extended": EXTENDED_KINDS,
}

# Default paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "src/exps_performance/results"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "src/exps_logistic/notebooks"


def filter_algorithm_names(text: str, algorithm_names: Set[str]) -> str:
    """Remove algorithm names from text to prevent trivial clustering."""
    for name in algorithm_names:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        text = pattern.sub("", text)
    return text


def remove_code_comments(text: str) -> str:
    """Remove code comments (# and // style) from text."""
    lines = text.split("\n")
    cleaned_lines = []
    in_multiline = False

    for line in lines:
        # Handle multi-line strings/comments (triple quotes)
        if '"""' in line or "'''" in line:
            in_multiline = not in_multiline
            cleaned_lines.append(line)
            continue

        if in_multiline:
            cleaned_lines.append(line)
            continue

        # Remove # comments (but preserve string content)
        # Simple approach: remove everything after # if not in a string
        if "#" in line:
            # Find # that's not inside quotes
            in_string = False
            quote_char = None
            result = []
            for i, char in enumerate(line):
                if char in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                if char == "#" and not in_string:
                    break
                result.append(char)
            line = "".join(result).rstrip()

        # Remove // comments (C-style)
        if "//" in line:
            idx = line.find("//")
            # Simple check: not inside a string
            before = line[:idx]
            if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                line = before.rstrip()

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def compute_silhouette(embeddings: np.ndarray, labels: List[str]) -> float:
    """Compute silhouette score for clustering quality."""
    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)

    # Need at least 2 clusters and more samples than clusters
    n_clusters = len(set(numeric_labels))
    if n_clusters < 2 or len(embeddings) <= n_clusters:
        return float("nan")

    return silhouette_score(embeddings, numeric_labels)


def draw_cluster_ellipses(
    ax: plt.Axes,
    coords: np.ndarray,
    kinds: List[str],
    color_map: Dict[str, tuple],
    n_std: float = 2.0,
    alpha: float = 0.15,
) -> None:
    """Draw ellipses around each cluster based on covariance."""
    unique_kinds = sorted(set(kinds))
    for kind in unique_kinds:
        mask = np.array([k == kind for k in kinds])
        points = coords[mask]

        if len(points) < 3:
            continue

        # Compute mean and covariance
        mean = np.mean(points, axis=0)
        cov = np.cov(points.T)

        # Handle degenerate cases
        if np.any(np.isnan(cov)) or np.linalg.det(cov) < 1e-10:
            continue

        # Compute eigenvalues and eigenvectors for ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Ellipse parameters
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])

        # Draw ellipse
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            facecolor=color_map.get(kind, (0.5, 0.5, 0.5)),
            edgecolor=color_map.get(kind, (0.5, 0.5, 0.5)),
            alpha=alpha,
            linewidth=1.5,
        )
        ax.add_patch(ellipse)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for embedding visualization."""
    p = argparse.ArgumentParser(
        description="Visualize CoT embeddings with PCA and t-SNE for extended algorithm kinds."
    )

    # Data source
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

    # Embedding settings
    p.add_argument(
        "--embed-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embeddings (cuda/cpu/None for auto)",
    )

    # Dimensionality reduction
    p.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30.0)",
    )
    p.add_argument(
        "--tsne-iter",
        type=int,
        default=1000,
        help="t-SNE iterations (default: 1000)",
    )

    # Visualization
    p.add_argument(
        "--rep",
        choices=["nl", "code", "both"],
        default="both",
        help="Representation type to visualize (default: both)",
    )
    p.add_argument(
        "--kinds",
        choices=["fg", "clrs", "nphard", "extended"],
        default="extended",
        help="Algorithm kinds preset to visualize (default: extended)",
    )
    p.add_argument(
        "--draw-circles",
        action="store_true",
        help="Draw ellipse circles around each cluster",
    )
    p.add_argument(
        "--max-samples-per-kind",
        type=int,
        default=500,
        help="Max samples per algorithm kind for subsampling (default: 500)",
    )
    p.add_argument(
        "--point-size",
        type=int,
        default=15,
        help="Scatter point size (default: 15)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Point transparency (default: 0.6)",
    )
    p.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[16, 14],
        help="Figure size in inches (default: 16 14)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )

    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default="embedding",
        help="Output filename prefix (default: embedding)",
    )

    # Text filtering options
    p.add_argument(
        "--filter-algo-names",
        action="store_true",
        help="Filter out algorithm names from CoT text",
    )
    p.add_argument(
        "--remove-comments",
        action="store_true",
        help="Remove code comments from CoT text (for code rep)",
    )

    # Misc
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--lda-task",
        action="store_true",
        help="Generate LDA vs residual PC plot for task-relevant clustering analysis",
    )
    p.add_argument(
        "--dual-clustering",
        action="store_true",
        help="Generate dual clustering figure (KIND vs CORRECTNESS) to show dissociation",
    )
    p.add_argument(
        "--sil-vs-hardness",
        action="store_true",
        help="Generate silhouette vs task hardness plot for NL/code/sim",
    )

    return p.parse_args()


def build_colormap(kinds: List[str]) -> Tuple[Dict[str, tuple], Dict[str, str]]:
    """
    Build color and marker mappings for algorithm kinds.

    Returns:
        color_map: dict mapping kind -> RGB color tuple
        marker_map: dict mapping kind -> marker shape
    """
    # Sort kinds by category for consistent ordering
    fg_list = sorted([k for k in kinds if k in FG_KINDS])
    clrs_list = sorted([k for k in kinds if k in CLRS_KINDS])
    nphard_list = sorted([k for k in kinds if k in NPHARD_KINDS])

    # Use distinct color palettes for each category
    fg_colors = sns.color_palette("Set2", n_colors=max(9, len(fg_list)))
    clrs_colors = sns.color_palette("tab20", n_colors=20)
    if len(clrs_list) > 20:
        clrs_colors = list(clrs_colors) + list(sns.color_palette("tab20b", n_colors=len(clrs_list) - 20))
    nphard_colors = sns.color_palette("Dark2", n_colors=max(5, len(nphard_list)))

    color_map = {}
    for i, k in enumerate(fg_list):
        color_map[k] = fg_colors[i]
    for i, k in enumerate(clrs_list):
        color_map[k] = clrs_colors[i]
    for i, k in enumerate(nphard_list):
        color_map[k] = nphard_colors[i]

    # Marker shapes by category
    marker_map = {}
    for k in fg_list:
        marker_map[k] = "o"  # Circle for FG
    for k in clrs_list:
        marker_map[k] = "^"  # Triangle for CLRS
    for k in nphard_list:
        marker_map[k] = "s"  # Square for NP-hard

    return color_map, marker_map


def load_and_subsample(
    results_dir: str,
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    max_samples_per_kind: int = 500,
    random_seed: int = 42,
    kinds_preset: str = "extended",
) -> pd.DataFrame:
    """
    Load data from JSONL files and optionally subsample for visualization.

    Returns:
        DataFrame with columns: kind, digits, rationale, rep, prompt
    """
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL result files found under {results_dir}")

    print(f"Found {len(jsonl_files)} JSONL files")
    df = create_big_df(jsonl_files)
    if df.empty:
        raise ValueError(f"No rows loaded from JSONL files under {results_dir}")

    print(f"Loaded {len(df)} total rows")

    # Filter by models/seeds if specified
    if models:
        df = df[df["model"].isin(models)]
        print(f"Filtered to {len(df)} rows for models: {models}")
    if seeds:
        df = df[df["seed"].isin(seeds)]
        print(f"Filtered to {len(df)} rows for seeds: {seeds}")

    # Convert to rationale format (creates nl/code rows)
    df = _convert_results_df(df, filter_algo=True)
    print(f"Converted to {len(df)} rationale rows")

    # Filter to selected kinds preset
    kinds_set = KINDS_PRESETS.get(kinds_preset, EXTENDED_KINDS)
    df = filter_by_kinds(df, kinds_set)
    print(f"Filtered to {len(df)} rows with {kinds_preset} kinds ({len(kinds_set)} types)")

    # Stratified subsample if needed
    if max_samples_per_kind > 0:
        subsampled = []
        for (kind, rep), group in df.groupby(["kind", "rep"]):
            if len(group) > max_samples_per_kind:
                subsampled.append(group.sample(n=max_samples_per_kind, random_state=random_seed))
            else:
                subsampled.append(group)
        df = pd.concat(subsampled, ignore_index=True)
        print(f"Subsampled to {len(df)} rows (max {max_samples_per_kind} per kind per rep)")

    return df


def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute sentence embeddings using SentenceTransformers.

    Returns:
        numpy array of shape (n_samples, embedding_dim)
    """
    print(f"Computing embeddings with {model_name}...")
    featurizer = SentenceTransformersFeaturizer(model_name, device=device)
    embeddings = featurizer.transform(texts)
    print(f"Computed embeddings: {embeddings.shape}")
    return embeddings


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Reduce embedding dimensions using PCA or t-SNE.

    Returns:
        numpy array of shape (n_samples, n_components)
    """
    print(f"Reducing dimensions with {method.upper()}...")
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=random_seed)
    elif method.lower() == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_seed,
            init="pca",
            learning_rate="auto",
        )
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    coords = reducer.fit_transform(embeddings)
    print(f"Reduced to {coords.shape}")
    return coords


def create_scatter(
    ax: plt.Axes,
    coords: np.ndarray,
    kinds: List[str],
    color_map: Dict[str, tuple],
    marker_map: Dict[str, str],
    title: str,
    point_size: int = 15,
    alpha: float = 0.6,
    draw_circles: bool = False,
) -> None:
    """Create a scatter plot visualization on the given axes."""
    # Draw cluster ellipses first (behind points)
    if draw_circles:
        draw_cluster_ellipses(ax, coords, kinds, color_map, n_std=1.8, alpha=0.2)

    # Plot each kind separately to handle different markers
    unique_kinds = sorted(set(kinds))
    for kind in unique_kinds:
        mask = np.array([k == kind for k in kinds])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color_map.get(kind, (0.5, 0.5, 0.5))],
            marker=marker_map.get(kind, "o"),
            s=point_size,
            alpha=alpha,
            label=kind,
            edgecolors="none",
        )

    # Title inside plot area to save space
    ax.text(0.5, 0.98, title, transform=ax.transAxes, fontsize=7, fontweight="bold",
            ha="center", va="top", bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8, edgecolor="none"))
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.002)
    # No border
    for spine in ax.spines.values():
        spine.set_visible(False)


def create_legend(
    fig: plt.Figure,
    color_map: Dict[str, tuple],
    marker_map: Dict[str, str],
    fontsize: float = 3,
    ncol: int = 44,
) -> None:
    """Create single-row ultra-compact legend."""
    handles = []
    for kind in sorted(color_map.keys()):
        # Just use colored marker, minimal text
        abbrev = kind[:4] if len(kind) > 4 else kind
        handle = Line2D(
            [0], [0],
            marker=marker_map.get(kind, "o"),
            color="w",
            markerfacecolor=color_map[kind],
            markersize=2,
            label=abbrev,
            linestyle="None",
        )
        handles.append(handle)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        bbox_to_anchor=(0.5, 0.001),
        handletextpad=0.01,
        columnspacing=0.08,
        labelspacing=0.02,
        borderpad=0,
    )


def generate_combined_figure(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Generate a 2x2 combined figure (NL/Code x PCA/t-SNE)."""
    # Build color and marker maps
    unique_kinds = sorted(df["kind"].unique())
    color_map, marker_map = build_colormap(unique_kinds)
    print(f"Found {len(unique_kinds)} unique kinds")

    # Determine which reps to plot
    if args.rep == "both":
        reps = ["nl", "code"]
    else:
        reps = [args.rep]

    # Build title suffix based on filtering options
    title_suffix = ""
    if args.filter_algo_names:
        title_suffix += " [no algo names]"
    if args.remove_comments:
        title_suffix += " [no comments]"

    # Create figure with maximum density - whitespace ~10%
    nrows = len(reps)
    # Ultra-compact: plots touch each other, minimal legend
    fig_width = 8
    fig_height = 5.8 if nrows == 2 else 3
    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.005, "hspace": 0.05, "left": 0.005, "right": 0.995, "top": 0.97, "bottom": 0.08},
    )
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # Title integrated into top margin
    fig.suptitle(f"CoT Embeddings{title_suffix}", fontsize=8, fontweight="bold", y=0.995)

    # Track silhouette scores
    silhouette_scores = {}

    for row_idx, rep in enumerate(reps):
        # Filter data by representation
        rep_df = filter_by_rep(df, rep)
        texts = rep_df["rationale"].tolist()
        kinds = rep_df["kind"].tolist()

        # Apply text filtering if requested
        if args.filter_algo_names:
            print(f"Filtering algorithm names from {rep} texts...")
            texts = [filter_algorithm_names(t, EXTENDED_KINDS) for t in texts]

        if args.remove_comments and rep == "code":
            print(f"Removing comments from {rep} texts...")
            texts = [remove_code_comments(t) for t in texts]

        # Compute embeddings
        embeddings = compute_embeddings(texts, args.embed_model, args.device)

        # Compute silhouette score on embeddings
        sil_emb = compute_silhouette(embeddings, kinds)
        silhouette_scores[f"{rep}_embedding"] = sil_emb
        print(f"  Silhouette (embedding, {rep}): {sil_emb:.4f}")

        # PCA
        pca_coords = reduce_dimensions(embeddings, method="pca", random_seed=args.seed)
        sil_pca = compute_silhouette(pca_coords, kinds)
        silhouette_scores[f"{rep}_pca"] = sil_pca
        print(f"  Silhouette (PCA, {rep}): {sil_pca:.4f}")

        create_scatter(
            axes[row_idx, 0],
            pca_coords,
            kinds,
            color_map,
            marker_map,
            f"{rep.upper()} - PCA (sil={sil_pca:.3f})",
            args.point_size,
            args.alpha,
            args.draw_circles,
        )

        # t-SNE
        tsne_coords = reduce_dimensions(
            embeddings,
            method="tsne",
            perplexity=args.perplexity,
            max_iter=args.tsne_iter,
            random_seed=args.seed,
        )
        sil_tsne = compute_silhouette(tsne_coords, kinds)
        silhouette_scores[f"{rep}_tsne"] = sil_tsne
        print(f"  Silhouette (t-SNE, {rep}): {sil_tsne:.4f}")

        create_scatter(
            axes[row_idx, 1],
            tsne_coords,
            kinds,
            color_map,
            marker_map,
            f"{rep.upper()} - t-SNE (sil={sil_tsne:.3f})",
            args.point_size,
            args.alpha,
            args.draw_circles,
        )

    # Print summary of silhouette scores
    print("\n" + "=" * 40)
    print("Silhouette Score Summary:")
    for key, val in sorted(silhouette_scores.items()):
        print(f"  {key}: {val:.4f}")
    print("=" * 40)

    # Legend (gridspec already handles tight layout)
    create_legend(fig, color_map, marker_map)

    # Build output filename
    suffix = ""
    if args.filter_algo_names:
        suffix += "_noalgo"
    if args.remove_comments:
        suffix += "_nocomments"

    # Save figure with minimal padding
    output_path = Path(args.output_dir) / f"{args.output_prefix}{suffix}_combined.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved figure to: {output_path}")
    plt.close(fig)


def load_data_with_correctness(
    results_dir: str,
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    max_samples_per_kind: int = 500,
    random_seed: int = 42,
    kinds_preset: str = "extended",
) -> pd.DataFrame:
    """Load data with correctness labels for task-relevant analysis."""
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL result files found under {results_dir}")

    print(f"Found {len(jsonl_files)} JSONL files")
    df = create_big_df(jsonl_files)
    if df.empty:
        raise ValueError(f"No rows loaded from JSONL files under {results_dir}")

    print(f"Loaded {len(df)} total rows")

    # Filter by models/seeds
    if models:
        df = df[df["model"].isin(models)]
    if seeds:
        df = df[df["seed"].isin(seeds)]

    # Filter by kinds
    kinds_set = KINDS_PRESETS.get(kinds_preset, EXTENDED_KINDS)
    df = df[df["kind"].isin(kinds_set)]
    print(f"Filtered to {len(df)} rows with {kinds_preset} kinds")

    # Build rows with correctness labels
    rows = []
    for _, row in df.iterrows():
        nl_text = str(row.get("nl_reasoning", "") or "").strip()
        code_text = str(row.get("sim_code", "") or "").strip()

        base = {
            "kind": row["kind"],
            "digits": int(row["digit"]),
        }

        if nl_text:
            rows.append({
                **base,
                "rationale": nl_text,
                "rep": "nl",
                "correct": bool(row.get("nl_correct", False)),
            })
        if code_text:
            rows.append({
                **base,
                "rationale": code_text,
                "rep": "code",
                "correct": bool(row.get("sim_correct", False)),
            })

    result_df = pd.DataFrame(rows)
    print(f"Created {len(result_df)} rationale rows with correctness labels")

    # Subsample if needed
    if max_samples_per_kind > 0:
        subsampled = []
        for (kind, rep), group in result_df.groupby(["kind", "rep"]):
            if len(group) > max_samples_per_kind:
                subsampled.append(group.sample(n=max_samples_per_kind, random_state=random_seed))
            else:
                subsampled.append(group)
        result_df = pd.concat(subsampled, ignore_index=True)
        print(f"Subsampled to {len(result_df)} rows")

    return result_df


def compute_lda_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, LinearDiscriminantAnalysis]:
    """
    Project embeddings onto LDA direction (task-relevant) and residual PC.

    Returns:
        lda_proj: 1D projection onto discriminant direction
        residual_pc: 1D projection onto top PC of residual
        lda_model: fitted LDA model
    """
    # LDA projection
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda_proj = lda.fit_transform(embeddings, labels)

    # Residual after removing LDA direction
    lda_direction = lda.scalings_.flatten()
    lda_direction = lda_direction / np.linalg.norm(lda_direction)

    # Project out LDA direction
    proj_onto_lda = embeddings @ lda_direction.reshape(-1, 1) @ lda_direction.reshape(1, -1)
    residual = embeddings - proj_onto_lda

    # Top PC of residual
    pca = PCA(n_components=1)
    residual_pc = pca.fit_transform(residual)

    return lda_proj.flatten(), residual_pc.flatten(), lda


def generate_lda_task_figure(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Generate LDA vs residual PC figure for task-relevant clustering analysis."""
    print("\n" + "=" * 60)
    print("Task-Relevant Clustering Analysis (LDA vs Residual PC)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    results = {}

    for ax, rep in zip(axes, ["nl", "code"]):
        # Filter by representation
        rep_df = df[df["rep"] == rep].copy()
        if len(rep_df) < 10:
            print(f"Skipping {rep}: not enough samples ({len(rep_df)})")
            continue

        texts = rep_df["rationale"].tolist()
        labels = rep_df["correct"].astype(int).values

        # Check class balance
        n_correct = labels.sum()
        n_incorrect = len(labels) - n_correct
        print(f"\n{rep.upper()}: {len(labels)} samples ({n_correct} correct, {n_incorrect} incorrect)")

        if n_correct < 5 or n_incorrect < 5:
            print(f"Skipping {rep}: insufficient class balance")
            continue

        # Compute embeddings
        embeddings = compute_embeddings(texts, args.embed_model, args.device)

        # LDA projection
        lda_proj, residual_pc, lda_model = compute_lda_projection(embeddings, labels)

        # Compute metrics
        # Silhouette in LDA direction (1D)
        sil_lda = silhouette_score(lda_proj.reshape(-1, 1), labels)

        # Silhouette in full embedding space
        sil_full = silhouette_score(embeddings, labels)

        # Between-class distance in LDA direction
        lda_correct = lda_proj[labels == 1]
        lda_incorrect = lda_proj[labels == 0]
        between_class_dist = abs(lda_correct.mean() - lda_incorrect.mean())
        within_class_std = (lda_correct.std() + lda_incorrect.std()) / 2

        # Fisher ratio
        fisher_ratio = between_class_dist / (within_class_std + 1e-10)

        results[rep] = {
            "sil_lda": sil_lda,
            "sil_full": sil_full,
            "fisher_ratio": fisher_ratio,
            "between_class_dist": between_class_dist,
        }

        print(f"  Silhouette (LDA direction): {sil_lda:.4f}")
        print(f"  Silhouette (full space): {sil_full:.4f}")
        print(f"  Fisher ratio: {fisher_ratio:.4f}")

        # Plot
        colors = ["#e74c3c" if c == 0 else "#2ecc71" for c in labels]
        ax.scatter(lda_proj, residual_pc, c=colors, alpha=0.5, s=20, edgecolors="none")

        # Add decision boundary
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Decision boundary")

        # Add class means
        ax.axvline(x=lda_correct.mean(), color="#2ecc71", linestyle=":", linewidth=2, alpha=0.8)
        ax.axvline(x=lda_incorrect.mean(), color="#e74c3c", linestyle=":", linewidth=2, alpha=0.8)

        ax.set_xlabel("LDA Direction (Task-Relevant)", fontsize=11)
        ax.set_ylabel("Top Residual PC (Task-Irrelevant)", fontsize=11)
        ax.set_title(
            f"{rep.upper()}\n"
            f"Sil(LDA)={sil_lda:.3f}, Sil(Full)={sil_full:.3f}, Fisher={fisher_ratio:.2f}",
            fontsize=10,
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="Correct"),
            Patch(facecolor="#e74c3c", label="Incorrect"),
            Line2D([0], [0], color="black", linestyle="--", label="Decision boundary"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary Comparison:")
    if "nl" in results and "code" in results:
        print(f"  NL   - Sil(LDA): {results['nl']['sil_lda']:.4f}, Sil(Full): {results['nl']['sil_full']:.4f}, Fisher: {results['nl']['fisher_ratio']:.2f}")
        print(f"  Code - Sil(LDA): {results['code']['sil_lda']:.4f}, Sil(Full): {results['code']['sil_full']:.4f}, Fisher: {results['code']['fisher_ratio']:.2f}")

        # Key insight
        if results["nl"]["sil_full"] > results["code"]["sil_full"] and results["code"]["sil_lda"] > results["nl"]["sil_lda"]:
            print("\n  ⚠ CONFIRMED: NL has higher full-space silhouette but lower LDA silhouette!")
            print("    → NL clusters well in task-IRRELEVANT dimensions")
            print("    → Code clusters better in task-RELEVANT dimension")
    print("=" * 60)

    plt.tight_layout()
    output_path = Path(args.output_dir) / f"{args.output_prefix}_lda_task.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved LDA task figure to: {output_path}")
    plt.close(fig)


def compute_multiclass_lda_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, LinearDiscriminantAnalysis]:
    """
    Project embeddings onto LDA directions for multiclass labels.

    Returns:
        lda_proj: 2D projection onto top discriminant directions
        lda_model: fitted LDA model
    """
    n_classes = len(np.unique(labels))
    # LDA can have at most min(n_features, n_classes - 1) components
    max_components = min(embeddings.shape[1], n_classes - 1, n_components)

    lda = LinearDiscriminantAnalysis(n_components=max_components)
    lda_proj = lda.fit_transform(embeddings, labels)

    # Pad with zeros if we couldn't get enough components
    if lda_proj.shape[1] < n_components:
        padding = np.zeros((lda_proj.shape[0], n_components - lda_proj.shape[1]))
        lda_proj = np.hstack([lda_proj, padding])

    return lda_proj, lda


def generate_dual_clustering_figure(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """
    Generate figure comparing LDA clustering by KIND vs LDA clustering by CORRECTNESS.

    Both rows use LDA projection to show task-relevant directions.
    """
    print("\n" + "=" * 60)
    print("Dual LDA Analysis: Kind vs Correctness")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    results = {"nl": {}, "code": {}}

    for col, rep in enumerate(["nl", "code"]):
        rep_df = df[df["rep"] == rep].copy()
        if len(rep_df) < 10:
            print(f"Skipping {rep}: not enough samples")
            continue

        texts = rep_df["rationale"].tolist()
        kinds = rep_df["kind"].values
        correct = rep_df["correct"].astype(int).values

        print(f"\n{rep.upper()}: {len(texts)} samples")

        # Compute embeddings
        embeddings = compute_embeddings(texts, args.embed_model, args.device)

        # Encode kinds as integers
        le = LabelEncoder()
        kind_labels = le.fit_transform(kinds)

        # === Row 0: LDA by KIND ===
        lda_kind_proj, lda_kind_model = compute_multiclass_lda_projection(embeddings, kind_labels, n_components=2)

        # Silhouette in LDA space for kind
        sil_kind_lda = silhouette_score(lda_kind_proj, kind_labels)
        sil_kind_full = silhouette_score(embeddings, kind_labels)

        ax_kind = axes[0, col]
        unique_kinds = sorted(set(kinds))
        color_map, marker_map = build_colormap(unique_kinds)

        for kind in unique_kinds:
            mask = kinds == kind
            ax_kind.scatter(
                lda_kind_proj[mask, 0], lda_kind_proj[mask, 1],
                c=[color_map.get(kind, (0.5, 0.5, 0.5))],
                marker=marker_map.get(kind, "o"),
                s=15, alpha=0.5, edgecolors="none"
            )

        ax_kind.set_xlabel("LDA1 (Kind)", fontsize=10)
        ax_kind.set_ylabel("LDA2 (Kind)", fontsize=10)
        ax_kind.set_title(
            f"{rep.upper()} - LDA by KIND\n"
            f"Sil(Full)={sil_kind_full:.3f}, Sil(LDA)={sil_kind_lda:.3f}",
            fontsize=11
        )

        # === Row 1: LDA by CORRECTNESS ===
        lda_correct_proj, residual_pc, _ = compute_lda_projection(embeddings, correct)
        sil_correct_lda = silhouette_score(lda_correct_proj.reshape(-1, 1), correct)
        sil_correct_full = silhouette_score(embeddings, correct)

        ax_correct = axes[1, col]
        colors = ["#e74c3c" if c == 0 else "#2ecc71" for c in correct]
        ax_correct.scatter(lda_correct_proj, residual_pc, c=colors, alpha=0.5, s=15, edgecolors="none")
        ax_correct.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

        ax_correct.set_xlabel("LDA Direction (Correctness)", fontsize=10)
        ax_correct.set_ylabel("Residual PC", fontsize=10)
        ax_correct.set_title(
            f"{rep.upper()} - LDA by CORRECTNESS\n"
            f"Sil(Full)={sil_correct_full:.3f}, Sil(LDA)={sil_correct_lda:.3f}",
            fontsize=11
        )

        # === Train/Test Logistic Regression ===
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, correct, test_size=0.2, random_state=42, stratify=correct
        )

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        generalization_gap = train_acc - test_acc

        # Store results
        results[rep] = {
            "sil_kind_full": sil_kind_full,
            "sil_kind_lda": sil_kind_lda,
            "sil_correct_full": sil_correct_full,
            "sil_correct_lda": sil_correct_lda,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "gen_gap": generalization_gap,
        }

        print(f"  KIND - Sil(Full): {sil_kind_full:.4f}, Sil(LDA): {sil_kind_lda:.4f}")
        print(f"  CORRECT - Sil(Full): {sil_correct_full:.4f}, Sil(LDA): {sil_correct_lda:.4f}")
        print(f"  LOGREG - Train: {train_acc:.4f}, Test: {test_acc:.4f}, Gap: {generalization_gap:.4f}")

    # Add legend for correctness
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Correct"),
        Patch(facecolor="#e74c3c", label="Incorrect"),
    ]
    axes[1, 1].legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Task-Relevant Clustering (LDA directions):")
    if "nl" in results and "code" in results:
        nl, code = results["nl"], results["code"]
        print(f"\n  Clustering by KIND (LDA space):")
        print(f"    NL:   {nl['sil_kind_lda']:.4f}")
        print(f"    Code: {code['sil_kind_lda']:.4f}")
        print(f"    → {'NL' if nl['sil_kind_lda'] > code['sil_kind_lda'] else 'Code'} clusters better by kind in task-relevant directions")

        print(f"\n  Clustering by CORRECTNESS (LDA space):")
        print(f"    NL:   {nl['sil_correct_lda']:.4f}")
        print(f"    Code: {code['sil_correct_lda']:.4f}")
        print(f"    → {'NL' if nl['sil_correct_lda'] > code['sil_correct_lda'] else 'Code'} clusters better by correctness in task-relevant directions")

        print(f"\n  Logistic Regression (Train/Test):")
        print(f"    NL:   Train={nl['train_acc']:.4f}, Test={nl['test_acc']:.4f}, Gap={nl['gen_gap']:.4f}")
        print(f"    Code: Train={code['train_acc']:.4f}, Test={code['test_acc']:.4f}, Gap={code['gen_gap']:.4f}")
        print(f"    → {'NL' if nl['test_acc'] > code['test_acc'] else 'Code'} has better TEST accuracy")

        # Check for the key finding
        if nl['sil_correct_lda'] > code['sil_correct_lda'] and code['test_acc'] > nl['test_acc']:
            print("\n  ✓ KEY FINDING CONFIRMED:")
            print("    NL has better in-sample LDA separation")
            print("    BUT Code has better out-of-sample prediction")
            print("    → NL OVERFITS, Code GENERALIZES better")
        elif nl['sil_correct_lda'] > code['sil_correct_lda'] and nl['test_acc'] > code['test_acc']:
            print("\n  NL dominates both metrics")
        elif code['test_acc'] > nl['test_acc']:
            print(f"\n  Code wins on test accuracy ({code['test_acc']:.4f} vs {nl['test_acc']:.4f})")
    print("=" * 60)

    plt.tight_layout()
    output_path = Path(args.output_dir) / f"{args.output_prefix}_dual_clustering.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved dual clustering figure to: {output_path}")
    plt.close(fig)


def load_data_with_all_reps(
    results_dir: str,
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    max_samples_per_kind: int = 500,
    random_seed: int = 42,
    kinds_preset: str = "fg",
) -> pd.DataFrame:
    """Load data with all representation types (nl, code, sim_reasoning)."""
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL result files found under {results_dir}")

    print(f"Found {len(jsonl_files)} JSONL files")
    df = create_big_df(jsonl_files)
    if df.empty:
        raise ValueError(f"No rows loaded from JSONL files under {results_dir}")

    print(f"Loaded {len(df)} total rows")

    # Filter by models/seeds
    if models:
        df = df[df["model"].isin(models)]
    if seeds:
        df = df[df["seed"].isin(seeds)]

    # Filter by kinds
    kinds_set = KINDS_PRESETS.get(kinds_preset, FG_KINDS)
    df = df[df["kind"].isin(kinds_set)]
    print(f"Filtered to {len(df)} rows with {kinds_preset} kinds")

    # Build rows with all representations
    rows = []
    for _, row in df.iterrows():
        nl_text = str(row.get("nl_reasoning", "") or "").strip()
        code_text = str(row.get("sim_code", "") or "").strip()
        sim_text = str(row.get("sim_reasoning", "") or "").strip()

        base = {
            "kind": row["kind"],
            "digits": int(row["digit"]),
        }

        if nl_text:
            rows.append({**base, "rationale": nl_text, "rep": "nl"})
        if code_text:
            rows.append({**base, "rationale": code_text, "rep": "code"})
        if sim_text:
            rows.append({**base, "rationale": sim_text, "rep": "sim"})

    result_df = pd.DataFrame(rows)
    print(f"Created {len(result_df)} rationale rows")

    # Subsample if needed
    if max_samples_per_kind > 0:
        subsampled = []
        for (kind, rep, digits), group in result_df.groupby(["kind", "rep", "digits"]):
            if len(group) > max_samples_per_kind:
                subsampled.append(group.sample(n=max_samples_per_kind, random_state=random_seed))
            else:
                subsampled.append(group)
        result_df = pd.concat(subsampled, ignore_index=True)
        print(f"Subsampled to {len(result_df)} rows")

    return result_df


def generate_silhouette_vs_hardness_figure(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Generate silhouette score vs task hardness (digits) for NL, code, sim."""
    print("\n" + "=" * 60)
    print("Silhouette vs Task Hardness Analysis")
    print("=" * 60)

    reps = ["nl", "code", "sim"]
    rep_labels = {"nl": "NL CoT", "code": "Code", "sim": "Simulation"}
    rep_colors = {"nl": "#3498db", "code": "#e74c3c", "sim": "#2ecc71"}

    # Get unique digits levels
    all_digits = sorted(df["digits"].unique())
    print(f"Digits levels: {all_digits}")

    results = {rep: {"digits": [], "silhouette": [], "n_samples": []} for rep in reps}

    for rep in reps:
        rep_df = df[df["rep"] == rep]
        if len(rep_df) == 0:
            print(f"No data for {rep}")
            continue

        print(f"\n{rep.upper()}:")
        for digits in all_digits:
            subset = rep_df[rep_df["digits"] == digits]
            if len(subset) < 50:  # Need enough samples
                print(f"  digits={digits}: skipping (only {len(subset)} samples)")
                continue

            texts = subset["rationale"].tolist()
            kinds = subset["kind"].values

            # Need at least 2 unique kinds
            if len(np.unique(kinds)) < 2:
                print(f"  digits={digits}: skipping (only {len(np.unique(kinds))} kinds)")
                continue

            # Compute embeddings
            embeddings = compute_embeddings(texts, args.embed_model, args.device)

            # Encode kinds
            le = LabelEncoder()
            kind_labels = le.fit_transform(kinds)

            # Compute silhouette
            sil = silhouette_score(embeddings, kind_labels)

            results[rep]["digits"].append(digits)
            results[rep]["silhouette"].append(sil)
            results[rep]["n_samples"].append(len(subset))

            print(f"  digits={digits}: sil={sil:.4f} (n={len(subset)})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for rep in reps:
        if len(results[rep]["digits"]) > 0:
            ax.plot(
                results[rep]["digits"],
                results[rep]["silhouette"],
                marker="o",
                linewidth=2,
                markersize=8,
                label=rep_labels[rep],
                color=rep_colors[rep],
            )

    ax.set_xlabel("Task Hardness (Digits)", fontsize=12)
    ax.set_ylabel("Silhouette Score (by Algorithm Kind)", fontsize=12)
    ax.set_title("Clustering Quality vs Task Hardness\n(Fine-Grained Tasks)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to integer digits
    ax.set_xticks(all_digits)

    plt.tight_layout()
    output_path = Path(args.output_dir) / f"{args.output_prefix}_sil_vs_hardness.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")
    plt.close(fig)

    # Summary table
    print("\n" + "=" * 60)
    print("Summary Table:")
    print(f"{'Digits':<8}", end="")
    for rep in reps:
        print(f"{rep_labels[rep]:<15}", end="")
    print()
    print("-" * 60)

    for i, digits in enumerate(all_digits):
        print(f"{digits:<8}", end="")
        for rep in reps:
            if digits in results[rep]["digits"]:
                idx = results[rep]["digits"].index(digits)
                sil = results[rep]["silhouette"][idx]
                print(f"{sil:<15.4f}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    print("=" * 60)


def main() -> None:
    """Main entry point for embedding visualization."""
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("CoT Embedding Visualization")
    print("=" * 60)
    print(f"Embedding model: {args.embed_model}")
    print(f"Representation: {args.rep}")
    print(f"Kinds preset: {args.kinds}")
    print(f"Max samples/kind: {args.max_samples_per_kind}")
    print(f"Perplexity: {args.perplexity}")
    print(f"Draw circles: {args.draw_circles}")
    print("=" * 60)

    # LDA task-relevant analysis mode
    if args.lda_task:
        df = load_data_with_correctness(
            args.results_dir,
            args.models,
            args.seeds,
            args.max_samples_per_kind,
            args.seed,
            args.kinds,
        )
        generate_lda_task_figure(df, args)
        print("Done!")
        return

    # Dual clustering analysis (KIND vs CORRECTNESS)
    if args.dual_clustering:
        df = load_data_with_correctness(
            args.results_dir,
            args.models,
            args.seeds,
            args.max_samples_per_kind,
            args.seed,
            args.kinds,
        )
        generate_dual_clustering_figure(df, args)
        print("Done!")
        return

    # Silhouette vs hardness analysis
    if args.sil_vs_hardness:
        df = load_data_with_all_reps(
            args.results_dir,
            args.models,
            args.seeds,
            args.max_samples_per_kind,
            args.seed,
            args.kinds,
        )
        generate_silhouette_vs_hardness_figure(df, args)
        print("Done!")
        return

    # Load and prepare data
    df = load_and_subsample(
        args.results_dir,
        args.models,
        args.seeds,
        args.max_samples_per_kind,
        args.seed,
        args.kinds,
    )

    # Generate visualization
    generate_combined_figure(df, args)

    print("Done!")


if __name__ == "__main__":
    main()
