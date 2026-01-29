#!/usr/bin/env python3
"""
Embedding analysis for discrimination experiment traces.

Embeds original (native NL) and translated traces using OpenAI embeddings,
calculates pairwise cosine similarity, and creates cluster visualizations.

Usage:
    uv run python src/exps_control_again/scripts/embedding_analysis.py
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import httpx
import numpy as np
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

load_dotenv()

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR

# OpenRouter config
BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Files to analyze
MODEL_FILES = {
    "Claude Opus 4": "opus4_judge_gpt4o_translator_n1000.json",
    "Gemini 2.5 Pro": "source_discrimination_20260117_145524.json",
    "Grok 4.1 Fast": "grok41fast_judge_gpt4o_translator_n1000.json",
}


def load_traces(filepath: Path, max_samples: int = 200) -> Tuple[List[str], List[str]]:
    """Load traces from JSON file, returning (native_traces, translated_traces)."""
    with open(filepath) as f:
        data = json.load(f)

    native_traces = []
    translated_traces = []

    # Handle both "trials" (newer format) and direct results
    trials = data.get("trials", data.get("results", {}).get("trials", []))

    # If trials not found at top level, check in results
    if not trials and "results" in data:
        # The trials might be stored differently - let's check the structure
        pass

    for trial in trials:
        # Skip control trials
        if trial.get("control_type"):
            continue

        trace = trial.get("trace", "")
        if not trace or len(trace) < 50:  # Skip very short traces
            continue

        true_label = trial.get("true_label")

        if true_label == 0:  # Native NL
            if len(native_traces) < max_samples:
                native_traces.append(trace[:2000])  # Truncate for embedding
        elif true_label == 1:  # Translated
            if len(translated_traces) < max_samples:
                translated_traces.append(trace[:2000])

        if len(native_traces) >= max_samples and len(translated_traces) >= max_samples:
            break

    return native_traces, translated_traces


def get_embeddings(texts: List[str], batch_size: int = 50) -> np.ndarray:
    """Get embeddings for texts using OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = httpx.post(
            f"{BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": batch,
            },
            timeout=60.0,
        )
        response.raise_for_status()

        result = response.json()
        batch_embeddings = [item["embedding"] for item in result["data"]]
        all_embeddings.extend(batch_embeddings)

        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return np.array(all_embeddings)


def calculate_similarity_stats(native_emb: np.ndarray, translated_emb: np.ndarray) -> dict:
    """Calculate cosine similarity statistics."""
    # Within-group similarities
    native_sim = cosine_similarity(native_emb)
    translated_sim = cosine_similarity(translated_emb)

    # Cross-group similarity
    cross_sim = cosine_similarity(native_emb, translated_emb)

    # Get upper triangle (excluding diagonal) for within-group
    native_upper = native_sim[np.triu_indices(len(native_sim), k=1)]
    translated_upper = translated_sim[np.triu_indices(len(translated_sim), k=1)]

    return {
        "native_within_mean": float(np.mean(native_upper)),
        "native_within_std": float(np.std(native_upper)),
        "translated_within_mean": float(np.mean(translated_upper)),
        "translated_within_std": float(np.std(translated_upper)),
        "cross_group_mean": float(np.mean(cross_sim)),
        "cross_group_std": float(np.std(cross_sim)),
        "n_native": len(native_emb),
        "n_translated": len(translated_emb),
    }


def plot_clusters(native_emb: np.ndarray, translated_emb: np.ndarray,
                  model_name: str, output_path: Path):
    """Create t-SNE cluster visualization."""
    # Combine embeddings
    all_emb = np.vstack([native_emb, translated_emb])
    labels = ["Native NL"] * len(native_emb) + ["Translated"] * len(translated_emb)

    # t-SNE reduction
    print(f"  Running t-SNE on {len(all_emb)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_emb) - 1))
    coords = tsne.fit_transform(all_emb)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    native_coords = coords[:len(native_emb)]
    translated_coords = coords[len(native_emb):]

    ax.scatter(native_coords[:, 0], native_coords[:, 1],
               c='#2ecc71', alpha=0.6, label='Native NL', s=50)
    ax.scatter(translated_coords[:, 0], translated_coords[:, 1],
               c='#e74c3c', alpha=0.6, label='Translated', s=50)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f'Embedding Clusters: {model_name}\n(Native NL vs Translated Traces)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_path}")


def main():
    print("=" * 60)
    print("Embedding Analysis for Discrimination Traces")
    print("=" * 60)

    all_stats = {}

    for model_name, filename in MODEL_FILES.items():
        print(f"\n[{model_name}]")
        filepath = RESULTS_DIR / filename

        if not filepath.exists():
            print(f"  WARNING: File not found: {filepath}")
            continue

        # Load traces
        print("  Loading traces...")
        native_traces, translated_traces = load_traces(filepath, max_samples=150)
        print(f"  Found {len(native_traces)} native, {len(translated_traces)} translated traces")

        if len(native_traces) < 10 or len(translated_traces) < 10:
            print("  WARNING: Not enough traces, skipping...")
            continue

        # Get embeddings
        print("  Getting embeddings for native traces...")
        native_emb = get_embeddings(native_traces)

        print("  Getting embeddings for translated traces...")
        translated_emb = get_embeddings(translated_traces)

        # Calculate similarity stats
        print("  Calculating similarity statistics...")
        stats = calculate_similarity_stats(native_emb, translated_emb)
        all_stats[model_name] = stats

        print(f"  Native within-group similarity: {stats['native_within_mean']:.4f} ± {stats['native_within_std']:.4f}")
        print(f"  Translated within-group similarity: {stats['translated_within_mean']:.4f} ± {stats['translated_within_std']:.4f}")
        print(f"  Cross-group similarity: {stats['cross_group_mean']:.4f} ± {stats['cross_group_std']:.4f}")

    # Plot for the first model (Claude Opus 4)
    print("\n" + "=" * 60)
    print("Creating cluster visualization for Claude Opus 4...")

    filepath = RESULTS_DIR / MODEL_FILES["Claude Opus 4"]
    native_traces, translated_traces = load_traces(filepath, max_samples=150)
    native_emb = get_embeddings(native_traces)
    translated_emb = get_embeddings(translated_traces)

    plot_clusters(native_emb, translated_emb, "Claude Opus 4",
                  OUTPUT_DIR / "embedding_clusters_opus4.png")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Average Pairwise Cosine Similarity")
    print("=" * 60)
    print(f"{'Model':<20} {'Native↔Native':<18} {'Trans↔Trans':<18} {'Native↔Trans':<18}")
    print("-" * 74)
    for model_name, stats in all_stats.items():
        print(f"{model_name:<20} {stats['native_within_mean']:.4f} ± {stats['native_within_std']:.4f}   "
              f"{stats['translated_within_mean']:.4f} ± {stats['translated_within_std']:.4f}   "
              f"{stats['cross_group_mean']:.4f} ± {stats['cross_group_std']:.4f}")

    # Save stats to JSON
    stats_path = OUTPUT_DIR / "embedding_similarity_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved stats to {stats_path}")


if __name__ == "__main__":
    main()
