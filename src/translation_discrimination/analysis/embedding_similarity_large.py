#!/usr/bin/env python3
"""
Embedding cosine similarity analysis using OpenAI text-embedding-3-large.

Pools task instances from all judge files (native NL traces come from a mix of
source models in recorded benchmark artifacts; translated traces from GPT-4o).

Computes three comparisons:
  - Native ↔ Translated  (same task instance)
  - Native ↔ Native      (across different instances)
  - Translated ↔ Translated (across different instances)

Usage:
    uv run python -m src.translation_discrimination.analysis.embedding_similarity_large
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import httpx
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy import stats as sp_stats
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR

BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-large"

# All judge files — each has different task instances from mixed source models
JUDGE_FILES = [
    "opus4_judge_gpt4o_translator_n1000.json",
    "gemini3pro_judge_gpt4o_translator_n1000.json",
    "grok41fast_judge_gpt4o_translator_n1000.json",
]


def load_all_pairs() -> List[Dict]:
    """Pool matched (native, translated) pairs from all judge files.

    Uses (question_text, trace_text) as a dedup key so the same instance
    appearing in multiple files is only counted once.
    """
    seen = set()
    pairs = []

    for filename in JUDGE_FILES:
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found, skipping")
            continue

        with open(filepath) as f:
            data = json.load(f)

        trials = data.get("trials", data.get("results", {}).get("trials", []))

        native_by_q: Dict[str, Dict] = {}
        translated_by_q: Dict[str, Dict] = {}

        for trial in trials:
            if trial.get("control_type"):
                continue
            trace = trial.get("trace", "")
            if not trace or len(trace) < 50:
                continue
            q = trial["question"]
            entry = {"trace": trace[:2000], "kind": trial["kind"]}

            if trial["true_label"] == 0:
                native_by_q[q] = entry
            elif trial["true_label"] == 1:
                translated_by_q[q] = entry

        for q in sorted(set(native_by_q) & set(translated_by_q)):
            # Dedup by (question, native_trace_prefix)
            key = (q, native_by_q[q]["trace"][:200])
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "question": q,
                "kind": native_by_q[q]["kind"],
                "native": native_by_q[q]["trace"],
                "translated": translated_by_q[q]["trace"],
            })

    return pairs


def get_embeddings(texts: List[str], batch_size: int = 50) -> np.ndarray:
    """Get embeddings via OpenRouter using text-embedding-3-large."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = httpx.post(
            f"{BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        batch_emb = [item["embedding"] for item in result["data"]]
        all_embeddings.extend(batch_emb)
        print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(all_embeddings)


def compute_stats(values: np.ndarray) -> Dict:
    """Mean, std, SEM, and 95% CI for an array of values."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    sem = float(sp_stats.sem(values))
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - 1.96 * sem,
        "ci_high": mean + 1.96 * sem,
        "n": len(values),
    }


def plot_three_bars(stats: Dict, output_path: Path):
    """3-bar chart: Native↔Native, Translated↔Translated, Native↔Translated."""
    labels = ["Native ↔ Native\n(same task, diff instance)", "Translated ↔\nTranslated\n(same task, diff instance)", "Native ↔\nTranslated\n(same instance)"]
    keys = ["native_native", "translated_translated", "native_translated"]
    means = [stats[k]["mean"] for k in keys]
    ci_lows = [stats[k]["ci_low"] for k in keys]
    ci_highs = [stats[k]["ci_high"] for k in keys]
    ns = [stats[k]["n"] for k in keys]

    err_low = np.abs(np.array(means) - np.array(ci_lows))
    err_high = np.abs(np.array(ci_highs) - np.array(means))

    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(labels))
    width = 0.5

    bars = ax.bar(
        x, means, width,
        yerr=[err_low, err_high],
        color=colors, edgecolor="black", linewidth=1,
        capsize=5, error_kw={"linewidth": 1.5},
    )

    for i, (bar, m, n) in enumerate(zip(bars, means, ns)):
        label_text = f"{m:.3f}\n(n={n:,})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err_high[i] + 0.005,
            label_text,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Mean Cosine Similarity", fontsize=13, fontweight="bold")
    ax.set_title(
        "Embedding Cosine Similarity: Native NL vs GPT-4o Translated\n"
        "(text-embedding-3-large · mixed source benchmark models)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, max(means) + 0.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved figure → {output_path.with_suffix('.png')} / .pdf")


def main():
    print("=" * 60)
    print("Paired Embedding Analysis — text-embedding-3-large")
    print("(pooled across all judge files)")
    print("=" * 60)

    # 1. Load all matched pairs
    print("\nLoading matched pairs from all judge files...")
    pairs = load_all_pairs()
    print(f"Total unique pairs: {len(pairs)}")

    if len(pairs) < 20:
        print("Not enough pairs, aborting.")
        return

    # 2. Embed all traces (with disk cache to avoid redundant API calls)
    native_texts = [p["native"] for p in pairs]
    translated_texts = [p["translated"] for p in pairs]

    cache_path = OUTPUT_DIR / "_embedding_cache_large.npz"
    if cache_path.exists():
        print(f"\nLoading cached embeddings from {cache_path}")
        cached = np.load(cache_path)
        native_emb = cached["native"]
        translated_emb = cached["translated"]
        if len(native_emb) == len(pairs) and len(translated_emb) == len(pairs):
            print(f"  Cache hit: {len(native_emb)} native, {len(translated_emb)} translated")
        else:
            print(f"  Cache size mismatch ({len(native_emb)} vs {len(pairs)}), re-embedding...")
            native_emb = None  # type: ignore
    else:
        native_emb = None  # type: ignore

    if native_emb is None:
        print("\nEmbedding native traces...")
        native_emb = get_embeddings(native_texts)
        print("Embedding translated traces...")
        translated_emb = get_embeddings(translated_texts)
        np.savez(cache_path, native=native_emb, translated=translated_emb)
        print(f"Saved embedding cache → {cache_path}")

    # 3. Compute the three comparisons — all conditioned on same task (kind)

    kinds = np.array([p["kind"] for p in pairs])
    unique_kinds = sorted(set(kinds))
    print(f"\nComputing similarities conditioned on same task kind ({len(unique_kinds)} kinds)...")

    # (a) Native ↔ Translated — same task instance
    same_instance_sims = np.array([
        cosine_similarity(native_emb[i : i + 1], translated_emb[i : i + 1])[0, 0]
        for i in range(len(pairs))
    ])
    nt_stats = compute_stats(same_instance_sims)

    # (b) Native ↔ Native — different instances of the SAME task kind
    nn_sims_list = []
    for kind in unique_kinds:
        idx = np.where(kinds == kind)[0]
        if len(idx) < 2:
            continue
        kind_matrix = cosine_similarity(native_emb[idx])
        upper = kind_matrix[np.triu_indices(len(kind_matrix), k=1)]
        nn_sims_list.append(upper)
    nn_all = np.concatenate(nn_sims_list)
    nn_stats = compute_stats(nn_all)

    # (c) Translated ↔ Translated — different instances of the SAME task kind
    tt_sims_list = []
    for kind in unique_kinds:
        idx = np.where(kinds == kind)[0]
        if len(idx) < 2:
            continue
        kind_matrix = cosine_similarity(translated_emb[idx])
        upper = kind_matrix[np.triu_indices(len(kind_matrix), k=1)]
        tt_sims_list.append(upper)
    tt_all = np.concatenate(tt_sims_list)
    tt_stats = compute_stats(tt_all)

    all_stats = {
        "native_native": nn_stats,
        "translated_translated": tt_stats,
        "native_translated": nt_stats,
    }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for label, key in [
        ("Native ↔ Native (same task, diff instance)", "native_native"),
        ("Translated ↔ Translated (same task, diff instance)", "translated_translated"),
        ("Native ↔ Translated (same instance)", "native_translated"),
    ]:
        s = all_stats[key]
        print(f"  {label}:")
        print(f"    Mean: {s['mean']:.4f} ± {s['std']:.4f}")
        print(f"    95% CI: [{s['ci_low']:.4f}, {s['ci_high']:.4f}]")
        print(f"    n: {s['n']:,}")

    # 4. Save stats
    stats_path = OUTPUT_DIR / "embedding_paired_stats_large.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved stats → {stats_path}")

    # 5. Plot
    plot_three_bars(all_stats, OUTPUT_DIR / "cosine_similarity_paired_large")


if __name__ == "__main__":
    main()
