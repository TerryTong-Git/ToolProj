#!/usr/bin/env python3
"""
Test whether linear separability varies by translator model.

Hypothesis: if high classifier accuracy is due to GPT-4o's stylistic uniformity,
different translators should produce different separability.

Pipeline:
  1. Load N samples (code + native NL) from recorded benchmark artifacts
  2. Translate code with each translator: GPT-4o, Gemini 2.0 Flash, Claude Sonnet 4.5
  3. Embed native + translated traces with text-embedding-3-large
  4. Fit logistic regression (5-fold CV) per translator
  5. Report AUC, accuracy, F1; plot embedding space with decision boundary

Usage:
    uv run python -m src.translation_discrimination.experiments.translator_separability
"""

import asyncio
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import httpx
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tqdm.asyncio import tqdm_asyncio

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_RESULTS_DIR

load_dotenv()

# --- Config ---
PERF_RESULTS_DIR = LEGACY_BENCHMARK_RESULTS_DIR
OUTPUT_DIR = Path(__file__).parent.parent / "results"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-large"

TRANSLATORS = {
    "GPT-4o": "openai/gpt-4o",
    "Gemini 2.0\nFlash": "google/gemini-2.0-flash-001",
    "Claude\nSonnet 4.5": "anthropic/claude-sonnet-4.5",
}

N_SAMPLES = 300  # per translator (same set for all)
CONCURRENCY = 20
SEED = 42


# --- Data loading ---


def load_samples(max_samples: int = N_SAMPLES) -> List[Dict]:
    """Load samples with code and native NL from recorded benchmark artifacts."""
    random.seed(SEED)
    samples_by_kind: Dict[str, list] = defaultdict(list)

    for jsonl_path in PERF_RESULTS_DIR.glob("**/res.jsonl"):
        model_name = jsonl_path.parent.parent.parent.name
        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sim_code = row.get("sim_code", "")
                nl_reasoning = row.get("nl_reasoning", "")
                question = row.get("question", "") or row.get("nl_question", "")
                kind = row.get("kind", "unknown")

                if not sim_code or not nl_reasoning or not question:
                    continue
                if len(nl_reasoning) < 100 or len(sim_code) < 50:
                    continue

                samples_by_kind[kind].append(
                    {
                        "kind": kind,
                        "question": question,
                        "native_nl": nl_reasoning,
                        "code": sim_code,
                        "source_model": model_name,
                    }
                )

    # Balance across kinds
    samples = []
    max_per_kind = max(1, max_samples // len(samples_by_kind))
    for kind, kind_samples in samples_by_kind.items():
        random.shuffle(kind_samples)
        samples.extend(kind_samples[:max_per_kind])

    random.shuffle(samples)
    return samples[:max_samples]


# --- Translation ---


async def call_llm_async(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int = 1500,
    temperature: float = 0.7,
) -> str:
    resp = await client.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=120.0,
    )
    if resp.status_code != 200:
        raise httpx.HTTPStatusError(f"{resp.status_code}: {resp.text[:300]}", request=resp.request, response=resp)
    return resp.json()["choices"][0]["message"]["content"].strip()


async def translate_one(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    api_key: str,
    translator_model: str,
    translator_prompt: str,
    sample: Dict,
    max_retries: int = 3,
) -> str:
    async with sem:
        user_content = f"**Problem:** {sample['question']}\n\n" f"**Code:**\n```python\n{sample['code']}\n```"
        messages = [
            {"role": "system", "content": translator_prompt},
            {"role": "user", "content": user_content},
        ]
        for attempt in range(max_retries):
            try:
                return await call_llm_async(client, api_key, translator_model, messages)
            except (httpx.ReadError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                if attempt == max_retries - 1:
                    print(f"    WARN: failed after {max_retries} retries: {e}")
                    return ""
                await asyncio.sleep(2**attempt)
        return ""


async def translate_batch(
    translator_name: str,
    translator_model: str,
    samples: List[Dict],
    translator_prompt: str,
    api_key: str,
) -> List[str]:
    """Translate all samples with one translator model."""
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        tasks = [translate_one(sem, client, api_key, translator_model, translator_prompt, s) for s in samples]
        results = await tqdm_asyncio.gather(*tasks, desc=f"  Translating ({translator_name})")
    failed = sum(1 for r in results if not r)
    if failed:
        print(f"    {failed}/{len(results)} translations failed")
    return results


# --- Embedding ---


def get_embeddings(texts: List[str], batch_size: int = 50) -> np.ndarray:
    api_key = os.getenv("OPENROUTER_API_KEY")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = httpx.post(
            f"{BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        batch_emb = [item["embedding"] for item in result["data"]]
        all_embeddings.extend(batch_emb)
        if (i // batch_size) % 4 == 0:
            print(f"    {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embeddings)


# --- Classification ---


def classify_and_report(native_emb: np.ndarray, translated_emb: np.ndarray, label: str):
    """Logistic regression with 5-fold CV. Returns metrics dict."""
    X = np.vstack([native_emb, translated_emb])
    y = np.array([0] * len(native_emb) + [1] * len(translated_emb))

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred, average="macro")
    report = classification_report(y, y_pred, target_names=["Native", "Translated"], output_dict=True)

    print(f"\n  [{label}]  Accuracy={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}")

    # Refit on full data for boundary
    clf.fit(X, y)
    return {"accuracy": acc, "auc": auc, "f1": f1, "report": report, "clf": clf, "X": X, "y": y}


# --- Plotting ---


def plot_all_translators(all_results: Dict, output_path: Path):
    """Side-by-side PCA embedding plots with decision boundaries."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        X, y = res["X"], res["y"]

        # PCA
        pca = PCA(n_components=2, random_state=SEED)
        X_2d = pca.fit_transform(X)

        # Refit on 2D for boundary visualisation
        clf_2d = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
        clf_2d.fit(X_2d, y)

        # Mesh
        pad = 1.0
        x_min, x_max = X_2d[:, 0].min() - pad, X_2d[:, 0].max() + pad
        y_min, y_max = X_2d[:, 1].min() - pad, X_2d[:, 1].max() + pad
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        Z = clf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

        cmap_bg = ListedColormap(["#d5f5d5", "#f5d5d5"])
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=cmap_bg, alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5, linestyles="--")

        native_mask = y == 0
        ax.scatter(X_2d[native_mask, 0], X_2d[native_mask, 1], c="#2ecc71", alpha=0.45, s=18, label="Native NL", edgecolors="none")
        ax.scatter(X_2d[~native_mask, 0], X_2d[~native_mask, 1], c="#e74c3c", alpha=0.45, s=18, label="Translated", edgecolors="none")

        ax.set_xlabel("PC 1", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("PC 2", fontsize=11)
        ax.set_title(f"{name}\nAcc={res['accuracy']:.1%}  AUC={res['auc']:.3f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Linear Separability by Translator Model\n" "(text-embedding-3-large · logistic regression · 5-fold CV)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\nSaved plot → {output_path.with_suffix('.png')} / .pdf")


def plot_metric_comparison(all_results: Dict, output_path: Path):
    """Bar chart comparing AUC and accuracy across translators."""
    names = list(all_results.keys())
    accs = [all_results[n]["accuracy"] for n in names]
    aucs = [all_results[n]["auc"] for n in names]
    f1s = [all_results[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, accs, width, label="Accuracy", color="#3498db", edgecolor="black", linewidth=0.8)
    ax.bar(x, aucs, width, label="AUC", color="#e74c3c", edgecolor="black", linewidth=0.8)
    ax.bar(x + width, f1s, width, label="Macro F1", color="#2ecc71", edgecolor="black", linewidth=0.8)

    # Value labels
    for i, (a, u, f) in enumerate(zip(accs, aucs, f1s)):
        ax.text(i - width, a + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i, u + 0.008, f"{u:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i + width, f + 0.008, f"{f:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Translator Model", fontsize=13, fontweight="bold")
    ax.set_title("Linear Separability: Native NL vs Translated\nby Translator Model (full 3072-d, 5-fold CV)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved metric plot → {output_path.with_suffix('.png')} / .pdf")


# --- Main ---


async def run():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    translator_prompt = (PROMPTS_DIR / "translator_native_10shot.md").read_text().strip()

    # 1. Load samples
    print("=" * 60)
    print("Translator Separability Experiment")
    print("=" * 60)
    print(f"\nLoading {N_SAMPLES} samples from recorded benchmark artifacts...")
    samples = load_samples(N_SAMPLES)
    print(f"  Loaded {len(samples)} samples across {len(set(s['kind'] for s in samples))} task kinds")

    # 2. Translate with each model (per-translator cache files)
    translations: Dict[str, List[str]] = {}

    for name, model_id in TRANSLATORS.items():
        safe_name = name.replace("\n", "_").replace(" ", "_")
        cache_path = OUTPUT_DIR / f"_trans_cache_{safe_name}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("n_samples") == len(samples) and cached.get("seed") == SEED:
                translations[name] = cached["texts"]
                print(f"  Cache hit: {name} ({len(cached['texts'])} translations)")
                continue
        print(f"\n  [{name}] ({model_id})")
        translated = await translate_batch(name, model_id, samples, translator_prompt, api_key)
        translations[name] = translated
        with open(cache_path, "w") as f:
            json.dump({"n_samples": len(samples), "seed": SEED, "texts": translated}, f)
        print(f"  Cached → {cache_path}")

    # 3. Embed native NL (shared across all translators)
    native_texts = [s["native_nl"][:2000] for s in samples]
    native_cache = OUTPUT_DIR / "_translator_native_emb_cache.npz"

    if native_cache.exists():
        print("\nLoading cached native embeddings...")
        native_emb = np.load(native_cache)["emb"]
        if len(native_emb) != len(samples):
            native_emb = None
    else:
        native_emb = None

    if native_emb is None:
        print("\nEmbedding native NL traces...")
        native_emb = get_embeddings(native_texts)
        np.savez(native_cache, emb=native_emb)

    # 4. Embed translations + classify
    all_results = {}
    for name in TRANSLATORS:
        print(f"\n--- {name} ---")
        trans_texts = [t[:2000] for t in translations[name]]

        emb_cache = OUTPUT_DIR / f"_translator_{name.replace(chr(10), '_')}_emb_cache.npz"
        if emb_cache.exists():
            print("  Loading cached translated embeddings...")
            translated_emb = np.load(emb_cache)["emb"]
            if len(translated_emb) != len(samples):
                translated_emb = None
        else:
            translated_emb = None

        if translated_emb is None:
            print("  Embedding translated traces...")
            translated_emb = get_embeddings(trans_texts)
            np.savez(emb_cache, emb=translated_emb)

        res = classify_and_report(native_emb, translated_emb, name)
        all_results[name] = res

    # 5. Save metrics (without non-serialisable objects)
    metrics = {}
    for name, res in all_results.items():
        metrics[name] = {
            "accuracy": res["accuracy"],
            "auc": res["auc"],
            "f1": res["f1"],
            "report": res["report"],
        }
    metrics_path = OUTPUT_DIR / "translator_separability_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics → {metrics_path}")

    # 6. Plots
    plot_all_translators(all_results, OUTPUT_DIR / "translator_separability_embeddings")
    plot_metric_comparison(all_results, OUTPUT_DIR / "translator_separability_comparison")

    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Translator':<22} {'Accuracy':<10} {'AUC':<10} {'F1':<10}")
    print("-" * 52)
    for name, res in all_results.items():
        print(f"{name.replace(chr(10), ' '):<22} {res['accuracy']:.4f}    {res['auc']:.4f}    {res['f1']:.4f}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
