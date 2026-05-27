#!/usr/bin/env python3
"""
Functional Similarity Experiment: Same vs Different Translator

Tests whether using the same model as translator (vs GPT-4o) affects:
1. Distributional similarity (embedding cosine similarity)
2. Functional similarity (LLM judge discrimination accuracy)

Source models: GPT-4o-mini, Claude Haiku 4.5, Gemini 2.5 Flash
Judge models: Claude Opus 4, Gemini 2.5 Pro, Grok 4.1 Fast

Usage:
    uv run python -m src.translation_discrimination.experiments.functional_similarity
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm_asyncio

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_RESULTS_DIR

load_dotenv()

# Paths
RESULTS_DIR = LEGACY_BENCHMARK_RESULTS_DIR
OUTPUT_DIR = Path(__file__).parent.parent / "results"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TRANSLATOR_PROMPT_PATH = PROMPTS_DIR / "translator_native_10shot.md"
CLASSIFIER_PROMPT_PATH = PROMPTS_DIR / "source_classifier.md"

# API Config
BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Model mappings
SOURCE_MODELS = {
    "GPT-4o-mini": {
        "dir_pattern": "gpt-4o-mini_seed*",
        "translator_model": "openai/gpt-4o-mini",
    },
    "Haiku 4.5": {
        "dir_pattern": "claude-haiku-4.5_seed*",
        "translator_model": "anthropic/claude-haiku-4.5",
    },
    "Gemini 2.5 Flash": {
        "dir_pattern": "gemini-2.5-flash_seed*",
        "translator_model": "google/gemini-2.5-flash",
    },
}

JUDGE_MODELS = {
    "Claude Opus 4": "anthropic/claude-opus-4",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro",
    "Grok 4.1 Fast": "x-ai/grok-4.1-fast",
}

DIFFERENT_TRANSLATOR = "openai/gpt-4o"

N_SAMPLES = 200  # Per source model


@dataclass
class Sample:
    kind: str
    question: str
    native_nl: str
    sim_code: str
    source_model: str


@dataclass
class TranslatedSample:
    sample: Sample
    translated_same: str  # Translated by same model
    translated_diff: str  # Translated by GPT-4o


def load_translator_prompt() -> str:
    return TRANSLATOR_PROMPT_PATH.read_text().strip()


def load_classifier_prompt() -> str:
    return CLASSIFIER_PROMPT_PATH.read_text().strip()


def load_samples_for_model(model_name: str, max_samples: int = N_SAMPLES) -> list[Sample]:
    """Load samples from a specific source model."""
    config = SOURCE_MODELS[model_name]
    pattern = config["dir_pattern"]

    samples = []
    for model_dir in RESULTS_DIR.glob(pattern):
        for jsonl_path in model_dir.glob("**/res.jsonl"):
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

                    samples.append(
                        Sample(
                            kind=kind,
                            question=question,
                            native_nl=nl_reasoning,
                            sim_code=sim_code,
                            source_model=model_name,
                        )
                    )

    random.shuffle(samples)
    return samples[:max_samples]


async def translate_code(code: str, model: str, prompt: str, client: httpx.AsyncClient) -> str:
    """Translate code to NL using specified model."""
    api_key = os.getenv("OPENROUTER_API_KEY")

    full_prompt = f"{prompt}\n\n{code}"

    try:
        response = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Translation error with {model}: {e}")
        return ""


async def translate_all_samples(
    samples: list[Sample],
    same_model: str,
    diff_model: str,
    prompt: str,
    concurrency: int = 20,
) -> list[TranslatedSample]:
    """Translate all samples with both same and different translators."""
    semaphore = asyncio.Semaphore(concurrency)

    async def translate_one(sample: Sample, client: httpx.AsyncClient) -> TranslatedSample:
        async with semaphore:
            translated_same = await translate_code(sample.sim_code, same_model, prompt, client)
            translated_diff = await translate_code(sample.sim_code, diff_model, prompt, client)
            return TranslatedSample(
                sample=sample,
                translated_same=translated_same,
                translated_diff=translated_diff,
            )

    async with httpx.AsyncClient() as client:
        tasks = [translate_one(s, client) for s in samples]
        results = await tqdm_asyncio.gather(*tasks, desc="Translating")

    return [r for r in results if r.translated_same and r.translated_diff]


def get_embeddings_sync(texts: list[str], batch_size: int = 25) -> np.ndarray:
    """Get embeddings synchronously."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(3):  # Retry up to 3 times
            try:
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
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                print(f"  Embedding attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise
                import time

                time.sleep(5)

    return np.array(all_embeddings)


async def judge_discrimination(
    native_nl: str,
    translated_nl: str,
    judge_model: str,
    classifier_prompt: str,
    client: httpx.AsyncClient,
) -> tuple[int, str]:
    """Have judge predict if trace is native (0) or translated (1)."""
    api_key = os.getenv("OPENROUTER_API_KEY")

    full_prompt = f"{classifier_prompt}\n\n---\n\nExplanation to classify:\n{translated_nl}"

    try:
        response = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": judge_model,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 256,
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse prediction
        content_upper = content.upper()
        if "PREDICTION: NATIVE" in content_upper or "PREDICTION:NATIVE" in content_upper:
            return 0, content
        elif "PREDICTION: TRANSLATED" in content_upper or "PREDICTION:TRANSLATED" in content_upper:
            return 1, content
        else:
            return -1, content  # Parse error
    except Exception as e:
        return -1, str(e)


async def run_discrimination_trials(
    samples: list[TranslatedSample],
    judge_model: str,
    classifier_prompt: str,
    use_same_translator: bool,
    concurrency: int = 20,
) -> dict:
    """Run discrimination trials and return accuracy stats."""
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(ts: TranslatedSample, true_label: int, client: httpx.AsyncClient) -> tuple[int, int, bool]:
        async with semaphore:
            trace = ts.sample.native_nl if true_label == 0 else (ts.translated_same if use_same_translator else ts.translated_diff)
            predicted, _ = await judge_discrimination(ts.sample.native_nl, trace, judge_model, classifier_prompt, client)
            correct = (predicted == true_label) if predicted != -1 else False
            return true_label, predicted, correct

    async with httpx.AsyncClient() as client:
        tasks = []
        for ts in samples:
            # Trial with native NL (label 0)
            tasks.append(judge_one(ts, 0, client))
            # Trial with translated NL (label 1)
            tasks.append(judge_one(ts, 1, client))

        results = await tqdm_asyncio.gather(*tasks, desc=f"Judging ({judge_model.split('/')[-1]})")

    # Calculate accuracy
    valid_results = [(t, p, c) for t, p, c in results if p != -1]
    if not valid_results:
        return {"accuracy": 0.5, "n": 0, "ci_low": 0.0, "ci_high": 1.0}

    correct = sum(1 for _, _, c in valid_results if c)
    n = len(valid_results)
    accuracy = correct / n

    # Wilson score CI
    z = 1.96
    p = accuracy
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    ci_low = max(0, center - spread)
    ci_high = min(1, center + spread)

    return {
        "accuracy": accuracy,
        "n": n,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def calculate_cosine_stats(native_emb: np.ndarray, translated_emb: np.ndarray) -> dict:
    """Calculate cross-group cosine similarity."""
    cross_sim = cosine_similarity(native_emb, translated_emb)

    # Diagonal = same sample, native vs translated
    diag_sim = np.diag(cross_sim)

    return {
        "mean": float(np.mean(diag_sim)),
        "std": float(np.std(diag_sim)),
        "n": len(diag_sim),
    }


def plot_clusters(
    native_emb: np.ndarray,
    translated_emb: np.ndarray,
    title: str,
    output_path: Path,
):
    """Create t-SNE cluster visualization."""
    all_emb = np.vstack([native_emb, translated_emb])

    print(f"  Running t-SNE on {len(all_emb)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_emb) - 1))
    coords = tsne.fit_transform(all_emb)

    fig, ax = plt.subplots(figsize=(10, 8))

    native_coords = coords[: len(native_emb)]
    translated_coords = coords[len(native_emb) :]

    ax.scatter(native_coords[:, 0], native_coords[:, 1], c="#2ecc71", alpha=0.6, label="Native NL", s=50)
    ax.scatter(translated_coords[:, 0], translated_coords[:, 1], c="#e74c3c", alpha=0.6, label="Translated", s=50)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot to {output_path}")


async def main():
    print("=" * 70)
    print("Functional Similarity Experiment: Same vs Different Translator")
    print("=" * 70)

    translator_prompt = load_translator_prompt()
    classifier_prompt = load_classifier_prompt()

    # Store all results
    all_results = {
        "distributional": {},  # source_model -> {same: stats, diff: stats}
        "functional": {},  # source_model -> judge_model -> {same: stats, diff: stats}
    }

    # Store embeddings for plotting
    gemini_embeddings = {}

    for source_name, source_config in SOURCE_MODELS.items():
        print(f"\n{'='*70}")
        print(f"Source Model: {source_name}")
        print("=" * 70)

        # Load samples
        print(f"\n[1] Loading samples from {source_name}...")
        samples = load_samples_for_model(source_name, N_SAMPLES)
        print(f"    Loaded {len(samples)} samples")

        if len(samples) < 50:
            print(f"    WARNING: Not enough samples, skipping {source_name}")
            continue

        # Translate with both same and different translators
        print(f"\n[2] Translating with same ({source_config['translator_model']}) and different ({DIFFERENT_TRANSLATOR})...")
        translated_samples = await translate_all_samples(
            samples,
            same_model=source_config["translator_model"],
            diff_model=DIFFERENT_TRANSLATOR,
            prompt=translator_prompt,
            concurrency=15,
        )
        print(f"    Got {len(translated_samples)} translated samples")

        if len(translated_samples) < 50:
            print(f"    WARNING: Not enough translated samples, skipping {source_name}")
            continue

        # Get embeddings
        print("\n[3] Computing embeddings...")
        native_texts = [ts.sample.native_nl[:2000] for ts in translated_samples]
        same_texts = [ts.translated_same[:2000] for ts in translated_samples]
        diff_texts = [ts.translated_diff[:2000] for ts in translated_samples]

        native_emb = get_embeddings_sync(native_texts)
        same_emb = get_embeddings_sync(same_texts)
        diff_emb = get_embeddings_sync(diff_texts)

        # Store Gemini embeddings for plotting
        if source_name == "Gemini 2.5 Flash":
            gemini_embeddings = {
                "native": native_emb,
                "same": same_emb,
                "diff": diff_emb,
            }

        # Calculate distributional similarity
        print("\n[4] Calculating distributional similarity...")
        same_stats = calculate_cosine_stats(native_emb, same_emb)
        diff_stats = calculate_cosine_stats(native_emb, diff_emb)

        all_results["distributional"][source_name] = {
            "same_translator": same_stats,
            "diff_translator": diff_stats,
        }

        print(f"    Same translator: {same_stats['mean']:.4f} ± {same_stats['std']:.4f}")
        print(f"    Diff translator: {diff_stats['mean']:.4f} ± {diff_stats['std']:.4f}")

        # Run functional discrimination with each judge
        print("\n[5] Running functional discrimination trials...")
        all_results["functional"][source_name] = {}

        for judge_name, judge_model in JUDGE_MODELS.items():
            print(f"\n    Judge: {judge_name}")

            # Same translator
            same_results = await run_discrimination_trials(
                translated_samples[:100],  # Use subset for speed
                judge_model,
                classifier_prompt,
                use_same_translator=True,
                concurrency=15,
            )

            # Different translator
            diff_results = await run_discrimination_trials(
                translated_samples[:100],
                judge_model,
                classifier_prompt,
                use_same_translator=False,
                concurrency=15,
            )

            all_results["functional"][source_name][judge_name] = {
                "same_translator": same_results,
                "diff_translator": diff_results,
            }

            print(f"      Same: {same_results['accuracy']*100:.1f}% [{same_results['ci_low']*100:.1f}%, {same_results['ci_high']*100:.1f}%]")
            print(f"      Diff: {diff_results['accuracy']*100:.1f}% [{diff_results['ci_low']*100:.1f}%, {diff_results['ci_high']*100:.1f}%]")

    # Create cluster plots for Gemini
    if gemini_embeddings:
        print(f"\n{'='*70}")
        print("Creating cluster plots for Gemini 2.5 Flash...")
        print("=" * 70)

        plot_clusters(
            gemini_embeddings["native"],
            gemini_embeddings["diff"],
            "Gemini 2.5 Flash: Native vs Translated (GPT-4o translator)",
            OUTPUT_DIR / "clusters_gemini_diff_translator.png",
        )

        plot_clusters(
            gemini_embeddings["native"],
            gemini_embeddings["same"],
            "Gemini 2.5 Flash: Native vs Translated (Same translator)",
            OUTPUT_DIR / "clusters_gemini_same_translator.png",
        )

    # Print summary tables
    print(f"\n{'='*70}")
    print("SUMMARY TABLES")
    print("=" * 70)

    # Distributional table
    print("\n## Distributional Similarity (Cosine Sim: Native ↔ Translated)")
    print(f"{'Source Model':<20} {'Same Translator':<25} {'Diff Translator (GPT-4o)':<25}")
    print("-" * 70)
    for source_name in SOURCE_MODELS:
        if source_name in all_results["distributional"]:
            same = all_results["distributional"][source_name]["same_translator"]
            diff = all_results["distributional"][source_name]["diff_translator"]
            print(f"{source_name:<20} {same['mean']:.4f} ± {same['std']:.4f}         {diff['mean']:.4f} ± {diff['std']:.4f}")

    # Functional table
    print("\n## Functional Similarity (Judge Accuracy - lower = more similar)")
    for judge_name in JUDGE_MODELS:
        print(f"\n### Judge: {judge_name}")
        print(f"{'Source Model':<20} {'Same Translator':<25} {'Diff Translator (GPT-4o)':<25}")
        print("-" * 70)
        for source_name in SOURCE_MODELS:
            if source_name in all_results["functional"] and judge_name in all_results["functional"][source_name]:
                same = all_results["functional"][source_name][judge_name]["same_translator"]
                diff = all_results["functional"][source_name][judge_name]["diff_translator"]
                print(
                    f"{source_name:<20} {same['accuracy']*100:.1f}% [{same['ci_low']*100:.0f}%-{same['ci_high']*100:.0f}%]       {diff['accuracy']*100:.1f}% [{diff['ci_low']*100:.0f}%-{diff['ci_high']*100:.0f}%]"
                )

    # Save results
    output_path = OUTPUT_DIR / f"functional_similarity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
