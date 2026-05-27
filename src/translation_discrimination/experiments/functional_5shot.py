#!/usr/bin/env python3
"""
Functional Similarity Experiment with 5-shot examples.

Tests whether using the same model as translator (vs GPT-4o) affects
LLM judge discrimination accuracy, using stronger judges and 5-shot prompting.

Usage:
    uv run python -m src.translation_discrimination.experiments.functional_5shot
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_RESULTS_DIR

load_dotenv()

# Paths
RESULTS_DIR = LEGACY_BENCHMARK_RESULTS_DIR
OUTPUT_DIR = Path(__file__).parent.parent / "results"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TRANSLATOR_PROMPT_PATH = PROMPTS_DIR / "translator_native_10shot.md"

# API Config
BASE_URL = "https://openrouter.ai/api/v1"

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
    "Claude Opus 4.5": "anthropic/claude-opus-4.5",
    # "Gemini 3.0 Pro": "google/gemini-3-pro-preview",  # Add back later
    # "GPT 5.2 Pro": "openai/gpt-5.2-pro",  # Add back later
}

DIFFERENT_TRANSLATOR = "openai/gpt-4o"

N_SAMPLES = 100  # Per source model


# 5-shot classifier prompt with real examples from data
CLASSIFIER_PROMPT_5SHOT = """You are analyzing explanations of algorithmic problem solutions.

TASK: Given a QUESTION and an EXPLANATION, determine whether the explanation was written by someone solving the problem naturally using language ("NATIVE") or by someone translating code into natural language ("TRANSLATED").

Use whatever semantic, stylistic, or structural cues help you discriminate between the two classes.

## Examples

### Example 1: NATIVE
Question: KMP matcher problem - find occurrences of key [2 3 3 1 1 1 0 1 0 3...] in string [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]
Explanation: "To determine the match of the key within the string using the KMP algorithm, we first need to understand that the string consists of only zeros followed by a few ones, while the key contains a variety of numbers. Given that the string does not contain the numbers present in the key, there are no positions where the key matches any part of the string. Therefore, the final answer for the match index is 0, indicating that there is no occurrence of the key in the string."

Classification: NATIVE

### Example 2: TRANSLATED
Question: KMP matcher problem - find occurrences of key [2 3 3 1 1 1 0 1 0 3...] in string [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]
Explanation: "The problem involves counting how many times a specific sequence (the 'key') appears as a contiguous subsequence within another sequence (the 'string'). Let's break down the thought process step-by-step. First, I have a sequence called `string`, which consists mainly of zeros followed by a few ones. Next, there's a second sequence called `key`. The goal is to determine how many times the `key` appears in the `string`. To do this, the code essentially slides over the `string` and checks each possible starting position. The loop runs from the start to a position where the remaining elements are still enough to match the length of `key`. As the loop iterates through each possible starting position (denoted by `i`), it compares the slice of `string` from `i` to `i + key_length` with the `key`. Because the string contains only 0s and 1s while the key contains 2s and 3s, no match will ever be found. The final answer is 0 occurrences."

Classification: TRANSLATED

### Example 3: NATIVE
Question: Compute: 7698930253608163 - 8774199854573939
Explanation: "To compute the difference between 7698930253608163 and 8774199854573939, we can align the numbers and perform the subtraction as follows: Start from the rightmost digit, subtract each digit while borrowing from the next column as necessary. The result of this subtraction is -1075269600965776."

Classification: NATIVE

### Example 4: TRANSLATED
Question: Compute: 7698930253608163 - 8774199854573939
Explanation: "The problem here involves a straightforward subtraction between two large integers, `number1` and `number2`. Let's break this down step-by-step. First, we identify the numbers involved: `number1` is 7698930253608163 and `number2` is 8774199854573939. The task is to subtract `number2` from `number1`. Since `number2` is larger than `number1`, the result will be negative. Python handles large integers natively with its `int` type, so we simply compute `number1 - number2` which yields -1075269600965776."

Classification: TRANSLATED

### Example 5: NATIVE
Question: Perform insertion sort on key: [0.548 0.715 0.603 ...]
Explanation: "To perform the insertion sort on the given list of numbers, we start with the second element and compare it to the first. If it is smaller, we swap them. We continue this process, gradually building up a sorted section of the list as we move through each element. After inserting each element into the correct position in the sorted portion, the final sorted list is obtained."

Classification: NATIVE

---

Now classify the following. Respond in EXACTLY this format:
PREDICTION: [NATIVE or TRANSLATED]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
REASONING: [1-2 sentences explaining your decision]

Question: {question}

Explanation: {explanation}
"""


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
    translated_same: str
    translated_diff: str


def load_translator_prompt() -> str:
    return TRANSLATOR_PROMPT_PATH.read_text().strip()


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
            timeout=90.0,
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
    concurrency: int = 15,
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


async def judge_discrimination(
    question: str,
    trace: str,
    judge_model: str,
    client: httpx.AsyncClient,
) -> tuple[int, str]:
    """Have judge predict if trace is native (0) or translated (1)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    full_prompt = CLASSIFIER_PROMPT_5SHOT.format(question=question, explanation=trace)

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
            return -1, content
    except Exception as e:
        return -1, str(e)


async def run_control_trials(
    samples: list[TranslatedSample],
    judge_model: str,
    control_type: str,  # "code_vs_nl" or "shuffled"
    concurrency: int = 15,
) -> dict:
    """Run control trials to verify judge has discriminative power."""
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(idx: int, ts: TranslatedSample, true_label: int, client: httpx.AsyncClient) -> tuple[int, int, bool]:
        async with semaphore:
            question = ts.sample.question  # Always use the current sample's question
            if control_type == "code_vs_nl":
                # Label 0 = native NL, Label 1 = raw code (should be easy)
                trace = ts.sample.native_nl if true_label == 0 else ts.sample.sim_code
            elif control_type == "shuffled":
                # Label 0 = native NL (matches question), Label 1 = native NL from DIFFERENT sample (mismatched)
                if true_label == 0:
                    trace = ts.sample.native_nl
                else:
                    # Get native NL from a different sample (mismatched question)
                    other_idx = (idx + len(samples) // 2) % len(samples)
                    trace = samples[other_idx].sample.native_nl
            else:
                raise ValueError(f"Unknown control type: {control_type}")

            predicted, _ = await judge_discrimination(question, trace, judge_model, client)
            correct = (predicted == true_label) if predicted != -1 else False
            return true_label, predicted, correct

    async with httpx.AsyncClient() as client:
        tasks = []
        for idx, ts in enumerate(samples[:50]):  # Use 50 samples for controls
            tasks.append(judge_one(idx, ts, 0, client))  # Native
            tasks.append(judge_one(idx, ts, 1, client))  # Code or shuffled translated

        results = await tqdm_asyncio.gather(*tasks, desc=f"Control ({control_type})")

    valid_results = [(t, p, c) for t, p, c in results if p != -1]
    if not valid_results:
        return {"accuracy": 0.5, "n": 0, "ci_low": 0.0, "ci_high": 1.0}

    correct = sum(1 for _, _, c in valid_results if c)
    n = len(valid_results)
    accuracy = correct / n

    z = 1.96
    p = accuracy
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom

    return {
        "accuracy": accuracy,
        "n": n,
        "ci_low": max(0, center - spread),
        "ci_high": min(1, center + spread),
    }


async def run_discrimination_trials(
    samples: list[TranslatedSample],
    judge_model: str,
    use_same_translator: bool,
    concurrency: int = 15,
) -> dict:
    """Run discrimination trials and return accuracy stats."""
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(ts: TranslatedSample, true_label: int, client: httpx.AsyncClient) -> tuple[int, int, bool]:
        async with semaphore:
            question = ts.sample.question
            trace = ts.sample.native_nl if true_label == 0 else (ts.translated_same if use_same_translator else ts.translated_diff)
            predicted, _ = await judge_discrimination(question, trace, judge_model, client)
            correct = (predicted == true_label) if predicted != -1 else False
            return true_label, predicted, correct

    async with httpx.AsyncClient() as client:
        tasks = []
        for ts in samples:
            tasks.append(judge_one(ts, 0, client))  # Native
            tasks.append(judge_one(ts, 1, client))  # Translated

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


async def main():
    print("=" * 70)
    print("Functional Similarity Experiment (5-shot, 100 samples, Strong Judges)")
    print("=" * 70)

    translator_prompt = load_translator_prompt()

    all_results = {}

    for source_name, source_config in SOURCE_MODELS.items():
        print(f"\n{'='*70}")
        print(f"Source Model: {source_name}")
        print("=" * 70)

        # Load samples
        print(f"\n[1] Loading {N_SAMPLES} samples from {source_name}...")
        samples = load_samples_for_model(source_name, N_SAMPLES)
        print(f"    Loaded {len(samples)} samples")

        if len(samples) < 20:
            print(f"    WARNING: Not enough samples, skipping {source_name}")
            continue

        # Translate
        print(f"\n[2] Translating with same ({source_config['translator_model']}) and diff ({DIFFERENT_TRANSLATOR})...")
        translated_samples = await translate_all_samples(
            samples,
            same_model=source_config["translator_model"],
            diff_model=DIFFERENT_TRANSLATOR,
            prompt=translator_prompt,
            concurrency=15,
        )
        print(f"    Got {len(translated_samples)} translated samples")

        if len(translated_samples) < 20:
            print(f"    WARNING: Not enough translated samples, skipping {source_name}")
            continue

        # Run judges
        print("\n[3] Running discrimination trials...")
        all_results[source_name] = {}

        for judge_name, judge_model in JUDGE_MODELS.items():
            print(f"\n    Judge: {judge_name}")

            # Run controls first
            print("      Running controls...")
            code_control = await run_control_trials(translated_samples, judge_model, "code_vs_nl", concurrency=15)
            shuffled_control = await run_control_trials(translated_samples, judge_model, "shuffled", concurrency=15)

            code_pass = "✅" if code_control["accuracy"] > 0.7 else "❌"
            shuf_pass = "✅" if shuffled_control["accuracy"] > 0.7 else "❌"
            print(f"      Control (code vs NL): {code_control['accuracy']*100:.1f}% {code_pass} (expect >70%)")
            print(f"      Control (shuffled):   {shuffled_control['accuracy']*100:.1f}% {shuf_pass} (expect >70%)")

            # Run main discrimination trials
            same_results = await run_discrimination_trials(
                translated_samples,
                judge_model,
                use_same_translator=True,
                concurrency=15,
            )

            diff_results = await run_discrimination_trials(
                translated_samples,
                judge_model,
                use_same_translator=False,
                concurrency=15,
            )

            all_results[source_name][judge_name] = {
                "same_translator": same_results,
                "diff_translator": diff_results,
                "control_code_vs_nl": code_control,
                "control_shuffled": shuffled_control,
            }

            # Check if CI contains 50%
            same_contains = "✅" if same_results["ci_low"] <= 0.5 <= same_results["ci_high"] else "❌"
            diff_contains = "✅" if diff_results["ci_low"] <= 0.5 <= diff_results["ci_high"] else "❌"

            print(
                f"      Same: {same_results['accuracy']*100:.1f}% [{same_results['ci_low']*100:.1f}%, {same_results['ci_high']*100:.1f}%] {same_contains} (n={same_results['n']})"
            )
            print(
                f"      Diff: {diff_results['accuracy']*100:.1f}% [{diff_results['ci_low']*100:.1f}%, {diff_results['ci_high']*100:.1f}%] {diff_contains} (n={diff_results['n']})"
            )

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    for judge_name in JUDGE_MODELS:
        print(f"\n### Judge: {judge_name}")
        print(f"{'Source Model':<20} {'Same Translator':<30} {'Diff Translator (GPT-4o)':<30}")
        print("-" * 80)
        for source_name in SOURCE_MODELS:
            if source_name in all_results and judge_name in all_results[source_name]:
                same = all_results[source_name][judge_name]["same_translator"]
                diff = all_results[source_name][judge_name]["diff_translator"]

                same_mark = "✅" if same["ci_low"] <= 0.5 <= same["ci_high"] else "❌"
                diff_mark = "✅" if diff["ci_low"] <= 0.5 <= diff["ci_high"] else "❌"

                same_str = f"{same['accuracy']*100:.1f}% [{same['ci_low']*100:.1f}%-{same['ci_high']*100:.1f}%] {same_mark}"
                diff_str = f"{diff['accuracy']*100:.1f}% [{diff['ci_low']*100:.1f}%-{diff['ci_high']*100:.1f}%] {diff_mark}"
                print(f"{source_name:<20} {same_str:<30} {diff_str:<30}")

    # Save
    output_path = OUTPUT_DIR / f"functional_5shot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
