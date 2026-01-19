#!/usr/bin/env python3
"""
Code-to-NL Translation Discrimination Experiment

Tests whether translated NL reasoning is distinguishable from original NL reasoning.
Supports multiple translator models via command-line arguments.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
from tqdm import tqdm

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Available translator models
TRANSLATOR_MODELS = {
    "haiku-4.5": "anthropic/claude-haiku-4.5",
    "gpt-5": "openai/gpt-5",
    "deepseek-v3": "deepseek/deepseek-chat-v3-0324",
    "gemini-2.5-flash": "google/gemini-2.5-flash-preview-05-20",
    "opus-4.0": "anthropic/claude-opus-4-0",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview-06-05",
}

# Default models
DEFAULT_TRANSLATOR = "gpt-5"
DEFAULT_DISCRIMINATOR = "anthropic/claude-opus-4.5"

# Global model settings (set by main)
TRANSLATOR_MODEL = None
DISCRIMINATOR_MODEL = None

def load_samples(n_samples: int = 200) -> list[dict]:
    """Load samples with both code and NL reasoning from experiment results."""
    results_dir = Path(__file__).parent / "results"

    all_samples = []

    # Load from multiple result files
    for jsonl_path in results_dir.glob("*/tb/run_*/res.jsonl"):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Need both sim_code and nl_reasoning
                    if record.get("sim_code") and record.get("nl_reasoning"):
                        all_samples.append({
                            "kind": record["kind"],
                            "question": record.get("nl_question", ""),
                            "code": record["sim_code"],
                            "nl_reasoning": record["nl_reasoning"],
                            "answer": record.get("answer", ""),
                        })
                except json.JSONDecodeError:
                    continue

    # Shuffle and select n_samples
    random.seed(42)
    random.shuffle(all_samples)

    # Try to get diverse tasks
    by_kind = {}
    for s in all_samples:
        by_kind.setdefault(s["kind"], []).append(s)

    selected = []
    kinds = list(by_kind.keys())
    idx = 0
    while len(selected) < n_samples and any(by_kind.values()):
        kind = kinds[idx % len(kinds)]
        if by_kind[kind]:
            selected.append(by_kind[kind].pop(0))
        idx += 1

    return selected[:n_samples]


def create_icl_examples(samples: list[dict], n_examples: int = 5) -> str:
    """Create in-context learning examples for code-to-NL translation."""
    examples = []
    for i, s in enumerate(samples[:n_examples]):
        examples.append(f"""Example {i+1}:
Code:
```python
{s['code']}
```

Natural Language Reasoning:
{s['nl_reasoning']}
""")
    return "\n---\n".join(examples)


def translate_code_to_nl(code: str, icl_prompt: str, max_retries: int = 3) -> str:
    """Use GPT 5.2 to translate code to natural language reasoning."""
    prompt = f"""{icl_prompt}

---

Now translate this code to natural language reasoning in the same style as the examples above.
Focus on explaining the algorithmic steps and logic, not the code syntax.
Match the tone, structure, and level of detail from the examples.

Code:
```python
{code}
```

Natural Language Reasoning:"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": TRANSLATOR_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.3,
                },
                timeout=90,
            )

            if response.status_code != 200:
                print(f"Translation error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return ""

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            print(f"Translation timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
        except Exception as e:
            print(f"Translation error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    return ""


def discriminate(question: str, reasoning: str, max_retries: int = 3) -> str:
    """Use Opus 4.5 via OpenRouter to classify whether reasoning is original or translated."""
    prompt = f"""You are an expert at distinguishing between original natural language reasoning written by an LLM, and reasoning that was translated from code.

Given an algorithmic problem and a reasoning trace, determine if the reasoning was:
- "original": Written directly as natural language by an LLM
- "translated": Translated from code to natural language by a post-processor

Original reasoning tends to be more narrative and sometimes makes reasoning errors.
Translated reasoning may have more structured, step-by-step logic matching code execution.

Problem context (first 500 chars):
{question[:500]}

Reasoning trace:
{reasoning}

Answer with just one word: "original" or "translated"."""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": DISCRIMINATOR_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0,
                },
                timeout=120,  # Longer timeout for Opus
            )

            if response.status_code != 200:
                print(f"Discrimination error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "error"

            result = response.json()
            return result["choices"][0]["message"]["content"].strip().lower()

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    return "error"


def run_experiment(n_samples: int = 200):
    """Run the full discrimination experiment."""
    print(f"Loading {n_samples} samples...")
    samples = load_samples(n_samples)
    print(f"Loaded {len(samples)} samples across {len(set(s['kind'] for s in samples))} task types")

    # Reserve first 5 for ICL examples, use rest for evaluation
    icl_samples = samples[:5]
    eval_samples = samples[5:n_samples]

    icl_prompt = create_icl_examples(icl_samples)
    print(f"Created ICL prompt with {len(icl_samples)} examples")

    results = []
    correct = 0
    total = 0

    print(f"\nRunning translation and discrimination on {len(eval_samples)} samples...")

    for sample in tqdm(eval_samples):
        # Translate code to NL
        translated_nl = translate_code_to_nl(sample["code"], icl_prompt)
        if not translated_nl:
            continue

        # Randomly decide which to show (to avoid position bias)
        show_translated = random.random() < 0.5

        if show_translated:
            reasoning = translated_nl
            ground_truth = "translated"
        else:
            reasoning = sample["nl_reasoning"]
            ground_truth = "original"

        # Get discrimination
        prediction = discriminate(sample["question"], reasoning)

        # Skip errors
        if prediction == "error":
            continue

        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "kind": sample["kind"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": is_correct,
            "original_nl": sample["nl_reasoning"][:200],
            "translated_nl": translated_nl[:200] if show_translated else None,
        })

        # Progress update every 20 samples
        if total % 20 == 0:
            acc = correct / total
            consistency = max(acc, 1 - acc)
            print(f"\n  Progress: {total}/{len(eval_samples)}, Accuracy: {acc:.1%}, Consistency: {consistency:.1%}")

    # Final results
    accuracy = correct / total if total > 0 else 0
    consistency = max(accuracy, 1 - accuracy)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Samples evaluated: {total}")
    print(f"Raw accuracy: {accuracy:.1%}")
    print(f"Consistency (|acc - 50%|): {consistency:.1%}")
    print(f"  - 50% = indistinguishable (random guessing)")
    print(f"  - 100% = perfectly distinguishable")

    # Per-task breakdown
    by_kind = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r["correct"])

    print("\nPer-task accuracy:")
    for kind, corrects in sorted(by_kind.items()):
        task_acc = sum(corrects) / len(corrects)
        print(f"  {kind}: {task_acc:.1%} ({len(corrects)} samples)")

    # Save results with timestamp and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = TRANSLATOR_MODEL.split("/")[-1]
    output_path = Path(__file__).parent / f"translation_discrimination_{model_short}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "translator": TRANSLATOR_MODEL,
                "discriminator": DISCRIMINATOR_MODEL,
                "n_samples": total,
                "timestamp": timestamp,
            },
            "summary": {
                "accuracy": accuracy,
                "consistency": consistency,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return accuracy, consistency


def main():
    global TRANSLATOR_MODEL, DISCRIMINATOR_MODEL

    parser = argparse.ArgumentParser(description="Run translation discrimination experiment")
    parser.add_argument("--translator", type=str, default=DEFAULT_TRANSLATOR,
                        choices=list(TRANSLATOR_MODELS.keys()),
                        help=f"Translator model (default: {DEFAULT_TRANSLATOR})")
    parser.add_argument("--discriminator", type=str, default=DEFAULT_DISCRIMINATOR,
                        help=f"Discriminator model (default: {DEFAULT_DISCRIMINATOR})")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to evaluate (default: 200)")
    parser.add_argument("--all", action="store_true",
                        help="Run all translator models sequentially")

    args = parser.parse_args()

    if args.all:
        # Run all models
        all_results = {}
        for name, model_id in TRANSLATOR_MODELS.items():
            print(f"\n{'='*60}")
            print(f"Running with translator: {name} ({model_id})")
            print(f"{'='*60}")
            TRANSLATOR_MODEL = model_id
            DISCRIMINATOR_MODEL = args.discriminator
            try:
                acc, cons = run_experiment(args.n_samples)
                all_results[name] = {"accuracy": acc, "consistency": cons}
            except Exception as e:
                print(f"Error with {name}: {e}")
                all_results[name] = {"error": str(e)}

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY: ALL MODELS")
        print("="*60)
        for name, result in all_results.items():
            if "error" in result:
                print(f"  {name}: ERROR - {result['error']}")
            else:
                print(f"  {name}: Acc={result['accuracy']:.1%}, Consistency={result['consistency']:.1%}")
    else:
        # Run single model
        TRANSLATOR_MODEL = TRANSLATOR_MODELS[args.translator]
        DISCRIMINATOR_MODEL = args.discriminator
        print(f"Translator: {args.translator} ({TRANSLATOR_MODEL})")
        print(f"Discriminator: {DISCRIMINATOR_MODEL}")
        run_experiment(args.n_samples)


if __name__ == "__main__":
    main()
