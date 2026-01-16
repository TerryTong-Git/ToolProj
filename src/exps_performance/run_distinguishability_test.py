#!/usr/bin/env python3
"""
Test whether Opus 4.5 can distinguish between original NL reasoning
and GPT 5.2 translated NL reasoning.

Two conditions:
1. Zero-shot: No examples, just classification
2. Few-shot: 5 examples, evaluate on remaining 5
"""

import json
import os
import random
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPUS_MODEL = "anthropic/claude-opus-4"


def load_results() -> list[dict]:
    """Load the indistinguishability test results."""
    results_path = Path(__file__).parent / "indistinguishability_results.json"
    with open(results_path, "r") as f:
        return json.load(f)


def call_opus(prompt: str) -> str:
    """Call Opus 4.5 via OpenRouter."""
    client = openai.OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    try:
        response = client.chat.completions.create(
            model=OPUS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


def create_zero_shot_prompt(problem: str, reasoning_a: str, reasoning_b: str) -> str:
    """Create zero-shot classification prompt."""
    return f"""You are given a problem and two different reasoning approaches to solve it.

Your task: Determine which reasoning style is "Style A" (one particular approach) and which is "Style B" (a different approach). The two styles have distinct characteristics.

**Problem:**
{problem[:500]}...

**Reasoning 1:**
{reasoning_a}

**Reasoning 2:**
{reasoning_b}

Based on the writing style, tone, and reasoning approach:
- Which reasoning (1 or 2) sounds more like someone EXPLAINING an algorithm step-by-step?
- Which reasoning (1 or 2) sounds more like someone THINKING through the problem naturally?

Answer in this format:
STEP-BY-STEP EXPLAINER: [1 or 2]
NATURAL THINKER: [1 or 2]
CONFIDENCE: [low/medium/high]
REASONING: [brief explanation of what distinguishes them]
"""


def create_few_shot_prompt(
    examples: list[dict], test_problem: str, test_a: str, test_b: str
) -> str:
    """Create few-shot classification prompt with labeled examples."""
    examples_text = ""
    for i, ex in enumerate(examples):
        examples_text += f"""
---
**Example {i+1} Problem:** {ex['problem'][:200]}...

**Type A (Step-by-step explainer):**
{ex['original'][:400]}...

**Type B (Natural thinker):**
{ex['translated'][:400]}...
---
"""

    return f"""You are given examples of two different reasoning styles for algorithmic problems.

**Type A**: Tends to explain algorithms step-by-step, often mentioning specific operations
**Type B**: Tends to reason through problems more conversationally and intuitively

Here are {len(examples)} labeled examples:
{examples_text}

Now classify the following NEW example:

**Test Problem:**
{test_problem[:500]}...

**Reasoning 1:**
{test_a}

**Reasoning 2:**
{test_b}

Which reasoning is Type A (step-by-step explainer) and which is Type B (natural thinker)?

Answer:
TYPE A (step-by-step): [1 or 2]
TYPE B (natural): [1 or 2]
CONFIDENCE: [low/medium/high]
REASONING: [what features helped you decide]
"""


def run_zero_shot_experiment(samples: list[dict]) -> list[dict]:
    """Run zero-shot classification on all samples."""
    results = []

    for i, sample in enumerate(samples):
        original = sample.get("original_nl_reasoning", "")
        translated = sample.get("translated_nl_reasoning", "")
        problem = sample.get("problem", "")

        # Skip samples with empty translations
        if not translated or not original:
            print(f"  Skipping sample {i+1} (empty reasoning)")
            continue

        # Randomize order to avoid position bias
        random.seed(i)  # Reproducible per-sample
        if random.random() > 0.5:
            reasoning_a, reasoning_b = original, translated
            ground_truth = {"step_by_step": 1, "natural": 2}
        else:
            reasoning_a, reasoning_b = translated, original
            ground_truth = {"step_by_step": 2, "natural": 1}

        prompt = create_zero_shot_prompt(problem, reasoning_a, reasoning_b)
        response = call_opus(prompt)

        results.append({
            "sample_idx": i + 1,
            "kind": sample.get("kind", "unknown"),
            "ground_truth": ground_truth,
            "opus_response": response,
            "original_first": ground_truth["step_by_step"] == 1,
        })

        print(f"  Sample {i+1} ({sample.get('kind', 'unknown')}): processed")

    return results


def run_few_shot_experiment(samples: list[dict]) -> list[dict]:
    """Run few-shot classification: 5 examples, test on remaining."""
    # Filter out samples with empty translations
    valid_samples = [
        s for s in samples
        if s.get("translated_nl_reasoning") and s.get("original_nl_reasoning")
    ]

    if len(valid_samples) < 6:
        print(f"  Only {len(valid_samples)} valid samples, need at least 6")
        return []

    # Split: first 5 as examples, rest as test
    example_samples = valid_samples[:5]
    test_samples = valid_samples[5:]

    # Prepare examples
    examples = [
        {
            "problem": s.get("problem", ""),
            "original": s.get("original_nl_reasoning", ""),
            "translated": s.get("translated_nl_reasoning", ""),
        }
        for s in example_samples
    ]

    results = []
    for i, sample in enumerate(test_samples):
        original = sample.get("original_nl_reasoning", "")
        translated = sample.get("translated_nl_reasoning", "")
        problem = sample.get("problem", "")

        # Randomize order
        random.seed(100 + i)
        if random.random() > 0.5:
            reasoning_a, reasoning_b = original, translated
            ground_truth = {"type_a": 1, "type_b": 2}
        else:
            reasoning_a, reasoning_b = translated, original
            ground_truth = {"type_a": 2, "type_b": 1}

        prompt = create_few_shot_prompt(examples, problem, reasoning_a, reasoning_b)
        response = call_opus(prompt)

        results.append({
            "sample_idx": i + 1,
            "kind": sample.get("kind", "unknown"),
            "ground_truth": ground_truth,
            "opus_response": response,
            "original_first": ground_truth["type_a"] == 1,
        })

        print(f"  Test sample {i+1} ({sample.get('kind', 'unknown')}): processed")

    return results


def parse_response(response: str, is_few_shot: bool = False) -> dict:
    """Parse Opus response to extract predictions."""
    response_lower = response.lower()

    if is_few_shot:
        # Look for TYPE A and TYPE B
        type_a_pred = None
        type_b_pred = None

        if "type a" in response_lower:
            for line in response.split("\n"):
                if "type a" in line.lower():
                    if "1" in line and "2" not in line:
                        type_a_pred = 1
                    elif "2" in line and "1" not in line:
                        type_a_pred = 2
                if "type b" in line.lower():
                    if "1" in line and "2" not in line:
                        type_b_pred = 1
                    elif "2" in line and "1" not in line:
                        type_b_pred = 2

        return {"type_a_pred": type_a_pred, "type_b_pred": type_b_pred}
    else:
        # Zero-shot parsing
        step_pred = None
        natural_pred = None

        for line in response.split("\n"):
            line_lower = line.lower()
            if "step-by-step" in line_lower or "explainer" in line_lower:
                if "1" in line and "2" not in line:
                    step_pred = 1
                elif "2" in line and "1" not in line:
                    step_pred = 2
            if "natural" in line_lower or "thinker" in line_lower:
                if "1" in line and "2" not in line:
                    natural_pred = 1
                elif "2" in line and "1" not in line:
                    natural_pred = 2

        return {"step_pred": step_pred, "natural_pred": natural_pred}


def main():
    print("=" * 70)
    print("Opus 4.5 Distinguishability Test")
    print("Can Opus tell apart original NL vs GPT 5.2 translated NL?")
    print("=" * 70)

    # Load data
    print("\n[1] Loading test data...")
    samples = load_results()
    print(f"    Loaded {len(samples)} samples")

    # Zero-shot experiment
    print("\n[2] Running ZERO-SHOT experiment...")
    print("    (No examples, just asking Opus to classify)")
    zero_shot_results = run_zero_shot_experiment(samples)

    # Few-shot experiment
    print("\n[3] Running FEW-SHOT experiment...")
    print("    (5 labeled examples, test on remaining)")
    few_shot_results = run_few_shot_experiment(samples)

    # Save all results
    output = {
        "zero_shot": zero_shot_results,
        "few_shot": few_shot_results,
    }
    output_path = Path(__file__).parent / "distinguishability_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[4] Results saved to: {output_path}")

    # Print qualitative summary
    print("\n" + "=" * 70)
    print("QUALITATIVE RESULTS")
    print("=" * 70)

    print("\n--- ZERO-SHOT RESULTS ---")
    for r in zero_shot_results:
        print(f"\nSample {r['sample_idx']} ({r['kind']}):")
        print(f"  Ground truth: Original={'1' if r['original_first'] else '2'}, Translated={'2' if r['original_first'] else '1'}")
        print(f"  Opus response:\n{r['opus_response'][:500]}")

    print("\n--- FEW-SHOT RESULTS ---")
    for r in few_shot_results:
        print(f"\nTest Sample {r['sample_idx']} ({r['kind']}):")
        print(f"  Ground truth: TypeA(original)={'1' if r['original_first'] else '2'}, TypeB(translated)={'2' if r['original_first'] else '1'}")
        print(f"  Opus response:\n{r['opus_response'][:500]}")

    # Calculate accuracy
    print("\n" + "=" * 70)
    print("ACCURACY SUMMARY")
    print("=" * 70)

    # Zero-shot accuracy
    zs_correct = 0
    zs_total = 0
    for r in zero_shot_results:
        parsed = parse_response(r["opus_response"], is_few_shot=False)
        gt = r["ground_truth"]
        if parsed["step_pred"] == gt["step_by_step"] and parsed["natural_pred"] == gt["natural"]:
            zs_correct += 1
        zs_total += 1

    if zs_total > 0:
        print(f"\nZero-shot accuracy: {zs_correct}/{zs_total} = {100*zs_correct/zs_total:.1f}%")

    # Few-shot accuracy
    fs_correct = 0
    fs_total = 0
    for r in few_shot_results:
        parsed = parse_response(r["opus_response"], is_few_shot=True)
        gt = r["ground_truth"]
        if parsed["type_a_pred"] == gt["type_a"] and parsed["type_b_pred"] == gt["type_b"]:
            fs_correct += 1
        fs_total += 1

    if fs_total > 0:
        print(f"Few-shot accuracy: {fs_correct}/{fs_total} = {100*fs_correct/fs_total:.1f}%")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
