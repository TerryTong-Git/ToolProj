#!/usr/bin/env python3
"""
Test indistinguishability of code-derived NL reasoning using GPT 5.2.

This script:
1. Loads 10 diverse samples from experiment results
2. Uses the optimized prompt to translate code to NL
3. Calls GPT 5.2 via OpenRouter
4. Outputs the translated reasoning for evaluation
"""

import json
import os
import random
from pathlib import Path

import openai

# Load environment
from dotenv import load_dotenv

load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model configuration - GPT 5.2 via OpenRouter
# Note: Check OpenRouter for the exact model ID when available
GPT_52_MODEL = "openai/gpt-5.2"  # Placeholder - update if different


def load_prompt() -> str:
    """Load the optimized indistinguishability prompt."""
    prompt_path = Path(__file__).parent / "icl_code_to_nl_indistinguishable.md"
    with open(prompt_path, "r") as f:
        return f.read()


def load_samples(n: int = 10) -> list[dict]:
    """Load diverse samples from experiment results."""
    results_dir = Path(__file__).parent / "results"

    # Get first available result directory
    result_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not result_dirs:
        raise FileNotFoundError("No result directories found")

    # Load samples from first available directory
    result_dir = result_dirs[0]
    tb_dirs = list((result_dir / "tb").iterdir())
    if not tb_dirs:
        raise FileNotFoundError(f"No tb directories in {result_dir}")

    jsonl_path = tb_dirs[0] / "res.jsonl"

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            # Only include samples with non-empty code
            if sample.get("code_question") and sample.get("code_question").strip():
                samples.append(sample)

    # Get diverse samples by kind
    by_kind: dict[str, list] = {}
    for s in samples:
        kind = s.get("kind", "unknown")
        if kind not in by_kind:
            by_kind[kind] = []
        by_kind[kind].append(s)

    # Sample evenly across kinds
    selected = []
    kinds = list(by_kind.keys())
    random.seed(42)  # Reproducible
    random.shuffle(kinds)

    while len(selected) < n and kinds:
        for kind in kinds[:]:
            if by_kind[kind] and len(selected) < n:
                sample = random.choice(by_kind[kind])
                by_kind[kind].remove(sample)
                selected.append(sample)
            if not by_kind[kind]:
                kinds.remove(kind)

    return selected[:n]


def call_gpt52(prompt: str, code: str, problem_desc: str) -> str:
    """Call GPT 5.2 via OpenRouter to translate code to NL."""

    # Build the full prompt
    full_prompt = prompt.replace("[PROBLEM_DESCRIPTION]", problem_desc).replace("[CODE_HERE]", code)

    client = openai.OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    try:
        response = client.chat.completions.create(
            model=GPT_52_MODEL,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


def extract_problem_description(sample: dict) -> str:
    """Extract problem description from sample."""
    # Use nl_question to extract the problem part
    nl_q = sample.get("nl_question", "")

    # Extract the problem portion
    if "algorithmic problem:" in nl_q.lower():
        parts = nl_q.split("algorithmic problem:")
        if len(parts) > 1:
            problem = parts[1].split("YOU ARE NEVER")[0].strip()
            return problem

    # Fallback: use kind and basic info
    kind = sample.get("kind", "unknown")
    return f"Problem type: {kind}"


def main():
    print("=" * 60)
    print("GPT 5.2 Code→NL Indistinguishability Test")
    print("=" * 60)

    # Load prompt
    print("\n[1] Loading optimized prompt...")
    prompt = load_prompt()
    print(f"    Prompt loaded ({len(prompt)} chars)")

    # Load samples
    print("\n[2] Loading 10 diverse samples...")
    samples = load_samples(10)
    print(f"    Loaded {len(samples)} samples from kinds: {set(s['kind'] for s in samples)}")

    # Process each sample
    print("\n[3] Calling GPT 5.2 for each sample...")
    results = []

    for i, sample in enumerate(samples):
        kind = sample.get("kind", "unknown")
        code = sample.get("code_question", "")
        problem_desc = extract_problem_description(sample)
        original_nl = sample.get("nl_reasoning", "")

        print(f"\n--- Sample {i+1}/{len(samples)}: {kind} ---")
        print(f"Problem: {problem_desc[:100]}...")

        # Call GPT 5.2
        translated_nl = call_gpt52(prompt, code, problem_desc)

        result = {
            "index": i + 1,
            "kind": kind,
            "problem": problem_desc,
            "code": code,
            "original_nl_reasoning": original_nl,
            "translated_nl_reasoning": translated_nl,
        }
        results.append(result)

        print(f"\n[Original NL]:\n{original_nl[:200]}...")
        print(f"\n[GPT 5.2 Translated NL]:\n{translated_nl[:200]}...")

    # Save results
    output_path = Path(__file__).parent / "indistinguishability_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[4] Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("DONE - Review results to evaluate indistinguishability")
    print("=" * 60)


if __name__ == "__main__":
    main()
