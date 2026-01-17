#!/usr/bin/env python3
"""
Distinguishability Test with Calibrated Baseline.

Tests whether a judge can distinguish original NL from translated code-to-NL,
using a calibrated baseline to correct for LLM "DIFFERENT" bias.

Evaluation:
- CONTROL pairs: Original_NL vs Original_NL (different samples) → measures bias
- TEST pairs: Original_NL vs Translated_NL → what we care about
- Adjusted Score = Test_SAME_rate - Baseline_SAME_rate (bias-corrected)

Usage:
    uv run python src/exps_control/run_distinguishability.py --n_samples 10
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "exps_performance" / "results"
OUTPUT_DIR = Path(__file__).parent / "results"
ICL_PROMPT_PATH = Path(__file__).parent / "prompts" / "icl_code_to_nl_FINAL.md"

# Models
TRANSLATOR_MODEL = "openai/gpt-5.2"
JUDGE_MODEL = "google/gemini-2.5-pro-preview-06-05"
BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class Sample:
    """A sample for the distinguishability test."""
    kind: str
    question: str
    original_nl: str
    sim_code: str
    translated_nl: str = ""
    judge_verdict: str = ""  # "SAME" or "DIFFERENT"
    judge_reasoning: str = ""


@dataclass
class Results:
    """Results with calibrated baseline."""
    n_samples: int
    # Control (baseline)
    control_total: int
    control_same: int
    baseline_same_rate: float
    # Test
    test_total: int
    test_same: int
    test_same_rate: float
    # Adjusted
    adjusted_score: float
    # By kind
    by_kind: dict


def load_icl_prompt() -> str:
    """Load the translation prompt."""
    if ICL_PROMPT_PATH.exists():
        return ICL_PROMPT_PATH.read_text().strip()
    raise FileNotFoundError(f"ICL prompt not found at {ICL_PROMPT_PATH}")


def load_samples(
    results_dir: Path,
    max_samples: int = 100,
    model_filter: Optional[str] = None,
) -> list[Sample]:
    """Load samples with both code and NL reasoning."""
    samples = []
    kind_counts: dict[str, int] = {}

    for jsonl_path in results_dir.glob("**/res.jsonl"):
        if model_filter and model_filter.lower() not in str(jsonl_path).lower():
            continue

        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)

                sim_code = row.get("sim_code", "")
                nl_reasoning = row.get("nl_reasoning", "")
                question = row.get("question", "")
                kind = row.get("kind", "unknown")

                if not sim_code or not nl_reasoning or not question:
                    continue
                if len(nl_reasoning) < 100 or len(sim_code) < 50:
                    continue

                # Diverse sampling - max 5 per kind
                if kind_counts.get(kind, 0) >= 5:
                    continue
                kind_counts[kind] = kind_counts.get(kind, 0) + 1

                samples.append(Sample(
                    kind=kind,
                    question=question,
                    original_nl=nl_reasoning,
                    sim_code=sim_code,
                ))

                if len(samples) >= max_samples:
                    return samples

    return samples


def call_llm(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> str:
    """Call an LLM via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def translate_code(api_key: str, icl_prompt: str, code: str) -> str:
    """Translate code to NL using the ICL prompt."""
    messages = [
        {"role": "system", "content": icl_prompt},
        {"role": "user", "content": f"```python\n{code}\n```"},
    ]
    return call_llm(api_key, TRANSLATOR_MODEL, messages, max_tokens=600, temperature=0.7)


def judge_pair(api_key: str, text_a: str, text_b: str) -> tuple[str, str]:
    """
    Ask judge if two texts are from SAME or DIFFERENT sources.
    Returns: (verdict, reasoning)
    """
    prompt = f"""You are analyzing two explanations of how to solve an algorithmic problem.

TEXT A:
{text_a[:1200]}

TEXT B:
{text_b[:1200]}

Determine if these two texts were written by the SAME author/source or DIFFERENT authors/sources.

Consider:
- Writing style, tone, and voice
- Level of formality vs casualness
- Structure and organization
- Use of technical terminology
- Explanation approach and depth

Respond with EXACTLY this format:
VERDICT: [SAME or DIFFERENT]
REASONING: [2-3 sentences explaining your decision]
"""

    response = call_llm(api_key, JUDGE_MODEL, [{"role": "user", "content": prompt}],
                        max_tokens=300, temperature=0.1)

    # Parse response
    verdict = ""
    reasoning = ""

    for line in response.split("\n"):
        line_upper = line.upper()
        if "VERDICT:" in line_upper:
            if "SAME" in line_upper and "DIFFERENT" not in line_upper:
                verdict = "SAME"
            elif "DIFFERENT" in line_upper:
                verdict = "DIFFERENT"
        elif "REASONING:" in line_upper:
            reasoning = line.split(":", 1)[-1].strip()

    # Fallback
    if not verdict:
        if "SAME" in response.upper()[:100]:
            verdict = "SAME"
        else:
            verdict = "DIFFERENT"

    return verdict, reasoning or response[:200]


def run_experiment(
    api_key: str,
    samples: list[Sample],
    icl_prompt: str,
) -> tuple[list[Sample], Results]:
    """
    Run distinguishability test with calibrated baseline.
    """
    print(f"\n[1/3] Translating {len(samples)} samples...")
    for sample in tqdm(samples, desc="Translating"):
        try:
            sample.translated_nl = translate_code(api_key, icl_prompt, sample.sim_code)
        except Exception as e:
            tqdm.write(f"Translation failed ({sample.kind}): {e}")
            sample.translated_nl = ""

    valid_samples = [s for s in samples if s.translated_nl]
    print(f"      {len(valid_samples)}/{len(samples)} translated successfully")

    # Phase 2: CONTROL pairs (baseline - two different originals)
    print(f"\n[2/3] Running CONTROL pairs (baseline)...")
    control_same = 0
    control_total = 0

    shuffled = valid_samples.copy()
    random.shuffle(shuffled)
    control_pairs = list(zip(shuffled[::2], shuffled[1::2]))

    for s1, s2 in tqdm(control_pairs, desc="Control"):
        try:
            verdict, _ = judge_pair(api_key, s1.original_nl, s2.original_nl)
            if verdict == "SAME":
                control_same += 1
            control_total += 1
        except Exception as e:
            tqdm.write(f"Control failed: {e}")

    baseline_same_rate = control_same / control_total if control_total > 0 else 0
    print(f"      Baseline SAME rate: {baseline_same_rate*100:.1f}% ({control_same}/{control_total})")

    # Phase 3: TEST pairs (original vs translated)
    print(f"\n[3/3] Running TEST pairs (original vs translated)...")
    test_same = 0
    test_total = 0
    by_kind: dict[str, dict] = defaultdict(lambda: {"same": 0, "total": 0})

    pbar = tqdm(valid_samples, desc="Test")
    for sample in pbar:
        # Randomize order
        if random.random() < 0.5:
            text_a, text_b = sample.original_nl, sample.translated_nl
        else:
            text_a, text_b = sample.translated_nl, sample.original_nl

        try:
            verdict, reasoning = judge_pair(api_key, text_a, text_b)
            sample.judge_verdict = verdict
            sample.judge_reasoning = reasoning

            by_kind[sample.kind]["total"] += 1
            if verdict == "SAME":
                test_same += 1
                by_kind[sample.kind]["same"] += 1
            test_total += 1

        except Exception as e:
            tqdm.write(f"Test failed ({sample.kind}): {e}")
            test_total += 1

        pbar.set_postfix({"same": f"{test_same}/{test_total}"})

    test_same_rate = test_same / test_total if test_total > 0 else 0
    adjusted_score = test_same_rate - baseline_same_rate

    # Compute per-kind stats
    kind_stats = {}
    for kind, data in by_kind.items():
        kind_stats[kind] = {
            "same_rate": data["same"] / data["total"] if data["total"] > 0 else 0,
            "total": data["total"],
        }

    results = Results(
        n_samples=len(valid_samples),
        control_total=control_total,
        control_same=control_same,
        baseline_same_rate=baseline_same_rate,
        test_total=test_total,
        test_same=test_same,
        test_same_rate=test_same_rate,
        adjusted_score=adjusted_score,
        by_kind=kind_stats,
    )

    return valid_samples, results


def print_results(results: Results):
    """Print final results."""
    print("\n" + "=" * 60)
    print("RESULTS (Calibrated)")
    print("=" * 60)

    print(f"\nCONTROL (two originals - measures bias):")
    print(f"  SAME rate: {results.baseline_same_rate*100:.1f}% ({results.control_same}/{results.control_total})")

    print(f"\nTEST (original vs translated):")
    print(f"  SAME rate: {results.test_same_rate*100:.1f}% ({results.test_same}/{results.test_total})")

    print(f"\nADJUSTED SCORE (test - baseline):")
    print(f"  {results.adjusted_score*100:+.1f}%")

    if results.adjusted_score > 0.1:
        print(f"  ✓ GOOD: Translations are MORE similar than baseline")
    elif results.adjusted_score > -0.1:
        print(f"  ~ NEUTRAL: Similar to baseline (indistinguishable)")
    else:
        print(f"  ✗ POOR: Translations are MORE distinguishable than baseline")

    print(f"\nBy problem kind:")
    for kind in sorted(results.by_kind.keys()):
        data = results.by_kind[kind]
        print(f"  {kind}: {data['same_rate']*100:.0f}% same (n={data['total']})")


def save_results(samples: list[Sample], results: Results, output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "results": asdict(results),
        "samples": [asdict(s) for s in samples],
    }

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Distinguishability Test (Calibrated)")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--model_filter", type=str, default=None, help="Filter by model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output) if args.output else OUTPUT_DIR / f"distinguishability_{timestamp}.json"

    print("=" * 60)
    print("DISTINGUISHABILITY TEST (Calibrated)")
    print("=" * 60)
    print(f"Samples: {args.n_samples}")
    print(f"Translator: {TRANSLATOR_MODEL}")
    print(f"Judge: {JUDGE_MODEL}")

    # Load ICL prompt
    icl_prompt = load_icl_prompt()
    print(f"ICL prompt: {len(icl_prompt)} chars")

    # Load samples
    print(f"\nLoading samples...")
    samples = load_samples(RESULTS_DIR, args.n_samples, args.model_filter)
    random.shuffle(samples)
    print(f"Loaded {len(samples)} samples")
    print(f"Kinds: {set(s.kind for s in samples)}")

    # Run experiment
    samples, results = run_experiment(api_key, samples, icl_prompt)

    # Print and save
    print_results(results)
    save_results(samples, results, output_file)


if __name__ == "__main__":
    main()
