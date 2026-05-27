#!/usr/bin/env python3
"""
Information Additivity Experiment.

Tests whether code provides additional information beyond NL reasoning.

Conditions:
1. x: Question only
2. x || z_nl: Question + NL reasoning trace
3. x || z_code: Question + code (not translated)
4. x || z_nl || z_code: Question + NL reasoning + code
5. mismatch: Question + mismatched NL reasoning + code (control)

Interpretation:
- If (4) ≈ (3) > (2): code contains additional info beyond NL
- If (4) > (3): NL and code are complementary

Usage:
    uv run python -m src.translation_additivity.cli information --n_samples 200
    uv run python -m src.translation_additivity.cli information --model claude-opus-4 --n_samples 100
"""

import argparse
import asyncio
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from scipy import stats
from tqdm.asyncio import tqdm_asyncio

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_RESULTS_DIR

load_dotenv()

# Paths
RESULTS_DIR = LEGACY_BENCHMARK_RESULTS_DIR
OUTPUT_DIR = Path(__file__).parent / "results"

# API config
BASE_URL = "https://openrouter.ai/api/v1"

# Model mapping: short name -> OpenRouter model ID
MODEL_MAP = {
    "claude-opus-4": "anthropic/claude-opus-4",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.5-flash": "google/gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
}


@dataclass
class Sample:
    """A sample for the additivity experiment."""
    kind: str
    question: str
    nl_reasoning: str
    sim_code: str
    gold_answer: str
    source_model: str
    index_in_kind: int


@dataclass
class Trial:
    """A single evaluation trial."""
    sample_id: str
    kind: str
    condition: str  # "x", "x_nl", "x_code", "x_nl_code", "mismatch"
    gold_answer: str
    predicted_answer: str
    correct: bool
    raw_response: str


@dataclass
class ConditionResults:
    """Results for a single condition."""
    condition: str
    n_trials: int
    n_correct: int
    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    by_kind: dict


@dataclass
class ExperimentResults:
    """Full experiment results."""
    model: str
    n_samples: int
    conditions: dict[str, ConditionResults]
    interpretation: str
    timestamp: str


# Prompt templates
CONDITION_X = """You are given an algorithmic problem. Determine the final answer.

## Problem
{question}

Respond with ONLY the final numerical answer. No explanation.
Answer:"""

CONDITION_X_NL = """You are given an algorithmic problem with a reasoning trace. Use the reasoning to determine the answer.

## Problem
{question}

## Reasoning Trace
{nl_reasoning}

Respond with ONLY the final numerical answer. No explanation.
Answer:"""

CONDITION_X_CODE = """You are given an algorithmic problem with solution code. Use the code to determine the answer.

## Problem
{question}

## Solution Code
```python
{code}
```

Respond with ONLY the final numerical answer. No explanation.
Answer:"""

CONDITION_X_NL_CODE = """You are given an algorithmic problem with both reasoning trace and solution code.

## Problem
{question}

## Reasoning Trace
{nl_reasoning}

## Solution Code
```python
{code}
```

Respond with ONLY the final numerical answer. No explanation.
Answer:"""

CONDITION_MISMATCH = """You are given an algorithmic problem with both reasoning trace and solution code.

## Problem
{question}

## Reasoning Trace
{mismatched_nl}

## Solution Code
```python
{code}
```

Respond with ONLY the final numerical answer. No explanation.
Answer:"""


def load_samples(
    results_dir: Path,
    source_model_filter: Optional[str] = None,
    max_samples: int = 500,
    max_per_kind: int = 50,
) -> list[Sample]:
    """Load samples from recorded benchmark results.

    If source_model_filter is provided, only load from that model's results.
    This ensures we use the same model for generation and evaluation.
    """
    samples_by_kind: dict[str, list[Sample]] = defaultdict(list)

    for jsonl_path in results_dir.glob("**/res.jsonl"):
        # Extract model name from path: results/model_seed/tb/run_xxx/res.jsonl
        model_dir = jsonl_path.parent.parent.parent.name
        source_model = model_dir.rsplit("_seed", 1)[0]

        # Filter by source model if specified
        if source_model_filter and source_model != source_model_filter:
            continue

        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract fields
                sim_code = row.get("sim_code", "")
                nl_reasoning = row.get("nl_reasoning", "")
                question = row.get("question", "") or row.get("nl_question", "")
                gold_answer = str(row.get("answer", ""))
                kind = row.get("kind", "unknown")
                index_in_kind = row.get("index_in_kind", -1)

                # Quality filters
                if not sim_code or not nl_reasoning or not question or not gold_answer:
                    continue
                if len(nl_reasoning) < 50 or len(sim_code) < 50:
                    continue

                # Clean code (remove markdown fences if present)
                if sim_code.startswith("```"):
                    sim_code = re.sub(r'^```\w*\n?', '', sim_code)
                    sim_code = re.sub(r'\n?```$', '', sim_code)

                samples_by_kind[kind].append(Sample(
                    kind=kind,
                    question=question,
                    nl_reasoning=nl_reasoning,
                    sim_code=sim_code,
                    gold_answer=gold_answer,
                    source_model=source_model,
                    index_in_kind=index_in_kind,
                ))

    # Apply per-kind limits
    samples = []
    for kind, kind_samples in samples_by_kind.items():
        random.shuffle(kind_samples)
        samples.extend(kind_samples[:max_per_kind])

    random.shuffle(samples)
    print(f"Loaded {len(samples)} samples across {len(samples_by_kind)} task types")
    if samples:
        print(f"Source model: {samples[0].source_model}")
    return samples[:max_samples]


def get_mismatch_nl(samples: list[Sample], current_sample: Sample) -> str:
    """Get NL reasoning from a DIFFERENT sample of the same kind."""
    same_kind = [s for s in samples if s.kind == current_sample.kind and s.index_in_kind != current_sample.index_in_kind]
    if same_kind:
        return random.choice(same_kind).nl_reasoning
    # Fallback: any different sample
    different = [s for s in samples if s.index_in_kind != current_sample.index_in_kind]
    if different:
        return random.choice(different).nl_reasoning
    return current_sample.nl_reasoning  # Last resort


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = answer.strip()
    # Remove common prefixes
    answer = re.sub(r'^(Answer:|The answer is|Result:)\s*', '', answer, flags=re.IGNORECASE)
    # Extract first number-like thing
    match = re.search(r'-?\d+\.?\d*', answer)
    if match:
        return match.group()
    return answer.strip()


def check_correct(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Numeric comparison with tolerance
    try:
        pred_float = float(pred_norm)
        gold_float = float(gold_norm)
        return abs(pred_float - gold_float) < 1e-6
    except (ValueError, TypeError):
        return False


async def call_llm_async(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """Call LLM via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(3):
        try:
            resp = await client.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2:
                print(f"API error after 3 attempts: {e}")
                return ""
            await asyncio.sleep(2 ** attempt)
    return ""


async def run_trial(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    sample: Sample,
    condition: str,
    all_samples: list[Sample],
) -> Trial:
    """Run a single trial for one condition."""

    # Build prompt based on condition
    if condition == "x":
        prompt = CONDITION_X.format(question=sample.question[:2000])
    elif condition == "x_nl":
        prompt = CONDITION_X_NL.format(
            question=sample.question[:2000],
            nl_reasoning=sample.nl_reasoning[:2000],
        )
    elif condition == "x_code":
        prompt = CONDITION_X_CODE.format(
            question=sample.question[:2000],
            code=sample.sim_code[:3000],
        )
    elif condition == "x_nl_code":
        prompt = CONDITION_X_NL_CODE.format(
            question=sample.question[:2000],
            nl_reasoning=sample.nl_reasoning[:2000],
            code=sample.sim_code[:3000],
        )
    elif condition == "mismatch":
        mismatched_nl = get_mismatch_nl(all_samples, sample)
        prompt = CONDITION_MISMATCH.format(
            question=sample.question[:2000],
            mismatched_nl=mismatched_nl[:2000],
            code=sample.sim_code[:3000],
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    response = await call_llm_async(client, api_key, model, prompt)
    correct = check_correct(response, sample.gold_answer)

    return Trial(
        sample_id=f"{sample.kind}_{sample.index_in_kind}",
        kind=sample.kind,
        condition=condition,
        gold_answer=sample.gold_answer,
        predicted_answer=normalize_answer(response),
        correct=correct,
        raw_response=response[:500],
    )


def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for proportion."""
    if n_total == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = (z / denom) * ((p * (1 - p) / n_total + z**2 / (4 * n_total**2)) ** 0.5)
    return max(0, center - margin), min(1, center + margin)


def compute_condition_results(trials: list[Trial], condition: str) -> ConditionResults:
    """Compute results for one condition."""
    cond_trials = [t for t in trials if t.condition == condition]
    n_total = len(cond_trials)
    n_correct = sum(1 for t in cond_trials if t.correct)

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    ci_low, ci_high = wilson_ci(n_correct, n_total)

    # By kind
    by_kind = defaultdict(lambda: {"correct": 0, "total": 0})
    for t in cond_trials:
        by_kind[t.kind]["total"] += 1
        if t.correct:
            by_kind[t.kind]["correct"] += 1

    by_kind_acc = {k: v["correct"] / v["total"] if v["total"] > 0 else 0 for k, v in by_kind.items()}

    return ConditionResults(
        condition=condition,
        n_trials=n_total,
        n_correct=n_correct,
        accuracy=accuracy,
        accuracy_ci_low=ci_low,
        accuracy_ci_high=ci_high,
        by_kind=dict(by_kind_acc),
    )


def interpret_results(conditions: dict[str, ConditionResults]) -> str:
    """Generate interpretation of results."""
    x = conditions["x"].accuracy
    x_nl = conditions["x_nl"].accuracy
    x_code = conditions["x_code"].accuracy
    x_nl_code = conditions["x_nl_code"].accuracy
    mismatch = conditions["mismatch"].accuracy

    lines = [
        "## Interpretation",
        "",
        f"Baseline (x only): {x:.1%}",
        f"+ NL reasoning: {x_nl:.1%} (Δ = {x_nl - x:+.1%})",
        f"+ Code: {x_code:.1%} (Δ = {x_code - x:+.1%})",
        f"+ NL + Code: {x_nl_code:.1%} (Δ = {x_nl_code - x:+.1%})",
        f"Mismatch control: {mismatch:.1%}",
        "",
    ]

    # Key comparisons
    if abs(x_nl_code - x_code) < 0.03 and x_code > x_nl + 0.05:
        lines.append("→ Code contains additional information beyond NL (Condition 4 ≈ 3 > 2)")
    elif x_nl_code > x_code + 0.03 and x_nl_code > x_nl + 0.03:
        lines.append("→ NL and Code are complementary (Condition 4 > 3 and 4 > 2)")
    elif x_nl > x_code:
        lines.append("→ NL reasoning more informative than code")
    elif x_code > x_nl:
        lines.append("→ Code more informative than NL reasoning")

    if mismatch < x_nl_code - 0.1:
        lines.append("→ Mismatch control confirms NL reasoning is being used (not just code)")

    return "\n".join(lines)


async def run_experiment(
    model: str,
    samples: list[Sample],
    concurrency: int = 32,
) -> list[Trial]:
    """Run the full experiment across all conditions."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Resolve model name
    full_model = MODEL_MAP.get(model, model)
    if not full_model.startswith(("openai/", "anthropic/", "google/", "meta/", "mistral")):
        full_model = f"openai/{model}"  # Default to openai prefix

    print(f"Using model: {full_model}")

    conditions = ["x", "x_nl", "x_code", "x_nl_code", "mismatch"]

    # Create all trial tasks
    all_tasks = []
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def bounded_trial(sample: Sample, condition: str) -> Trial:
            async with sem:
                return await run_trial(client, api_key, full_model, sample, condition, samples)

        for sample in samples:
            for condition in conditions:
                all_tasks.append(bounded_trial(sample, condition))

        print(f"Running {len(all_tasks)} trials ({len(samples)} samples × {len(conditions)} conditions)")
        trials = await tqdm_asyncio.gather(*all_tasks, desc="Running trials")

    return trials


def main():
    parser = argparse.ArgumentParser(description="Information Additivity Experiment")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use for evaluation")
    parser.add_argument("--source_model", default=None, help="Filter source data by model (e.g., claude-opus-4)")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--max_per_kind", type=int, default=20, help="Max samples per task type")
    parser.add_argument("--concurrency", type=int, default=32, help="API concurrency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load samples
    samples = load_samples(
        RESULTS_DIR,
        source_model_filter=args.source_model,
        max_samples=args.n_samples,
        max_per_kind=args.max_per_kind,
    )

    if not samples:
        print("No samples found!")
        return

    # Run experiment
    trials = asyncio.run(run_experiment(args.model, samples, args.concurrency))

    # Compute results
    conditions_results = {}
    for cond in ["x", "x_nl", "x_code", "x_nl_code", "mismatch"]:
        conditions_results[cond] = compute_condition_results(trials, cond)

    interpretation = interpret_results(conditions_results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nModel: {args.model}")
    print(f"Samples: {len(samples)}")
    print()

    print(f"{'Condition':<15} {'Accuracy':>10} {'95% CI':>20} {'N':>6}")
    print("-" * 55)
    for cond in ["x", "x_nl", "x_code", "x_nl_code", "mismatch"]:
        r = conditions_results[cond]
        ci_str = f"[{r.accuracy_ci_low:.1%}, {r.accuracy_ci_high:.1%}]"
        print(f"{cond:<15} {r.accuracy:>10.1%} {ci_str:>20} {r.n_trials:>6}")

    print()
    print(interpretation)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"additivity_{args.model}_{timestamp}.json"

    results = ExperimentResults(
        model=args.model,
        n_samples=len(samples),
        conditions={k: asdict(v) for k, v in conditions_results.items()},
        interpretation=interpretation,
        timestamp=timestamp,
    )

    with output_path.open("w") as f:
        json.dump(asdict(results), f, indent=2)

    # Also save trials for detailed analysis
    trials_path = OUTPUT_DIR / f"additivity_{args.model}_{timestamp}_trials.jsonl"
    with trials_path.open("w") as f:
        for t in trials:
            f.write(json.dumps(asdict(t)) + "\n")

    print(f"\nResults saved to: {output_path}")
    print(f"Trials saved to: {trials_path}")


if __name__ == "__main__":
    main()
