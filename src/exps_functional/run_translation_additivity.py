#!/usr/bin/env python3
"""
Translation Additivity Experiment.

Tests whether NL translated from code provides the same information as native NL.

Conditions:
1. x: Question only
2. x + z_nl_native: Question + native NL reasoning (from Arm1)
3. x + z_nl_translated: Question + NL translated from code (by same model)

Usage:
    uv run python src/exps_functional/run_translation_additivity.py --n_samples 100
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

import httpx
from dotenv import load_dotenv
from scipy import stats
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "exps_performance" / "results"
OUTPUT_DIR = Path(__file__).parent / "results"

# API config
BASE_URL = "https://openrouter.ai/api/v1"

# Model mapping
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
    """A sample for the experiment."""
    kind: str
    question: str
    nl_reasoning_native: str  # Native NL from Arm1
    sim_code: str  # Code from Arm2
    gold_answer: str
    source_model: str
    index_in_kind: int
    nl_reasoning_translated: str = ""  # Will be filled by translation


@dataclass
class Trial:
    """A single evaluation trial."""
    sample_id: str
    kind: str
    condition: str  # "x", "x_nl_native", "x_nl_translated"
    gold_answer: str
    predicted_answer: str
    correct: bool
    raw_response: str


# Native 10-shot ICL Translation prompt (problem-solving version)
TRANSLATE_PROMPT = '''# Natural Language Problem Solver

You are given code that solves an algorithmic problem. Your task is to **reason through the problem step-by-step using natural language** and arrive at the answer.

Do NOT describe or translate the code mechanically. Instead, solve the problem yourself using natural reasoning, as if you were explaining your thought process to a colleague.

## Guidelines

1. **Think like a human** - use exploratory reasoning, not procedural code execution
2. **Be conversational** - use phrases like "Let me check", "I notice that", "This means"
3. **Skip obvious steps** - don't enumerate every single iteration
4. **Focus on insights** - explain WHY something works, not just WHAT happens
5. **Use natural structure** - paragraphs are fine, avoid numbered lists unless truly needed

## Examples

---

### Example 1: Topological Sort

**Input (Code):**
```python
def solution():
    A = [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]]
    # Return the first node in topological order
```

**Output (Reasoning):**
To find topological order, I need to identify which nodes have no incoming edges - those come first.

Let me count in-degrees by looking at each column:
- Node 0: A[3][0]=1, A[5][0]=1 → in-degree = 2
- Node 1: A[5][1]=1 → in-degree = 1
- Node 2: A[0][2]=1, A[1][2]=1, A[3][2]=1, A[5][2]=1 → in-degree = 4
- Node 3: A[5][3]=1 → in-degree = 1
- Node 4: A[3][4]=1 → in-degree = 1
- Node 5: No incoming edges → in-degree = 0

Node 5 has in-degree 0, so it must come first in any valid topological ordering.

The answer is 5.

---

### Example 2: Quickselect (Median)

**Input (Code):**
```python
def solution():
    key = [0.963, 0.383, 0.791, 0.528, 0.568, 0.925, 0.071, 0.087]
    # Find median
```

**Output (Reasoning):**
To find the median, I first mentally sort the 8 numbers.

Sorted: [0.071, 0.087, 0.383, 0.528, 0.568, 0.791, 0.925, 0.963]

With 8 elements (even count), the median is the average of the 4th and 5th values.
- 4th value: 0.528
- 5th value: 0.568

Median = (0.528 + 0.568) / 2 = 1.096 / 2 = 0.548

The answer is 0.548.

---

### Example 3: Longest Common Subsequence

**Input (Code):**
```python
def solution():
    S = "dxkxwqnq"
    T = "bcclfxzv"
    # Find LCS length
```

**Output (Reasoning):**
I need to find the longest subsequence common to both strings.

Looking at S = "dxkxwqnq" and T = "bcclfxzv", let me identify common characters:
- 'x' appears in both: S has 'x' at positions 1, 3; T has 'x' at position 5

Are there other common characters? Going through: d, k, w, q, n don't appear in T. And b, c, l, f, z, v don't appear in S.

So the only common character is 'x'. The LCS is just "x" with length 1.

The answer is 1.

---

### Example 4: DAG Shortest Paths

**Input (Code):**
```python
def solution():
    s = 1  # source node
    A = [[0.0, 0.0, 0.0, 0.0], [0.995, 0.0, 0.414, 0.474],
         [0.623, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    # Find shortest path distance from s to node 0
```

**Output (Reasoning):**
Starting from node 1, I need to find the shortest path to reach node 0.

From the adjacency matrix, node 1 has edges to:
- Node 0 with weight 0.995
- Node 2 with weight 0.414
- Node 3 with weight 0.474

Node 2 has an edge to node 0 with weight 0.623.

So I have two paths from node 1 to node 0:
1. Direct: 1 → 0 with total weight 0.995
2. Via node 2: 1 → 2 → 0 with total weight 0.414 + 0.623 = 1.037

The direct path is shorter. The shortest distance is 0.995.

The answer is 0.995.

---

### Example 5: Binary Search

**Input (Code):**
```python
def solution():
    key = [0.02, 0.071, 0.383, 0.437, 0.544, 0.602, 0.715, 0.891]
    target = 0.5
    # Find insertion point
```

**Output (Reasoning):**
I need to find where 0.5 would be inserted to maintain sorted order.

Looking at the sorted list: [0.02, 0.071, 0.383, 0.437, 0.544, 0.602, 0.715, 0.891]

The value 0.5 falls between 0.437 (index 3) and 0.544 (index 4).

So the insertion point is index 4.

The answer is 4.

---

## Your Task

Now solve the following problem using the same natural reasoning approach. Given the code, understand the problem and explain your solution process conversationally. End with "The answer is X."

**Code:**
```python
{code}
```
'''

# Evaluation prompts
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


# Task types used in ICL examples (must be held out from evaluation)
ICL_EXAMPLE_TASKS = {
    "topological_sort",
    "quickselect",
    "lcs",
    "dag_shortest_paths",
    "binary_search",
}


def load_samples(
    results_dir: Path,
    source_model_filter: str | None = None,
    max_samples: int = 500,
    max_per_kind: int = 50,
    exclude_icl_tasks: bool = True,
) -> list[Sample]:
    """Load samples from exps_performance results.

    Args:
        exclude_icl_tasks: If True, excludes task types used in ICL examples
                          to ensure held-out evaluation.
    """
    samples_by_kind: dict[str, list[Sample]] = defaultdict(list)

    for jsonl_path in results_dir.glob("**/res.jsonl"):
        model_dir = jsonl_path.parent.parent.parent.name
        source_model = model_dir.rsplit("_seed", 1)[0]

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

                # Filter to simple answer formats only (no arrays/lists)
                if '[' in gold_answer or ',' in gold_answer:
                    continue

                # Exclude ICL example tasks for held-out evaluation
                if exclude_icl_tasks and kind in ICL_EXAMPLE_TASKS:
                    continue

                # Clean code
                if sim_code.startswith("```"):
                    sim_code = re.sub(r'^```\w*\n?', '', sim_code)
                    sim_code = re.sub(r'\n?```$', '', sim_code)

                samples_by_kind[kind].append(Sample(
                    kind=kind,
                    question=question,
                    nl_reasoning_native=nl_reasoning,
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
    print(f"Task types: {sorted(samples_by_kind.keys())}")
    if exclude_icl_tasks:
        print(f"(Held-out evaluation: excluded ICL tasks {ICL_EXAMPLE_TASKS})")
    print(f"(Filtered to simple answer formats only)")
    return samples[:max_samples]


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = answer.strip()
    answer = re.sub(r'^(Answer:|The answer is|Result:)\s*', '', answer, flags=re.IGNORECASE)
    match = re.search(r'-?\d+\.?\d*', answer)
    if match:
        return match.group()
    return answer.strip()


def check_correct(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

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
    max_tokens: int = 500,
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


async def translate_code_to_nl(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    samples: list[Sample],
    concurrency: int = 32,
) -> list[Sample]:
    """Translate code to NL for all samples."""
    sem = asyncio.Semaphore(concurrency)

    async def translate_one(sample: Sample) -> Sample:
        async with sem:
            prompt = TRANSLATE_PROMPT.format(code=sample.sim_code[:3000])
            translation = await call_llm_async(client, api_key, model, prompt, max_tokens=1000)
            sample.nl_reasoning_translated = translation
            return sample

    print(f"Translating {len(samples)} code samples to NL...")
    tasks = [translate_one(s) for s in samples]
    translated = await tqdm_asyncio.gather(*tasks, desc="Translating")

    # Filter out failed translations
    valid = [s for s in translated if s.nl_reasoning_translated and len(s.nl_reasoning_translated) > 50]
    print(f"Successfully translated {len(valid)}/{len(samples)} samples")
    return valid


async def run_evaluation(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    samples: list[Sample],
    concurrency: int = 32,
) -> list[Trial]:
    """Run evaluation for all conditions."""
    sem = asyncio.Semaphore(concurrency)
    conditions = ["x", "x_nl_native", "x_nl_translated"]

    async def eval_one(sample: Sample, condition: str) -> Trial:
        async with sem:
            if condition == "x":
                prompt = CONDITION_X.format(question=sample.question[:2000])
            elif condition == "x_nl_native":
                prompt = CONDITION_X_NL.format(
                    question=sample.question[:2000],
                    nl_reasoning=sample.nl_reasoning_native[:2000],
                )
            elif condition == "x_nl_translated":
                prompt = CONDITION_X_NL.format(
                    question=sample.question[:2000],
                    nl_reasoning=sample.nl_reasoning_translated[:2000],
                )
            else:
                raise ValueError(f"Unknown condition: {condition}")

            response = await call_llm_async(client, api_key, model, prompt, max_tokens=100)
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

    all_tasks = []
    for sample in samples:
        for condition in conditions:
            all_tasks.append(eval_one(sample, condition))

    print(f"Running {len(all_tasks)} evaluation trials...")
    trials = await tqdm_asyncio.gather(*all_tasks, desc="Evaluating")
    return trials


def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if n_total == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = (z / denom) * ((p * (1 - p) / n_total + z**2 / (4 * n_total**2)) ** 0.5)
    return max(0, center - margin), min(1, center + margin)


def print_results(trials: list[Trial], model: str):
    """Print results summary."""
    conditions = ["x", "x_nl_native", "x_nl_translated"]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Total samples: {len(trials) // 3}")
    print()

    print(f"{'Condition':<20} {'Accuracy':>10} {'95% CI':>22} {'N':>6}")
    print("-" * 62)

    results = {}
    for cond in conditions:
        cond_trials = [t for t in trials if t.condition == cond]
        n_total = len(cond_trials)
        n_correct = sum(1 for t in cond_trials if t.correct)
        acc = n_correct / n_total if n_total > 0 else 0
        ci_low, ci_high = wilson_ci(n_correct, n_total)
        results[cond] = acc
        ci_str = f"[{ci_low:.1%}, {ci_high:.1%}]"
        print(f"{cond:<20} {acc:>10.1%} {ci_str:>22} {n_total:>6}")

    print()
    print("INTERPRETATION:")
    print(f"  Baseline (x only):     {results['x']:.1%}")
    print(f"  + Native NL:           {results['x_nl_native']:.1%} (Δ = {results['x_nl_native'] - results['x']:+.1%})")
    print(f"  + Translated NL:       {results['x_nl_translated']:.1%} (Δ = {results['x_nl_translated'] - results['x']:+.1%})")
    print()

    gap = results['x_nl_native'] - results['x_nl_translated']
    if abs(gap) < 0.03:
        print("→ Translated NL ≈ Native NL (translation preserves information)")
    elif gap > 0.03:
        print(f"→ Native NL > Translated NL by {gap:.1%} (translation loses information)")
    else:
        print(f"→ Translated NL > Native NL by {-gap:.1%} (code contains extra information)")

    return results


async def main_async(args):
    """Main async entry point."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    full_model = MODEL_MAP.get(args.model, args.model)
    if not full_model.startswith(("openai/", "anthropic/", "google/", "meta/", "mistral")):
        full_model = f"openai/{args.model}"

    print(f"Using model: {full_model}")

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

    async with httpx.AsyncClient() as client:
        # Step 1: Translate code to NL
        samples = await translate_code_to_nl(client, api_key, full_model, samples, args.concurrency)

        if not samples:
            print("No valid translations!")
            return

        # Step 2: Run evaluation
        trials = await run_evaluation(client, api_key, full_model, samples, args.concurrency)

    # Print and save results
    results = print_results(trials, args.model)

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save trials
    trials_path = OUTPUT_DIR / f"translation_{args.model}_{timestamp}_trials.jsonl"
    with trials_path.open("w") as f:
        for t in trials:
            f.write(json.dumps(asdict(t)) + "\n")

    # Save summary
    summary_path = OUTPUT_DIR / f"translation_{args.model}_{timestamp}.json"
    with summary_path.open("w") as f:
        json.dump({
            "model": args.model,
            "n_samples": len(samples),
            "results": results,
            "timestamp": timestamp,
        }, f, indent=2)

    print(f"\nResults saved to: {summary_path}")
    print(f"Trials saved to: {trials_path}")


def main():
    parser = argparse.ArgumentParser(description="Translation Additivity Experiment")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model for translation and evaluation")
    parser.add_argument("--source_model", default=None, help="Filter source data by model")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--max_per_kind", type=int, default=20, help="Max samples per task type")
    parser.add_argument("--concurrency", type=int, default=32, help="API concurrency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
