#!/usr/bin/env python3
"""
Source-Label Discrimination Test.

Instead of pairwise SAME/DIFFERENT, we directly ask:
"Did this trace come from the native-NL channel or the translated-NL channel?"

This directly estimates distinguishability between joint distributions p(x)p(z|x).

Dataset construction (balanced):
- label 0: z ~ p_NL(·|x)      (native NL explanation)
- label 1: z ~ p_translated(·|x)  (code-to-NL translation)

Reports:
- Accuracy with confidence intervals
- Positive controls to show judge has discriminative power
- If accuracy ≈ 50% and controls succeed → evidence for indistinguishability

Usage:
    uv run python -m src.translation_discrimination.cli --n_samples 200
"""

# mypy: ignore-errors

import argparse
import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from scipy import stats
from tqdm.asyncio import tqdm_asyncio

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_RESULTS_DIR

load_dotenv()

# Paths
RESULTS_DIR = LEGACY_BENCHMARK_RESULTS_DIR
OUTPUT_DIR = Path(__file__).parent / "results"
PROMPTS_DIR = Path(__file__).parent / "prompts"
CLASSIFIER_PROMPT_PATH = PROMPTS_DIR / "source_classifier.md"
TRANSLATOR_PROMPT_PATH = Path(__file__).parent / "prompts" / "translator_native_10shot.md"

# Models (defaults, can be overridden via CLI)
DEFAULT_TRANSLATOR_MODEL = "openai/gpt-4o"
DEFAULT_JUDGE_MODEL = "google/gemini-2.5-pro-preview-06-05"
BASE_URL = "https://openrouter.ai/api/v1"

# Global model settings (set by main())
TRANSLATOR_MODEL = DEFAULT_TRANSLATOR_MODEL
JUDGE_MODEL = DEFAULT_JUDGE_MODEL
REASONING_EFFORT = None  # None, "low", "medium", "high"


@dataclass
class Sample:
    """A sample for source discrimination."""

    kind: str
    question: str
    original_nl: str
    sim_code: str
    model: str = ""
    translated_nl: str = ""


@dataclass
class Trial:
    """A single discrimination trial."""

    kind: str
    question: str
    trace: str  # The explanation shown to judge
    true_label: int  # 0 = native NL, 1 = translated
    predicted_label: int  # Judge's prediction
    judge_confidence: str  # "high", "medium", "low"
    judge_reasoning: str
    correct: bool


@dataclass
class ControlTrial:
    """A positive control trial (easy cases)."""

    control_type: str  # "code_vs_nl" or "random_vs_nl"
    trace: str
    true_label: int  # 0 = NL, 1 = obviously different
    predicted_label: int
    correct: bool


@dataclass
class Results:
    """Experiment results."""

    n_trials: int
    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    # AUC (if we have confidence scores)
    auc: float
    # Confusion matrix
    confusion_matrix: dict  # {"tp": x, "tn": x, "fp": x, "fn": x}
    # By kind
    by_kind: dict
    # Positive controls
    control_accuracy: float
    control_n: int
    # Interpretation
    judge_has_power: bool  # True if control accuracy > 70%
    indistinguishable: bool  # True if accuracy CI contains 50%


def load_classifier_prompt() -> str:
    """Load the source classifier prompt."""
    if CLASSIFIER_PROMPT_PATH.exists():
        return CLASSIFIER_PROMPT_PATH.read_text().strip()
    raise FileNotFoundError(f"Classifier prompt not found at {CLASSIFIER_PROMPT_PATH}")


def load_translator_prompt() -> str:
    """Load the translation prompt."""
    if TRANSLATOR_PROMPT_PATH.exists():
        return TRANSLATOR_PROMPT_PATH.read_text().strip()
    raise FileNotFoundError(f"Translator prompt not found at {TRANSLATOR_PROMPT_PATH}")


def load_samples(
    results_dir: Path,
    max_samples: int = 100,
    kind_filter: Optional[set[str]] = None,
    max_per_kind: Optional[int] = None,
) -> list[Sample]:
    """Load samples with both code and NL reasoning.

    Loads samples ensuring diversity across kinds by collecting all valid
    samples first, then applying per-kind limits.
    """
    # First pass: collect all valid samples grouped by kind
    samples_by_kind: dict[str, list[Sample]] = defaultdict(list)

    for jsonl_path in results_dir.glob("**/res.jsonl"):
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
                # Use question field, fall back to nl_question (CLRS tasks use nl_question)
                question = row.get("question", "") or row.get("nl_question", "")
                kind = row.get("kind", "unknown")

                if not sim_code or not nl_reasoning or not question:
                    continue
                if len(nl_reasoning) < 100 or len(sim_code) < 50:
                    continue

                if kind_filter is not None and kind not in kind_filter:
                    continue

                samples_by_kind[kind].append(
                    Sample(
                        kind=kind,
                        question=question,
                        original_nl=nl_reasoning,
                        sim_code=sim_code,
                        model=model_name,
                    )
                )

    # Second pass: apply per-kind limits and collect final samples
    samples = []
    for kind, kind_samples in samples_by_kind.items():
        random.shuffle(kind_samples)
        limit = max_per_kind if max_per_kind else len(kind_samples)
        samples.extend(kind_samples[:limit])

    # Shuffle and apply total limit
    random.shuffle(samples)
    return samples[:max_samples]


async def call_llm_async(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 500,
    temperature: float = 0.7,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Call an LLM via OpenRouter.

    Args:
        reasoning_effort: Optional reasoning effort level ("low", "medium", "high").
            Only applies to models that support it (e.g., Claude with extended thinking,
            OpenAI o-series models).
    """
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

    # Add reasoning effort if specified (OpenRouter unified reasoning parameter)
    # See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    if reasoning_effort:
        payload["reasoning"] = {
            "effort": reasoning_effort,
        }

    resp = await client.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    if resp.status_code != 200:
        error_detail = resp.text[:500] if resp.text else "No error details"
        raise httpx.HTTPStatusError(f"{resp.status_code}: {error_detail}", request=resp.request, response=resp)
    return resp.json()["choices"][0]["message"]["content"].strip()


async def translate_code_async(
    client: httpx.AsyncClient,
    api_key: str,
    translator_prompt: str,
    code: str,
    question: str = "",
) -> str:
    """Translate code to NL explanation."""
    user_content = f"**Problem:** {question}\n\n**Code:**\n```python\n{code}\n```" if question else f"```python\n{code}\n```"
    messages = [
        {"role": "system", "content": translator_prompt},
        {"role": "user", "content": user_content},
    ]
    return await call_llm_async(client, api_key, TRANSLATOR_MODEL, messages, max_tokens=1500, temperature=0.7)


async def classify_source_async(
    client: httpx.AsyncClient,
    api_key: str,
    classifier_prompt: str,
    question: str,
    trace: str,
) -> tuple[int, str, str]:
    """
    Ask judge to classify: native NL (0) or translated (1)?
    Returns: (predicted_label, confidence, reasoning)
    """
    prompt = classifier_prompt.format(question=question[:500], trace=trace[:2000])

    response = await call_llm_async(
        client,
        api_key,
        JUDGE_MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.1,
        reasoning_effort=REASONING_EFFORT,
    )

    # Parse response
    predicted_label = -1
    confidence = "medium"
    reasoning = ""

    response_upper = response.upper()

    for line in response.split("\n"):
        line_upper = line.upper()
        if "PREDICTION:" in line_upper or "SOURCE:" in line_upper:
            if "NATIVE" in line_upper or "ORIGINAL" in line_upper or "0" in line:
                predicted_label = 0
            elif "TRANSLATED" in line_upper or "CODE" in line_upper or "1" in line:
                predicted_label = 1
        elif "CONFIDENCE:" in line_upper:
            if "HIGH" in line_upper:
                confidence = "high"
            elif "LOW" in line_upper:
                confidence = "low"
            else:
                confidence = "medium"
        elif "REASONING:" in line_upper:
            reasoning = line.split(":", 1)[-1].strip()

    # Fallback parsing
    if predicted_label == -1:
        if "NATIVE" in response_upper or "ORIGINAL" in response_upper:
            predicted_label = 0
        elif "TRANSLATED" in response_upper:
            predicted_label = 1
        else:
            predicted_label = random.randint(0, 1)  # Random guess if unclear

    return predicted_label, confidence, reasoning or response[:200]


def compute_accuracy_ci(correct: list[bool], confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute accuracy with Wilson score confidence interval."""
    n = len(correct)
    if n == 0:
        return 0.0, 0.0, 0.0

    k = sum(correct)
    p_hat = k / n

    # Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    return p_hat, max(0, center - margin), min(1, center + margin)


async def run_experiment(
    api_key: str,
    samples: list[Sample],
    translator_prompt: str,
    classifier_prompt: str,
    concurrency: int = 32,
) -> tuple[list[Trial], list[ControlTrial], Results]:
    """Run source discrimination experiment."""
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=120) as client:
        # Phase 1: Translate all samples
        print(f"\n[1/3] Translating {len(samples)} samples...")

        async def translate_one(sample: Sample) -> tuple[Sample, str, Optional[str]]:
            async with semaphore:
                try:
                    translated = await translate_code_async(client, api_key, translator_prompt, sample.sim_code, sample.question)
                    return sample, translated, None
                except Exception as e:
                    return sample, "", str(e)

        tasks = [translate_one(s) for s in samples]
        results = await tqdm_asyncio.gather(*tasks, desc="Translating")

        for sample, translated, error in results:
            sample.translated_nl = translated
            if error:
                print(f"Translation failed ({sample.kind}): {error}")

        valid_samples = [s for s in samples if s.translated_nl]
        print(f"      {len(valid_samples)}/{len(samples)} translated successfully")

        # Phase 2: Create balanced trial set
        print("\n[2/3] Running discrimination trials...")

        # Each sample contributes two trials: one native, one translated
        trial_inputs = []
        for sample in valid_samples:
            # Native NL trial (label 0)
            trial_inputs.append(
                {
                    "sample": sample,
                    "trace": sample.original_nl,
                    "true_label": 0,
                }
            )
            # Translated trial (label 1)
            trial_inputs.append(
                {
                    "sample": sample,
                    "trace": sample.translated_nl,
                    "true_label": 1,
                }
            )

        random.shuffle(trial_inputs)

        async def run_trial(inp: dict) -> tuple[Trial, Optional[str]]:
            sample = inp["sample"]
            async with semaphore:
                try:
                    pred, conf, reasoning = await classify_source_async(client, api_key, classifier_prompt, sample.question, inp["trace"])
                    return Trial(
                        kind=sample.kind,
                        question=sample.question[:200],
                        trace=inp["trace"][:500],
                        true_label=inp["true_label"],
                        predicted_label=pred,
                        judge_confidence=conf,
                        judge_reasoning=reasoning,
                        correct=(pred == inp["true_label"]),
                    ), None
                except Exception as e:
                    return Trial(
                        kind=sample.kind,
                        question=sample.question[:200],
                        trace=inp["trace"][:500],
                        true_label=inp["true_label"],
                        predicted_label=-1,
                        judge_confidence="",
                        judge_reasoning="",
                        correct=False,
                    ), str(e)

        trial_tasks = [run_trial(inp) for inp in trial_inputs]
        trial_results = await tqdm_asyncio.gather(*trial_tasks, desc="Classifying")

        trials = []
        for trial, error in trial_results:
            if error:
                print(f"Trial failed: {error}")
            if trial.predicted_label != -1:
                trials.append(trial)

        # Phase 3: Positive controls (easy cases)
        print("\n[3/3] Running positive controls...")

        # Control: Raw code vs NL explanation (should be easy to distinguish)
        control_inputs = []
        for sample in valid_samples[:50]:  # Use subset for controls
            # NL explanation (label 0)
            control_inputs.append(
                {
                    "sample": sample,
                    "trace": sample.original_nl,
                    "true_label": 0,
                    "control_type": "code_vs_nl",
                }
            )
            # Raw code (label 1 - obviously different)
            control_inputs.append(
                {
                    "sample": sample,
                    "trace": f"```python\n{sample.sim_code}\n```",
                    "true_label": 1,
                    "control_type": "code_vs_nl",
                }
            )

        random.shuffle(control_inputs)

        async def run_control(inp: dict) -> tuple[ControlTrial, Optional[str]]:
            sample = inp["sample"]
            async with semaphore:
                try:
                    pred, _, _ = await classify_source_async(client, api_key, classifier_prompt, sample.question, inp["trace"])
                    return ControlTrial(
                        control_type=inp["control_type"],
                        trace=inp["trace"][:500],
                        true_label=inp["true_label"],
                        predicted_label=pred,
                        correct=(pred == inp["true_label"]),
                    ), None
                except Exception as e:
                    return ControlTrial(
                        control_type=inp["control_type"],
                        trace=inp["trace"][:500],
                        true_label=inp["true_label"],
                        predicted_label=-1,
                        correct=False,
                    ), str(e)

        control_tasks = [run_control(inp) for inp in control_inputs]
        control_results = await tqdm_asyncio.gather(*control_tasks, desc="Controls")

        controls = []
        for ctrl, error in control_results:
            if error:
                print(f"Control failed: {error}")
            if ctrl.predicted_label != -1:
                controls.append(ctrl)

        # Compute results
        accuracy, ci_low, ci_high = compute_accuracy_ci([t.correct for t in trials])

        # Confusion matrix: true_label=0 is Native, true_label=1 is Translated
        # TP = correctly identified as Translated (true=1, pred=1)
        # TN = correctly identified as Native (true=0, pred=0)
        # FP = Native misclassified as Translated (true=0, pred=1)
        # FN = Translated misclassified as Native (true=1, pred=0)
        tp = sum(1 for t in trials if t.true_label == 1 and t.predicted_label == 1)
        tn = sum(1 for t in trials if t.true_label == 0 and t.predicted_label == 0)
        fp = sum(1 for t in trials if t.true_label == 0 and t.predicted_label == 1)
        fn = sum(1 for t in trials if t.true_label == 1 and t.predicted_label == 0)
        confusion = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

        # AUC approximation using confidence levels as scores
        # Map confidence to pseudo-probability for translated
        conf_to_score = {"high": 0.9, "medium": 0.6, "low": 0.55}
        y_true = []
        y_score = []
        for t in trials:
            y_true.append(t.true_label)
            base_score = conf_to_score.get(t.judge_confidence, 0.5)
            # If predicted translated, score is base; if predicted native, score is 1-base
            if t.predicted_label == 1:
                y_score.append(base_score)
            else:
                y_score.append(1 - base_score)

        # Compute AUC
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_true, y_score)
        except Exception:
            # Fallback: estimate from accuracy
            auc = accuracy

        # By kind
        by_kind = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
        for trial in trials:
            by_kind[trial.kind]["total"] += 1
            if trial.correct:
                by_kind[trial.kind]["correct"] += 1
            if trial.true_label == 1 and trial.predicted_label == 1:
                by_kind[trial.kind]["tp"] += 1
            elif trial.true_label == 0 and trial.predicted_label == 0:
                by_kind[trial.kind]["tn"] += 1
            elif trial.true_label == 0 and trial.predicted_label == 1:
                by_kind[trial.kind]["fp"] += 1
            elif trial.true_label == 1 and trial.predicted_label == 0:
                by_kind[trial.kind]["fn"] += 1

        kind_stats = {}
        for kind, data in by_kind.items():
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0
            acc_val, acc_ci_low, acc_ci_high = compute_accuracy_ci([t.correct for t in trials if t.kind == kind])
            kind_stats[kind] = {
                "accuracy": acc,
                "accuracy_ci": [acc_ci_low, acc_ci_high],
                "total": data["total"],
                "confusion": {"tp": data["tp"], "tn": data["tn"], "fp": data["fp"], "fn": data["fn"]},
            }

        # Control accuracy
        control_correct = [c.correct for c in controls]
        control_accuracy = sum(control_correct) / len(control_correct) if control_correct else 0

        # Interpretation
        judge_has_power = bool(control_accuracy > 0.70)
        indistinguishable = bool(ci_low <= 0.50 <= ci_high)

        results = Results(
            n_trials=len(trials),
            accuracy=accuracy,
            accuracy_ci_low=ci_low,
            accuracy_ci_high=ci_high,
            auc=auc,
            confusion_matrix=confusion,
            by_kind=kind_stats,
            control_accuracy=control_accuracy,
            control_n=len(controls),
            judge_has_power=judge_has_power,
            indistinguishable=indistinguishable,
        )

        return trials, controls, results


def print_results(results: Results):
    """Print results summary."""
    print("\n" + "=" * 70)
    print("SOURCE DISCRIMINATION RESULTS")
    print("=" * 70)

    print("\nMAIN TEST (native vs translated):")
    print(f"  Accuracy: {results.accuracy*100:.1f}% [{results.accuracy_ci_low*100:.1f}%, {results.accuracy_ci_high*100:.1f}%]")
    print(f"  AUC:      {results.auc:.3f}")
    print(f"  N trials: {results.n_trials}")

    print("\nCONFUSION MATRIX:")
    cm = results.confusion_matrix
    print("                    Predicted")
    print("                 Native  Translated")
    print(f"  Actual Native    {cm['tn']:4d}      {cm['fp']:4d}")
    print(f"  Actual Translated{cm['fn']:4d}      {cm['tp']:4d}")

    # Precision/Recall for detecting translated
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0
    print(f"\n  Precision (Translated): {precision*100:.1f}%")
    print(f"  Recall (Translated):    {recall*100:.1f}%")

    print("\nPOSITIVE CONTROLS (NL vs raw code):")
    print(f"  Accuracy: {results.control_accuracy*100:.1f}% (n={results.control_n})")
    print(f"  Judge has power: {'YES' if results.judge_has_power else 'NO'}")

    print("\nINTERPRETATION:")
    if results.judge_has_power and results.indistinguishable:
        print("  ✓ STRONG EVIDENCE: Judge has power but cannot distinguish")
        print("    → Supports Assumption 2 (indistinguishability)")
    elif results.judge_has_power and not results.indistinguishable:
        print("  ✗ DISTINGUISHABLE: Judge can tell them apart")
        print("    → Assumption 2 may not hold")
    elif not results.judge_has_power:
        print("  ? INCONCLUSIVE: Judge lacks discriminative power")
        print("    → Cannot draw conclusions")

    print("\nBY PROBLEM KIND:")
    print(f"  {'Kind':<20} {'Acc':>6} {'95% CI':>16} {'N':>5}  {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3}")
    print(f"  {'-'*20} {'-'*6} {'-'*16} {'-'*5}  {'-'*3} {'-'*3} {'-'*3} {'-'*3}")
    for kind in sorted(results.by_kind.keys()):
        data = results.by_kind[kind]
        ci = data.get("accuracy_ci", [0, 0])
        cm = data.get("confusion", {})
        print(
            f"  {kind:<20} {data['accuracy']*100:5.1f}% [{ci[0]*100:4.1f}%, {ci[1]*100:4.1f}%] {data['total']:5d}  {cm.get('tp',0):3d} {cm.get('tn',0):3d} {cm.get('fp',0):3d} {cm.get('fn',0):3d}"
        )


def save_results(
    trials: list[Trial],
    controls: list[ControlTrial],
    results: Results,
    output_path: Path,
):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "results": asdict(results),
        "trials": [asdict(t) for t in trials],
        "controls": [asdict(c) for c in controls],
    }

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")


def main():
    global TRANSLATOR_MODEL, JUDGE_MODEL, REASONING_EFFORT

    parser = argparse.ArgumentParser(description="Source Discrimination Test")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples (each gives 2 trials)")
    parser.add_argument("--kinds", type=str, default=None, help="Comma-separated kinds to include")
    parser.add_argument("--max_per_kind", type=int, default=None, help="Max samples per kind")
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrent requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument(
        "--translator_model", type=str, default=DEFAULT_TRANSLATOR_MODEL, help=f"Translator model (default: {DEFAULT_TRANSLATOR_MODEL})"
    )
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL, help=f"Judge/discriminator model (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument(
        "--reasoning_effort", type=str, default=None, choices=["low", "medium", "high"], help="Reasoning effort for judge model (low/medium/high)"
    )
    args = parser.parse_args()

    # Set global model configuration
    TRANSLATOR_MODEL = args.translator_model
    JUDGE_MODEL = args.judge_model
    REASONING_EFFORT = args.reasoning_effort

    kind_filter = set(args.kinds.split(",")) if args.kinds else None
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output) if args.output else OUTPUT_DIR / f"source_discrimination_{timestamp}.json"

    print("=" * 60)
    print("SOURCE DISCRIMINATION TEST")
    print("=" * 60)
    print(f"Samples: {args.n_samples} (→ {args.n_samples * 2} trials)")
    print(f"Translator: {TRANSLATOR_MODEL}")
    print(f"Judge: {JUDGE_MODEL}")
    if REASONING_EFFORT:
        print(f"Reasoning effort: {REASONING_EFFORT}")
    print(f"Concurrency: {args.concurrency}")

    # Load prompts
    translator_prompt = load_translator_prompt()
    classifier_prompt = load_classifier_prompt()
    print(f"Translator prompt: {len(translator_prompt)} chars")
    print(f"Classifier prompt: {len(classifier_prompt)} chars")

    # Load samples
    print("\nLoading samples...")
    samples = load_samples(RESULTS_DIR, args.n_samples, kind_filter, args.max_per_kind)
    random.shuffle(samples)
    print(f"Loaded {len(samples)} samples")
    print(f"Kinds: {set(s.kind for s in samples)}")
    print(f"Models: {set(s.model for s in samples)}")

    # Run
    trials, controls, results = asyncio.run(run_experiment(api_key, samples, translator_prompt, classifier_prompt, args.concurrency))

    print_results(results)
    save_results(trials, controls, results, output_file)


if __name__ == "__main__":
    main()
