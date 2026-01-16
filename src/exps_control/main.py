#!/usr/bin/env python3
"""
Prefilled CoT Comparison Experiment.

This experiment tests whether translated code reasoning performs similarly
to original NL reasoning when prefilled as Chain-of-Thought.

Experiment design:
1. Load existing results with generated code (sim_code) and NL reasoning (nl_reasoning)
2. Translate code to NL using GPT 5.2 with the consistency-optimized prompt
3. Create two conditions:
   - Condition A: Original problem + original NL reasoning (prefilled) → answer
   - Condition B: Original problem + translated code-to-NL (prefilled) → answer
4. Compare accuracy between conditions

Usage:
    uv run python src/exps_control/main.py --n_samples 50 --model claude-sonnet-4
"""

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

from src.exps_control.data_loader import Sample, get_unique_kinds, load_samples, sample_per_kind
from src.exps_control.translator import CodeToNLTranslator

load_dotenv()

# Default paths
RESULTS_DIR = Path(__file__).parent.parent / "exps_performance" / "results"
OUTPUT_DIR = Path(__file__).parent / "results"


def create_prefilled_prompt(question: str, reasoning: str, answer_format: str) -> str:
    """Create a prompt with prefilled reasoning (simulating assistant turn).

    Args:
        question: The original problem question
        reasoning: The reasoning to prefill (NL or translated)
        answer_format: Format instructions for the final answer

    Returns:
        The complete prompt with prefilled reasoning
    """
    return f"""{question}

Here is my reasoning so far:
{reasoning}

Based on the reasoning above, provide the final answer.
{answer_format}"""


def extract_answer(response: str) -> str:
    """Extract the answer from a model response."""
    # Try to find JSON-like answer
    import re

    # Look for common answer patterns
    patterns = [
        r'"answer"\s*:\s*"?([^"}\n]+)"?',
        r'"result"\s*:\s*"?([^"}\n]+)"?',
        r"answer[:\s]+([^\n]+)",
        r"result[:\s]+([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: return last line or first number found
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if lines:
        return lines[-1]

    return response.strip()


def check_answer(predicted: str, gold: str) -> bool:
    """Check if the predicted answer matches the gold answer."""
    # Normalize both
    pred_norm = str(predicted).strip().lower()
    gold_norm = str(gold).strip().lower()

    # Direct match
    if pred_norm == gold_norm:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred_norm.replace(",", ""))
        gold_num = float(gold_norm.replace(",", ""))
        return abs(pred_num - gold_num) < 1e-6
    except (ValueError, TypeError):
        pass

    # Check if gold is contained in prediction
    if gold_norm in pred_norm:
        return True

    return False


class PrefillExperiment:
    """Run the prefilled CoT comparison experiment."""

    def __init__(
        self,
        evaluator_model: str = "openai/gpt-4o",
        translator_model: str = "openai/gpt-5.2",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.evaluator_model = evaluator_model
        self.translator_model = translator_model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        self.translator = CodeToNLTranslator(
            model=translator_model,
            api_key=self.api_key,
            base_url=base_url,
        )

    def call_llm(self, prompt: str, timeout: float = 60.0) -> str:
        """Call the evaluator LLM to get the final answer."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.evaluator_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    def run_single(self, sample: Sample, answer_format: str = "Provide only the final numeric answer.") -> Sample:
        """Run experiment on a single sample.

        Args:
            sample: The sample to process
            answer_format: Format instructions for the answer

        Returns:
            Updated sample with results
        """
        # Step 1: Translate code to NL if not already done
        if not sample.translated_reasoning and sample.sim_code:
            sample.translated_reasoning = self.translator.translate(sample.sim_code)

        # Step 2: Run with original NL reasoning prefilled
        if sample.nl_reasoning:
            nl_prompt = create_prefilled_prompt(sample.question, sample.nl_reasoning, answer_format)
            nl_response = self.call_llm(nl_prompt)
            sample.nl_prefill_answer = extract_answer(nl_response)
            sample.nl_prefill_correct = check_answer(sample.nl_prefill_answer, sample.answer)

        # Step 3: Run with translated reasoning prefilled
        if sample.translated_reasoning:
            trans_prompt = create_prefilled_prompt(sample.question, sample.translated_reasoning, answer_format)
            trans_response = self.call_llm(trans_prompt)
            sample.translated_prefill_answer = extract_answer(trans_response)
            sample.translated_prefill_correct = check_answer(sample.translated_prefill_answer, sample.answer)

        return sample

    def run_batch(
        self,
        samples: list[Sample],
        answer_format: str = "Provide only the final numeric answer.",
        progress_interval: int = 10,
    ) -> list[Sample]:
        """Run experiment on a batch of samples.

        Args:
            samples: List of samples to process
            answer_format: Format instructions for the answer
            progress_interval: How often to print progress

        Returns:
            List of updated samples with results
        """
        print(f"\n[1] Translating {len(samples)} code samples...")
        for i, sample in enumerate(samples):
            if (i + 1) % progress_interval == 0:
                print(f"    {i + 1}/{len(samples)} translated...")
            if sample.sim_code and not sample.translated_reasoning:
                try:
                    sample.translated_reasoning = self.translator.translate(sample.sim_code)
                except Exception as e:
                    print(f"    Warning: Translation failed for {sample.kind}: {e}")
                    sample.translated_reasoning = ""
        print(f"    Done! {len(samples)} translated.")

        print("\n[2] Running prefilled CoT evaluation...")
        for i, sample in enumerate(samples):
            if (i + 1) % progress_interval == 0:
                nl_acc = sum(1 for s in samples[:i+1] if s.nl_prefill_correct) / (i + 1) * 100
                trans_acc = sum(1 for s in samples[:i+1] if s.translated_prefill_correct) / (i + 1) * 100
                print(f"    {i + 1}/{len(samples)} evaluated... NL: {nl_acc:.1f}%, Translated: {trans_acc:.1f}%")

            # Run with NL prefill
            if sample.nl_reasoning:
                try:
                    nl_prompt = create_prefilled_prompt(sample.question, sample.nl_reasoning, answer_format)
                    nl_response = self.call_llm(nl_prompt)
                    sample.nl_prefill_answer = extract_answer(nl_response)
                    sample.nl_prefill_correct = check_answer(sample.nl_prefill_answer, sample.answer)
                except Exception as e:
                    print(f"    Warning: NL evaluation failed for {sample.kind}: {e}")

            # Run with translated prefill
            if sample.translated_reasoning:
                try:
                    trans_prompt = create_prefilled_prompt(sample.question, sample.translated_reasoning, answer_format)
                    trans_response = self.call_llm(trans_prompt)
                    sample.translated_prefill_answer = extract_answer(trans_response)
                    sample.translated_prefill_correct = check_answer(sample.translated_prefill_answer, sample.answer)
                except Exception as e:
                    print(f"    Warning: Translated evaluation failed for {sample.kind}: {e}")

        print("    Done!")
        return samples


def compute_statistics(samples: list[Sample]) -> dict:
    """Compute statistics from experiment results."""
    n_total = len(samples)
    n_nl = sum(1 for s in samples if s.nl_reasoning)
    n_translated = sum(1 for s in samples if s.translated_reasoning)

    nl_correct = sum(1 for s in samples if s.nl_prefill_correct)
    trans_correct = sum(1 for s in samples if s.translated_prefill_correct)

    # Agreement: both correct or both wrong
    both_evaluated = [s for s in samples if s.nl_reasoning and s.translated_reasoning]
    agreement = sum(1 for s in both_evaluated if s.nl_prefill_correct == s.translated_prefill_correct)

    return {
        "n_samples": n_total,
        "n_with_nl": n_nl,
        "n_with_translated": n_translated,
        "nl_accuracy": nl_correct / n_nl if n_nl > 0 else 0,
        "nl_correct": nl_correct,
        "translated_accuracy": trans_correct / n_translated if n_translated > 0 else 0,
        "translated_correct": trans_correct,
        "agreement_rate": agreement / len(both_evaluated) if both_evaluated else 0,
        "n_both_evaluated": len(both_evaluated),
    }


def print_results(stats: dict, samples: list[Sample]) -> None:
    """Print experiment results."""
    print("\n" + "=" * 70)
    print("PREFILLED COT COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nSamples: {stats['n_samples']}")
    print(f"  With NL reasoning: {stats['n_with_nl']}")
    print(f"  With translated reasoning: {stats['n_with_translated']}")

    print("\nAccuracy:")
    print(f"  Original NL prefill:    {stats['nl_accuracy']*100:.1f}% ({stats['nl_correct']}/{stats['n_with_nl']})")
    print(f"  Translated prefill:     {stats['translated_accuracy']*100:.1f}% ({stats['translated_correct']}/{stats['n_with_translated']})")

    print(f"\nAgreement (both correct or both wrong): {stats['agreement_rate']*100:.1f}%")

    # Per-kind breakdown
    from collections import defaultdict
    by_kind = defaultdict(lambda: {"nl_correct": 0, "trans_correct": 0, "total": 0})
    for s in samples:
        by_kind[s.kind]["total"] += 1
        if s.nl_prefill_correct:
            by_kind[s.kind]["nl_correct"] += 1
        if s.translated_prefill_correct:
            by_kind[s.kind]["trans_correct"] += 1

    print("\nPer-kind breakdown:")
    for kind in sorted(by_kind.keys()):
        k = by_kind[kind]
        nl_acc = k["nl_correct"] / k["total"] * 100 if k["total"] > 0 else 0
        trans_acc = k["trans_correct"] / k["total"] * 100 if k["total"] > 0 else 0
        print(f"  {kind}: NL={nl_acc:.0f}%, Trans={trans_acc:.0f}% (n={k['total']})")


def save_results(samples: list[Sample], stats: dict, output_path: Path) -> None:
    """Save experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "samples": [asdict(s) for s in samples],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prefilled CoT Comparison Experiment")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to use")
    parser.add_argument("--n_per_kind", type=int, default=5, help="Max samples per problem kind")
    parser.add_argument("--model", type=str, default=None, help="Filter by source model (partial match)")
    parser.add_argument("--evaluator", type=str, default="openai/gpt-4o", help="Model to use for evaluation")
    parser.add_argument("--translator", type=str, default="openai/gpt-5.2", help="Model to use for translation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR), help="Path to results directory")

    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("PREFILLED COT COMPARISON EXPERIMENT")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Evaluator model: {args.evaluator}")
    print(f"  Translator model: {args.translator}")
    print(f"  Max samples: {args.n_samples}")
    print(f"  Max per kind: {args.n_per_kind}")
    print(f"  Model filter: {args.model or '(all)'}")
    print(f"  Seed: {args.seed}")

    # Load samples
    print(f"\nLoading samples from {args.results_dir}...")
    samples = load_samples(
        Path(args.results_dir),
        model_filter=args.model,
        require_code=True,
        require_nl=True,
    )
    print(f"  Found {len(samples)} samples with both code and NL")

    # Sample per kind
    if args.n_per_kind:
        samples = sample_per_kind(samples, args.n_per_kind)
        print(f"  Sampled to {len(samples)} ({args.n_per_kind} per kind)")

    # Limit total
    if len(samples) > args.n_samples:
        samples = random.sample(samples, args.n_samples)
        print(f"  Limited to {len(samples)} samples")

    kinds = get_unique_kinds(samples)
    print(f"  Problem kinds: {len(kinds)}")

    # Run experiment
    experiment = PrefillExperiment(
        evaluator_model=args.evaluator,
        translator_model=args.translator,
    )

    samples = experiment.run_batch(samples)

    # Compute and print results
    stats = compute_statistics(samples)
    print_results(stats, samples)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"prefill_comparison_{timestamp}.json"

    save_results(samples, stats, output_path)


if __name__ == "__main__":
    main()
