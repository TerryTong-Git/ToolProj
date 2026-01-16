#!/usr/bin/env python3
"""
Iterative adversarial prompt optimization.

Loop:
1. GPT 5.2 translates code → NL using current prompt
2. Opus 4.5 tries to distinguish original vs translated
3. GPT 5.2 receives Opus's feedback and optimizes the prompt
4. Repeat until Opus accuracy drops below threshold
"""

import json
import os
import random
from pathlib import Path
from datetime import datetime

import openai
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GPT_52_MODEL = "openai/gpt-5.2"
OPUS_MODEL = "anthropic/claude-opus-4"

# Target: Opus accuracy below this threshold
TARGET_ACCURACY = 0.55  # 55% = near random guessing (50% is chance)


def call_model(model: str, prompt: str, max_tokens: int = 1000) -> str:
    """Call a model via OpenRouter."""
    client = openai.OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 if "gpt" in model else 0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


def load_samples() -> list[dict]:
    """Load test samples with original NL reasoning."""
    results_path = Path(__file__).parent / "indistinguishability_results.json"
    with open(results_path, "r") as f:
        samples = json.load(f)
    # Filter to samples with non-empty original reasoning
    return [s for s in samples if s.get("original_nl_reasoning")]


def get_initial_prompt() -> str:
    """Load the mimic-style prompt (matches original structure without algorithm names)."""
    # Use the new mimic prompt that matches original style
    prompt_path = Path(__file__).parent / "icl_code_to_nl_mimic.md"
    if prompt_path.exists():
        print(f"    Loading mimic-style prompt")
        with open(prompt_path, "r") as f:
            return f.read()

    # Fallback to original
    prompt_path = Path(__file__).parent / "icl_code_to_nl_indistinguishable.md"
    with open(prompt_path, "r") as f:
        return f.read()


def translate_with_prompt(icl_prompt: str, code: str) -> str:
    """Use GPT 5.2 to translate code to NL using the ICL prompt.

    IMPORTANT: GPT 5.2 only sees the code, NOT the problem description.
    This prevents data leakage - the translator must infer the problem from code alone.
    """
    # Remove problem placeholder or replace with generic text
    # GPT 5.2 must figure out what the code does from the code itself
    full_prompt = icl_prompt.replace("[PROBLEM_DESCRIPTION]", "(Infer from the code below)").replace("[CODE_HERE]", code)
    return call_model(GPT_52_MODEL, full_prompt, max_tokens=600)


def test_opus_distinguishability(samples: list[dict], translations: list[str]) -> tuple[float, list[dict]]:
    """Test if Opus can distinguish original from translated. Returns accuracy and details."""
    results = []
    correct = 0
    total = 0

    for i, (sample, translated) in enumerate(zip(samples, translations)):
        original = sample.get("original_nl_reasoning", "")
        problem = sample.get("problem", "")[:300]

        if not translated or len(translated) < 50:
            continue

        # Randomize order
        random.seed(i * 7)
        if random.random() > 0.5:
            r1, r2 = original, translated
            gt_step, gt_natural = 1, 2
        else:
            r1, r2 = translated, original
            gt_step, gt_natural = 2, 1

        opus_prompt = f"""Two people solved this problem. One explains algorithms step-by-step (Type A). One thinks through it naturally (Type B).

Problem: {problem}...

Reasoning 1:
{r1[:600]}

Reasoning 2:
{r2[:600]}

Which is Type A (algorithmic explainer) and which is Type B (natural thinker)?
Answer format:
TYPE_A: [1 or 2]
TYPE_B: [1 or 2]
CONFIDENCE: [low/medium/high]
KEY_TELLS: [what gave it away]"""

        response = call_model(OPUS_MODEL, opus_prompt, max_tokens=300)

        # Parse response
        pred_a, pred_b = None, None
        for line in response.split("\n"):
            if "TYPE_A" in line.upper():
                pred_a = 1 if "1" in line else (2 if "2" in line else None)
            if "TYPE_B" in line.upper():
                pred_b = 1 if "1" in line else (2 if "2" in line else None)

        is_correct = (pred_a == gt_step and pred_b == gt_natural)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "sample_idx": i,
            "kind": sample.get("kind"),
            "correct": is_correct,
            "opus_response": response,
            "key_tells": response.split("KEY_TELLS:")[-1][:200] if "KEY_TELLS:" in response else ""
        })

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def optimize_prompt_with_feedback(current_prompt: str, opus_feedback: list[dict], iteration: int) -> str:
    """Ask GPT 5.2 to optimize the prompt based on Opus's detection patterns."""

    # Collect Opus's key tells
    key_tells = [r.get("key_tells", "") for r in opus_feedback if r.get("correct")]
    tells_summary = "\n".join([f"- {t[:200]}" for t in key_tells[:5] if t])

    # Count accuracy
    correct_count = len([r for r in opus_feedback if r.get("correct")])
    total_count = len(opus_feedback)
    accuracy_pct = (correct_count / total_count * 100) if total_count > 0 else 0

    optimization_prompt = f"""You are in an ADVERSARIAL GAME optimizing a prompt that translates code into natural language reasoning.

## SITUATION
- A discriminator (Claude Opus 4) distinguishes "algorithmic explanations" from "natural human thinking"
- Current accuracy: {correct_count}/{total_count} = {accuracy_pct:.1f}% (we need < 55%)
- Iteration: {iteration + 1}

## DISCRIMINATOR'S EXACT DETECTION SIGNALS (from this round):
{tells_summary}

## THE CORE PROBLEM
The discriminator keeps detecting these patterns in your outputs:
1. **Algorithm naming** - Any mention of Kruskal, BFS, DP, binary search, etc. is instant detection
2. **Textbook structure** - "First... Second... Finally..." or "Step 1... Step 2..."
3. **Formal definitions** - "X is used to...", "The algorithm works by..."
4. **Technical jargon** - "parenthesization", "recurrence", "invariant", "scalar multiplications"
5. **Exhaustive enumeration** - Walking through every case/pass systematically

## YOUR MISSION
Rewrite the prompt so translations sound like a REAL PERSON who just figured out the problem:
- Start mid-thought ("So basically..." / "The trick is..." / "What matters here is...")
- Use imprecise but correct language ("that table thing", "tracking which ones connect")
- Include human artifacts (brief hesitations, quick self-corrections, "wait actually...")
- Be SELECTIVE - only mention 1-2 key insights, skip obvious details
- Sound slightly informal, like explaining to a friend

## CURRENT PROMPT:
{current_prompt[:3000]}

## OUTPUT
Write the COMPLETE new prompt with examples. Be aggressive about removing any detectable patterns. The examples are CRITICAL - they teach the style.

NEW OPTIMIZED PROMPT:
"""

    return call_model(GPT_52_MODEL, optimization_prompt, max_tokens=5000)


def run_optimization_loop(max_iterations: int = 10):
    """Main optimization loop."""
    print("=" * 70)
    print("ITERATIVE ADVERSARIAL PROMPT OPTIMIZATION")
    print(f"Target: Reduce Opus accuracy below {TARGET_ACCURACY*100:.0f}%")
    print("=" * 70)

    samples = load_samples()[:7]  # Use 7 samples for speed
    print(f"\nUsing {len(samples)} samples for evaluation")

    current_prompt = get_initial_prompt()
    history = []

    # Start fresh with new mimic approach
    start_iter = 11  # Continue numbering from where we left off

    for iteration in range(start_iter, start_iter + max_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print("=" * 70)

        # Step 1: Translate all samples with current prompt
        # NOTE: GPT 5.2 only sees CODE, not problem description (prevents data leakage)
        print(f"\n[1] Translating {len(samples)} samples with GPT 5.2 (code-only, no problem)...")
        translations = []
        for i, sample in enumerate(samples):
            code = sample.get("code", "")
            translated = translate_with_prompt(current_prompt, code)
            translations.append(translated)
            print(f"    Sample {i+1}: {len(translated)} chars")

        # Step 2: Test Opus distinguishability
        print(f"\n[2] Testing Opus 4.5 distinguishability...")
        accuracy, opus_results = test_opus_distinguishability(samples, translations)
        print(f"    Opus accuracy: {accuracy*100:.1f}%")

        # Log iteration
        history.append({
            "iteration": iteration,
            "accuracy": accuracy,
            "opus_feedback": opus_results,
            "prompt_length": len(current_prompt),
        })

        # Check if we've succeeded
        if accuracy <= TARGET_ACCURACY:
            print(f"\n🎉 SUCCESS! Opus accuracy dropped to {accuracy*100:.1f}%")
            print("Translations are now indistinguishable!")
            break

        # Step 3: Show Opus's detection patterns
        print(f"\n[3] Opus detection patterns:")
        for r in opus_results[:3]:
            if r.get("correct"):
                print(f"    - {r['kind']}: {r['key_tells'][:100]}...")

        # Step 4: Optimize prompt with GPT 5.2
        print(f"\n[4] GPT 5.2 optimizing prompt based on feedback...")
        new_prompt = optimize_prompt_with_feedback(current_prompt, opus_results, iteration)

        if "ERROR" in new_prompt or len(new_prompt) < 500:
            print("    ERROR: Failed to generate new prompt, retrying...")
            continue

        current_prompt = new_prompt
        print(f"    New prompt: {len(current_prompt)} chars")

        # Save intermediate prompt
        prompt_path = Path(__file__).parent / f"prompt_iteration_{iteration}.md"
        with open(prompt_path, "w") as f:
            f.write(current_prompt)
        print(f"    Saved to: {prompt_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nAccuracy progression:")
    for h in history:
        status = "✓ SUCCESS" if h["accuracy"] <= TARGET_ACCURACY else ""
        print(f"  Iteration {h['iteration']}: {h['accuracy']*100:.1f}% {status}")

    # Save final results
    output_path = Path(__file__).parent / "optimization_history.json"
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to: {output_path}")

    # Save final prompt
    final_prompt_path = Path(__file__).parent / "icl_code_to_nl_FINAL.md"
    with open(final_prompt_path, "w") as f:
        f.write(current_prompt)
    print(f"Final prompt saved to: {final_prompt_path}")

    return history


if __name__ == "__main__":
    run_optimization_loop(max_iterations=5)
