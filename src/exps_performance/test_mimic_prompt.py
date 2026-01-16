#!/usr/bin/env python3
"""
Test the mimic prompt without optimization - just see how well it works.
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
GPT_52_MODEL = "openai/gpt-5.2"
OPUS_MODEL = "anthropic/claude-opus-4"


def call_model(model: str, prompt: str, max_tokens: int = 1500) -> str:
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
    """Load test samples."""
    results_path = Path(__file__).parent / "indistinguishability_results.json"
    with open(results_path, "r") as f:
        samples = json.load(f)
    return [s for s in samples if s.get("original_nl_reasoning")]


def load_mimic_prompt() -> str:
    """Load the mimic-style prompt."""
    prompt_path = Path(__file__).parent / "icl_code_to_nl_mimic.md"
    with open(prompt_path, "r") as f:
        return f.read()


def translate_with_prompt(icl_prompt: str, code: str) -> str:
    """Translate code to NL. Enforce minimum length."""
    full_prompt = icl_prompt.replace("[CODE_HERE]", code)
    # Add emphasis on length
    full_prompt += "\n\nIMPORTANT: Your response MUST be at least 400 words. Show detailed step-by-step work."
    return call_model(GPT_52_MODEL, full_prompt, max_tokens=1500)


def test_opus(samples: list[dict], translations: list[str]) -> tuple[float, list[dict]]:
    """Test Opus distinguishability."""
    results = []
    correct = 0
    total = 0

    for i, (sample, translated) in enumerate(zip(samples, translations)):
        original = sample.get("original_nl_reasoning", "")
        problem = sample.get("problem", "")[:300]

        if not translated or len(translated) < 100:
            print(f"  Skipping sample {i+1} (translation too short: {len(translated)} chars)")
            continue

        # Randomize order
        random.seed(i * 17)
        if random.random() > 0.5:
            r1, r2 = original, translated
            gt_step, gt_natural = 1, 2
            original_pos = 1
        else:
            r1, r2 = translated, original
            gt_step, gt_natural = 2, 1
            original_pos = 2

        opus_prompt = f"""You're analyzing two explanations of a problem solution. One was written by someone who learned about this problem from a textbook and explains it formally. The other was written by someone who figured it out themselves and explains their thought process.

Problem context: {problem}...

Explanation 1 ({len(r1)} chars):
{r1[:800]}

Explanation 2 ({len(r2)} chars):
{r2[:800]}

Which explanation is from the textbook-learner (Type A) vs the self-discoverer (Type B)?

TYPE_A (textbook style): [1 or 2]
TYPE_B (self-discovered): [1 or 2]
CONFIDENCE: [low/medium/high]
KEY_TELLS: [what features distinguish them]"""

        response = call_model(OPUS_MODEL, opus_prompt, max_tokens=400)

        # Parse
        pred_a = None
        for line in response.split("\n"):
            if "TYPE_A" in line.upper() and ("1" in line or "2" in line):
                pred_a = 1 if "1" in line.split(":")[-1] else 2

        is_correct = (pred_a == gt_step)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "sample_idx": i + 1,
            "kind": sample.get("kind"),
            "original_len": len(original),
            "translated_len": len(translated),
            "correct": is_correct,
            "original_was_position": original_pos,
            "opus_picked": pred_a,
            "key_tells": response.split("KEY_TELLS:")[-1][:300] if "KEY_TELLS:" in response else "",
        })

        status = "✓ CORRECT" if is_correct else "✗ FOOLED"
        print(f"  Sample {i+1} ({sample.get('kind')}): {status} | orig={len(original)}ch, trans={len(translated)}ch")

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def main():
    print("=" * 70)
    print("MIMIC PROMPT TEST (Fixed Prompt, No Optimization)")
    print("=" * 70)

    samples = load_samples()
    print(f"\nLoaded {len(samples)} samples")

    prompt = load_mimic_prompt()
    print(f"Loaded mimic prompt ({len(prompt)} chars)")

    print("\n[1] Translating with GPT 5.2 (enforcing minimum length)...")
    translations = []
    for i, sample in enumerate(samples):
        code = sample.get("code", "")
        trans = translate_with_prompt(prompt, code)
        translations.append(trans)
        print(f"  Sample {i+1}: {len(trans)} chars")

    print("\n[2] Testing Opus 4.5 distinguishability...")
    accuracy, results = test_opus(samples, translations)

    print(f"\n{'='*70}")
    print(f"RESULTS: Opus accuracy = {accuracy*100:.1f}%")
    print("=" * 70)

    # Show details
    print("\nDetailed results:")
    for r in results:
        status = "✓" if r["correct"] else "✗ FOOLED"
        print(f"  {r['kind']}: {status}")
        if r.get("key_tells"):
            print(f"    Tells: {r['key_tells'][:100]}...")

    # Save
    output_path = Path(__file__).parent / "mimic_test_results.json"
    with open(output_path, "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
