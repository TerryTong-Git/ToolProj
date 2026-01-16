#!/usr/bin/env python3
"""
Optimize for CONSISTENCY: translations should be indistinguishable from originals.

Key changes from previous approach:
1. ALLOW algorithm naming (match original's style)
2. Neutral Opus prompt (no style bias)
3. 10 train + 10 eval samples
4. Target 50% accuracy (random chance = perfect consistency)
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict

import openai
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GPT_52_MODEL = "openai/gpt-5.2"
OPUS_MODEL = "anthropic/claude-opus-4"

TARGET_ACCURACY = 0.45  # Want ~50% (random chance), allow some margin


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


def load_diverse_samples(n_train: int = 10, n_eval: int = 10) -> tuple[list, list]:
    """Load diverse samples from multiple problem kinds."""
    results_dir = Path(__file__).parent / "results"
    samples_by_kind = defaultdict(list)

    for model_dir in sorted(results_dir.iterdir())[:5]:
        if not model_dir.is_dir():
            continue
        tb_dir = model_dir / "tb"
        if not tb_dir.exists():
            continue
        for run_dir in tb_dir.iterdir():
            jsonl = run_dir / "res.jsonl"
            if jsonl.exists():
                with open(jsonl) as f:
                    for line in f:
                        sample = json.loads(line)
                        kind = sample.get("kind", "")
                        nl = sample.get("nl_reasoning", "")
                        code = sample.get("code_question", "")
                        if nl and code and len(nl) > 200 and len(code) > 50:
                            samples_by_kind[kind].append({
                                "kind": kind,
                                "code": code,
                                "original_nl": nl,
                            })
                break

    # Sample evenly across kinds
    all_samples = []
    kinds = list(samples_by_kind.keys())
    random.seed(42)
    random.shuffle(kinds)

    # Take 1-2 from each kind until we have enough
    needed = n_train + n_eval
    for kind in kinds:
        if len(all_samples) >= needed:
            break
        samples = samples_by_kind[kind]
        random.shuffle(samples)
        all_samples.extend(samples[:2])

    random.shuffle(all_samples)
    train = all_samples[:n_train]
    eval_set = all_samples[n_train:n_train + n_eval]

    return train, eval_set


# Initial prompt seeded from icl_code_to_nl_FINAL.md
INITIAL_PROMPT = """# NEW OPTIMIZED ICL PROMPT: Code → "me talking it through after I finally got it"

You get a chunk of code. You explain what it's doing like you're texting a friend after staring at it for way too long and it *finally* clicked. Sound human, a bit messy, but still correct on the main idea.

## How to sound (this matters more than being "complete")
- Write like you're mid-thought: "So basically…", "The trick is…", "Wait—actually…"
- Use first-person: "I think", "I'm pretty sure", "I noticed"
- Keep it *selective*: hit **1–2 key insights** and stop. Don't narrate every loop.
- Tie it to the code with **2–4 concrete anchors** (variable names, constants, conditions).
  - Example anchors: `parent`, `find(x)`, `best[i][j]`, `if a > b:`, `swapped = False`, `999999`
- It's okay to be fuzzy on tiny details ("I think this is just…") as long as the big point is right.
- 1–3 short paragraphs. Aim ~120–220 words unless the code is massive.
- End with exactly one plain closing line that starts with: **"So in the end it returns …"**

## Hard "don't do this" (these are instant giveaways)
- Don't name famous techniques or textbook labels. Avoid words like:
  - *Kruskal, Prim, Dijkstra, BFS, DFS, dynamic programming, greedy, binary search, topological, minimum spanning tree, recurrence, invariant, parenthesization,* etc.
- Don't do lecture structure: no "This code implements…", no "First/Second/Finally", no "Step 1/2/3".
- Don't define terms like a textbook ("X is used to…").
- Don't do exhaustive tracing ("then i=0, then i=1…").

---

## Example 1 (picking cheap links without re-linking the same blob)

**Code:**
```python
def solution():
    edges = [(0,7,0.546), (2,7,0.247), (6,7,0.033), (8,9,0.138)]
    edges.sort(key=lambda x: x[2])
    parent = list(range(10))

    def find(x):
        return x if parent[x] == x else find(parent[x])

    def union(a, b):
        parent[find(a)] = find(b)

    count = 0
    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            count += 1
    return count
```

**Good explanation:**

So basically it's trying to accept some connections, but only when they actually change something. The giveaway is `parent = list(range(10))` plus that little `find(x)` helper — it keeps "who belongs with who" by chasing `parent[x]` until it hits a self-pointer (`parent[x] == x`).

Then `edges.sort(... x[2])` means it's looking at the smallest `w` first (like `0.033` before `0.546`). The main gate is `if find(u) != find(v)`: if both ends already lead to the same representative, that edge is kinda pointless and it skips it. Otherwise it does `union(u, v)` and bumps `count`.

It's not even adding up weights here — it's just counting how many edges were "actually useful."
So in the end it returns the number of accepted edges.

---

## Example 2 (the "best cost for a slice" table)

**Code:**
```python
def solve(dims):
    n = len(dims) - 1
    best = [[0]*n for _ in range(n)]
    split = [[-1]*n for _ in range(n)]

    for gap in range(1, n):
        for i in range(n-gap):
            j = i + gap
            best[i][j] = 10**18
            for k in range(i, j):
                cost = best[i][k] + best[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
                if cost < best[i][j]:
                    best[i][j] = cost
                    split[i][j] = k
    return best[0][n-1], split
```

**Good explanation:**

The trick here is that `best[i][j]` is treating "from i to j" as one chunk, and it's trying to find the cheapest way to break that chunk into two smaller chunks. I noticed it because it sets `best[i][j] = 10**18` and then tries a bunch of `k` values, updating when `cost < best[i][j]`.

The `gap` loop is basically forcing it to fill the table from small spans to bigger spans, so when it computes `best[i][j]` it already has `best[i][k]` and `best[k+1][j]` lying around. And that weird-looking multiply `dims[i]*dims[k+1]*dims[j+1]` is the "price" of doing the final combine after you pick where to cut.

`split[i][j] = k` is just keeping the winning cut position so you can reconstruct the choices later.
So in the end it returns the smallest total cost and the split table it recorded.

---

## Example 3 (the "keep swapping neighbors until it calms down" thing)

**Code:**
```python
def sort_nums(a):
    n = len(a)
    swapped = True
    while swapped:
        swapped = False
        for i in range(n-1):
            if a[i] > a[i+1]:
                a[i], a[i+1] = a[i+1], a[i]
                swapped = True
    return a
```

**Good explanation:**

What's going on is it keeps scanning the list and only cares about adjacent pairs (`a[i]` and `a[i+1]`). If it catches one out of order (`if a[i] > a[i+1]`), it swaps them and flips `swapped = True` so it knows "okay, we're not done yet."

The `while swapped:` loop is basically "do another pass if anything changed last time." So it repeats those neighbor swaps until it makes a full pass where nothing triggers, meaning the list has kind of settled into order. The exact number of passes isn't tracked — it just stops naturally when `swapped` stays `False`.

It's simple, and you can almost picture bigger numbers getting nudged to the right over repeated passes.
So in the end it returns the sorted list `a`.

---

## Your turn
Given new code, produce **only** the human-style explanation (1–3 short paragraphs) and finish with exactly:
**"So in the end it returns …"**

**Code:**
```python
[CODE_HERE]
```
"""


def translate_code(prompt: str, code: str, target_len: int = None) -> str:
    """Translate code to NL using the prompt."""
    full_prompt = prompt.replace("[CODE_HERE]", code)
    if target_len:
        full_prompt += f"\n\nIMPORTANT: Keep your response to approximately {target_len} characters (about {target_len//5} words). Be concise!"
    return call_model(GPT_52_MODEL, full_prompt, max_tokens=600)


def test_opus_consistency(samples: list, translations: list) -> tuple[float, list]:
    """
    Test if Opus can distinguish original from translated.
    Uses NEUTRAL prompt - no style assumptions.
    """
    results = []
    correct = 0
    total = 0

    for i, (sample, translated) in enumerate(zip(samples, translations)):
        original = sample["original_nl"]

        if not translated or len(translated) < 100:
            continue

        # Randomize positions
        random.seed(i * 31)
        if random.random() > 0.5:
            text_a, text_b = original, translated
            answer_key = "A"  # Original is A
        else:
            text_a, text_b = translated, original
            answer_key = "B"  # Original is B

        # NEUTRAL prompt - no style bias
        opus_prompt = f"""You are given two explanations of an algorithm problem solution. One is the ORIGINAL explanation, one is a REPRODUCTION.

Your task: Determine which is the original and which is the reproduction.

Text A:
{text_a[:900]}

Text B:
{text_b[:900]}

Which text is the ORIGINAL? Answer with just the letter (A or B), then explain briefly.

ANSWER: [A or B]
REASONING: [brief explanation]"""

        response = call_model(OPUS_MODEL, opus_prompt, max_tokens=300)

        # Parse response
        opus_pick = None
        first_line = response.strip().split("\n")[0].upper()
        if "A" in first_line and "B" not in first_line:
            opus_pick = "A"
        elif "B" in first_line and "A" not in first_line:
            opus_pick = "B"
        elif "ANSWER: A" in response.upper() or "ANSWER:A" in response.upper():
            opus_pick = "A"
        elif "ANSWER: B" in response.upper() or "ANSWER:B" in response.upper():
            opus_pick = "B"

        is_correct = opus_pick == answer_key
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "kind": sample["kind"],
            "original_len": len(original),
            "translated_len": len(translated),
            "correct": is_correct,
            "answer_key": answer_key,
            "opus_pick": opus_pick,
            "response_preview": response[:200],
        })

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def optimize_prompt(current_prompt: str, eval_results: list, iteration: int) -> str:
    """Ask GPT 5.2 to optimize the prompt based on what Opus detected."""

    correct_cases = [r for r in eval_results if r["correct"]]
    incorrect_cases = [r for r in eval_results if not r["correct"]]
    accuracy = len(correct_cases) / len(eval_results) if eval_results else 0

    # Different strategies based on accuracy
    if accuracy < 0.40:
        # Accuracy too LOW - translations look MORE original than originals
        # Need to make translations LESS polished
        direction = """
## PROBLEM: Translations look MORE polished than originals!
Opus keeps picking translations as "original" because they're too clean/structured.

## FIX: Make translations LESS formal:
- Add minor hesitations ("I think...", "basically...")
- Use more casual language
- Don't over-explain
- Include some imprecision
- Match the rougher style of actual human explanations"""
    else:
        # Accuracy too HIGH - originals look more original
        direction = """
## PROBLEM: Originals still distinguishable from translations.

## FIX: Match original style more closely:
- Use same technical terms
- Match sentence structure
- Similar level of detail"""

    patterns = []
    for r in (correct_cases if correct_cases else incorrect_cases)[:5]:
        patterns.append(f"- {r['kind']}: {r['response_preview'][:150]}")
    patterns_text = "\n".join(patterns) if patterns else "None identified"

    optimization_prompt = f"""You are optimizing a prompt for code-to-NL translation.

## GOAL
Make translations INDISTINGUISHABLE from originals. Target: ~50% accuracy (random chance).
Current accuracy: {len(correct_cases)}/{len(eval_results)} = {accuracy*100:.1f}%
{direction}

## OPUS FEEDBACK:
{patterns_text}

## CURRENT PROMPT:
{current_prompt[:2500]}

Output the COMPLETE improved prompt:
"""

    return call_model(GPT_52_MODEL, optimization_prompt, max_tokens=4000)


def main():
    print("=" * 70)
    print("CONSISTENCY OPTIMIZATION")
    print("Goal: Translations indistinguishable from originals (50% accuracy)")
    print("=" * 70)

    # Load data
    print("\n[1] Loading 10 train + 10 eval samples...")
    train_samples, eval_samples = load_diverse_samples(10, 10)
    print(f"    Train: {len(train_samples)} samples - {set(s['kind'] for s in train_samples)}")
    print(f"    Eval: {len(eval_samples)} samples - {set(s['kind'] for s in eval_samples)}")

    current_prompt = INITIAL_PROMPT
    history = []

    for iteration in range(1, 11):  # 10 iterations
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print("=" * 70)

        # Translate eval samples (match original length)
        print(f"\n[2] Translating {len(eval_samples)} eval samples (matching original length)...")
        translations = []
        for i, sample in enumerate(eval_samples):
            orig_len = len(sample["original_nl"])
            # Target similar length to original (within 50% margin)
            target_len = max(200, min(orig_len + 100, 800))
            trans = translate_code(current_prompt, sample["code"], target_len=target_len)
            translations.append(trans)
            print(f"    {i+1}. {sample['kind']}: orig={orig_len}ch, trans={len(trans)}ch")

        # Test with Opus (neutral prompt)
        print(f"\n[3] Testing Opus consistency detection...")
        accuracy, results = test_opus_consistency(eval_samples, translations)
        print(f"    Opus accuracy: {accuracy*100:.1f}%")

        for r in results:
            status = "✓ detected" if r["correct"] else "✗ FOOLED"
            print(f"      {r['kind']}: {status}")

        history.append({
            "iteration": iteration,
            "accuracy": accuracy,
            "results": results,
        })

        # Check success (want ~50%, so accuracy should be between 40-60%)
        if 0.40 <= accuracy <= 0.60:
            print(f"\n🎉 SUCCESS! Accuracy {accuracy*100:.1f}% is near 50% (random chance)")
            break

        # Check if too low (inverted detection)
        if accuracy < 0.40:
            print(f"\n⚠️  Accuracy too LOW ({accuracy*100:.1f}%) - translations look MORE original than originals!")

        # Optimize prompt
        print(f"\n[4] Optimizing prompt...")
        new_prompt = optimize_prompt(current_prompt, results, iteration)
        if len(new_prompt) > 500:
            current_prompt = new_prompt
            print(f"    New prompt: {len(current_prompt)} chars")
        else:
            print(f"    WARNING: Failed to generate new prompt")

    # Final summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for h in history:
        status = "✓" if h["accuracy"] <= TARGET_ACCURACY else ""
        print(f"  Iteration {h['iteration']}: {h['accuracy']*100:.1f}% {status}")

    # Save
    output = {
        "history": history,
        "final_prompt": current_prompt,
    }
    output_path = Path(__file__).parent / "consistency_optimization_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")

    # Save final prompt
    prompt_path = Path(__file__).parent / "icl_consistency_FINAL.md"
    with open(prompt_path, "w") as f:
        f.write(current_prompt)
    print(f"Final prompt: {prompt_path}")


if __name__ == "__main__":
    main()
