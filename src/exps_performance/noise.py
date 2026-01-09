from __future__ import annotations

import random
import re
import string
from typing import Callable, Dict

NoiseFunc = Callable[[str, float, int], str]

DISTRACTORS = [
    "Note: historical context may not change the calculation.",
    "Reminder: intermediate steps should remain consistent.",
    "Background: unrelated metadata is sometimes noisy.",
    "Caution: ignore anecdotal remarks while solving.",
    "Hint: double-check arithmetic despite distractions.",
]


def clamp_sigma(sigma: float) -> float:
    """Clamp sigma into [0, 1]."""
    return max(0.0, min(1.0, float(sigma)))


def perturb_numerical(question: str, sigma: float, seed: int) -> str:
    """Replace a fraction of digits with random digits."""
    sigma = clamp_sigma(sigma)
    if sigma <= 0.0:
        return question
    rng = random.Random(seed)
    chars = list(question)
    digit_positions = [i for i, ch in enumerate(chars) if ch.isdigit()]
    if not digit_positions:
        return question
    n_changes = max(1, int(len(digit_positions) * sigma))
    indices = rng.sample(digit_positions, min(n_changes, len(digit_positions)))
    for idx in indices:
        original = chars[idx]
        replacement_pool = [d for d in string.digits if d != original]
        chars[idx] = rng.choice(replacement_pool)
    return "".join(chars)


def _perturb_numeric_with_sampler(question: str, sigma: float, seed: int, sampler: Callable[[random.Random, float], int]) -> str:
    """Shared helper for numeric noise driven by a sampler (e.g., Gaussian or Uniform)."""
    sigma = clamp_sigma(sigma)
    if sigma <= 0.0:
        return question
    rng = random.Random(seed)
    chars = list(question)
    digit_positions = [i for i, ch in enumerate(chars) if ch.isdigit()]
    if not digit_positions:
        return question
    n_changes = max(1, int(len(digit_positions) * sigma))
    indices = rng.sample(digit_positions, min(n_changes, len(digit_positions)))
    for idx in indices:
        original = chars[idx]
        delta = sampler(rng, sigma)
        new_val = (int(original) + delta) % 10
        # ensure a change occurred
        if str(new_val) == original:
            replacement_pool = [d for d in string.digits if d != original]
            new_val = int(rng.choice(replacement_pool))
        chars[idx] = str(new_val)
    return "".join(chars)


def perturb_gaussian(question: str, sigma: float, seed: int) -> str:
    """Numeric perturbation where digit deltas follow a Gaussian step."""

    def sampler(rng: random.Random, s: float) -> int:
        step = rng.gauss(0.0, max(1e-3, s) * 5.0)
        return int(round(step))

    return _perturb_numeric_with_sampler(question, sigma, seed, sampler)


def perturb_uniform(question: str, sigma: float, seed: int) -> str:
    """Numeric perturbation where digit deltas follow a uniform step."""

    def sampler(rng: random.Random, s: float) -> int:
        width = max(1, int(round(9 * s)))
        return rng.randint(-width, width)

    return _perturb_numeric_with_sampler(question, sigma, seed, sampler)


def perturb_textual(question: str, sigma: float, seed: int) -> str:
    """Character-level noise: insertions, deletions, swaps."""
    sigma = clamp_sigma(sigma)
    if sigma <= 0.0:
        return question
    rng = random.Random(seed)
    chars = list(question)
    ops = max(1, int(len(chars) * sigma))
    alphabet = string.ascii_letters
    for _ in range(ops):
        if not chars:
            break
        op = rng.choice(["insert", "delete", "swap"])
        if op == "insert":
            pos = rng.randrange(len(chars) + 1)
            chars.insert(pos, rng.choice(alphabet))
        elif op == "delete":
            pos = rng.randrange(len(chars))
            chars.pop(pos)
        else:  # swap
            if len(chars) < 2:
                continue
            pos = rng.randrange(len(chars) - 1)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def _shuffle_list_segment(segment: str, rng: random.Random) -> str:
    inner = segment.strip()[1:-1]  # drop brackets
    tokens = [tok.strip() for tok in inner.split(",")]
    if len(tokens) <= 1:
        return segment
    rng.shuffle(tokens)
    return "[" + ", ".join(tokens) + "]"


def perturb_structural(question: str, sigma: float, seed: int) -> str:
    """
    Shuffle list/matrix elements with probability proportional to sigma.
    Falls back to word-level shuffle if no bracketed list is found.
    """
    sigma = clamp_sigma(sigma)
    if sigma <= 0.0:
        return question
    rng = random.Random(seed)
    pattern = re.compile(r"\[[^\[\]]+\]")
    matches = list(pattern.finditer(question))
    if not matches:
        words = question.split()
        if len(words) <= 2:
            return question
        n_swaps = max(1, int(len(words) * sigma))
        for _ in range(n_swaps):
            i = rng.randrange(len(words))
            j = rng.randrange(len(words))
            words[i], words[j] = words[j], words[i]
        return " ".join(words)

    new_parts = []
    last = 0
    for m in matches:
        new_parts.append(question[last : m.start()])
        segment = m.group(0)
        if rng.random() < sigma:
            segment = _shuffle_list_segment(segment, rng)
        new_parts.append(segment)
        last = m.end()
    new_parts.append(question[last:])
    return "".join(new_parts)


def perturb_irrelevant(question: str, sigma: float, seed: int) -> str:
    """Inject distractor sentences proportional to sigma."""
    sigma = clamp_sigma(sigma)
    if sigma <= 0.0:
        return question
    rng = random.Random(seed)
    n_sent = max(1, int(max(1, len(question.split()) // 25) * sigma))
    chosen = rng.choices(DISTRACTORS, k=n_sent)
    # Insert at a random position between sentences; default to append.
    if "." in question:
        parts = question.split(".")
        insert_at = rng.randrange(len(parts))
        parts.insert(insert_at, " ".join(chosen))
        return ".".join(parts).strip()
    return question + " " + " ".join(chosen)


NOISE_FUNCS: Dict[str, NoiseFunc] = {
    "numerical": perturb_numerical,
    "gaussian": perturb_gaussian,
    "uniform": perturb_uniform,
    "textual": perturb_textual,
    "structural": perturb_structural,
    "irrelevant": perturb_irrelevant,
}


def perturb(question: str, noise_type: str, sigma: float, seed: int) -> str:
    """Dispatch perturbation based on noise type."""
    if noise_type not in NOISE_FUNCS:
        raise ValueError(f"Unknown noise type: {noise_type}")
    func = NOISE_FUNCS[noise_type]
    return func(question, sigma, seed)
