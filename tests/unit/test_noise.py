import itertools
from difflib import SequenceMatcher

import pytest

from src.exps_performance.noise import (
    DISTRACTORS,
    clamp_sigma,
    perturb,
    perturb_irrelevant,
    perturb_numerical,
    perturb_structural,
    perturb_textual,
)

BASE = "Compute: 123 + 456. Items: [1, 2, 3, 4]."


def diff_count(a: str, b: str) -> int:
    return sum(1 for x, y in itertools.zip_longest(a, b, fillvalue="") if x != y)


@pytest.mark.parametrize(
    "func",
    [perturb_numerical, perturb_textual, perturb_structural, perturb_irrelevant],
)
def test_sigma_zero_no_change(func):
    assert func(BASE, 0.0, 0) == BASE


def test_reproducibility():
    out1 = perturb_numerical(BASE, 0.5, 7)
    out2 = perturb_numerical(BASE, 0.5, 7)
    out3 = perturb_numerical(BASE, 0.5, 8)
    assert out1 == out2
    assert out1 != out3


def test_numerical_monotonicity():
    low = diff_count(BASE, perturb_numerical(BASE, 0.1, 1))
    high = diff_count(BASE, perturb_numerical(BASE, 0.8, 1))
    assert high >= low


def test_textual_monotonicity():
    low = SequenceMatcher(None, BASE, perturb_textual(BASE, 0.1, 2)).ratio()
    high = SequenceMatcher(None, BASE, perturb_textual(BASE, 0.8, 2)).ratio()
    assert high <= low


def test_structural_changes_lists():
    out = perturb_structural("List: [1,2,3,4]", 0.9, 3)
    assert out != "List: [1,2,3,4]"
    assert out.startswith("List:")


def test_irrelevant_injection():
    out = perturb_irrelevant("Simple sentence.", 0.5, 4)
    assert any(sent in out for sent in DISTRACTORS)


def test_perturb_dispatch():
    assert perturb(BASE, "numerical", 0.2, 5) != BASE
    with pytest.raises(ValueError):
        perturb(BASE, "unknown", 0.1, 0)


def test_clamp_sigma_bounds():
    assert clamp_sigma(-1.0) == 0.0
    assert clamp_sigma(1.5) == 1.0
