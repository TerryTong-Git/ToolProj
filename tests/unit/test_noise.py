import itertools
import json
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.exps_performance import noise_runner
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


def test_gaussian_uniform_dispatch_changes():
    base = "Compute 12 + 3."
    assert perturb(base, "gaussian", 0.4, 1) != base
    assert perturb(base, "uniform", 0.4, 2) != base


def test_noise_runner_serialization(tmp_path, monkeypatch):
    class FakeRecord:
        def __init__(self, kind: str):
            self.kind = kind
            self.digits = 2
            self.nl_correct = True
            self.sim_correct = False
            self.controlsim_correct = False
            self.code_correct = True

        def model_dump(self):
            return {
                "kind": self.kind,
                "digits": self.digits,
                "nl_correct": self.nl_correct,
                "sim_correct": self.sim_correct,
                "controlsim_correct": self.controlsim_correct,
                "code_correct": self.code_correct,
            }

    def fake_dataset(kinds, n, digits_list):
        return [SimpleNamespace(question="Compute 1+2", kind=k, digits=digits_list[0], answer="3") for k in kinds]

    def fake_run_all_arms(data, args, client):
        for q in data:
            q.record = FakeRecord(q.kind)
        return data

    monkeypatch.setattr(noise_runner, "seed_all_and_setup", lambda args: None)
    monkeypatch.setattr(noise_runner, "llm", lambda args: "client")
    monkeypatch.setattr(noise_runner, "make_dataset", fake_dataset)
    monkeypatch.setattr(noise_runner, "_run_all_arms", fake_run_all_arms)

    args = noise_runner.NoiseArgs(
        model="dummy/model",
        backend="dummy",
        n=1,
        kinds=["add"],
        digits_list=[2],
        sigma=[0.0, 0.25],
        noise_types=["gaussian"],
        root=str(tmp_path),
        save_path=str(tmp_path / "noise.json"),
    )

    noise_runner.run(args)

    payload = json.loads(Path(args.save_path).read_text())
    assert isinstance(payload, list)
    assert payload, "results should not be empty"
    summary = payload[-1]
    assert summary["noise_types"] == args.noise_types
    assert summary["noise_levels"] == args.sigma
    first = payload[0]
    assert "noise_type" in first and "sigma" in first
    assert first["noise_type"] == "gaussian"
