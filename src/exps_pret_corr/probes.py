#!/usr/bin/env python3
"""
Probes and evaluators for semantic task labeling.

This module provides:
1. Probe generation for 9 task types (add, sub, mul, lcs, knap, rod, ilp_assign, ilp_prod, ilp_partition)
2. Ground-truth evaluators for each task (reusing algorithms from exps_performance)
3. LLM-based semantic labeling via denotation matching
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

# Reuse algorithm implementations from exps_performance
from src.exps_performance.algorithms import (
    assignment_min_cost,
    knap_01_max_value,
    lcs_len,
    partition_min_diff,
    rod_cut_max,
)

TASK_KINDS = ["add", "sub", "mul", "lcs", "knap", "rod", "ilp_assign", "ilp_prod", "ilp_partition"]


# ------------------------------------------------------------------------------
# Probe generation
# ------------------------------------------------------------------------------


def _rand_int_with_digits(d: int) -> int:
    """Generate a random integer with exactly d digits."""
    lo = 10 ** (d - 1)
    hi = 10**d - 1
    return random.randint(lo, hi)


def gen_probes(num_per_kind: int, seed: int = 0) -> Dict[str, List[dict]]:
    """
    Generate probe instances for each task kind.

    Args:
        num_per_kind: Number of probes to generate per task type
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping task kind to list of probe dictionaries
    """
    random.seed(seed)
    R: Dict[str, List[dict]] = {k: [] for k in TASK_KINDS}

    for _ in range(num_per_kind):
        # Arithmetic: add
        a, b = _rand_int_with_digits(3), _rand_int_with_digits(3)
        R["add"].append({"a": a, "b": b})

        # Arithmetic: sub (ensure a >= b)
        a, b = _rand_int_with_digits(3), _rand_int_with_digits(3)
        if b > a:
            a, b = b, a
        R["sub"].append({"a": a, "b": b})

        # Arithmetic: mul
        a, b = _rand_int_with_digits(3), _rand_int_with_digits(3)
        R["mul"].append({"a": a, "b": b})

        # LCS: two random strings
        L = random.randint(6, 10)
        s1 = "".join(random.choice(string.ascii_uppercase[:6]) for _ in range(L))
        s2 = "".join(random.choice(string.ascii_uppercase[:6]) for _ in range(L))
        R["lcs"].append({"s1": s1, "s2": s2})

        # Knapsack
        n = random.randint(5, 8)
        weights = [random.randint(1, 9) for _ in range(n)]
        values = [random.randint(1, 20) for _ in range(n)]
        cap = max(5, sum(weights) // 2)
        R["knap"].append({"w": weights, "v": values, "cap": cap})

        # Rod cutting
        n = random.randint(6, 10)
        prices = [random.randint(1, 12) + i for i in range(n)]
        R["rod"].append({"prices": prices, "n": n})

        # ILP assignment
        m = random.randint(3, 5)
        nn = random.randint(3, 5)
        C = [[random.randint(1, 12) for _ in range(nn)] for _ in range(m)]
        R["ilp_assign"].append({"C": C})

        # ILP production
        m = random.randint(2, 4)
        a1 = [random.randint(1, 9) for _ in range(m)]
        b1 = [random.randint(8, 25) for _ in range(m)]
        p1 = [random.randint(5, 20) for _ in range(m)]
        R["ilp_prod"].append({"a": a1, "b": b1, "p": p1})

        # ILP partition
        n = random.randint(8, 12)
        arr = [random.randint(1, 25) for _ in range(n)]
        R["ilp_partition"].append({"arr": arr})

    return R


# ------------------------------------------------------------------------------
# Evaluators (ground-truth solvers)
# Reuse implementations from exps_performance.algorithms where available
# ------------------------------------------------------------------------------


def eval_add(x: dict) -> int:
    """Simple addition."""
    return int(x["a"] + x["b"])


def eval_sub(x: dict) -> int:
    """Simple subtraction."""
    return int(x["a"] - x["b"])


def eval_mul(x: dict) -> int:
    """Simple multiplication."""
    return int(x["a"] * x["b"])


def eval_lcs(x: dict) -> int:
    """Compute LCS length. Wraps lcs_len from algorithms."""
    return lcs_len(x["s1"], x["s2"])


def eval_knap(x: dict) -> int:
    """Solve 0/1 knapsack. Wraps knap_01_max_value from algorithms."""
    return knap_01_max_value(x["w"], x["v"], x["cap"])


def eval_rod(x: dict) -> int:
    """Solve rod cutting. Wraps rod_cut_max from algorithms."""
    # rod_cut_max expects prices list where prices[i] is price for length i+1
    return rod_cut_max(x["prices"][: x["n"]])


def eval_ilp_assign(x: dict) -> int:
    """Solve assignment problem. Wraps assignment_min_cost from algorithms."""
    return assignment_min_cost(x["C"])


def eval_ilp_prod(x: dict) -> int:
    """Solve ILP production (independent bounded knapsacks)."""
    # Simple case not in algorithms.py - keep local implementation
    a, b, p = x["a"], x["b"], x["p"]
    tot = 0
    for ai, bi, pi in zip(a, b, p):
        xi = bi // ai
        tot += pi * xi
    return tot


def eval_ilp_partition(x: dict) -> int:
    """Solve partition problem. Wraps partition_min_diff from algorithms."""
    return partition_min_diff(x["arr"])


EVALUATORS: Dict[str, Callable[[dict], int]] = {
    "add": eval_add,
    "sub": eval_sub,
    "mul": eval_mul,
    "lcs": eval_lcs,
    "knap": eval_knap,
    "rod": eval_rod,
    "ilp_assign": eval_ilp_assign,
    "ilp_prod": eval_ilp_prod,
    "ilp_partition": eval_ilp_partition,
}


def precompute_reference_answers(R_by_kind: Dict[str, List[dict]]) -> Dict[str, List[str]]:
    """Compute reference answers for each probe using ground-truth evaluators."""
    return {k: [str(EVALUATORS[k](x)) for x in probes] for k, probes in R_by_kind.items()}


# ------------------------------------------------------------------------------
# LLM-based labeling
# ------------------------------------------------------------------------------


def serialize_probe(x: dict) -> str:
    """Serialize a probe dictionary to a compact string."""
    items = []
    for k, v in x.items():
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                # 2D list
                inner = "; ".join("[" + ",".join(map(str, row)) + "]" for row in v)
                items.append(f"{k}=[{inner}]")
            else:
                items.append(f"{k}=[{','.join(map(str, v))}]")
        else:
            items.append(f"{k}={v}")
    return "; ".join(items)


def create_apply_prompt(procedure_text: str, probe: dict) -> str:
    """Create a prompt to apply procedure text to a probe input."""
    return (
        "Apply the following procedure exactly to the input. "
        "Return only the final numeric answer on one line.\n\n"
        f"PROCEDURE:\n{procedure_text}\n\nINPUT:\n{serialize_probe(probe)}\n\nANSWER:"
    )


def normalize_answer(s: str) -> str:
    """Normalize model output to extract numeric answer."""
    s = s.strip()
    s = s.splitlines()[-1].strip() if s else ""
    s = re.sub(r"^(answer|final|result)\s*[:=-]\s*", "", s, flags=re.I)
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return m.group(0) if m else "⊥"


@dataclass
class LabelingResult:
    """Result from semantic labeling."""

    label: Optional[str]
    confidence: float
    scores: Dict[str, float]


class SemanticLabeler:
    """Labels text documents by semantic task via LLM probing."""

    def __init__(
        self,
        model_name: str = "google/gemma-2-9b-it",
        probes_per_kind: int = 16,
        max_new_tokens: int = 32,
        batch_size: int = 32,
        conf_threshold: float = 0.6,
        seed: int = 0,
        dtype: str = "auto",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.probes_per_kind = probes_per_kind
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.seed = seed
        self.dtype = dtype
        self.device = device

        # Lazy loading
        self._tok: Optional[Any] = None
        self._model: Optional[Any] = None
        self._probes: Optional[Dict[str, List[dict]]] = None
        self._ref_answers: Optional[Dict[str, List[str]]] = None

    def _ensure_loaded(self, load_model: bool = False) -> None:
        """Lazily load probes (and model if requested)."""
        if self._probes is None:
            self._probes = gen_probes(self.probes_per_kind, seed=self.seed)
            self._ref_answers = precompute_reference_answers(self._probes)

        if load_model and self._tok is None:
            self._tok, self._model = self._build_model()

    # Allow tests to bypass wrappers by storing original method
    _ensure_loaded.__wrapped__ = _ensure_loaded  # type: ignore[attr-defined]

    def _build_model(self) -> Tuple[Any, Any]:
        """Load HuggingFace causal LM."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def pick_dtype(flag: str) -> "torch.dtype":
            if flag == "auto":
                if torch.cuda.is_available():
                    return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
                return torch.float32
            return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[flag]

        tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id or 0

        torch_dtype = pick_dtype(self.dtype) if self.device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype, device_map="auto" if self.device != "cpu" else None
        ).eval()

        if self.device == "cpu":
            model.to("cpu")

        return tok, model

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate completions for a batch of prompts."""
        import torch

        assert self._tok is not None, "Tokenizer not loaded"
        assert self._model is not None, "Model not loaded"

        enc = self._tok(prompts, return_tensors="pt", padding=True, truncation=True)
        for k in enc:
            enc[k] = enc[k].to(self._model.device)

        with torch.inference_mode():
            gen = self._model.generate(**enc, do_sample=False, temperature=0.0, max_new_tokens=self.max_new_tokens)

        outs = self._tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return [normalize_answer(o) for o in outs]

    def label(self, text: str) -> LabelingResult:
        """
        Assign a semantic label to text by applying it to probes.

        Args:
            text: The procedure/document text to label

        Returns:
            LabelingResult with label, confidence, and per-kind scores
        """
        self._ensure_loaded(load_model=True)

        assert self._probes is not None, "Probes not loaded"
        assert self._ref_answers is not None, "Reference answers not loaded"

        scores: Dict[str, float] = {}
        for kind, probes in self._probes.items():
            prompts = [create_apply_prompt(text, x) for x in probes]
            outs: List[str] = []
            for i in range(0, len(prompts), self.batch_size):
                outs.extend(self._generate_batch(prompts[i : i + self.batch_size]))

            ref = self._ref_answers[kind]
            match = sum(1 for a, b in zip(outs, ref) if a == b)
            scores[kind] = match / max(1, len(ref))

        best_kind = max(scores.items(), key=lambda kv: kv[1])[0]
        conf = scores[best_kind]
        label = best_kind if conf >= self.conf_threshold else None

        return LabelingResult(label=label, confidence=conf, scores=scores)
