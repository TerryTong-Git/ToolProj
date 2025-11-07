from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from algorithms import assignment_min_cost, knap_01_max_value, lcs_len, partition_min_diff, prodplan_max_profit, rod_cut_max
from datasets import load_dataset


@dataclass
class Problem:
    kind: str
    digits: int = 0
    a: int = 0
    b: int = 0
    # general payload for DP/ILP
    data: Dict[str, Any] = field(default_factory=lambda: {})

    def text(self) -> str:
        k = self.kind
        if k in ("add", "sub", "mul", "mix"):
            if k == "add":
                return f"Compute: {self.a} + {self.b}"
            if k == "sub":
                return f"Compute: {self.a} - {self.b}"
            if k == "mul":
                return f"Compute: {self.a} * {self.b}"
            if k == "mix":
                return f"Compute: ({self.a} + {self.b}) * {self.a}"
        elif k == "lcs":
            s = self.data["s"]
            t = self.data["t"]
            return f'Compute the length of the Longest Common Subsequence (LCS) between strings:\nS = "{s}"\nT = "{t}"'
        elif k == "knap":
            w = self.data["weights"]
            v = self.data["values"]
            C = self.data["capacity"]
            return (
                "0/1 Knapsack: Given item weights W and values V and capacity C, " "compute the maximum total value.\n" f"W = {w}\nV = {v}\nC = {C}"
            )
        elif k == "rod":
            prices = self.data["prices"]
            N = len(prices)
            return (
                "Rod cutting: Given a rod of length N and price list P[1..N], " "compute the maximum obtainable revenue.\n" f"N = {N}\nP = {prices}"
            )
        elif k == "ilp_assign":
            C = self.data["cost"]
            return (
                "Assignment problem: Given an n×n cost matrix C, assign each worker to one task "
                "minimizing the total cost. Return the minimum total cost as an integer. \n"
                f"C = {C}"
            )
        elif k == "ilp_prod":
            prof = self.data["profit"]
            cons = self.data["consumption"]
            caps = self.data["capacity"]
            ub = self.data["upper_bound"]
            return (
                "Production planning: Choose integer quantities x_j ≥ 0 to maximize total profit sum_j profit[j]*x_j, "
                "subject to resource constraints sum_j consumption[i][j]*x_j ≤ capacity[i]. Return the max profit.\n"
                f"profit = {prof}\nconsumption (rows=resources) = {cons}\ncapacity = {caps}\nupper_bounds = {ub}"
            )
        elif k == "ilp_partition":
            w = self.data["weights"]
            return (
                "Partition: Split the items into two groups to minimize the absolute difference between the sums. "
                "Return the minimum difference as an integer.\n"
                f"weights = {w}"
            )
        raise ValueError("unknown kind")

    # ---- Ground-truth evaluators ----
    def ground_truth(self) -> int:
        k = self.kind
        if k in ("add", "sub", "mul", "mix"):
            if k == "add":
                return self.a + self.b
            if k == "sub":
                return self.a - self.b
            if k == "mul":
                return self.a * self.b
            if k == "mix":
                return (self.a + self.b) * self.a
        elif k == "lcs":
            return lcs_len(self.data["s"], self.data["t"])
        elif k == "knap":
            return knap_01_max_value(self.data["weights"], self.data["values"], self.data["capacity"])
        elif k == "rod":
            return rod_cut_max(self.data["prices"])
        elif k == "ilp_assign":
            return assignment_min_cost(self.data["cost"])
        elif k == "ilp_prod":
            return prodplan_max_profit(self.data)
        elif k == "ilp_partition":
            return partition_min_diff(self.data["weights"])
        raise ValueError("unknown kind")


@dataclass
class GSM8KProblem(Problem):
    kind: str = "gsm8k"
    digits: int = 0
    a: int = 0
    b: int = 0
    data: Dict[str, Any] = field(default_factory=lambda: {})

    def text(self) -> str:
        return self.data["question"]

    def ground_truth(self) -> int:
        return parse_gsm8k_gold(self.data["answer"])


def load_gsm8k() -> Sequence[Problem]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, ex in enumerate(ds):
        if check_parse_gsm8k_gold(ex["answer"]) is None:
            continue
        problem = GSM8KProblem(
            data={
                "question": ex["question"],
                "answer": ex["answer"],
            }
        )
        items.append(problem)
    return items


def parse_gsm8k_gold(ans: str) -> int:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1))  # type: ignore


def check_parse_gsm8k_gold(ans: str) -> Optional[int]:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1)) if m else None


@dataclass
class CLRS30(Problem):
    kind: str = "gsm8k"
    digits: int = 0
    a: int = 0
    b: int = 0
    data: Dict[str, Any] = field(default_factory=lambda: {})

    def text(self) -> str:
        return self.data["question"]

    def ground_truth(self) -> int:
        return parse_gsm8k_gold(self.data["answer"])


def load_CLRS30() -> Sequence[Problem]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, ex in enumerate(ds):
        if check_parse_gsm8k_gold(ex["answer"]) is None:
            continue
        problem = CLRS30(
            data={
                "question": ex["question"],
                "answer": ex["answer"],
            }
        )
        items.append(problem)
    return items


def parse_CLRS30_gold(ans: str) -> int:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1))  # type: ignore


def check_parse_CLRS30_gold(ans: str) -> Optional[int]:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1)) if m else None


@dataclass
class NPHARDEVALProblem(Problem):
    kind: str = "NPHARDEVAL"
    digits: int = 0
    a: int = 0
    b: int = 0
    data: Dict[str, Any] = field(default_factory=lambda: {})

    def text(self) -> str:
        return self.data["question"]

    def ground_truth(self) -> int:
        return parse_NPHARDEVAL_gold(self.data["answer"])


def load_NPHARDEVAL() -> Sequence[Problem]:
    ds = load_dataset("openai/NPHARDEVAL", "main", split="test")
    items = []
    for i, ex in enumerate(ds):
        if check_parse_NPHARDEVAL_gold(ex["answer"]) is None:
            continue
        problem = NPHARDEVALProblem(
            data={
                "question": ex["question"],
                "answer": ex["answer"],
            }
        )
        items.append(problem)
    return items


def parse_NPHARDEVAL_gold(ans: str) -> int:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1))  # type: ignore


def check_parse_NPHARDEVAL_gold(ans: str) -> Optional[int]:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1)) if m else None
