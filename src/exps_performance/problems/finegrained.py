from __future__ import annotations

from abc import abstractmethod
from dataclasses import field
from typing import Any, Dict

from algorithms import assignment_min_cost, knap_01_max_value, lcs_len, partition_min_diff, prodplan_max_profit, rod_cut_max
from problems import Problem


class FineGrainedProblems(Problem):
    def format_one(self):
        return self.instantiate_prompt()

    def decision_check(self, answer, problem_text=None):
        return int(answer == self.ground_truth())

    @abstractmethod
    def ground_truth(self):
        pass

    @abstractmethod
    def instantiate_prompt(self):
        raise NotImplementedError


class ArithmeticProblems(FineGrainedProblems):
    a: int
    b: int


class DPProblem(FineGrainedProblems):
    data: Dict[str, Any] = field(default_factory=lambda: {})


class ILPProblem(FineGrainedProblems):
    data: Dict[str, Any] = field(default_factory=lambda: {})


class Addition(ArithmeticProblems):
    def instantiate_prompt(self) -> str:
        return f"Compute: {self.a} + {self.b}"

    def ground_truth(self) -> int:
        return self.a + self.b


class Subtraction(ArithmeticProblems):
    def instantiate_prompt(self) -> str:
        return f"Compute: {self.a} - {self.b}"

    def ground_truth(self) -> int:
        return self.a - self.b


class Multiplication(ArithmeticProblems):
    def instantiate_prompt(self) -> str:
        return f"Compute: {self.a} * {self.b}"

    def ground_truth(self) -> int:
        return self.a * self.b


class LCS(DPProblem):
    def format_one(self) -> str:
        s = self.data["s"]
        t = self.data["t"]
        return f'Compute the length of the Longest Common Subsequence (LCS) between strings:\nS = "{s}"\nT = "{t}"'

    def ground_truth(self) -> int:
        return lcs_len(self.data["s"], self.data["t"])


class KnapSack(DPProblem):
    def format_one(self) -> str:
        w = self.data["weights"]
        v = self.data["values"]
        C = self.data["capacity"]
        return "0/1 Knapsack: Given item weights W and values V and capacity C, " "compute the maximum total value.\n" f"W = {w}\nV = {v}\nC = {C}"

    def ground_truth(self) -> int:
        return knap_01_max_value(self.data["weights"], self.data["values"], self.data["capacity"])


class RodCutting(DPProblem):
    def format_one(self) -> str:
        prices = self.data["prices"]
        N = len(prices)
        return "Rod cutting: Given a rod of length N and price list P[1..N], " "compute the maximum obtainable revenue.\n" f"N = {N}\nP = {prices}"

    def ground_truth(self) -> int:
        return rod_cut_max(self.data["prices"])


class ILPAssign(ILPProblem):
    def format_one(self) -> str:
        C = self.data["cost"]
        return (
            "Assignment problem: Given an n×n cost matrix C, assign each worker to one task "
            "minimizing the total cost. Return the minimum total cost as an integer. \n"
            f"C = {C}"
        )

    def ground_truth(self) -> int:
        return assignment_min_cost(self.data["cost"])


class ILPPartition(ILPProblem):
    def format_one(self) -> str:
        w = self.data["weights"]
        return (
            "Partition: Split the items into two groups to minimize the absolute difference between the sums. "
            "Return the minimum difference as an integer.\n"
            f"weights = {w}"
        )

    def ground_truth(self) -> int:
        return partition_min_diff(self.data["weights"])


class ILPProd(ILPProblem):
    def format_one(self) -> str:
        prof = self.data["profit"]
        cons = self.data["consumption"]
        caps = self.data["capacity"]
        ub = self.data["upper_bound"]
        return (
            "Production planning: Choose integer quantities x_j ≥ 0 to maximize total profit sum_j profit[j]*x_j, "
            "subject to resource constraints sum_j consumption[i][j]*x_j ≤ capacity[i]. Return the max profit.\n"
            f"profit = {prof}\nconsumption (rows=resources) = {cons}\ncapacity = {caps}\nupper_bounds = {ub}"
        )

    def ground_truth(self) -> int:
        return prodplan_max_profit(self.data)
