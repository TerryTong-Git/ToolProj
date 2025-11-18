from __future__ import annotations

from abc import abstractmethod
from dataclasses import field
from typing import Any, Dict

from pydantic import BaseModel, Field

from src.exps_performance.algorithms import assignment_min_cost, knap_01_max_value, lcs_len, partition_min_diff, prodplan_max_profit, rod_cut_max
from src.exps_performance.problems import Problem

sppPrompts = (
    "Description: The Shortest Path Problem (SPP) involves finding the shortest path between two nodes in a weighted graph."
    "Question: You need to find the shortest path between node {start_node} and node {end_node} in a graph. The graph's edges and their weights are given. {edges}. "
    " \n ONLY RETURN ACCORDING TO THE FORMAT!!! THIS IS REALLY IMPORTANT. DO USE PROSE BEFORE OR AFTER THE FORMAT. Here are the format instructions: {format_instructions}"
)

sppPrompts_nl = (
    "Description: The Shortest Path Problem (SPP) involves finding the shortest path between two nodes in a weighted graph."
    "Question: You need to find the shortest path between node {start_node} and node {end_node} in a graph. The graph's edges and their weights are given. {edges}. "
    " \n ONLY RETURN STRUCTURED OUTPUT ACCORDING TO THE FORMAT!!! THIS IS REALLY IMPORTANT. DO NOT TALK TO ME, OR GIVE ME A DESCRIPTION, JUST GIVE THE FORMAT. DO USE PROSE OR CODE BEFORE OR AFTER THE GIVEN FORMAT. YOU ARE NEVER ALLOWED TO USE CODE. Here are the format instructions: {format_instructions}"
)

sim_template = "Simulate the execution of the provided code: {code} \n ONLY RETURN ACCORDING TO THE FORMAT!!! THIS IS REALLY IMPORTANT. DO USE PROSE BEFORE OR AFTER THE FORMAT. Here are the format instructions: {format_instructions}"


# a local variable called answer should hold the answer? Then when I run the code, I can extract the local variable?
class CodeReasoning(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block of import statements. Define all necessary imports that are necessary to execute the below code")
    code: str = Field(description="The code block without imports that solves the problem. This should be a self-contained function.")
    code_answer: str = Field(
        description="The final piece of the code block that calls the function specified in the code block and puts the answer in a local variable 'answer'. ALWAYS USE THE VARIABLE NAME 'answer'. Make sure the code is executable in its entirety by defining all the variables necessary for execution from the text, e.g. the graph variables. For example, for a function FUNC(a,b) with a=1 and b=2, set answer = Function(1,2)"
    )
    simulation: str = Field(description="The attempt at simulating the code in natural language reasoning to give the final answer.")


class AdditionCodeReasoning(CodeReasoning):
    Answer: str = Field(
        description="This is part of the final answer, and the sum that the code simulation gives. This should not be a piece of code, but rather, an instance of the answer. Give just the answer as an integer. For example: '0'. Answers without this are completely wrong"
    )


class NLReasoning(BaseModel):
    reasoning: str = Field(
        description="The attempt at simulating the problem in natural language reasoning to give the final answer. YOU ARE NEVER ALLOWED TO GENERATE CODE."
    )


class AdditionNLReasoning(NLReasoning):
    Path: str = Field(
        description="This is part of the final answer, and the path that the natural language reasoning simulation gives. Give the answer as an integer. For example: '0'. Answers without this are completely wrong"
    )


class ControlledCodeSim(BaseModel):
    simulation: str = Field(
        description="The attempt at simulating the code in natural language reasoning to give the final answer. ALL NECESSARY INFORMATION IS IN THE CODE PROVIDED. If you don't know, just say you don't know."
    )


class AdditionSim(ControlledCodeSim):
    Answer: str = Field(
        description="This is part of the final answer, and the sum that the code simulation gives. This should not be a piece of code, but rather, an instance of the answer. Give just the answer as an integer. For example: '0'. Answers without this are completely wrong"
    )


class FineGrainedProblems(Problem):
    def format_one(self, q: Any) -> str:
        return self.instantiate_prompt()

    def decision_check(self, answer, problem_text=None):
        return int(answer == self.ground_truth())

    @abstractmethod
    def ground_truth(self):
        raise NotImplementedError

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
    def format_one(self, q: Any) -> str:
        s = self.data["s"]
        t = self.data["t"]
        return f'Compute the length of the Longest Common Subsequence (LCS) between strings:\nS = "{s}"\nT = "{t}"'

    def ground_truth(self) -> int:
        return lcs_len(self.data["s"], self.data["t"])


class KnapSack(DPProblem):
    def format_one(self, q: Any) -> str:
        w = self.data["weights"]
        v = self.data["values"]
        C = self.data["capacity"]
        return "0/1 Knapsack: Given item weights W and values V and capacity C, " "compute the maximum total value.\n" f"W = {w}\nV = {v}\nC = {C}"

    def ground_truth(self) -> int:
        return knap_01_max_value(self.data["weights"], self.data["values"], self.data["capacity"])


class RodCutting(DPProblem):
    def format_one(self, q: Any) -> str:
        prices = self.data["prices"]
        N = len(prices)
        return "Rod cutting: Given a rod of length N and price list P[1..N], " "compute the maximum obtainable revenue.\n" f"N = {N}\nP = {prices}"

    def ground_truth(self) -> int:
        return rod_cut_max(self.data["prices"])


class ILPAssign(ILPProblem):
    def format_one(self, q: Any) -> str:
        C = self.data["cost"]
        return (
            "Assignment problem: Given an n×n cost matrix C, assign each worker to one task "
            "minimizing the total cost. Return the minimum total cost as an integer. \n"
            f"C = {C}"
        )

    def ground_truth(self) -> int:
        return assignment_min_cost(self.data["cost"])


class ILPPartition(ILPProblem):
    def format_one(self, q: Any) -> str:
        w = self.data["weights"]
        return (
            "Partition: Split the items into two groups to minimize the absolute difference between the sums. "
            "Return the minimum difference as an integer.\n"
            f"weights = {w}"
        )

    def ground_truth(self) -> int:
        return partition_min_diff(self.data["weights"])


class ILPProd(ILPProblem):
    def format_one(self, q: Any) -> str:
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
