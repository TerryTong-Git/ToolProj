from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from typing import Any, List, Type

from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from src.exps_performance.algorithms import assignment_min_cost, knap_01_max_value, lcs_len, partition_min_diff, prodplan_max_profit, rod_cut_max
from src.exps_performance.logger import Record
from src.exps_performance.problems import CheckAndFormat, Question
from src.exps_performance.utils import rand_string, sample_int

clrs_desc = "Description: You are going to be given a set of algorithmic problem." "Question: Solve the following algorithmic problem: \n {question}"
func_typing = "int"


class FgAnswer(BaseModel):
    Answer: str = Field(description="The answer to the algorithmic problem. Type: int. Example: 1 ", default="")


@dataclass
class FgQuestion(Question):
    kind: str = "finegrained"
    digits: int = 0
    answer: str = ""
    question: str = ""
    record: Record = field(default_factory=Record)

    @property
    def util_pointer(self) -> Type["FgCheckAndFormat"]:
        return FgCheckAndFormat


class FgCheckAndFormat(CheckAndFormat):
    k: str

    def __init__(self, prob_type: str, n: int = 10, digits_list: List[int] = [2], seed: int = 1):
        super().__init__(prob_type, func_typing, clrs_desc, FgAnswer)
        self.instancetype = FgQuestion
        self.n = n
        self.digits_list = digits_list
        self.seed = seed

    def loaded_data_to_class(self, data: Any) -> Any:
        return data

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(str(code))
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"
        if isinstance(evaluated, int):
            return True
        else:
            return False

    # rename to code to class
    def get_field_kwargs(self, result: Any) -> dict[str, str]:
        return dict(Answer=str(result))

    @property
    def prompt(self) -> PromptTemplate:
        return self.prompt_template("question") if self.prob_type != "sim" else self.prompt_template("code")

    def format_one(self, q: FgQuestion) -> str:
        if self.prob_type == "sim":
            return str(self.prompt.format_prompt(code=q.code).to_string())
        prompt_text = self.prompt.format_prompt(question=q.question)
        return str(prompt_text.to_string())

    def decision_check(self, instance: FgAnswer, solution: BaseModel) -> tuple[bool, str]:
        str_ans = solution.Answer
        return str_ans == instance.answer, ""

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        rng1 = random.Random(2)
        a = sample_int(d, rng)
        b = sample_int(d, rng1)
        question = f"Compute: {a} + {b}"
        answer = int(float(a) + float(b))
        return FgQuestion(kind=self.k, digits=d, question=question, answer=str(answer))

    def load_data(self) -> list[FgQuestion]:
        rng = random.Random(self.seed)
        problems = []
        D = max(1, len(self.digits_list))
        per = max(1, self.n // (D))
        for d in self.digits_list:
            for _ in range(per):
                problems.append(self.make_problem(rng, d))
        return list(problems)


class AddCheckAndFormat(FgCheckAndFormat):
    k: str = "add"


class SubCheckAndFormat(FgCheckAndFormat):
    k: str = "sub"

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        rng1 = random.Random(2)
        a = sample_int(d, rng)
        b = sample_int(d, rng1)
        question = f"Compute: {a} - {b}"
        if b > a:
            a, b = b, a
        answer = int(float(a) - float(b))
        return FgQuestion(kind=self.k, digits=d, question=question, answer=str(answer))


class MulCheckAndFormat(FgCheckAndFormat):
    k: str = "mul"

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        rng1 = random.Random(2)
        a = sample_int(d, rng)
        b = sample_int(d, rng1)
        question = f"Compute: {a} * {b}"
        answer = int(float(a) * float(b))
        return FgQuestion(kind=self.k, digits=d, question=question, answer=str(answer))


class LcsCheckAndFormat(FgCheckAndFormat):
    k: str = "lcs"

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        n = int(d)  # complexity O(n^2)
        rng1 = random.Random(2)
        s = rand_string(rng, alpha="abcdefghijklmnopqrstuvwxyz", n=n)
        t = rand_string(rng1, alpha="abcdefghijklmnopqrstuvwxyz", n=n)

        question = f'Compute the length of the Longest Common Subsequence (LCS) between strings:\nS = "{s}"\nT = "{t}"'
        answer = lcs_len(s, t)
        return FgQuestion(kind="lcs", digits=d, question=question, answer=str(answer))


class Knap01CheckAndFormat(FgCheckAndFormat):
    k: str = "knapsack"

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        # complexity O(items * Capacity)
        capacity = max(2, int(d))
        n_items = max(2, int(d))
        # scale magnitudes gently with d to keep runtimes sane
        w_max = max(2, 32)
        v_max = max(2, 32)
        weights = [rng.randint(1, w_max) for _ in range(n_items)]
        values = [rng.randint(1, v_max) for _ in range(n_items)]
        question = (
            "0/1 Knapsack: Given item weights W and values V and capacity C, "
            "compute the maximum total value.\n"
            f"W = {weights}\nV = {values}\nC = {capacity}"
        )
        answer = knap_01_max_value(weights, values, capacity)
        return FgQuestion(kind="knap", digits=d, question=question, answer=str(answer))


class RodCheckAndFormat(FgCheckAndFormat):
    k: str = "rod"

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        N = max(2, int(d))  # O(n^2)
        price_max = 32
        prices = [rng.randint(1, price_max) for _ in range(N)]
        question = (
            "Rod cutting: Given a rod of length N and price list P[1..N], " "compute the maximum obtainable revenue.\n" f"N = {N}\nP = {prices}"
        )
        answer = rod_cut_max(prices)
        return FgQuestion(kind="rod", digits=d, question=question, answer=str(answer))


class IlpAssignCheckAndFormat(FgCheckAndFormat):
    k: str = "ilp_assign"

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        n = max(2, int(d))  # O(2^n)
        C = [[rng.randint(1, 32) for _ in range(n)] for __ in range(n)]
        question = (
            "Assignment problem: Given an n×n cost matrix C, assign each worker to one task "
            "minimizing the total cost. Return the minimum total cost as an integer. \n"
            f"C = {C}"
        )
        answer = assignment_min_cost(C)
        return FgQuestion(kind="ilp_assign", digits=d, question=question, answer=str(answer))


class IlpPartitionCheckAndFormat(FgCheckAndFormat):
    k: str = "ilp_partition"

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        n_items = max(2, int(d))
        w_max = 32
        weights = [rng.randint(1, w_max) for _ in range(n_items)]
        question = (
            "Partition: Split the items into two groups to minimize the absolute difference between the sums. "
            "Return the minimum difference as an integer.\n"
            f"weights = {weights}"
        )
        answer = partition_min_diff(weights)
        return FgQuestion(kind="ilp_partition", digits=d, question=question, answer=str(answer))


class IlpProdCheckAndFormat(FgCheckAndFormat):
    k: str = "ilp_prod"

    def make_problem(self, rng: random.Random, d: int) -> FgQuestion:
        # scale #products/#resources and magnitudes with d, but cap to keep fallback feasible
        P = max(2, int(d))
        R = max(2, int(d))
        profit = [rng.randint(2, 32) for _ in range(P)]
        consumption = [[rng.randint(2, 32) for _ in range(P)] for __ in range(R)]
        capacity = [rng.randint(2, 32) for _ in range(R)]
        upper = []
        for j in range(P):  # upper bound 32 so there is slack
            ub_j = min(32, min((capacity[i] // max(1, consumption[i][j]) for i in range(R)), default=32))
            upper.append(int(max(3, ub_j)))
        data = {
            "profit": profit,
            "consumption": consumption,
            "capacity": capacity,
            "upper_bound": upper,
        }
        question = (
            "Production planning: Choose integer quantities x_j ≥ 0 to maximize total profit sum_j profit[j]*x_j, "
            "subject to resource constraints sum_j consumption[i][j]*x_j ≤ capacity[i]. Return the max profit.\n"
            f"profit = {profit}\nconsumption (rows=resources) = {consumption}\ncapacity = {capacity}\nupper_bounds = {upper}"
        )
        answer = prodplan_max_profit(data)
        return FgQuestion(kind="ilp_prod", digits=d, question=question, answer=str(answer))
