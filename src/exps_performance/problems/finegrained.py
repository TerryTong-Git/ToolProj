from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from typing import List, Sequence

from pydantic import BaseModel, Field

from src.exps_performance.algorithms import assignment_min_cost, knap_01_max_value, lcs_len, partition_min_diff, prodplan_max_profit, rod_cut_max
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

    @property
    def util_pointer(self):
        return FgCheckAndFormat


class FgCheckAndFormat(CheckAndFormat):
    k: str

    def __init__(self, prob_type, n: int = 10, digits_list: List[int] = [2], seed: int = 1):
        super().__init__(prob_type, func_typing, clrs_desc, FgAnswer)
        self.instancetype = FgQuestion
        self.n = n
        self.digits_list = digits_list
        self.seed = seed

    def loaded_data_to_class(self, data):
        return data

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(code)
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"
        if isinstance(evaluated, int):
            return True
        else:
            return False

    # rename to code to class
    def get_field_kwargs(self, result):
        return dict(Answer=str(result))

    @property
    def prompt(self):
        return self.prompt_template(["question"]) if self.prob_type != "sim" else self.prompt_template(["code"])

    def format_one(self, q: FgQuestion) -> str:
        if self.prob_type == "sim":
            return self.prompt.format_prompt(code=q.code).to_string()
        prompt_text = self.prompt.format_prompt(question=q.question)
        return prompt_text.to_string()

    def decision_check(self, instance: FgAnswer, solution: BaseModel):
        str_ans = solution.Answer
        return int(str_ans == instance.answer), ""

    def make_problem(self, rng, d):
        a = sample_int(d, rng)
        b = sample_int(d, rng)
        question = f"Compute: {a} + {a}"
        answer = a + b
        return FgQuestion(kind=self.k, digits=d, question=question, answer=str(answer))

    def load_data(self) -> Sequence[FgQuestion]:
        rng = random.Random(self.seed)
        problems = []
        D = max(1, len(self.digits_list))
        per = max(1, self.n // (D))
        for d in self.digits_list:
            for _ in range(per):
                problems.append(self.make_problem(rng, d))
        return problems


class AddCheckAndFormat(FgCheckAndFormat):
    k: str = "add"


class SubCheckAndFormat(FgCheckAndFormat):
    k: str = "sub"

    def make_problem(self, rng, d):
        a = sample_int(d, rng)
        b = sample_int(d, rng)
        question = f"Compute: {a} - {a}"
        if b > a:
            a, b = b, a
        answer = a - b
        return FgQuestion(kind=self.k, digits=d, question=question, answer=str(answer))


class MulCheckAndFormat(FgCheckAndFormat):
    k: str = "mul"

    def make_problem(self, rng, d):
        a = sample_int(d, rng)
        b = sample_int(d, rng)
        question = f"Compute: {a} * {a}"
        answer = a * b
        return FgQuestion(kind=self.k, digits=d, question=question, answer=str(answer))


class LcsCheckAndFormat(FgCheckAndFormat):
    k: str = "lcs"

    def make_problem(self, rng, d):
        n = max(2, int(d))  # max 2 digit lcs
        s = rand_string(rng, alpha="abcd", n=n)
        t = rand_string(rng, alpha="abcd", n=n + rng.randint(-1, 1))
        question = f'Compute the length of the Longest Common Subsequence (LCS) between strings:\nS = "{s}"\nT = "{t}"'
        answer = lcs_len(s, t)
        return FgQuestion(kind="lcs", digits=d, question=question, answer=str(answer))


class Knap01CheckAndFormat(FgCheckAndFormat):
    k: str = "knapsack"

    def make_problem(self, rng, d):
        n_items = max(3, int(d))
        # scale magnitudes gently with d to keep runtimes sane
        w_max = max(5, 2 * d)  # this cap may make some results disingenuous
        v_max = max(10, 4 * d)
        weights = [rng.randint(1, w_max) for _ in range(n_items)]
        values = [rng.randint(1, v_max) for _ in range(n_items)]
        capacity = max(1, int(0.5 * sum(weights)))
        question = (
            "0/1 Knapsack: Given item weights W and values V and capacity C, "
            "compute the maximum total value.\n"
            f"W = {weights}\nV = {values}\nC = {capacity}"
        )
        answer = knap_01_max_value(weights, values, capacity)
        return FgQuestion(kind="knap", digits=d, question=question, answer=str(answer))


class RodCheckAndFormat(FgCheckAndFormat):
    k: str = "rod"

    def make_problem(self, rng, d):
        N = max(2, int(d))
        price_max = max(5, 3 * d)
        prices = [rng.randint(1, price_max) for _ in range(N)]
        question = (
            "Rod cutting: Given a rod of length N and price list P[1..N], " "compute the maximum obtainable revenue.\n" f"N = {N}\nP = {prices}"
        )
        answer = rod_cut_max(prices)
        return FgQuestion(kind="rod", digits=d, question=question, answer=str(answer))


class IlpAssignCheckAndFormat(FgCheckAndFormat):
    k: str = "ilp_assign"

    def make_problem(self, rng, d):
        n = max(2, min(int(d), 7))  # cap n for brute-force fallback safety
        C = [[rng.randint(1, max(6, 3 * d)) for _ in range(n)] for __ in range(n)]
        question = (
            "Assignment problem: Given an n×n cost matrix C, assign each worker to one task "
            "minimizing the total cost. Return the minimum total cost as an integer. \n"
            f"C = {C}"
        )
        answer = assignment_min_cost(C)
        return FgQuestion(kind="ilp_assign", digits=d, question=question, answer=answer)


class IlpPartitionCheckAndFormat(FgCheckAndFormat):
    k: str = "ilp_partition"

    def make_problem(self, rng, d):
        n_items = max(4, min(int(d), 24))
        w_max = max(6, 3 * d)
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

    def make_problem(self, rng, d):
        # scale #products/#resources and magnitudes with d, but cap to keep fallback feasible
        P = max(2, min(2 + d // 3, 6))  # benchmark runtime of problems
        R = max(2, min(2 + d // 4, 4))
        profit = [rng.randint(3, max(8, 3 * d)) for _ in range(P)]
        consumption = [[rng.randint(1, max(3, d)) for _ in range(P)] for __ in range(R)]
        # capacity scaled so some slack exists; upper bounds smallish (<= 10)
        capacity = [rng.randint(max(6, 2 * d), max(10, 4 * d)) for _ in range(R)]
        upper = []
        for j in range(P):
            ub_j = min(10, min((capacity[i] // max(1, consumption[i][j]) for i in range(R)), default=10))
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
