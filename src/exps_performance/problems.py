from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from algorithms import assignment_min_cost, knap_01_max_value, lcs_len, partition_min_diff, prodplan_max_profit, rod_cut_max
from datasets import load_dataset
from prompts import bspPrompts, edpPrompts, gcp_dPrompts, gcpPrompts, kspPrompts, mfpPrompts, mspPrompts, sppPrompts, tsp_dPrompts
from tqdm import tqdm


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


# NPHARD EVAL PROBLEMS #


class NPHardEvalProblem:
    def format(self, qs):
        all_prompts = []
        for q in tqdm(qs):
            prompt_text = self.format_one(q)
            all_prompts.append(prompt_text)
        # output = run_opensource_models(args, MODEL, all_prompts)
        # return output

    def instantiate_prompt(self, kwargs):
        return self.p["Intro"] + "\n" + self.p["Initial_question"].format(**kwargs) + "\n" + self.p["Output_format"] + "\n"

    def format_one(self, q):
        raise NotImplementedError

    def check(self):
        raise NotImplementedError


class GCP_D(NPHardEvalProblem):
    def __init__(self):
        self.p = gcp_dPrompts

    def format_one(self, q):
        number_of_colors = q.split("\n")[0].split()[-2]  # last character of the first line
        number_of_vertices = q.split("\n")[1].split(" ")[2]  # third word of the second line
        prompt_text = (
            self.instantiate_prompt(dict(total_vertices=number_of_vertices, number_of_colors=number_of_colors)) + "\n The graph is below: \n"
        )
        for line in q.split("\n")[2:]:
            vertex_list = line.split(" ")
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1], vertex_list[2])
            prompt_text += this_line + "\n"
        return prompt_text


class KSP(NPHardEvalProblem):
    def __init__(self):
        self.p = kspPrompts

    def format_one(self, q):
        knapsack_capacity = q["knapsack_capacity"]
        items = q["items"]
        prompt_text = self.instantiate_prompt(dict(knapsack_capacity=knapsack_capacity)) + "\n The items details are as below: \n"
        for item in items:
            this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
            prompt_text += this_line + "\n"
        return prompt_text


class TSP(NPHardEvalProblem):
    def __init__(self):
        self.p = tsp_dPrompts

    def format_one(self, q):
        threshold = q.iloc[-1, 0]  # therashold is the last row
        adj_matrix = q.iloc[:-1].values  # distance matrix is the rest of the rows
        total_cities = adj_matrix.shape[0]  # exclude the last row
        prompt_text = (
            self.instantiate_prompt(dict(total_cities=total_cities, distance_limit=threshold)) + "The distances between cities are below: \n"
        )

        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i < j:  # only use the upper triangle
                    this_line = "The distance between City {} and City {} is {}.".format(i, j, adj_matrix[i, j])
                    prompt_text += this_line + "\n"
        return prompt_text


class GCP(NPHardEvalProblem):
    def __init__(self):
        self.p = gcpPrompts

    def format_one(self, q):
        chromatic_number = q.split("\n")[0][-1]  # last character of the first line
        number_of_vertices = q.split("\n")[1].split(" ")[2]  # third word of the second line
        prompt_text = self.instantiate_prompt(dict(max_vertices=number_of_vertices, max_colors=chromatic_number)) + "\n The graph is below: \n"
        for line in q.split("\n")[2:]:
            vertex_list = line.split(" ")
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1], vertex_list[2])
            prompt_text += this_line + "\n"
        return prompt_text


class MSP(NPHardEvalProblem):
    def __init__(self):
        self.p = mspPrompts

    def format_one(self, q) -> str:
        participants = q["participants"]
        prompt_text = (
            self.instantiate_prompt(dict(total_participants=participants, total_timeslots=q["time_slots"]))
            + "\n The meetings and participants details are as below: \n"
        )

        for meeting in q["meetings"]:
            this_line = "Meeting {} is with duration {}.".format(meeting["id"], meeting["duration"])
            prompt_text += this_line + "\n"
        for j in participants.keys():
            this_line = "Participant {} is available at time slots {} and has meetings {}.".format(
                j, participants[j]["available_slots"], participants[j]["meetings"]
            )
            prompt_text += this_line + "\n"
        return prompt_text


class BSP(NPHardEvalProblem):
    def __init__(self):
        self.p = bspPrompts

    def format_one(self, q):
        target_value = q["target"]
        # TO-DO: fix data not being sorted
        array = sorted(q["array"])
        prompt_text = (
            self.instantiate_prompt(dict(target_value=target_value)) + "\n The sorted array elements are: " + ", ".join(map(str, array)) + "\n"
        )
        return prompt_text


class EDP(NPHardEvalProblem):
    def __init__(self):
        self.p = edpPrompts

    def format_one(self, q):
        string_a = q["string_a"]
        string_b = q["string_b"]
        prompt_text = self.instantiate_prompt(dict(string_a=string_a, string_b=string_b))
        prompt_text += "Answer:\n"
        return prompt_text


class MFP(NPHardEvalProblem):
    def __init__(self):
        self.p = mfpPrompts

    def format_one(self, q):
        source_node = q["source"]
        sink_node = q["sink"]
        edges = q["edges"]
        prompt_text = (
            self.instantiate_prompt(dict(source_node=source_node, sink_node=sink_node))
            + "\n\n"
            + "Here is a network description. The capacities of the network's edges are as follows: \n"
        )
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
            prompt_text += this_line + "\n"
        prompt_text += "Answer:\n"
        return prompt_text


class SPP(NPHardEvalProblem):
    def __init__(self):
        self.p = sppPrompts

    def format_one(self, q):
        start_node = q["nodes"][0]
        end_node = q["nodes"][-1]
        edges = q["edges"]
        prompt_text = self.instantiate_prompt(dict(start_node=start_node, end_node=end_node)) + "\n The graph's edges and weights are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
            prompt_text += this_line + "\n"
        prompt_text += "Answer:\n"
        return prompt_text
