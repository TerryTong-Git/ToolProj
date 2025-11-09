from __future__ import annotations

import ast
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from algorithms import assignment_min_cost, knap_01_max_value, lcs_len, partition_min_diff, prodplan_max_profit, rod_cut_max
from clrs.huggingface_generators import clrs_generator
from datasets import load_dataset
from prompts import bspPrompts, edpPrompts, gcp_dPrompts, gcpPrompts, kspPrompts, mfpPrompts, mspPrompts, sppPrompts, tsp_dPrompts, tspPrompts
from tqdm import tqdm
from utils import read_dimacs_format


def load_NPHardEval() -> Sequence[Problem]:
    all_subclasses = NPHardEvalProblem.__subclasses__()
    file_path = os.path.join(Path(__name__).parent, "Data_V2")
    all_data: List[Problem] = []
    for CLASS in all_subclasses:
        if CLASS is NPHardEvalProblem:
            continue
        data = CLASS().load_data(os.path.join(file_path, CLASS.folder_name))  # type: ignore[abstract]
        all_data += data
    return all_data


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


def load_CLRS30() -> Sequence[Problem]:
    clrs = CLRS()
    return clrs.load_data()


# make this an ABC.
@dataclass
class Problem:
    kind: str = "null"
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

    def decision_check(self, answer, problem_text=None):
        return int(answer == self.ground_truth())


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

    def decision_check(self, answer, problem_text=None):
        return int(answer == self.ground_truth())


@dataclass
class CLRSProblem:
    kind: str = "clrs"
    digits: int = 0
    answer: str = ""
    text_data: str = ""

    def text(
        self,
    ):
        return self.text_data

    def decision_check(self, computed_ans, problem_text=None):
        str_ans = str(computed_ans)
        import pdb

        pdb.set_trace()
        return int(str_ans == self.answer)

    def ground_truth(self):
        return self.answer


def parse_gsm8k_gold(ans: str) -> int:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1))  # type: ignore


def check_parse_gsm8k_gold(ans: str) -> Optional[int]:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1)) if m else None


class CLRS:
    def load_data(self):
        _DEFAULT_VAL_ALGOS_AND_LENGTHS = {
            "activity_selector": list(range(4, 41)),
            "articulation_points": list(range(4, 20)),
            "bellman_ford": list(range(4, 33)),
            "bfs": list(range(4, 42)),
            "binary_search": list(range(4, 65)),
            "bridges": list(range(4, 8)),
            "bubble_sort": list(range(4, 12)),
            "dag_shortest_paths": list(range(4, 20)),
            "dfs": list(range(4, 21)),
            "dijkstra": list(range(4, 26)),
            "find_maximum_subarray_kadane": list(range(4, 65)),
            "floyd_warshall": list(range(4, 12)),
            "graham_scan": list(range(4, 32)),
            "heapsort": list(range(4, 12)),
            "insertion_sort": list(range(4, 26)),
            "jarvis_march": list(range(4, 14)),
            "kmp_matcher": list(range(4, 65)),
            "lcs_length": list(range(4, 13)),
            "matrix_chain_order": list(range(4, 13)),
            "minimum": list(range(4, 65)),
            "mst_kruskal": list(range(4, 11)),
            "mst_prim": list(range(4, 27)),
            "naive_string_matcher": list(range(4, 65)),
            "optimal_bst": list(range(4, 11)),
            "quickselect": list(range(4, 65)),
            "quicksort": list(range(4, 13)),
            "segments_intersect": list(range(4, 65)),
            "strongly_connected_components": list(range(4, 17)),
            "task_scheduling": list(range(4, 42)),
            "topological_sort": list(range(4, 22)),
        }
        _DEFAULT_VAL_NUMBER_OF_SAMPLES = 2000
        _DEFAULT_VAL_SEEDS = [0]
        for seed in _DEFAULT_VAL_SEEDS:
            data = clrs_generator(_DEFAULT_VAL_ALGOS_AND_LENGTHS, _DEFAULT_VAL_NUMBER_OF_SAMPLES, use_hints=False, seed=seed)
        return [CLRSProblem(d["algo_name"], d["length"], answer=re.sub(r"\s+", "", d["answer"]), text_data=d["question"]) for d in data][:100]


# NPHARD EVAL PROBLEMS #
class NPHardProblem(Problem):  # to conform to Problem interface
    def __init__(self, text):
        self.text_data = text

    def text(self):
        return self.text_data

    def ground_truth(self):  # dummy
        return None


class NPHardEvalProblem(ABC):
    folder_name: str

    def format(self, qs) -> Sequence[Problem]:
        all_prompts = []
        for q in tqdm(qs):
            prompt_text = self.format_one(q)
            all_prompts.append(NPHardProblem(prompt_text))
        return all_prompts

    @abstractmethod
    def load_data(self, data_path: str) -> Sequence[Problem]:
        raise NotImplementedError

    def instantiate_prompt(self, kwargs):
        return self.p["Intro"] + "\n" + self.p["Initial_question"].format(**kwargs) + "\n" + self.p["Output_format"] + "\n"

    @abstractmethod
    def format_one(self, q):
        raise NotImplementedError

    @abstractmethod
    def decision_check(self, q, output):
        raise NotImplementedError


class GCP_D(NPHardEvalProblem):
    folder_name = "GCP_Decision"

    def __init__(self):
        self.p = gcp_dPrompts

    def load_data(self, data_path):
        all_data = []
        n = 10
        start = n - 9
        for file_num in range(start, n):
            with open(data_path + "decision_data_GCP_{}.txt".format(file_num)) as f:
                data = f.read()
            all_data += data.split("\n\n")[:-1]
        return self.format(all_data)

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

    def decision_check(self, q, output):
        number_of_colors = int(q.split("\n")[0].split()[-2])
        return self.gcp_decision_check(q, output, number_of_colors)

    def gcp_greedy_solution(self, adjacency_list):
        """Provides a greedy solution to the GCP problem.

        :param adjacency_list: A dictionary of the adjacency list.
        :return: A tuple of (num_colors, coloring).
        """
        G = nx.Graph()
        G.add_nodes_from(adjacency_list.keys())
        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                G.add_edge(vertex, neighbor)
        coloring = nx.coloring.greedy_color(G, strategy="largest_first")
        num_colors = max(coloring.values()) + 1
        return num_colors, coloring

    def gcp_decision_check(self, dimacs_str, answer, k_colors):
        """
        Check if the given GCP instance is feasible with k_colors.

        :param dimacs_str: The DIMACS format string of the GCP instance.
        :param answer: The answer returned by the model.
        :param k_colors: The target number of colors.
        :return: A tuple of (is_correct, message).
        """
        num_vertices, adjacency_list = read_dimacs_format(dimacs_str)
        try:
            is_feasible = answer.get("Feasible", "no").lower() == "yes"
        except ValueError:
            return False, "Feasible key not found"
        num_colors, coloring = self.gcp_greedy_solution(adjacency_list)
        exist_optimal = num_colors <= k_colors
        if is_feasible != exist_optimal:
            if exist_optimal:
                return False, f"Feasibility mismatch: {coloring}"
            else:
                return False, f"Feasibility mismatch: {is_feasible} vs {exist_optimal}"
        return True, "Feasible" if is_feasible else "Infeasible"


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

    def load_data(self, data_path):
        with open(data_path + "ksp_instances.json", "r") as f:
            all_data = json.load(f)
        return all_data

    def decision_check(self, q, output):
        return self.kspCheck(q, output)

    def ksp_optimal_solution(self, knapsacks, capacity):
        """Provides the optimal solution for the KSP instance with dynamic programming.

        :param knapsacks: A dictionary of the knapsacks.
        :param capacity: The capacity of the knapsack.
        :return: The optimal value.
        """
        # Create a one-dimensional array to store intermediate solutions
        dp = [0] * (capacity + 1)
        for itemId, (weight, value) in knapsacks.items():
            for w in range(capacity, weight - 1, -1):
                dp[w] = max(dp[w], value + dp[w - weight])

        return dp[capacity]

    # KSP
    def kspCheck(self, instance, solution):
        """Validates the solution for the KSP instance.

        :param instance: A dictionary of the KSP instance.
        :param solution: A dictionary of the solution.
        :return: A tuple of (is_correct, message).
        """
        # Change string key to integer key and value to boolean
        items = instance.get("items", [])
        knapsacks = {item["id"]: (item["weight"], item["value"]) for item in items}

        ksp_optimal_value = self.ksp_optimal_solution(knapsacks, instance["knapsack_capacity"])

        try:
            is_feasible = solution.get("Feasible", "").lower() == "yes"
        except ValueError:
            return False, "Output format is incorrect."
        if is_feasible != (ksp_optimal_value > 0):
            return False, f"The solution is {is_feasible} but the optimal solution is {ksp_optimal_value > 0}."

        try:
            total_value = int(solution.get("TotalValue", -1))
            selectedItems = list(map(int, solution.get("SelectedItemIds", [])))
        except ValueError:
            return False, "Output format is incorrect."

        if len(set(selectedItems)) != len(selectedItems):
            return False, "Duplicate items are selected."

        total_weight = 0
        cum_value = 0

        # Calculate total weight and value of selected items
        for item in selectedItems:
            if knapsacks.get(item, False):
                weight, value = knapsacks[item]
                total_weight += weight
                cum_value += value
            else:
                return False, f"Item {item} does not exist."

        # Check if the item weight exceeds the knapsack capacity
        if total_weight > instance["knapsack_capacity"]:
            return False, f"Total weight {total_weight} exceeds knapsack capacity {instance['knapsack_capacity']}."

        if total_value != cum_value:
            return False, f"The total value {total_value} does not match the cumulative value {cum_value} of the selected items."

        if total_value != ksp_optimal_value:
            return False, f"The total value {total_value} does not match the optimal value {ksp_optimal_value}."

        return True, f"The solution is valid with total weight {total_weight} and total value {total_value}."


class TSP_D(NPHardEvalProblem):
    def __init__(self):
        self.p = tsp_dPrompts

    def load_data(self, data_path):
        n = 11
        start = n - 10
        all_data = []
        for level in range(start, n):
            for file_num in range(10):
                df = pd.read_csv(data_path + "decision_data_TSP_level_{}_instance_{}.csv".format(level, file_num + 1), header=None, index_col=False)
                all_data.append(df)
        return all_data

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

    def tsp_approx(self, distance_matrix):
        """Returns an approximate solution to the TSP problem.

        :param distance_matrix: A 2D numpy array representing the distance matrix.
        :return: A list of the cities in the order they were visited.
        """
        G = nx.from_numpy_array(distance_matrix)
        return nx.approximation.traveling_salesman_problem(G)

    def tsp_decision_check(self, distance_matrix, threshold, tour):
        """
        Checks if a given TSP tour is valid and within the threshold distance.

        :param distance_matrix: A 2D numpy array representing the distance matrix.
        :param threshold: The maximum distance allowed.
        :param tour: A dictionary containing the feasibility.
        """
        try:
            is_feasible = tour.get("Feasible", "no").lower() == "yes"
        except:  # noqa: E722
            return False, "Output format incorrect"

        # Calculate the approxed distance of the tour
        tours = self.tsp_approx(distance_matrix)
        tour_distance = sum(distance_matrix[tours[i], tours[i + 1]] for i in range(len(tours) - 1)) + distance_matrix[tours[-1], tours[0]]

        if is_feasible != (tour_distance <= threshold):
            return False, f"Feasibility mismatch: {is_feasible} vs {tour_distance} > {threshold}"
        return True, "Feasible: {} <= {}".format(tour_distance, threshold)

    def decision_check(self, q, output):
        threshold = q.iloc[-1, 0]  # therashold is the last row
        distance_matrix = q.iloc[:-1].values  # distance matrix is the rest of the rows
        return self.tsp_decision_check(distance_matrix, threshold, output)


class TSP(NPHardEvalProblem):
    def __init__(self):
        self.p = tspPrompts

    def format_one(self, q):
        total_cities = q.shape[0]
        prompt_text = self.instantiate_prompt(dict(total_cities=total_cities)) + "The distances between cities are below: \n"

        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                if i < j:  # only use the upper triangle
                    this_line = "The path between City {} and City {} is with distance {}.".format(i, j, q.iloc[i, j])
                    prompt_text += this_line + "\n"

        return prompt_text

    def load_data(self, data_path):
        n = 11
        all_data = []
        start = n - 10
        for level in range(start, n):
            for file_num in range(10):
                # read np arrary
                df = pd.read_csv(
                    data_path + "synthesized_data_TSP_level_{}_instance_{}.csv".format(level, file_num + 1), header=None, index_col=False
                )
                # transform df to
                all_data.append(df)
        return all_data

    def greedy_tsp(self, distance_matrix):
        """
        Solve the Traveling Salesman Problem using a greedy algorithm.

        :param distance_matrix: 2D numpy array where the element at [i, j] is the distance between city i and j
        :return: A tuple containing a list of the cities in the order they were visited and the total distance
        """
        num_cities = distance_matrix.shape[0]
        unvisited_cities = set(range(num_cities))
        current_city = np.random.choice(list(unvisited_cities))
        tour = [current_city]
        total_distance = 0

        while unvisited_cities:
            unvisited_cities.remove(current_city)
            if unvisited_cities:
                # Find the nearest unvisited city
                distances_to_unvisited = distance_matrix[current_city][list(unvisited_cities)]
                nearest_city = list(unvisited_cities)[np.argmin(distances_to_unvisited)]
                tour.append(nearest_city)
                # Update the total distance
                total_distance += distance_matrix[current_city, nearest_city]
                current_city = nearest_city

        # Return to start
        total_distance += distance_matrix[current_city, tour[0]]
        tour.append(tour[0])

        return tour, total_distance

    def tspCheck(self, distance_matrix, final_answer_element):
        """
        Check if the TSP solution is complete and if the distance matches the greedy solution.

        :param tour_string: String representing the TSP tour in the format "0->1->2->...->N->0"
        :param distance_matrix: 2D numpy array representing the distances between cities
        :return: Boolean indicating whether the tour is complete and matches the greedy distance
        """
        # convert distance_matrix to numpy array
        distance_matrix = np.array(distance_matrix)

        # Convert the tour string to a list of integers
        # convert solution to dictionary
        if final_answer_element == "":
            return False
        elif final_answer_element is None:
            return False
        else:
            if isinstance(final_answer_element, str):
                try:
                    tour_string = ast.literal_eval(final_answer_element)["Path"]
                    if tour_string is None:
                        return False
                except:  # noqa: E722
                    try:
                        tour_string = ast.literal_eval("{" + final_answer_element + "}")["Path"]
                        if tour_string is None:
                            return False
                    except:  # noqa: E722
                        return False
            else:
                try:
                    tour_string = ast.literal_eval(final_answer_element.text)["Path"]
                    if tour_string is None:
                        return False
                except:  # noqa: E722
                    return False
        try:
            tour = list(map(int, tour_string.split("->")))
        except:  # noqa: E722
            return False
        # we could also prinpt `reasoning_element` to see the reasoning of the answer
        # we could also print the final distance of the tour by `final_answer_element['Distance']`

        # Check if tour is a cycle
        if tour[0] != tour[-1]:
            return False, "The tour must start and end at the same city."

        # Check if all cities are visited
        if len(tour) != len(distance_matrix) + 1:
            return False, "The tour does not visit all cities exactly once."

        # Calculate the distance of the provided tour
        tour_distance = sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

        # Find the greedy tour distance for comparison
        greedy_tour, greedy_distance = self.greedy_tsp(distance_matrix)

        # Check if the provided tour distance is equal to the greedy tour distance
        if tour_distance != greedy_distance:
            return False, f"The tour distance ({tour_distance}) does not match the greedy solution ({greedy_distance})."

        return True, "The solution is complete and matches the greedy solution distance."


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
