from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Any, Type, cast

import networkx as nx
from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NpCheckAndFormat, NpQuestion
from src.exps_performance.utils import read_dimacs_format

gcp_desc = (
    "Description: The Graph Coloring Problem (Decision Version, GCP-D) involves determining if it is possible to color the vertices of a graph using a given number of colors, ensuring no two adjacent vertices have the same color."
    "Question: Find out if the vertices of a graph with {total_vertices} vertices can be colored using only {number_of_colors} colors, such that no adjacent vertices share the same color. \n {graph}"
)

func_typing = "bool"  # (Path, TotalDistance)


class GcpdAnswer(BaseModel):
    Feasible: str = Field(description="The feasibility. Type: bool. Return 'True' or 'False'. ", default="")


@dataclass
class GcpdQuestion(NpQuestion):
    kind: str = "gcp_d"
    type: str = "code"  # could be sim, nl etc
    dimacs_str: str = ""
    code: str = ""

    @property
    def util_pointer(self) -> Type[NpCheckAndFormat]:
        return GcpdCheckAndFormat


class GcpdCheckAndFormat(NpCheckAndFormat):
    def __init__(self, prob_type: str):
        super().__init__(prob_type, func_typing, gcp_desc, GcpdAnswer)
        self.instancetype = GcpdQuestion

    # tied to inputs, may not be called input
    def loaded_data_to_class(self, data: Any) -> dict[str, str]:
        return dict(dimacs_str=str(data))

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(str(code))
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"

        if not isinstance(evaluated, dict):
            return False  # "Not a dict"
        else:
            for vertex, color in evaluated.items():
                if not isinstance(vertex, int):
                    return False
                if not isinstance(color, str) or len(color) > 1:
                    return False
        return True

    # tied to code
    def get_field_kwargs(self, result: Any) -> dict[str, str]:
        return dict(Feasible=str(result))

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self) -> Any:
        return self.prompt_template(["total_vertices", "number_of_colors", "graph"]) if self.prob_type != "sim" else self.prompt_template("code")

    def format_one(self, q: GcpdQuestion) -> str:
        if self.prob_type == "sim":
            return str(self.prompt.format_prompt(code=q.code).to_string())
        inp = q.dimacs_str
        number_of_colors = inp.split("\n")[0].split()[-2]  # last character of the first line
        number_of_vertices = inp.split("\n")[1].split(" ")[2]  # third word of the second line
        graph = "\n The graph is below: \n"

        for line in inp.split("\n")[2:]:
            vertex_list = line.split(" ")
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1], vertex_list[2])
            graph += this_line + "\n"
        prompt_text = self.prompt.format_prompt(total_vertices=number_of_vertices, number_of_colors=number_of_colors, graph=graph)
        return str(prompt_text.to_string())

    def decision_check(self, q: GcpdQuestion, output: BaseModel) -> tuple[bool, str]:
        number_of_colors = int(q.dimacs_str.split("\n")[0].split()[-2])
        return self.gcp_decision_check(q.dimacs_str, output, number_of_colors)

    def gcp_greedy_solution(self, adjacency_list: dict) -> tuple[int, dict[int, int]]:
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

    def gcp_decision_check(self, dimacs_str: str, answer: BaseModel, k_colors: int) -> tuple[bool, str]:
        """
        Check if the given GCP instance is feasible with k_colors.

        :param dimacs_str: The DIMACS format string of the GCP instance.
        :param answer: The answer returned by the model.
        :param k_colors: The target number of colors.
        :return: A tuple of (is_correct, message).
        """
        num_vertices, adjacency_list = read_dimacs_format(dimacs_str)
        try:
            is_feasible = ast.literal_eval(str(answer.Feasible))
        except (SyntaxError, ValueError):
            return False, "Feasible key not found"
        num_colors, coloring = self.gcp_greedy_solution(adjacency_list)
        exist_optimal = num_colors <= k_colors
        if is_feasible != exist_optimal:
            if exist_optimal:
                return False, f"Feasibility mismatch: {coloring}"
            else:
                return False, f"Feasibility mismatch: {is_feasible} vs {exist_optimal}"
        return True, "Feasible" if is_feasible else "Infeasible"

    def load_data(self) -> list[GcpdQuestion]:
        data = []
        n = 10
        start = n - 9
        for file_num in range(start, n):
            with open(os.path.join(self.folder_name, "GCP_Decision", "decision_data_GCP_{}.txt".format(file_num))) as f:
                d = f.read()
            data += d.split("\n\n")[:-1]
        problem_cls = cast(type[GcpdQuestion], self.instancetype)
        data_func = self.loaded_data_to_class
        all_data = []
        for d in data:
            payload = data_func(d)
            all_data.append(problem_cls(dimacs_str=payload["dimacs_str"]))
        return list(all_data)
