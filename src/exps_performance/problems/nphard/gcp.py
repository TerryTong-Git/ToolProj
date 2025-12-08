from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Any, Tuple, Type, cast

from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NpCheckAndFormat, NpQuestion
from src.exps_performance.utils import read_dimacs_format

gcp_desc = (
    "Background: Graph coloring refers to the problem of coloring vertices of a graph in such a way that no two adjacent vertices have the same color. "
    "Question: There are {max_vertices} vertices 1 to {max_vertices} in a graph. You may use {max_colors} colors with alphabets from A, B, C,... to color the graph.\n{graph}"
)

func_typing = "Dict[int, str]"  # (Path, TotalDistance)


@dataclass
class GcpQuestion(NpQuestion):
    kind: str = "gcp"
    type: str = "code"  # could be sim, nl etc
    dimacs_str: str = ""
    code: str = ""

    @property
    def util_pointer(self) -> Type[NpCheckAndFormat]:
        return GcpCheckAndFormat


class GcpAnswer(BaseModel):
    Colors: str = Field(description="The color assignment for each vertex. Type: Dict[int, str]. For example {1: 'A', 2: 'B' }", default="")


class GcpCheckAndFormat(NpCheckAndFormat):
    def __init__(self, prob_type: str):
        super().__init__(prob_type, func_typing, gcp_desc, GcpAnswer)
        self.instancetype = GcpQuestion

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
        return dict(Colors=str(result))

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self) -> Any:
        return self.prompt_template(["max_vertices", "max_colors", "graph"]) if self.prob_type != "sim" else self.prompt_template("code")

    def format_one(self, q: GcpQuestion) -> str:
        inp = q.dimacs_str
        if self.prob_type == "sim":
            return str(self.prompt.format_prompt(code=q.code).to_string())
        chromatic_number = inp.split("\n")[0][-1]  # last character of the first line
        number_of_vertices = inp.split("\n")[1].split(" ")[2]  # third word of the second line
        graph = "\n The graph is below: \n"
        for line in inp.split("\n")[2:]:
            vertex_list = line.split(" ")
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1], vertex_list[2])
            graph += this_line + "\n"
        prompt_text = self.prompt.format_prompt(max_vertices=number_of_vertices, max_colors=chromatic_number, graph=graph)
        return str(prompt_text.to_string())

    def gcpCheck(self, dimacs_str: str, answer: str) -> Tuple[bool, str]:
        num_vertices, adjacency_list = read_dimacs_format(dimacs_str)
        try:
            answer_colors = ast.literal_eval(str(answer))
        except (SyntaxError, ValueError):
            return False, "wrong format"
        if not isinstance(answer_colors, dict):
            return False, "wrong format"
        # Check if all colors in the answer are valid
        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                try:
                    if answer_colors[vertex] == answer_colors[neighbor]:
                        return False, f"Invalid coloring: Vertex {vertex} and {neighbor} have the same color."
                except:  # noqa
                    return False, "Invalid input."  # dealing with hullucination
        return True, f"Valid coloring found with {len(set(answer_colors.values()))} colors: {answer_colors}"

    def decision_check(self, q: GcpQuestion, output: BaseModel) -> tuple[bool, str]:
        return self.gcpCheck(q.dimacs_str, output.Colors)

    def load_data(self) -> list[GcpQuestion]:
        n = 11
        start = n - 10
        data = []
        for file_num in range(start, n):
            with open(os.path.join(self.folder_name, "GCP", "synthesized_data_GCP_{}.txt".format(file_num))) as f:
                d = f.read()
            data += d.split("\n\n")[:-1]
        problem_cls = cast(type[GcpQuestion], self.instancetype)
        data_func = self.loaded_data_to_class
        all_data = []
        for d in data:
            payload = data_func(d)
            all_data.append(problem_cls(dimacs_str=payload["dimacs_str"]))
        return list(all_data)
