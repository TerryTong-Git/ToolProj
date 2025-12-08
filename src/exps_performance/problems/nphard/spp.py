import ast
import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Type, cast

import networkx as nx
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NpCheckAndFormat, NpQuestion
from src.exps_performance.utils import cast_float_to_int

spp_desc = (
    "Description: The Shortest Path Problem (SPP) involves finding the shortest path between two nodes in a weighted graph."
    "Question: You need to find the shortest path between node {start_node} and node {end_node} in a graph. The graph's edges and their weights are given. {edges}."
)

func_typing = "Tuple[List[int], int]"  # (Path, TotalDistance)


class SppAnswer(BaseModel):
    Path: List[int] = Field(description="The path. Type: list[int]. For example: '[0,1,2,3]' ", default_factory=list)
    TotalDistance: str = Field(description="The distance. Type: int. For example: 8. ", default="")


@dataclass
class SppQuestion(NpQuestion):
    kind: str = "spp"
    type: str = "code"  # could be sim, nl etc
    nodes: List[int] = field(default_factory=list)
    edges: List[dict[str, Any]] = field(default_factory=list)
    complexity_level: int = -1
    code: str = ""

    @property
    def util_pointer(self) -> Type[NpCheckAndFormat]:
        return cast(Type[NpCheckAndFormat], SppCheckAndFormat)


class SppCheckAndFormat(NpCheckAndFormat):
    def __init__(self, prob_type: str):
        super().__init__(prob_type, func_typing, spp_desc, SppAnswer)
        self.instancetype = SppQuestion

    def loaded_data_to_class(self, data: Any) -> Any:
        return data

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(str(code))
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"
        if isinstance(evaluated, tuple) and len(evaluated) == 2:
            return True
        else:
            return False

    def get_field_kwargs(self, result: Tuple[List[int], int] | str) -> dict[str, object]:
        """
        Return native types for pydantic validation. Accepts a tuple or its string form.
        """
        if isinstance(result, str):
            try:
                result = ast.literal_eval(result)
            except (SyntaxError, ValueError, TypeError):
                return {"Path": [], "TotalDistance": ""}
        if not isinstance(result, tuple) or len(result) != 2:
            return {"Path": [], "TotalDistance": ""}
        path, total_distance = result
        path = cast_float_to_int(path)
        total_distance = cast_float_to_int(total_distance)
        return {"Path": path, "TotalDistance": str(total_distance)}

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self) -> PromptTemplate:
        return self.prompt_template(["start_node", "end_node", "edges"]) if self.prob_type != "sim" else self.prompt_template("code")

    def format_one(self, q: SppQuestion) -> str:
        if self.prob_type == "sim":
            return str(self.prompt.format_prompt(code=q.code).to_string())
        start_node = q.nodes[0]
        end_node = q.nodes[-1]
        edges = q.edges

        edge_string = "\n The graph's edges and weights are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."  # type: ignore
            edge_string += this_line + "\n"
        prompt_text = self.prompt.format_prompt(start_node=start_node, end_node=end_node, edges=edge_string)
        string_prompt = prompt_text.to_string()
        return str(string_prompt)

    def load_data(self) -> list[SppQuestion]:
        with open(os.path.join(self.folder_name, "SPP", "spp_instances.json"), "r") as f:
            data = json.load(f)
        problem = self.instancetype  # type: ignore
        data_func = self.loaded_data_to_class  # type: ignore #for some reason can only see base class type...
        all_data = [problem(**data_func(d)) for d in data]
        return list(all_data)

    def ssp_optimal_solution(self, instance: SppQuestion, source: int, target: int) -> Tuple[int | None, list[int] | None]:
        """Provides the optimal solution for the SSP instance.

        :param instance: The SSP instance as a dictionary with 'nodes' and 'edges'.
        :param source: The source node.
        :param target: The destination node.
        :return: The optimal shortest path length and path.
        """
        G = nx.Graph()
        G.add_nodes_from(instance.nodes)
        G.add_weighted_edges_from([(edge["from"], edge["to"], edge["weight"]) for edge in instance.edges])
        shortest_path_length = None
        shortest_path = None
        if nx.has_path(G, source=source, target=target):
            shortest_path_length = nx.shortest_path_length(G, source=source, target=target, weight="weight")
            shortest_path = nx.shortest_path(G, source=source, target=target, weight="weight")
        return shortest_path_length, shortest_path

    # SPP

    def decision_check(
        self, instance: SppQuestion, solution: BaseModel, start_node: int | None = None, end_node: int | None = None
    ) -> Tuple[bool, str]:
        """Validate the solution of the SPP problem.

        :param instance: The instance dictionary with nodes and edges.
        :param solution: The solution dictionary with the path and total distance.
        :param start_node: The start node.
        :param end_node: The end node.
        :return: A tuple of (is_correct, message).
        """

        # take string and parse it.
        all_none = True
        for key, value in vars(solution).items():
            is_none = value == ""
            all_none &= is_none
        if all_none:
            return (False, "Parse error")

        # Get the start and end nodes
        # Curently, the start and end nodes are the first and last nodes in the instance
        if start_node is None:
            start_node = instance.nodes[0]
        if end_node is None:
            end_node = instance.nodes[-1]

        # Convert solution to dictionary, know that it conforms to schema.
        path_string = solution.Path
        cost_string = solution.TotalDistance

        # Calculate the optimal solution
        ssp_optimal_length, ssp_optimal_path = self.ssp_optimal_solution(instance, start_node, end_node)

        if ssp_optimal_length is None:
            if isinstance(cost_string, int) or cost_string.isdigit():
                return False, f"No path between from node {start_node} to node {end_node}."
            else:
                return True, "No path found from node {start_node} to node {end_node}."

        try:
            path = ast.literal_eval(str(path_string))  # expecting a string [0,1,2,3]
        except SyntaxError:  # something wrong witht he parse, e.g. '0 trying to cast to int.
            path = []
        try:
            total_cost = int(cost_string)
        except ValueError:  # could not cast
            total_cost = -1

        # Check if path starts and ends with the correct nodes

        if not isinstance(path, list) or not path or path[0] != start_node or path[-1] != end_node:
            return False, "The path does not start or end at the correct nodes."

        # Check if the path is continuous and calculate the cost
        calculated_cost = 0

        # TODO: lambda funcs are unclear, change it
        is_in_edge = lambda edge, from_node, to_node: (edge["from"] == from_node and edge["to"] == to_node) or (  # noqa
            edge["from"] == to_node and edge["to"] == from_node
        )
        for i in range(len(path) - 1):
            from_node, to_node = path[i], path[i + 1]
            edge = next((edge for edge in instance.edges if is_in_edge(edge, from_node, to_node)), None)

            if not edge:
                return False, f"No edge found from node {from_node} to node {to_node}."

            calculated_cost += edge["weight"]  # type: ignore

        # Check if the calculated cost matches the total cost provided in the solution
        if calculated_cost != total_cost:
            return False, f"The calculated cost ({calculated_cost}) does not match the provided total cost ({total_cost})."

        if calculated_cost != ssp_optimal_length:
            # spp_optimal_path = "->".join(map(str, ssp_optimal_path))
            return False, f"The calculated cost ({calculated_cost}) does not match the optimal solution ({ssp_optimal_length}): {ssp_optimal_path}."
        return True, "The solution is valid."
