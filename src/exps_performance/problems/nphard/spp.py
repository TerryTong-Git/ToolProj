import ast
import json
from typing import Any, Tuple

import networkx as nx
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NPHardEvalProblem

sppPrompts = (
    "Description: The Shortest Path Problem (SPP) involves finding the shortest path between two nodes in a weighted graph."
    "Question: You need to find the shortest path between node {start_node} and node {end_node} in a graph. The graph's edges and their weights are given. {edges}. "
    "FOLLOW THE FORMAT CAREFULLY. Here are the format instructions: {format_instructions}"
)

sppPrompts_nl = (
    "Description: The Shortest Path Problem (SPP) involves finding the shortest path between two nodes in a weighted graph."
    "Question: You need to find the shortest path between node {start_node} and node {end_node} in a graph. The graph's edges and their weights are given. {edges}. "
    "YOU ARE NEVER ALLOWED TO USE CODE. FOLLOW THE FORMAT CAREFULLY. Here are the format instructions: {format_instructions}"
)

sim_template = "Simulate the execution of the provided code: {code} \n. ALL NECESSARY INFORMATION IS IN THE CODE PROVIDED. FOLLOW THE FORMAT CAREFULLY. Here are the format instructions: {format_instructions}"


default_code_instr = """
The code block that specifies a function 'solution()' that defines all variables, imports and IMPLEMENTS the actual code to solve the problem that can be executed. Begin and end code with ```python```. For example an INCORRECT way to solve the problem (Don't copy method, but only formatting) but is formatted correctly:       

```python
def solution():
    import numpy as np
    variable = [0,1,2,3]
    out = np.sum(variable) 
    return out
```

""".strip()


# easier to format as a list, but string
# a local variable called answer should hold the answer? Then when I run the code, I can extract the local variable?
class SPPCodeReasoning(BaseModel):
    code: str = Field(
        description=default_code_instr + "Here are the required types: def solution() -> tuple[list[int], int]",
        default="",
    )
    simulation: str = Field(description="The attempt at simulating the code in natural language reasoning to give the final answer.", default="")
    Path: str = Field(description="The path. Type: list[int]. For example: '[0,1,2,3]' ", default="")
    TotalDistance: str = Field(description="The distance. Type: int. For example: 8. ", default="")


class SPPNLReasoning(BaseModel):
    reasoning: str = Field(
        description="The attempt at simulating the problem in natural language reasoning to give the final answer.",
        default="",
    )
    Path: str = Field(description="The path. Type: list[int]. For example: '[0,1,2,3]' ", default="")
    TotalDistance: str = Field(description="The distance. Type: int. For example: 8. ", default="")


class ControlledCodeSim(BaseModel):
    simulation: str = Field(
        description="The attempt at simulating the code in natural language reasoning to give the final answer.",
        default="",
    )
    Path: str = Field(description="The path. Type: list[int]. For example: '[0,1,2,3]' ", default="")
    TotalDistance: str = Field(description="The distance. Type: int. For example: 8. ", default="")


PROB_TYPES = {"sim": ControlledCodeSim, "code": SPPCodeReasoning, "nl": SPPNLReasoning}
PROMPTS = {"sim": sim_template, "code": sppPrompts, "nl": sppPrompts_nl}


# have a record class keep track of parse failure statistics

# incorporate this decision logic into a base class to share.


class SPP(NPHardEvalProblem):
    def __init__(self, prob_type):
        assert prob_type in list(PROB_TYPES.keys())
        self.prob_type = prob_type
        self.p = sppPrompts
        self.parser = PydanticOutputParser(pydantic_object=PROB_TYPES[prob_type])  # Retry Output parser?
        if prob_type != "sim":
            self.prompt = PromptTemplate(
                template=PROMPTS[prob_type],
                input_variables=["start_node", "end_node", "edges"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )
        else:
            self.prompt = PromptTemplate(
                template=PROMPTS[prob_type],
                input_variables=["code"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )

    def format_one(self, q: Any) -> str:
        if self.prob_type == "sim":
            return self.prompt.format_prompt(code=q).to_string()
        start_node = q["nodes"][0]
        end_node = q["nodes"][-1]
        edges = q["edges"]

        edge_string = "\n The graph's edges and weights are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
            edge_string += this_line + "\n"
        prompt_text = self.prompt.format_prompt(start_node=start_node, end_node=end_node, edges=edge_string)
        return prompt_text.to_string()

    # returns either instance of code, nl, or sim class, or err.
    def parse_output(self, output) -> BaseModel:  # returns one of the pydantic objects
        try:
            return self.parser.parse(output)  # ok
        except OutputParserException:
            # another way to default to blanks
            return PROB_TYPES[self.prob_type]()  # err

    @staticmethod
    def load_data(data_path):
        with open(data_path + "spp_instances.json", "r") as f:
            all_data = json.load(f)
        return all_data

    def ssp_optimal_solution(self, instance, source, target):
        """Provides the optimal solution for the SSP instance.

        :param instance: The SSP instance as a dictionary with 'nodes' and 'edges'.
        :param source: The source node.
        :param target: The destination node.
        :return: The optimal shortest path length and path.
        """
        G = nx.Graph()
        G.add_nodes_from(instance["nodes"])
        G.add_weighted_edges_from([(edge["from"], edge["to"], edge["weight"]) for edge in instance["edges"]])
        shortest_path_length = None
        shortest_path = None
        if nx.has_path(G, source=source, target=target):
            shortest_path_length = nx.shortest_path_length(G, source=source, target=target, weight="weight")
            shortest_path = nx.shortest_path(G, source=source, target=target, weight="weight")
        return shortest_path_length, shortest_path

    # SPP
    def decision_check(self, instance: dict, solution: SPPCodeReasoning, start_node=None, end_node=None) -> Tuple[bool, str]:
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
            start_node = instance["nodes"][0]
        if end_node is None:
            end_node = instance["nodes"][-1]

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
            path = ast.literal_eval(path_string)  # expecting a string [0,1,2,3]
        except SyntaxError:  # something wrong witht he parse, e.g. '0 trying to cast to int.
            path = []
        try:
            total_cost = int(cost_string)
        except ValueError:  # could not cast
            total_cost = -1

        # Check if path starts and ends with the correct nodes
        if not path or path[0] != start_node or path[-1] != end_node:
            return False, "The path does not start or end at the correct nodes."

        # Check if the path is continuous and calculate the cost
        calculated_cost = 0
        is_in_edge = lambda edge, from_node, to_node: (edge["from"] == from_node and edge["to"] == to_node) or (  # noqa
            edge["from"] == to_node and edge["to"] == from_node
        )
        for i in range(len(path) - 1):
            from_node, to_node = path[i], path[i + 1]
            edge = next((edge for edge in instance["edges"] if is_in_edge(edge, from_node, to_node)), None)

            if not edge:
                return False, f"No edge found from node {from_node} to node {to_node}."

            calculated_cost += edge["weight"]

        # Check if the calculated cost matches the total cost provided in the solution
        if calculated_cost != total_cost:
            return False, f"The calculated cost ({calculated_cost}) does not match the provided total cost ({total_cost})."

        if calculated_cost != ssp_optimal_length:
            # spp_optimal_path = "->".join(map(str, ssp_optimal_path))
            return False, f"The calculated cost ({calculated_cost}) does not match the optimal solution ({ssp_optimal_length}): {ssp_optimal_path}."

        return True, "The solution is valid."
