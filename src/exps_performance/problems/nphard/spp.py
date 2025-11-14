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
    " \n ONLY RETURN ACCORDING TO THE FORMAT!!! THIS IS REALLY IMPORTANT. DO USE PROSE BEFORE OR AFTER THE FORMAT. Here are the format instructions: {format_instructions}"
)

sppPrompts_nl = (
    "Description: The Shortest Path Problem (SPP) involves finding the shortest path between two nodes in a weighted graph."
    "Question: You need to find the shortest path between node {start_node} and node {end_node} in a graph. The graph's edges and their weights are given. {edges}. "
    " \n ONLY RETURN STRUCTURED OUTPUT ACCORDING TO THE FORMAT!!! THIS IS REALLY IMPORTANT. DO NOT TALK TO ME, OR GIVE ME A DESCRIPTION, JUST GIVE THE FORMAT. DO USE PROSE OR CODE BEFORE OR AFTER THE GIVEN FORMAT. YOU ARE NEVER ALLOWED TO USE CODE. Here are the format instructions: {format_instructions}"
)

sim_template = "Simulate the execution of the provided code: {code} \n ONLY RETURN ACCORDING TO THE FORMAT!!! THIS IS REALLY IMPORTANT. DO USE PROSE BEFORE OR AFTER THE FORMAT. Here are the format instructions: {format_instructions}"


# a local variable called answer should hold the answer? Then when I run the code, I can extract the local variable?
class SPPCodeReasoning(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block of import statements")
    code: str = Field(description="The code block without imports that solves the problem.")
    print_statement: str = Field(
        description="The final piece of the code block that prints the variables to the required output format. For this question, it is JSON format: {'Path': '0->1->2->3','TotalDistance': '8'}. You must print something like this."
    )
    simulation: str = Field(description="The attempt at simulating the code in natural language reasoning to give the final answer.")
    Path: str = Field(
        description="This is part of the final answer, and the path that the code simulation gives. This should not be a piece of code, but rather, an instance of the answer. Give the answer separated by arrows. For example: '0->1->2->3'. Answers without this are completely wrong"
    )
    TotalDistance: str = Field(
        description="This is part of the final answer, and the total distance the code simulation gives. This should not be a piece of code, but rather, an instance of the answer. For example: 8. Answers without this are completely wrong. "
    )


class SPPNLReasoning(BaseModel):
    reasoning: str = Field(
        description="The attempt at simulating the problem in natural language reasoning to give the final answer. YOU ARE NEVER ALLOWED TO GENERATE CODE."
    )
    Path: str = Field(
        description="This is part of the final answer, and the path that the natural language reasoning simulation gives. Give the answer separated by arrows. For example: '0->1->2->3'. Answers without this are completely wrong"
    )
    TotalDistance: str = Field(
        description="This is part of the final answer, and the total distance the natural language reasoning simulation gives. For example: 8. Answers without this are completely wrong. "
    )


class ControlledCodeSim(BaseModel):
    simulation: str = Field(description="The attempt at simulating the code in natural language reasoning to give the final answer. ")
    Path: str = Field(
        description="This is part of the final answer, and the path that the code simulation gives. This should not be a piece of code, but rather, an instance of the answer. Give the answer separated by arrows. For example: '0->1->2->3'. Answers without this are completely wrong"
    )
    TotalDistance: str = Field(
        description="This is part of the final answer, and the total distance the code simulation gives. This should not be a piece of code, but rather, an instance of the answer. For example: 8. Answers without this are completely wrong. "
    )


PROB_TYPES = {"sim": ControlledCodeSim, "code": SPPCodeReasoning, "nl": SPPNLReasoning}
PROMPTS = {"sim": sim_template, "code": sppPrompts, "nl": sppPrompts_nl}


# have a record class keep track of parse failure statistics
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
            return self.prompt.format(q).to_string()
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
    def parse_output(self, output) -> Any:
        try:
            return self.parser.parse(output)  # ok
        except OutputParserException:
            return 0  # err

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

        path = list(map(int, path_string.split("->")))
        total_cost = int(cost_string)

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
