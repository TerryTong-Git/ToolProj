from __future__ import annotations

import networkx as nx
from problems.nphardeval import NPHardEvalProblem
from prompts import gcp_dPrompts
from utils import read_dimacs_format


class GCP_D(NPHardEvalProblem):
    folder_name = "GCP_Decision"

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

    @staticmethod
    def load_data(data_path):
        all_data = []
        n = 10
        start = n - 9
        for file_num in range(start, n):
            with open(data_path + "decision_data_GCP_{}.txt".format(file_num)) as f:
                data = f.read()
            all_data += data.split("\n\n")[:-1]
        return all_data
