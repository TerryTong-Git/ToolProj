import json
from collections import defaultdict

import networkx as nx

from src.exps_performance.problems.nphardeval import NPHardEvalProblem
from src.exps_performance.problems.prompts import mfpPrompts


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

    def mfp_optimal_solution(self, num_nodes, edge_capacities, source, target):
        """Provides the optimal solution for the MFP instance.

        :param num_nodes: The number of nodes in the graph.
        :param edge_capacities: A dictionary of the edge capacities.
        :param source: The source node.
        :param target: The target node.
        :return: The optimal maximum flow.
        """
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for edge_name, edge_capacity in edge_capacities.items():
            from_node, to_node = map(int, edge_name.split("->"))
            G.add_edge(from_node, to_node, weight=edge_capacity)
        max_flow = None
        if nx.has_path(G, source=source, target=target):
            max_flow = nx.maximum_flow_value(G, source, target, capacity="weight")
        return max_flow

    def decision_check(self, q, output):
        """Validate the output of the MFP problem.

        :param q: The q dictionary with nodes, edges, source, and sink.
        :param output: The output dictionary with the maximum flow and flows.
        :return: A tuple of (is_correct, message).
        """
        # Get the start and end nodes
        # Curently, the start and end nodes are the first and last nodes in the q
        num_nodes = q["nodes"]
        start_node = q["source"]
        end_node = q["sink"]

        # Initialize edge flows
        edges = q["edges"]
        edge_name_func = lambda from_node, to_node: f"{from_node}->{to_node}" if from_node < to_node else f"{to_node}->{from_node}"  # noqa
        edge_capacities = defaultdict(int)
        for edge in edges:
            edge_name = edge_name_func(edge["from"], edge["to"])
            edge_capacities[edge_name] += int(edge["capacity"])
        edge_flows = {edge_name: 0 for edge_name in edge_capacities.keys()}

        # Convert output to dictionary
        print(output)
        try:
            flows = output["Flows"]
            print(flows)
        except:  # noqa
            flows = {}
        try:
            max_flow = output["MaxFlow"]
            print(max_flow)
        except:  # noqa
            max_flow = -1
        print("------------------")

        # Get the optimal solution
        mfp_optimal_flow = self.mfp_optimal_solution(num_nodes, edge_capacities, start_node, end_node)

        if isinstance(max_flow, str):
            if max_flow.isdigit():
                max_flow = int(max_flow)
            elif mfp_optimal_flow is None:
                return True, f"There is no path from the start node to the end node, and the solution is {max_flow}."
            else:
                return False, f"The problem should be feasible ({mfp_optimal_flow}), but the solution is {max_flow}."

        if mfp_optimal_flow is None:
            if max_flow > 0:
                return False, "The problem should be infeasible."
            else:
                return True, "There is no path from the start node to the end node."
        elif max_flow < 0:
            return False, f"The problem should be feasible ({mfp_optimal_flow}), but the solution is {max_flow}."

        # Initialize node flows
        node_flows = [0 for _ in range(num_nodes)]
        node_flows[start_node] = max_flow
        node_flows[end_node] = -max_flow

        # Check if the flow is valid
        try:
            for edge, flow in flows.items():
                flow = int(flow)
                from_node, to_node = map(int, edge.split("->"))
                node_flows[from_node] -= flow
                node_flows[to_node] += flow
                edge_name = edge_name_func(from_node, to_node)
                edge_flow = flow
                if from_node > to_node:
                    edge_flow = -flow
                if edge_name not in edge_flows:
                    return False, f"Edge {edge} does not exist."
                edge_flows[edge_name] += edge_flow
        except:  # noqa
            return False, "The solution is not a valid dictionary."

        # Check the node conservation
        for node_id, node_flow in enumerate(node_flows):
            if node_flow != 0:
                return False, f"Node {node_id} is not conserved."

        # Check the edge capacities
        for edge_name, edge_flow in edge_flows.items():
            edge_capacity = edge_capacities[edge_name]
            if abs(edge_flow) > edge_capacity:
                return False, f"Edge {edge_name} with {edge_flow} exceeds its capacity {edge_capacity}."

        # Check if the flow is optimal
        if max_flow != mfp_optimal_flow:
            return False, f"The calculated flow ({max_flow}) does not match the optimal solution ({mfp_optimal_flow})."
        return True, "The solution is valid."

    @staticmethod
    def load_data(data_path):
        with open(data_path + "mfp_instances.json", "r") as f:
            all_data = json.load(f)
        return all_data
