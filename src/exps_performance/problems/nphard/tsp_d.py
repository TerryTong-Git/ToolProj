import networkx as nx
import pandas as pd

from src.exps_performance.problems.nphardeval import NPHardEvalProblem
from src.exps_performance.problems.prompts import tsp_dPrompts


class TSP_D(NPHardEvalProblem):
    def __init__(self):
        self.p = tsp_dPrompts

    @staticmethod
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
