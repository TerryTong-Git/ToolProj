import ast
import os
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NpCheckAndFormat, NpQuestion

tsp_desc = (
    "Description: The Traveling Salesman Problem (Decision Version, TSP-D) focuses on determining if a salesman can complete a route, visiting each city at least once, with the total travel distance being less than a specified value."
    "Question: Check if it's possible for a salesman to visit each of the {total_cities} cities at least once and return to the starting city with the total distance less than {distance_limit}. The distances between each pair of cities are given. \n {citystring}"
)


@dataclass
class TspdQuestion(NpQuestion):
    kind: str = "TSP_D"
    type: str = "code"  # could be sim, nl etc
    distance_matrix: pd.DataFrame = field(default_factory=[])  # type: ignore
    threshold: int = -1
    code: str = ""

    @property
    def util_pointer(self):
        return TspdCheckAndFormat


class TspdAnswer(BaseModel):
    Feasible: str = Field(description="The feasibility. Type: Bool. Answer with True or False.", default="")


func_typing = "Bool"  # (Feasible yes or no)


class TspdCheckAndFormat(NpCheckAndFormat):
    def __init__(self, prob_type):
        super().__init__(prob_type, func_typing, tsp_desc, TspdAnswer)
        self.instancetype = TspdQuestion

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self):
        return self.prompt_template(["total_cities", "distance_limit", "citystring"]) if self.prob_type != "sim" else self.prompt_template(["code"])

    def load_data(self):
        n = 11
        start = n - 10
        all_data = []
        for level in range(start, n):
            for file_num in range(10):
                file_name = os.path.join(self.folder_name, "TSP_Decision", "decision_data_TSP_level_{}_instance_{}.csv".format(level, file_num + 1))
                df = pd.read_csv(file_name, header=None, index_col=False)
                all_data.append(df)
        return all_data

    def format_one(self, q: TspdQuestion):
        if self.prob_type == "sim":
            return self.prompt.format_prompt(code=q.code).to_string()
        dm = q.distance_matrix
        total_cities = dm.shape[0]
        citystring = "The distances between cities are below: \n"

        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                if i < j:  # only use the upper triangle
                    this_line = "The path between City {} and City {} is with distance {}.".format(i, j, dm.iloc[i, j])
                    citystring += this_line + "\n"
        prompt_text = self.prompt.format_prompt(total_cities=total_cities, distance_limit=str(q.threshold), citystring=citystring)
        return prompt_text.to_string()

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(code)
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"
        return isinstance(evaluated, bool)

    def loaded_data_to_class(self, data):
        threshold = data.iloc[-1, 0]  # therashold is the last row
        distance_matrix = data.iloc[:-1].values
        return dict(threshold=threshold, distance_matrix=pd.DataFrame(distance_matrix))

    def tsp_approx(self, distance_matrix: pd.DataFrame):
        """Returns an approximate solution to the TSP problem.

        :param distance_matrix: A 2D numpy array representing the distance matrix.
        :return: A list of the cities in the order they were visited.
        """
        G = nx.from_numpy_array(distance_matrix.to_numpy())
        return nx.approximation.traveling_salesman_problem(G)

    def tsp_decision_check(self, distance_matrix: pd.DataFrame, threshold, tour):
        """
        Checks if a given TSP tour is valid and within the threshold distance.

        :param distance_matrix: A 2D numpy array representing the distance matrix.
        :param threshold: The maximum distance allowed.
        :param tour: A dictionary containing the feasibility.
        """
        is_feasible = tour.Feasible

        # Calculate the approxed distance of the tour
        tours = self.tsp_approx(distance_matrix)
        np_distance_matrix = distance_matrix.to_numpy()
        tour_distance = sum(np_distance_matrix[tours[i], tours[i + 1]] for i in range(len(tours) - 1)) + np_distance_matrix[tours[-1], tours[0]]

        if is_feasible != (tour_distance <= threshold):
            return False, f"Feasibility mismatch: {is_feasible} vs {tour_distance} > {threshold}"
        return True, "Feasible: {} <= {}".format(tour_distance, threshold)

    def decision_check(self, q: TspdQuestion, output: BaseModel):
        return self.tsp_decision_check(q.distance_matrix, q.threshold, output)
