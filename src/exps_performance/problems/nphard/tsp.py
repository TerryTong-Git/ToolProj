import ast

import numpy as np
import pandas as pd

from src.exps_performance.problems.nphardeval import NPHardEvalProblem
from src.exps_performance.problems.prompts import tspPrompts


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

    @staticmethod
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
