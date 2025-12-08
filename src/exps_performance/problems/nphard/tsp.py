import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type, cast

import numpy as np
import pandas as pd
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NpCheckAndFormat, NpQuestion
from src.exps_performance.utils import cast_float_to_int

tsp_desc = (
    "Description: The traveling salesman problem (TSP) is a classic optimization problem that aims to find the shortest possible route that visits a set of cities, with each city being visited exactly once and the route returning to the original city."
    "Question: You must find the shortest path that visits all {total_cities} cities, labelled from 1 to {total_cities}. The distances between each pair of cities are provided.\n {citystring}"
)


@dataclass
class TspQuestion(NpQuestion):
    kind: str = "tsp"
    type: str = "code"  # could be sim, nl etc
    distance_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)  # type: ignore
    code: str = ""

    @property
    def util_pointer(self) -> Type[NpCheckAndFormat]:
        return cast(Type[NpCheckAndFormat], TspCheckAndFormat)


class TspAnswer(BaseModel):
    Path: List[int] = Field(description="The path. Type: list[int]. For example: '[0,1,2,3]' ", default_factory=list)
    TotalDistance: str = Field(description="The distance. Type: int. For example: 8. ", default="")


func_typing = "Tuple[List[int], int]"  # (Path, TotalDistance)


class TspCheckAndFormat(NpCheckAndFormat):
    def __init__(self, prob_type: str):
        super().__init__(prob_type, func_typing, tsp_desc, TspAnswer)
        self.instancetype = TspQuestion

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self) -> PromptTemplate:
        return self.prompt_template(["total_cities", "citystring"]) if self.prob_type != "sim" else self.prompt_template("code")

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(str(code))
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"
        if isinstance(evaluated, tuple) and len(evaluated) == 2:
            return True
        else:
            return False

    def format_one(self, q: TspQuestion) -> str:
        if self.prob_type == "sim":
            return str(self.prompt.format_prompt(code=q.code).to_string())
        dm = q.distance_matrix
        total_cities = dm.shape[0]
        citystring = "The distances between cities are below: \n"

        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                if i < j:  # only use the upper triangle
                    this_line = "The path between City {} and City {} is with distance {}.".format(i, j, dm.iloc[i, j])
                    citystring += this_line + "\n"
        prompt_text = self.prompt.format_prompt(total_cities=total_cities, citystring=citystring)
        return str(prompt_text.to_string())

    def load_data(self) -> list[TspQuestion]:
        n = 11
        data = []
        start = n - 10
        for level in range(start, n):
            for file_num in range(10):
                csv_name = os.path.join(self.folder_name, "TSP", f"synthesized_data_TSP_level_{level}_instance_{file_num + 1}.csv")
                jsonl_path = Path(csv_name).with_suffix(".jsonl")
                if jsonl_path.exists():
                    df = pd.read_json(jsonl_path, lines=True)
                elif os.path.exists(csv_name):
                    df = pd.read_csv(csv_name, header=None, index_col=False)
                    # Persist converted JSONL for future runs.
                    df.to_json(jsonl_path, orient="records", lines=True)
                else:
                    raise FileNotFoundError(f"Missing TSP data file: {jsonl_path} (or {csv_name})")
                data.append(df)
        problem_cls = cast(type[TspQuestion], self.instancetype)
        data_func = self.loaded_data_to_class
        all_data = []
        for d in data:
            payload = data_func(d)
            all_data.append(problem_cls(distance_matrix=payload["distance_matrix"]))
        return list(all_data)

    def loaded_data_to_class(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return dict(distance_matrix=data)

    def greedy_tsp(self, distance_matrix: np.ndarray) -> tuple[list[int], float]:
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

        return tour, float(total_distance)

    # tied to the formatting
    def get_field_kwargs(self, result: tuple[list[int], float] | str) -> dict[str, object]:
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

    def decision_check(self, instance: TspQuestion, solution: BaseModel) -> tuple[bool, str]:
        """
        Check if the TSP solution is complete and if the distance matches the greedy solution.

        :param tour_string: String representing the TSP tour in the format "0->1->2->...->N->0"
        :param distance_matrix: 2D numpy array representing the distances between cities
        :return: Boolean indicating whether the tour is complete and matches the greedy distance
        """
        # convert distance_matrix to numpy array
        distance_matrix = np.array(instance.distance_matrix)
        tour = solution.Path
        if not tour:
            return False, "The tour is empty"
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
