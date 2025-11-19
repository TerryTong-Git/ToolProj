import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NPHardEvalProblem, NPHardEvalProblemUtil

tspPrompts = (
    "Description: The traveling salesman problem (TSP) is a classic optimization problem that aims to find the shortest possible route that visits a set of cities, with each city being visited exactly once and the route returning to the original city.",
    "Question: You must find the shortest path that visits all {total_cities} cities, labelled from 1 to {total_cities}. The distances between each pair of cities are provided.\n {citystring}",
    "FOLLOW THE FORMAT CAREFULLY. Here are the format instructions: {format_instructions}",
)

tspPrompts_nl = (
    "Description: The traveling salesman problem (TSP) is a classic optimization problem that aims to find the shortest possible route that visits a set of cities, with each city being visited exactly once and the route returning to the original city.",
    "Question: You must find the shortest path that visits all {total_cities} cities, labelled from 1 to {total_cities}. The distances between each pair of cities are provided.\n {citystring}",
    "YOU ARE NEVER ALLOWED TO USE CODE. FOLLOW THE FORMAT CAREFULLY. Here are the format instructions: {format_instructions}",
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
class TSPCodeReasoning(BaseModel):
    code: str = Field(
        description=default_code_instr + "Here are the required types: def solution() -> tuple[list[int], int]",
        default="",
    )
    simulation: str = Field(description="The attempt at simulating the code in natural language reasoning to give the final answer.", default="")
    Path: str = Field(description="The path. Type: list[int]. For example: '[0,1,2,3]' ", default="")
    TotalDistance: str = Field(description="The distance. Type: int. For example: 8. ", default="")


class TSPNLReasoning(BaseModel):
    reasoning: str = Field(
        description="The attempt at simulating the problem in natural language reasoning to give the final answer.",
        default="",
    )
    Path: str = Field(description="The path. Type: list[int]. For example: '[0,1,2,3]' ", default="")
    TotalDistance: str = Field(description="The distance. Type: int. For example: 8. ", default="")


class TSPControlledCodeSim(BaseModel):
    simulation: str = Field(
        description="The attempt at simulating the code in natural language reasoning to give the final answer.",
        default="",
    )
    Path: str = Field(description="The path. Type: list[int]. For example: '[0,1,2,3]' ", default="")
    TotalDistance: str = Field(description="The distance. Type: int. For example: 8. ", default="")


@dataclass
class TSP(NPHardEvalProblem):
    kind: str = "tsp"
    type: str = "code"  # could be sim, nl etc
    distance_matrix: pd.DataFrame = field(default_factory=pd.DataFrame([]))  # type: ignore
    complexity_level: int = -1
    formatted_prompt: str = ""
    code: str = ""

    @property
    def util_pointer(self):
        return TSPUtil


class TSPUtil(NPHardEvalProblemUtil):
    def __init__(self, prob_type):
        PROB_TYPES = {"sim": TSPControlledCodeSim, "code": TSPCodeReasoning, "nl": TSPNLReasoning}
        PROMPTS = {"sim": sim_template, "code": tspPrompts, "nl": tspPrompts_nl}
        self.PROB_TYPES = PROB_TYPES
        self.PROMPTS = PROMPTS
        assert prob_type in list(PROB_TYPES.keys())
        self.prob_type = prob_type
        self.p = tspPrompts
        self.parser = PydanticOutputParser(pydantic_object=PROB_TYPES[prob_type])  # Retry Output parser?
        self.instancetype = TSP

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self):
        if self.prob_type != "sim":
            return PromptTemplate(
                template=self.PROMPTS[self.prob_type],
                input_variables=["total_cities", "citystring"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )
        else:
            return PromptTemplate(
                template=self.PROMPTS[self.prob_type],
                input_variables=["code"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )

    def format_one(self, q: TSP):
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
        prompt_text = self.prompt.format_prompt(total_cities=total_cities, citystring=citystring)
        return prompt_text

    def load_data(self):
        n = 11
        all_data = []
        start = n - 10
        for level in range(start, n):
            for file_num in range(10):
                # read np arrary
                file_name = os.path.join(self.folder_name, "TSP", "synthesized_data_TSP_level_{}_instance_{}.csv".format(level, file_num + 1))
                df = pd.read_csv(file_name, header=None, index_col=False)
                # transform df to
                all_data.append(df)
        return all_data

    def loaded_data_to_class(self, data):
        return dict(distance_matrix=data)

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

    def decision_check(self, instance: TSP, solution: TSPCodeReasoning):
        """
        Check if the TSP solution is complete and if the distance matches the greedy solution.

        :param tour_string: String representing the TSP tour in the format "0->1->2->...->N->0"
        :param distance_matrix: 2D numpy array representing the distances between cities
        :return: Boolean indicating whether the tour is complete and matches the greedy distance
        """
        # convert distance_matrix to numpy array
        distance_matrix = np.array(instance.distance_matrix)
        tour = solution.Path
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
