from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NpCheckAndFormat, NpQuestion

ksp_desc = (
    "Description: The Knapsack Problem (KSP) asks whether a subset of items, each with a given weight and value, can be chosen to fit into a knapsack of fixed capacity, maximizing the total value without exceeding the capacity."
    "Question: Determine if a subset of items can be selected to fit into a knapsack with a capacity of {knapsack_capacity}, maximizing value without exceeding the capacity. Item weights and values are provided. \n {itemweights}"
)

func_typing = "Tuple[bool, int, List[int]]"  # (Path, TotalDistance)


class KspAnswer(BaseModel):
    Feasible: str = Field(description="The feasibility. Type: bool. Return 'True' or 'False'. ", default="")
    TotalValue: str = Field(description="The total value of knapsack. Type: int. For example: 8. ", default="")
    SelectedItemIds: str = Field(description="The total value of knapsack. Type: List[int]. For example: [1,2,3]. ", default="")


@dataclass
class KspQuestion(NpQuestion):
    kind: str = "ksp"
    type: str = "code"  # could be sim, nl etc
    knapsack_capacity: int = -1
    items: List[Dict[str, int]] = field(default_factory=[])  # type: ignore
    code: str = ""

    @property
    def util_pointer(self):
        return KspCheckAndFormat


class KspCheckAndFormat(NpCheckAndFormat):
    def __init__(self, prob_type):
        super().__init__(prob_type, func_typing, ksp_desc, KspAnswer)
        self.instancetype = KspQuestion

    def loaded_data_to_class(self, data):
        return data

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(code)
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"

        if not isinstance(evaluated, tuple):
            return False  # "Not a dict"
        else:
            if len(evaluated) != 3:
                return False
            if not isinstance(evaluated[0], bool):
                return False
            if not isinstance(evaluated[1], int):
                return False
            if not isinstance(evaluated[2], list):
                return False
            for e in evaluated[2]:
                if not isinstance(e, int):
                    return False
        return True

    # tied to code
    def get_field_kwargs(self, result):
        return dict(Feasible=str(result[0]), TotalValue=str(result[1]), SelectedItemIds=str(result[2]))

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self):
        return self.prompt_template(["knapsacks", "itemweights"]) if self.prob_type != "sim" else self.prompt_template(["code"])

    def format_one(self, q: KspQuestion):
        if self.prob_type == "sim":
            return self.prompt.format_prompt(code=q.code).to_string()
        knapsack_capacity = q.knapsack_capacity
        items = q.items
        itemweights = "\n The items details are as below: \n"
        for item in items:
            this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
            itemweights += this_line + "\n"
        prompt_text = self.prompt.format_prompt(knapsack_capacity=knapsack_capacity, itemweights=itemweights)
        return prompt_text.to_string()

    def load_data(self):
        with open(os.path.join(self.folder_name, "KSP", "ksp_instances.json"), "r") as f:
            all_data = json.load(f)
        return all_data

    def decision_check(self, q, output):
        return self.kspCheck(q, output)

    def ksp_optimal_solution(self, knapsacks, capacity):
        """Provides the optimal solution for the KSP instance with dynamic programming.

        :param knapsacks: A dictionary of the knapsacks.
        :param capacity: The capacity of the knapsack.
        :return: The optimal value.
        """
        # Create a one-dimensional array to store intermediate solutions
        dp = [0] * (capacity + 1)
        for itemId, (weight, value) in knapsacks.items():
            for w in range(capacity, weight - 1, -1):
                dp[w] = max(dp[w], value + dp[w - weight])

        return dp[capacity]

    # KSP
    def kspCheck(self, instance: KspQuestion, solution: BaseModel):
        """Validates the solution for the KSP instance.

        :param instance: A dictionary of the KSP instance.
        :param solution: A dictionary of the solution.
        :return: A tuple of (is_correct, message).
        """
        # Change string key to integer key and value to boolean
        items = instance.items
        knapsacks = {item["id"]: (item["weight"], item["value"]) for item in items}
        ksp_optimal_value = self.ksp_optimal_solution(knapsacks, instance.knapsack_capacity)

        try:
            is_feasible = ast.literal_eval(solution.Feasible)
        except (SyntaxError, ValueError):
            return False, "Output format is incorrect."
        if not isinstance(is_feasible, bool):
            return False, "type is wrong"
        if is_feasible != (ksp_optimal_value > 0):
            return False, f"The solution is {is_feasible} but the optimal solution is {ksp_optimal_value > 0}."

        try:
            total_value = int(ast.literal_eval(solution.TotalValue))
            selectedItems = ast.literal_eval(solution.SelectedItemIds)
        except (ValueError, SyntaxError):
            return False, "Output format is incorrect."
        if len(set(selectedItems)) != len(selectedItems):
            return False, "Duplicate items are selected."

        total_weight = 0
        cum_value = 0

        # Calculate total weight and value of selected items
        for item in selectedItems:
            if knapsacks.get(item, False):
                weight, value = knapsacks[item]
                total_weight += weight
                cum_value += value
            else:
                return False, f"Item {item} does not exist."

        # Check if the item weight exceeds the knapsack capacity
        if total_weight > instance.knapsack_capacity:
            return False, f"Total weight {total_weight} exceeds knapsack capacity {instance.knapsack_capacity}."

        if total_value != cum_value:
            return False, f"The total value {total_value} does not match the cumulative value {cum_value} of the selected items."

        if total_value != ksp_optimal_value:
            return False, f"The total value {total_value} does not match the optimal value {ksp_optimal_value}."

        return True, f"The solution is valid with total weight {total_weight} and total value {total_value}."
