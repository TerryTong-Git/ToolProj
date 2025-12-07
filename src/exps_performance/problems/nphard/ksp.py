from __future__ import annotations

import json

from src.exps_performance.problems.nphardeval import NPHardEvalProblem
from src.exps_performance.problems.prompts import kspPrompts


class KSP(NPHardEvalProblem):
    def __init__(self):
        self.p = kspPrompts

    def format_one(self, q):
        knapsack_capacity = q["knapsack_capacity"]
        items = q["items"]
        prompt_text = self.instantiate_prompt(dict(knapsack_capacity=knapsack_capacity)) + "\n The items details are as below: \n"
        for item in items:
            this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
            prompt_text += this_line + "\n"
        return prompt_text

    @staticmethod
    def load_data(data_path):
        with open(data_path + "ksp_instances.json", "r") as f:
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
    def kspCheck(self, instance, solution):
        """Validates the solution for the KSP instance.

        :param instance: A dictionary of the KSP instance.
        :param solution: A dictionary of the solution.
        :return: A tuple of (is_correct, message).
        """
        # Change string key to integer key and value to boolean
        items = instance.get("items", [])
        knapsacks = {item["id"]: (item["weight"], item["value"]) for item in items}

        ksp_optimal_value = self.ksp_optimal_solution(knapsacks, instance["knapsack_capacity"])

        try:
            is_feasible = solution.get("Feasible", "").lower() == "yes"
        except ValueError:
            return False, "Output format is incorrect."
        if is_feasible != (ksp_optimal_value > 0):
            return False, f"The solution is {is_feasible} but the optimal solution is {ksp_optimal_value > 0}."

        try:
            total_value = int(solution.get("TotalValue", -1))
            selectedItems = list(map(int, solution.get("SelectedItemIds", [])))
        except ValueError:
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
        if total_weight > instance["knapsack_capacity"]:
            return False, f"Total weight {total_weight} exceeds knapsack capacity {instance['knapsack_capacity']}."

        if total_value != cum_value:
            return False, f"The total value {total_value} does not match the cumulative value {cum_value} of the selected items."

        if total_value != ksp_optimal_value:
            return False, f"The total value {total_value} does not match the optimal value {ksp_optimal_value}."

        return True, f"The solution is valid with total weight {total_weight} and total value {total_value}."
