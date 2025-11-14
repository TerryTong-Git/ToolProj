import json

from src.exps_performance.problems.nphardeval import NPHardEvalProblem
from src.exps_performance.problems.prompts import bspPrompts


class BSP(NPHardEvalProblem):
    def __init__(self):
        self.p = bspPrompts

    def format_one(self, q):
        target_value = q["target"]
        # TO-DO: fix data not being sorted
        array = sorted(q["array"])
        prompt_text = (
            self.instantiate_prompt(dict(target_value=target_value)) + "\n The sorted array elements are: " + ", ".join(map(str, array)) + "\n"
        )
        return prompt_text

    def decision_check(self, q, output):
        """Check if the binary search solution is valid.

        :param instance: The instance dictionary with array and target value.
        :param solution: The solution dictionary with the position of the target value.
        :return: A tuple of (is_correct, message).
        """
        array = sorted(q["array"])
        target_value = q["target"]
        if isinstance(output, str):
            return False, "The solution is invalid."
        try:
            position = int(output["Position"])
        except:  # noqa E722
            return False, "The solution is invalid."
        if position == -1 or position >= len(array):
            return False, "The solution is invalid."
        elif array[position] != target_value:
            return False, "The target index is incorrect."
        return True, "The solution is valid."

    @staticmethod
    def load_data(data_path):
        with open(data_path + "bsp_instances.json", "r") as f:
            all_data = json.load(f)
        return all_data
