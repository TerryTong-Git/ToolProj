import json

from src.exps_performance.problems.nphardeval import NPHardEvalProblem
from src.exps_performance.problems.prompts import edpPrompts


class EDP(NPHardEvalProblem):
    def __init__(self):
        self.p = edpPrompts

    def format_one(self, q):
        string_a = q["string_a"]
        string_b = q["string_b"]
        prompt_text = self.instantiate_prompt(dict(string_a=string_a, string_b=string_b))
        prompt_text += "Answer:\n"
        return prompt_text

    @staticmethod
    def compute_min_edit_distance(string_a, string_b):
        """Computes the minimum edit distance between two strings using dynamic programming."""
        m, n = len(string_a), len(string_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif string_a[i - 1] == string_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    def decision_check(self, q, output):
        """Check if the edit distance solution is valid.

        :param instance: The instance dictionary with 'string_a' and 'string_b'.
        :param solution: The solution dictionary with the reported 'edit_distance'.
        :return: A tuple of (is_correct, message).
        """
        string_a = q["string_a"]
        string_b = q["string_b"]
        try:
            reported_distance = int(output.get("Operations", -1))
        except:  # noqa
            reported_distance = -1

        actual_distance = self.compute_min_edit_distance(string_a, string_b)

        if reported_distance == -1:
            return False, "No solution provided."
        elif reported_distance != actual_distance:
            return False, f"The reported edit distance ({reported_distance}) is incorrect. Actual distance: {actual_distance}."
        return True, "The solution is valid."

    @staticmethod
    def load_data(data_path):
        with open(data_path + "edp_instances.json", "r") as f:
            all_data = json.load(f)
        return all_data
