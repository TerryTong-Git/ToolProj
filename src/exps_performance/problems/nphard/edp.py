import ast
import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NPHardEvalProblem, NPHardEvalProblemUtil

edp_desc = (
    "Description: The Edit Distance Problem (EDP) involves finding the minimum number of operations required to transform one string into another, where each operation is either an insertion, deletion, or substitution of a single character."
    "Question: Find the minimum number of operations required to transform the first string {string_a} into the second string {string_b}. The operations are insertion, deletion, and substitution of a single character, each requiring 1 edit operation. \n"
)

func_typing = "int"


class EDPModel(BaseModel):
    Operations: str = Field(description="The number of edits. Type: int. For example: 8. ", default="")


@dataclass
class EDP(NPHardEvalProblem):
    kind: str = "edp"
    type: str = "code"  # could be sim, nl etc
    string_a: str = ""
    string_b: str = ""
    code: str = ""

    @property
    def util_pointer(self):
        return EDPUtil


class EDPUtil(NPHardEvalProblemUtil):
    def __init__(self, prob_type):
        super().__init__(prob_type, func_typing, edp_desc, EDPModel)
        self.instancetype = EDP

    # tied to inputs, may not be called input
    def loaded_data_to_class(self, data):
        return data

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(code)
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"
        if not isinstance(evaluated, int):
            return False
        return True

    # tied to code
    def get_field_kwargs(self, result):
        return dict(Operations=str(result))

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self):
        return self.prompt_template(["string_a", "string_b"]) if self.prob_type != "sim" else self.prompt_template(["code"])

    def format_one(self, q: EDP):
        if self.prob_type == "sim":
            return self.prompt.format_prompt(code=q.code).to_string()
        string_a = q.string_a
        string_b = q.string_b
        prompt_text = self.prompt.format_prompt(string_a=string_a, string_b=string_b)
        return prompt_text.to_string() + "Answer:\n"

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

    def decision_check(self, q: EDP, output: BaseModel):
        """Check if the edit distance solution is valid.

        :param instance: The instance dictionary with 'string_a' and 'string_b'.
        :param solution: The solution dictionary with the reported 'edit_distance'.
        :return: A tuple of (is_correct, message).
        """
        string_a = q.string_a
        string_b = q.string_b
        try:
            reported_distance = ast.literal_eval(output.Operations)
        except (SyntaxError, ValueError):  # noqa
            reported_distance = -1

        actual_distance = self.compute_min_edit_distance(string_a, string_b)

        if reported_distance == -1:
            return False, "No solution provided."
        elif reported_distance != actual_distance:
            return False, f"The reported edit distance ({reported_distance}) is incorrect. Actual distance: {actual_distance}."
        return True, "The solution is valid."

    def load_data(self):
        with open(os.path.join(self.folder_name, "EDP", "edp_instances.json"), "r") as f:
            all_data = json.load(f)
        return all_data
