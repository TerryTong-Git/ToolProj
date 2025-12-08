import ast
import json
import os
from dataclasses import dataclass, field
from typing import Any, Type, cast

from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NpCheckAndFormat, NpQuestion

bsp_desc = (
    "Description: The Binary Search Problem (BSP) deals with finding the position of a target value within a sorted array using a binary search algorithm, which efficiently narrows down the search range."
    "Question: Find the position of the target value {target_value} in the sorted array. The index begins with 0. The array elements are provided. \n {arr}"
)

func_typing = "int"


class BspAnswer(BaseModel):
    Position: str = Field(description="Position resulting from search. Type: int. For example: 8. ", default="")


@dataclass
class BspQuestion(NpQuestion):
    kind: str = "edp"
    type: str = "code"  # could be sim, nl etc
    target: str = ""
    array: list[int] = field(default_factory=list)
    code: str = ""

    def util_pointer(self) -> Type[NpCheckAndFormat]:
        return cast(Type[NpCheckAndFormat], BspCheckAndFormat)


class BspCheckAndFormat(NpCheckAndFormat):
    def __init__(self, prob_type: str):
        super().__init__(prob_type, func_typing, bsp_desc, BspAnswer)
        self.instancetype = BspQuestion

    # tied to inputs, may not be called input
    def loaded_data_to_class(self, data: Any) -> Any:
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
    def get_field_kwargs(self, result: Any) -> dict[str, str]:
        return dict(Position=str(result))

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self) -> Any:
        return self.prompt_template(["target_value", "arr"]) if self.prob_type != "sim" else self.prompt_template("code")

    def format_one(self, q: BspQuestion) -> str:
        if self.prob_type == "sim":
            return str(self.prompt.format_prompt(code=q.code).to_string())
        target_value = q.target
        array = sorted(q.array)
        array_formatted = "\n The sorted array elements are: " + ", ".join(map(str, array)) + "\n"
        prompt_text = self.prompt.format_prompt(target_value=target_value, arr=array_formatted).to_string() + "Answer:\n"

        return str(prompt_text)

    def decision_check(self, q: BspQuestion, output: BspAnswer) -> tuple[bool, str]:
        """Check if the binary search solution is valid.

        :param instance: The instance dictionary with array and target value.
        :param solution: The solution dictionary with the position of the target value.
        :return: A tuple of (is_correct, message).
        """
        array = sorted(q.array)
        target_value = q.target
        if isinstance(output, str):
            return False, "The solution is invalid."
        try:
            position = int(ast.literal_eval(output.Position))
        except (SyntaxError, ValueError):  # noqa E722
            return False, "The solution is invalid."
        if position == -1 or position >= len(array):
            return False, "The solution is invalid."
        elif array[position] != target_value:
            return False, "The target index is incorrect."
        return True, "The solution is valid."

    def load_data(self) -> list[BspQuestion]:
        with open(os.path.join(self.folder_name, "BSP", "bsp_instances.json"), "r") as f:
            data = json.load(f)
        problem = self.instancetype  # type: ignore
        data_func = self.loaded_data_to_class  # type: ignore #for some reason can only see base class type...
        all_data = [problem(**data_func(d)) for d in data]
        return list(all_data)
