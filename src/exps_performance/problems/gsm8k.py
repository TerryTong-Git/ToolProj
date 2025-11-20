import ast
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from datasets import load_dataset
from pydantic import BaseModel, Field

from src.exps_performance.problems import CheckAndFormat, Question

gsm8k_desc = "Description: You are going to be given a set of math problem." "Question: Solve the following math problems: \n {question}"

func_typing = "int"


class Gsm8kAnswer(BaseModel):
    Answer: str = Field(description="The answer to the math problem. Type: int. Example: 1 ", default="")


@dataclass
class Gsm8kQuestion(Question):
    kind: str = "gsm8k"
    digits: int = 0
    answer: str = ""
    question: str = ""

    @property
    def util_pointer(self):
        return Gsm8kCheckAndFormat


class Gsm8kCheckAndFormat(CheckAndFormat):
    def __init__(self, prob_type):
        super().__init__(prob_type, func_typing, gsm8k_desc, Gsm8kAnswer)
        self.instancetype = Gsm8kQuestion

    def loaded_data_to_class(self, data):
        return data

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(code)
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"
        if isinstance(evaluated, int):
            return True
        else:
            return False

    # rename to code to class
    def get_field_kwargs(self, result):
        return dict(Answer=str(result))

    @property
    def prompt(self):
        return self.prompt_template(["question"]) if self.prob_type != "sim" else self.prompt_template(["code"])

    def format_one(self, q: Gsm8kQuestion) -> str:
        if self.prob_type == "sim":
            return self.prompt.format_prompt(code=q.code).to_string()
        prompt_text = self.prompt.format_prompt(question=q.question)
        return prompt_text.to_string()

    def decision_check(self, instance: Gsm8kAnswer, solution: BaseModel):
        str_ans = solution.Answer
        return int(str_ans == instance.answer), ""

    def parse_gsm8k_gold(self, ans: str) -> int:
        m = re.search(r"####\s*(-?\d+)", ans)
        return int(m.group(1))  # type: ignore

    def check_parse_gsm8k_gold(self, ans: str) -> Optional[int]:
        m = re.search(r"####\s*(-?\d+)", ans)
        return int(m.group(1)) if m else None

    def load_data(self) -> Sequence[Gsm8kQuestion]:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for i, ex in enumerate(ds):
            if self.check_parse_gsm8k_gold(ex["answer"]) is None:
                continue
            problem = Gsm8kQuestion(question=ex["question"], answer=str(self.parse_gsm8k_gold(ex["answer"])))
            items.append(problem)
        return items
