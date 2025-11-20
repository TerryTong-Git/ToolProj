import ast
import re
from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, Field

from src.exps_performance.clrs.huggingface_generators import clrs_generator
from src.exps_performance.problems import CheckAndFormat, Question

clrs_desc = "Description: You are going to be given a set of algorithmic problem." "Question: Solve the following algorithmic problem: \n {question}"

func_typing = "int"


class ClrsAnswer(BaseModel):
    Answer: str = Field(description="The answer to the algorithmic problem. Type: int. Example: 1 ", default="")


@dataclass
class ClrsQuestion(Question):
    kind: str = "clrs"
    digits: int = 0
    answer: str = ""
    text_data: str = ""

    @property
    def util_pointer(self):
        return ClrsCheckAndFormat


class ClrsCheckAndFormat(CheckAndFormat):
    def __init__(self, prob_type):
        super().__init__(prob_type, func_typing, clrs_desc, ClrsAnswer)
        self.instancetype = ClrsQuestion

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

    def format_one(self, q: ClrsQuestion) -> str:
        if self.prob_type == "sim":
            return self.prompt.format_prompt(code=q.code).to_string()
        prompt_text = self.prompt.format_prompt(question=q.text_data)
        return prompt_text.to_string()

    def decision_check(self, instance: ClrsAnswer, solution: BaseModel):
        str_ans = solution.Answer
        return int(str_ans == instance.answer), ""

    def load_data(self) -> Sequence[ClrsQuestion]:
        _DEFAULT_VAL_ALGOS_AND_LENGTHS = {
            "activity_selector": list(range(4, 41)),
            "articulation_points": list(range(4, 20)),
            "bellman_ford": list(range(4, 33)),
            "bfs": list(range(4, 42)),
            "binary_search": list(range(4, 65)),
            "bridges": list(range(4, 8)),
            "bubble_sort": list(range(4, 12)),
            "dag_shortest_paths": list(range(4, 20)),
            "dfs": list(range(4, 21)),
            "dijkstra": list(range(4, 26)),
            "find_maximum_subarray_kadane": list(range(4, 65)),
            "floyd_warshall": list(range(4, 12)),
            "graham_scan": list(range(4, 32)),
            "heapsort": list(range(4, 12)),
            "insertion_sort": list(range(4, 26)),
            "jarvis_march": list(range(4, 14)),
            "kmp_matcher": list(range(4, 65)),
            "lcs_length": list(range(4, 13)),
            "matrix_chain_order": list(range(4, 13)),
            "minimum": list(range(4, 65)),
            "mst_kruskal": list(range(4, 11)),
            "mst_prim": list(range(4, 27)),
            "naive_string_matcher": list(range(4, 65)),
            "optimal_bst": list(range(4, 11)),
            "quickselect": list(range(4, 65)),
            "quicksort": list(range(4, 13)),
            "segments_intersect": list(range(4, 65)),
            "strongly_connected_components": list(range(4, 17)),
            "task_scheduling": list(range(4, 42)),
            "topological_sort": list(range(4, 22)),
        }
        _DEFAULT_VAL_NUMBER_OF_SAMPLES = 2000
        _DEFAULT_VAL_SEEDS = [0]
        for seed in _DEFAULT_VAL_SEEDS:
            data = clrs_generator(_DEFAULT_VAL_ALGOS_AND_LENGTHS, _DEFAULT_VAL_NUMBER_OF_SAMPLES, use_hints=False, seed=seed)
        return [ClrsQuestion(d["algo_name"], d["length"], answer=re.sub(r"\s+", "", d["answer"]), text_data=d["question"]) for d in data][:100]
