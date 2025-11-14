from __future__ import annotations

from abc import abstractmethod
from typing import Any

from src.exps_performance.problems import Problem


class NPHardEvalProblem(Problem):
    folder_name: str

    def ground_truth(self):  # dummy
        return None

    def instantiate_prompt(self, kwargs):
        return (
            self.p["Intro"]
            + "\n"
            + self.p["Initial_question"].format(**kwargs)
            + "\n"
            + +self.p["Output_content"]
            + "\n"
            + self.p["Output_format"]
            + "\n"
        )

    @abstractmethod
    def format_one(self, q: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_data(self, data_path):
        raise NotImplementedError
