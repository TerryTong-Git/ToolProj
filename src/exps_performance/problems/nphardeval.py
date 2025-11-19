from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.exps_performance.problems import Problem, ProblemUtil


@dataclass
class NPHardEvalProblem(Problem):
    def util_pointer(self):
        return NPHardEvalProblemUtil


@dataclass
class NPHardEvalProblemUtil(ProblemUtil):
    @property
    def folder_name(self):
        if "." in __name__:
            path = "/".join(__name__.split("."))
        else:
            path = __name__
        return os.path.join(Path(path).parent.parent, "Data_V2")

    # def ground_truth(self):  # dummy
    #     return None

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
    def load_data(self):
        raise NotImplementedError
