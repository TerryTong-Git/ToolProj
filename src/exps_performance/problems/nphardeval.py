from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.exps_performance.problems import CheckAndFormat, Question


@dataclass
class NpQuestion(Question):
    def util_pointer(self):
        return NpCheckAndFormat


class NpCheckAndFormat(CheckAndFormat):
    def __init__(self, prob_type, func_typing, desc, probModel):
        super().__init__(prob_type, func_typing, desc, probModel)

    @property
    def folder_name(self):
        if "." in __name__:
            path = "/".join(__name__.split("."))
        else:
            path = __name__
        return os.path.join(Path(path).parent.parent, "Data_V2")
