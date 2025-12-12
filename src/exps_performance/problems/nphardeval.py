from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type

from src.exps_performance.logger import Record
from src.exps_performance.problems import CheckAndFormat, Question


@dataclass
class NpQuestion(Question):
    record: Record = field(default_factory=Record)

    @property
    def util_pointer(self) -> Type["NpCheckAndFormat"]:
        return NpCheckAndFormat


class NpCheckAndFormat(CheckAndFormat):
    def __init__(self, prob_type: str, func_typing: str = "", desc: str = "", probModel: Any | None = None):
        if not func_typing or not desc or probModel is None:
            raise ValueError("func_typing, desc, and probModel are required")
        super().__init__(prob_type, func_typing, desc, probModel)

    @property
    def folder_name(self) -> str:
        if "." in __name__:
            path = "/".join(__name__.split("."))
        else:
            path = __name__
        return os.path.join(Path(path).parent.parent, "Data_V2")
