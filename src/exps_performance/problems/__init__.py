from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProblemUtil(ABC):
    @abstractmethod
    def decision_check(self, q, output):
        raise NotImplementedError

    @abstractmethod
    def format_one(self, q: Any) -> str:
        raise NotImplementedError


@dataclass
class Problem(ABC):
    kind: str = "null"  # e.g. clrs, finegrain, gsm8k etc
    digits: int = 0
    code: str = ""

    @property
    @abstractmethod
    def util_pointer(self):
        raise NotImplementedError


# (d.util_pointer)(self.run_type).format_one(d)
