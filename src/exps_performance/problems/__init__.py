from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Problem(ABC):
    kind: str = "null"
    digits: int = 0

    @abstractmethod
    def decision_check(self, q, output):
        raise NotImplementedError

    @abstractmethod
    def format_one(self, q: Any) -> str:
        raise NotImplementedError
