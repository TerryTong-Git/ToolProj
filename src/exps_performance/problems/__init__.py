from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Problem(ABC):
    kind: str = "null"
    digits: int = 0

    @abstractmethod
    def decision_check(self, q, output):
        raise NotImplementedError

    @abstractmethod
    def format_one(self):
        return self.instantiate_prompt()
