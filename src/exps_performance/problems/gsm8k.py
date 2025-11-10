import re
from dataclasses import field
from typing import Any, Dict, Optional

from problems import Problem


class GSM8KProblem(Problem):
    kind: str = "gsm8k"
    digits: int = 0
    data: Dict[str, Any] = field(default_factory=lambda: {})

    def format(self) -> str:
        return self.data["question"]

    def ground_truth(self) -> int:
        return self.parse_gsm8k_gold(self.data["answer"])

    def decision_check(self, answer, problem_text=None):
        return int(answer == self.ground_truth())

    @staticmethod
    def parse_gsm8k_gold(ans: str) -> int:
        m = re.search(r"####\s*(-?\d+)", ans)
        return int(m.group(1))  # type: ignore

    @staticmethod
    def check_parse_gsm8k_gold(ans: str) -> Optional[int]:
        m = re.search(r"####\s*(-?\d+)", ans)
        return int(m.group(1)) if m else None
