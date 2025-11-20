import re
from dataclasses import field
from typing import Any, Dict, Optional

from src.exps_performance.problems import Problem, ProblemUtil


# what interface for the tests to work?
class GSM8KUtil(ProblemUtil):
    def ground_truth(self, data) -> int:
        return self.parse_gsm8k_gold(data["answer"])

    def decision_check(self, answer, problem_text=None):
        return int(answer == self.ground_truth())

    @staticmethod
    def parse_gsm8k_gold(ans: str) -> int:
        m = re.search(r"####\s*(-?\d+)", ans)
        return int(m.group(1))  # type: ignore

    def format_one(self, data) -> str:
        return data["question"]

    @staticmethod
    def check_parse_gsm8k_gold(ans: str) -> Optional[int]:
        m = re.search(r"####\s*(-?\d+)", ans)
        return int(m.group(1)) if m else None


class GSM8KProblem(Problem):
    kind: str = "gsm8k"
    digits: int = 0
    data: Dict[str, Any] = field(default_factory=lambda: {})
