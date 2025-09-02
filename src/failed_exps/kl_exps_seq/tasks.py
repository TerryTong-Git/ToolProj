# cot_jointprob_exp/tasks.py
from dataclasses import dataclass
from typing import List, Dict, Any
import random

@dataclass
class Task:
    theta: str     # "add" | "sub" | "mul"
    a: int
    b: int

    @property
    def x_nl(self) -> str:
        if self.theta == "add":
            return f"Add {self.a} and {self.b}."
        if self.theta == "sub":
            return f"Subtract {self.b} from {self.a}."
        if self.theta == "mul":
            return f"Multiply {self.a} by {self.b}."
        raise ValueError(self.theta)

    @property
    def y(self) -> int:
        if self.theta == "add":
            return self.a + self.b
        if self.theta == "sub":
            return self.a - self.b
        if self.theta == "mul":
            return self.a * self.b
        raise ValueError(self.theta)

def make_taskset(n_per_theta: int, seed: int = 0,
                 thetas=("add","sub","mul"),
                 a_range=(2,99), b_range=(2,99)) -> List[Task]:
    rng = random.Random(seed)
    tasks: List[Task] = []
    for theta in thetas:
        for _ in range(n_per_theta):
            a = rng.randint(*a_range)
            b = rng.randint(*b_range)
            # avoid negatives for sub if you want; feel free to remove
            if theta == "sub" and b > a:
                a, b = b, a
            tasks.append(Task(theta, a, b))
    return tasks
