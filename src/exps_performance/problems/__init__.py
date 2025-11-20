from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class CheckAndFormat(ABC):
    @abstractmethod
    def decision_check(self, q, output):
        raise NotImplementedError

    @abstractmethod
    def format_one(self, q: Any) -> str:
        raise NotImplementedError


@dataclass
class Question(ABC):
    kind: str = "null"  # e.g. clrs, finegrain, gsm8k etc
    digits: int = 0
    code: str = ""

    @property
    @abstractmethod
    def util_pointer(self):
        raise NotImplementedError


DEFAULT_CODE_INSTR = """
The code block that specifies a function 'solution()' that defines all variables, imports and IMPLEMENTS the actual code to solve the problem that can be executed. Begin and end code with ```python```. For example an INCORRECT way to solve the problem (Don't copy method, but only formatting) but is formatted correctly:       

```python
def solution():
    import numpy as np
    variable = [0,1,2,3]
    out = np.sum(variable) 
    return out
```

""".strip()

FORMATTING = "FOLLOW THE FORMAT CAREFULLY. Here are the format instructions: {format_instructions}"
NL_INSTRUCT = "YOU ARE NEVER ALLOWED TO USE CODE."
SIM_TEMPLATE = "Simulate the execution of the provided code: {code} \n. ALL NECESSARY INFORMATION IS IN THE CODE PROVIDED " + FORMATTING


def get_prompts(desc):
    Prompts = desc + FORMATTING
    Prompts_nl = desc + NL_INSTRUCT + FORMATTING
    return Prompts, Prompts_nl, SIM_TEMPLATE


def get_prompt_classes(SpecificModel: BaseModel, func_typing: str):
    class CodeReasoning(SpecificModel):
        code: str = Field(
            description=DEFAULT_CODE_INSTR + f"Here are the required types: def solution() -> {func_typing}",
            default="",
        )
        simulation: str = Field(description="The attempt at simulating the code in natural language reasoning to give the final answer.", default="")

    class NLReasoning(SpecificModel):
        reasoning: str = Field(
            description="The attempt at simulating the problem in natural language reasoning to give the final answer.",
            default="",
        )

    class ControlledCodeSim(SpecificModel):
        simulation: str = Field(
            description="The attempt at simulating the code in natural language reasoning to give the final answer.",
            default="",
        )

    return CodeReasoning, NLReasoning, ControlledCodeSim


# (d.util_pointer)(self.run_type).format_one(d)
