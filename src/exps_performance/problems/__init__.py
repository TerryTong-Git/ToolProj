from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Type, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from src.exps_performance.logger import Record


class CheckAndFormat(ABC):
    def __init__(self, prob_type: str, func_typing: str, desc: str, answerClass: BaseModel) -> None:
        available = ["code", "nl", "sim"]
        PROB_TYPES = {name: classes for name, classes in zip(available, get_prompt_classes(answerClass, func_typing))}
        PROMPTS = {name: classes for name, classes in zip(available, get_prompts(desc))}
        self.PROB_TYPES = PROB_TYPES
        self.PROMPTS = PROMPTS
        assert prob_type in list(PROB_TYPES.keys())
        self.prob_type = prob_type
        self.parser = PydanticOutputParser(pydantic_object=PROB_TYPES[prob_type])

    @abstractmethod
    def decision_check(self, q: Any, output: Any) -> Tuple[bool, Any]:
        raise NotImplementedError

    def parse_output(self, output: Any) -> Tuple[BaseModel, str]:  # returns one of the pydantic objects
        try:
            return self.parser.parse(output), "ok"  # ok
        except OutputParserException as e:
            return self.PROB_TYPES[self.prob_type](), e  # err

    def prompt_template(self, input_var: Union[str, Sequence[str]]) -> PromptTemplate:
        input_vars = [input_var] if isinstance(input_var, str) else list(input_var)
        return PromptTemplate(
            template=self.PROMPTS[self.prob_type],
            input_variables=input_vars,
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    @abstractmethod
    def format_one(self, q: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_data(self) -> list[Any]:
        raise NotImplementedError


@dataclass
class Question(ABC):
    kind: str = "null"  # e.g. clrs, finegrain, gsm8k etc
    digits: int = 0
    code: str = ""

    question: str = ""  # formatted prompt
    answer: str = ""  # gold answer
    record: Record = Record()  # solution

    @property
    @abstractmethod
    def util_pointer(self) -> Any:
        raise NotImplementedError


DEFAULT_CODE_INSTR = """
The code block that specifies a function 'solution()' that defines all variables, imports and IMPLEMENTS the actual code to solve the problem that can be executed. You may use packages Pulp, Numpy, Pandas, Torch, Scipy. Begin and end code with ```python```.Do not \"\"\"python For example an INCORRECT way to solve the problem (Don't copy method, but only formatting) but is formatted correctly:       

```python
def solution():
    import numpy as np
    variable = [0,1,2,3]
    out = np.sum(variable) 
    return int(out)
```

""".strip()

FORMATTING = "FOLLOW THE FORMAT CAREFULLY. Here are the format instructions: {format_instructions}"
NL_INSTRUCT = "YOU ARE NEVER ALLOWED TO USE CODE."
SIM_TEMPLATE = "Simulate the execution of the provided code: {code} \n. ALL NECESSARY INFORMATION IS IN THE CODE PROVIDED " + FORMATTING


def get_prompts(desc: str) -> Tuple[str, str, str]:
    Prompts = desc + FORMATTING
    Prompts_nl = desc + NL_INSTRUCT + FORMATTING
    return Prompts, Prompts_nl, SIM_TEMPLATE


def get_prompt_classes(SpecificModel: BaseModel, func_typing: str) -> Tuple[Type[BaseModel], Type[BaseModel], Type[BaseModel]]:
    class CodeReasoning(SpecificModel):
        code: str = Field(
            description=DEFAULT_CODE_INSTR + f"Here are the required types: def solution() -> {func_typing}",
            default="",
        )
        simulation: str = Field(description="The attempt at simulating the code in natural language reasoning to give the final answer.", default="")

    class NLReasoning(SpecificModel):
        simulation: str = Field(
            description="The attempt at simulating the problem in natural language reasoning to give the final answer.",
            default="",
        )

    class ControlledCodeSim(SpecificModel):
        simulation: str = Field(
            description="The attempt at simulating the code in natural language reasoning to give the final answer.",
            default="",
        )

    return CodeReasoning, NLReasoning, ControlledCodeSim
