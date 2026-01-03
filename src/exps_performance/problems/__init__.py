import ast
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Type, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from src.exps_performance.logger import Record
from src.exps_performance.utils import remove_json_backticks, remove_python_triple_quote


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

    def _strip_fences(self, text: str) -> str:
        """Remove common markdown fences/backticks that break JSON parsing."""
        text = remove_json_backticks(text)
        if "```" in text:
            text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text.strip(), flags=re.DOTALL)
            text = text.replace("```", "")
        return text.strip()

    def _extract_json_span(self, text: str) -> str:
        """Keep content between the first opening and last closing brace/bracket."""
        starts = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
        ends = [pos for pos in (text.rfind("}"), text.rfind("]")) if pos != -1]
        if not starts or not ends:
            return text
        start = min(starts)
        end = max(ends)
        return text[start : end + 1] if start < end else text

    def _strip_trailing_commas(self, text: str) -> str:
        return re.sub(r",(\s*[}\]])", r"\1", text)

    def _normalize_single_quotes(self, text: str) -> str:
        """Swap single quotes to double quotes when no double quotes are present."""
        if '"' not in text and "'" in text:
            return text.replace("'", '"')
        return text

    def _same_as_default(self, parsed: BaseModel, default_model: BaseModel) -> bool:
        try:
            if hasattr(parsed, "model_dump") and hasattr(default_model, "model_dump"):
                return bool(parsed.model_dump() == default_model.model_dump())
            if hasattr(parsed, "dict") and hasattr(default_model, "dict"):
                return bool(parsed.dict() == default_model.dict())  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001
            return False
        return False

    def parse_output(self, output: Any) -> Tuple[BaseModel, str]:  # returns one of the pydantic objects
        model_cls = self.PROB_TYPES[self.prob_type]
        default_model = model_cls()
        last_err: str | Exception = "parse_failed"

        # First attempt: strict LangChain parser.
        try:
            parsed = self.parser.parse(output)
            if not self._same_as_default(parsed, default_model):
                return parsed, "ok"  # ok
            last_err = "empty_parse"
        except OutputParserException as e:
            last_err = e  # keep for diagnostics

        text = output if isinstance(output, str) else str(output)
        text = remove_python_triple_quote(text)
        text = self._strip_fences(text)
        text = text.strip()

        candidates = []
        base = self._extract_json_span(text)
        candidates.append(base)

        trimmed = self._strip_trailing_commas(base)
        if trimmed != base:
            candidates.append(trimmed)

        normalized = self._normalize_single_quotes(trimmed)
        if normalized not in candidates:
            candidates.append(normalized)

        # De-duplicate while preserving order.
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand not in seen:
                unique_candidates.append(cand)
                seen.add(cand)

        for cand in unique_candidates:
            try:
                loaded = json.loads(cand)
                if isinstance(loaded, dict):
                    return model_cls(**loaded), "ok"
            except Exception as json_err:  # noqa: BLE001
                last_err = json_err

            # Try pythonic dicts / numbers (e.g., single quotes) via literal_eval.
            try:
                loaded = ast.literal_eval(cand)
                if isinstance(loaded, dict):
                    return model_cls(**loaded), "ok"
            except Exception as lit_err:  # noqa: BLE001
                last_err = lit_err

        return default_model, str(last_err)

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
