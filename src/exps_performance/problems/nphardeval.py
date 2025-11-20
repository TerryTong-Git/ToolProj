from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel

from src.exps_performance.problems import CheckAndFormat, Question, get_prompt_classes, get_prompts


@dataclass
class NpQuestion(Question):
    def util_pointer(self):
        return NpCheckAndFormat


class NpCheckAndFormat(CheckAndFormat):
    def __init__(self, prob_type, func_typing, desc, probModel):
        available = ["code", "nl", "sim"]
        PROB_TYPES = {name: classes for name, classes in zip(available, get_prompt_classes(probModel, func_typing))}
        PROMPTS = {name: classes for name, classes in zip(available, get_prompts(desc))}
        self.PROB_TYPES = PROB_TYPES
        self.PROMPTS = PROMPTS
        assert prob_type in list(PROB_TYPES.keys())
        self.prob_type = prob_type
        self.parser = PydanticOutputParser(pydantic_object=PROB_TYPES[prob_type])  # Retry Output parser?

    @property
    def folder_name(self):
        if "." in __name__:
            path = "/".join(__name__.split("."))
        else:
            path = __name__
        return os.path.join(Path(path).parent.parent, "Data_V2")

    def parse_output(self, output) -> BaseModel:  # returns one of the pydantic objects
        try:
            return self.parser.parse(output)  # ok
        except OutputParserException:
            return self.PROB_TYPES[self.prob_type]()  # err

    def prompt_template(self, input_var):
        return PromptTemplate(
            template=self.PROMPTS[self.prob_type],
            input_variables=[input_var],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    # def ground_truth(self):  # dummy
    #     return None

    @abstractmethod
    def format_one(self, q: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_data(self):
        raise NotImplementedError
