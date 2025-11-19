from pydantic import BaseModel, Field

default_code_instr = """
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
            description=default_code_instr + f"Here are the required types: def solution() -> {func_typing}",
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
