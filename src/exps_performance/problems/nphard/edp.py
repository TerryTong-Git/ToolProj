from problems.nphardeval import NPHardEvalProblem
from prompts import edpPrompts


class EDP(NPHardEvalProblem):
    def __init__(self):
        self.p = edpPrompts

    def format_one(self, q):
        string_a = q["string_a"]
        string_b = q["string_b"]
        prompt_text = self.instantiate_prompt(dict(string_a=string_a, string_b=string_b))
        prompt_text += "Answer:\n"
        return prompt_text
