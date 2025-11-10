from problems.nphardeval import NPHardEvalProblem
from prompts import sppPrompts


class SPP(NPHardEvalProblem):
    def __init__(self):
        self.p = sppPrompts

    def format_one(self, q):
        start_node = q["nodes"][0]
        end_node = q["nodes"][-1]
        edges = q["edges"]
        prompt_text = self.instantiate_prompt(dict(start_node=start_node, end_node=end_node)) + "\n The graph's edges and weights are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
            prompt_text += this_line + "\n"
        prompt_text += "Answer:\n"
        return prompt_text
