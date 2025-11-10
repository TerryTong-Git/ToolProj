from problems.nphardeval import NPHardEvalProblem
from prompts import mfpPrompts


class MFP(NPHardEvalProblem):
    def __init__(self):
        self.p = mfpPrompts

    def format_one(self, q):
        source_node = q["source"]
        sink_node = q["sink"]
        edges = q["edges"]
        prompt_text = (
            self.instantiate_prompt(dict(source_node=source_node, sink_node=sink_node))
            + "\n\n"
            + "Here is a network description. The capacities of the network's edges are as follows: \n"
        )
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
            prompt_text += this_line + "\n"
        prompt_text += "Answer:\n"
        return prompt_text
