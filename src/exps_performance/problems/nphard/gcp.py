from __future__ import annotations

from problems.nphardeval import NPHardEvalProblem
from prompts import gcpPrompts


class GCP(NPHardEvalProblem):
    def __init__(self):
        self.p = gcpPrompts

    def format_one(self, q):
        chromatic_number = q.split("\n")[0][-1]  # last character of the first line
        number_of_vertices = q.split("\n")[1].split(" ")[2]  # third word of the second line
        prompt_text = self.instantiate_prompt(dict(max_vertices=number_of_vertices, max_colors=chromatic_number)) + "\n The graph is below: \n"
        for line in q.split("\n")[2:]:
            vertex_list = line.split(" ")
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1], vertex_list[2])
            prompt_text += this_line + "\n"
        return prompt_text
