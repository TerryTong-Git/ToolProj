from __future__ import annotations

import ast

from src.exps_performance.problems.nphardeval import NPHardEvalProblem
from src.exps_performance.problems.prompts import gcpPrompts
from src.exps_performance.utils import read_dimacs_format


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

    def gcpCheck(self, dimacs_str, answer_str):
        num_vertices, adjacency_list = read_dimacs_format(dimacs_str)
        answer_colors = self.parse_answer(answer_str)

        # Check if all colors in the answer are valid
        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                try:
                    if answer_colors[vertex] == answer_colors[neighbor]:
                        print(f"Invalid coloring: Vertex {vertex} and {neighbor} have the same color.")
                        return False
                except:  # noqa
                    print("Invalid input.")  # dealing with hullucination
                    return False

        print(f"Valid coloring found with {len(set(answer_colors.values()))} colors: {answer_colors}")
        return True

    def decision_check(self, q, output):
        return self.gcpCheck(q, output)

    def parse_answer(self, llm_string):
        # all_answers, reasoning_element = parse_xml_to_dict(llm_string)
        all_answers = ""  # fix the parsing

        if all_answers == "":
            return {}
        elif all_answers is None:
            return {}
        else:
            if isinstance(all_answers, str):
                try:
                    all_answers = ast.literal_eval(all_answers)
                except:  # noqa
                    try:
                        all_answers = ast.literal_eval("{" + all_answers + "}")
                    except:  # noqa
                        return {}
            else:
                all_answers = ast.literal_eval(all_answers.text)
        # answer_dict = {}
        # for pair in all_answers:
        #     vertex, color = pair.split(":")
        #     answer_dict[int(vertex)] = color
        # convert key type to int
        all_answers = {int(k): v for k, v in all_answers.items()}
        return all_answers  # answer_dict

    @staticmethod
    def load_data(data_path):
        n = 11
        start = n - 10
        all_data = []
        for file_num in range(start, n):
            with open(data_path + "synthesized_data_GCP_{}.txt".format(file_num)) as f:
                data = f.read()
            all_data += data.split("\n\n")[:-1]
        return all_data
