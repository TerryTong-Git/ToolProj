from src.exps_performance.problems import Problem


class CLRSProblem(Problem):
    kind: str = "clrs"
    digits: int = 0
    answer: str = ""
    text_data: str = ""

    def format(
        self,
    ):
        return self.text_data

    def decision_check(self, computed_ans, problem_text=None):
        str_ans = str(computed_ans)
        return int(str_ans == self.answer)

    def ground_truth(self):
        return self.answer
