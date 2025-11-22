import copy
from typing import List, Tuple

from src.exps_performance.core.executor import ProgramChatInterface
from src.exps_performance.llm import run_batch
from src.exps_performance.problems import Question
from src.exps_performance.problems.clrs import ClrsCheckAndFormat
from src.exps_performance.problems.finegrained import (
    AddCheckAndFormat,
    IlpAssignCheckAndFormat,
    IlpPartitionCheckAndFormat,
    IlpProdCheckAndFormat,
    Knap01CheckAndFormat,
    LcsCheckAndFormat,
    MulCheckAndFormat,
    RodCheckAndFormat,
    SubCheckAndFormat,
)
from src.exps_performance.problems.gsm8k import Gsm8kCheckAndFormat
from src.exps_performance.problems.nphard.bsp import BspCheckAndFormat
from src.exps_performance.problems.nphard.edp import EdpCheckAndFormat
from src.exps_performance.problems.nphard.gcp import GcpCheckAndFormat
from src.exps_performance.problems.nphard.gcp_d import GcpdCheckAndFormat
from src.exps_performance.problems.nphard.ksp import KspCheckAndFormat
from src.exps_performance.problems.nphard.msp import MspCheckAndFormat
from src.exps_performance.problems.nphard.spp import SppCheckAndFormat
from src.exps_performance.problems.nphard.tsp import TspCheckAndFormat
from src.exps_performance.problems.nphard.tsp_d import TspdCheckAndFormat
from src.exps_performance.utils import clean_code_llm

FG_PROBS = {
    "add": AddCheckAndFormat,
    "sub": SubCheckAndFormat,
    "mul": MulCheckAndFormat,
    "ilp_assign": IlpAssignCheckAndFormat,
    "ilp_partition": IlpPartitionCheckAndFormat,
    "ilp_prod": IlpProdCheckAndFormat,
    "lcs": LcsCheckAndFormat,
    "rod": RodCheckAndFormat,
    "knap": Knap01CheckAndFormat,
}
CLRS = {"clrs": ClrsCheckAndFormat}
GSM8K = {"gsm8k": Gsm8kCheckAndFormat}

NPHARD = {
    "bsp": BspCheckAndFormat,
    "edp": EdpCheckAndFormat,
    "spp": SppCheckAndFormat,
    "tsp": TspCheckAndFormat,
    "tspd": TspdCheckAndFormat,
    "gcp": GcpCheckAndFormat,
    "gcpd": GcpdCheckAndFormat,
    "msp": MspCheckAndFormat,
    "ksp": KspCheckAndFormat,
}


class BaseArm:
    run_type: str
    set_name: str

    def __init__(self, data_subset: List[Question], default_args, client):
        self.problems: List[Question] = data_subset
        self.default_args = default_args
        self.client = client

    def run(self) -> Tuple[float, List[Question]]:
        examples = [d.util_pointer(self.run_type).format_one(d) for d in self.problems]
        messages = [[{"role": "user", "content": e}] for e in examples]
        answers = run_batch(messages, self.default_args, self.client)
        parsed_answer = self._parse(answers)
        actual_parsed = [p[0] for p in parsed_answer]
        acc, sequence_parity = self._count_correct(actual_parsed)
        edited_problems = self.set_record(answers, parsed_answer, examples, sequence_parity)
        self.parsed_answer = actual_parsed  # for testing
        return acc, edited_problems

    def _count_correct(self, parsed_answer):
        total_correct = []
        count = 0
        for q, a in zip(self.problems, parsed_answer):
            pUtil = q.util_pointer(self.run_type)
            correct, reason = pUtil.decision_check(q, a)
            count += 1 if correct else 0
            total_correct.append(correct)
        return count / len(self.problems), total_correct

    def _parse(self, answers: List[str]):
        self.parse_fail = 0
        all_parsed = []
        for q, a in zip(self.problems, answers):
            pUtil = q.util_pointer(self.run_type)
            parsed_output, err = pUtil.parse_output(a)
            default = pUtil.PROB_TYPES[self.run_type]
            if parsed_output == default:
                self.parse_fail += 1
                parsed_output, err = self.rerun(q, parsed_output, pUtil, default)

            all_parsed.append((parsed_output, err))
        return all_parsed

    def each_record(self, q: Question, a, p, e, s) -> Question:
        setattr(q.record, self.set_name + "_question", e)
        if self.run_type != "code":
            setattr(q.record, self.set_name + "_reasoning", p[0].simulation)
        setattr(q.record, self.set_name + "_answer", a)
        setattr(q.record, self.set_name + "_parse_err", p[1] != "ok")
        setattr(q.record, self.set_name + "_err_msg", p[1])
        setattr(q.record, self.set_name + "_correct", s)
        return q

    def set_record(self, answers, parsed, examples, sequence_parity):
        edited_problems = []
        for q, a, p, e, s in zip(self.problems, answers, parsed, examples, sequence_parity):
            changed_q = self.each_record(q, a, p, e, s)
            copied_q = copy.deepcopy(changed_q)
            edited_problems.append(copied_q)
        assert edited_problems != [], "nothing added"
        self.edited_problems = edited_problems
        return edited_problems

    def rerun(self, problem, parsed, pUtil, default):
        default = pUtil.PROB_TYPE[self.run_type]
        count = 0
        while parsed == default and count < 3:
            a = run_batch([[{"role": "user", "content": problem}]], self.default_args, self.client)[0]
            parsed, err = pUtil.parse_output(a)
            count += 1
        print(f"Reran parsing {count} times")
        return parsed, err


class Arm2(BaseArm):
    run_type: str = "code"
    set_name: str = "sim"

    def each_record(self, q: Question, a, p, e, s):
        q.record.question = str(q.question)
        q.record.answer = str(q.answer)
        q.code = p[0].code
        q.record.seed = self.default_args.seed
        q = super().each_record(q, a, p, e, s)
        return q


class Arm3(BaseArm):
    run_type: str = "code"
    set_name: str = "code"

    def run(self) -> Tuple[float, List[Question]]:
        sequence_parity = []
        parsed_answer = []
        total_correct = 0
        examples = []
        answers = []
        self.parse_fail = 0
        for p in self.problems:
            pUtil = p.util_pointer(self.run_type)
            cleaned_code = clean_code_llm(p.code)
            examples.append(cleaned_code)
            assert "```" not in cleaned_code
            code, gen_err = self.extract_locals(cleaned_code)
            assert isinstance(code, str), "code not a string"
            answers.append(code)
            parsed, parse_err = pUtil.parse_output(code)
            correct, reason = pUtil.decision_check(p, parsed)
            sequence_parity.append(correct)

            total_correct += 1 if correct else 0
            if pUtil.type_check_code(code):
                self.parse_fail += 1
            parsed_answer.append((parsed, (parse_err, gen_err)))
        actual_parsed = [p[0] for p in parsed_answer]
        self.parsed_answer = actual_parsed
        edited_problems = self.set_record(answers, parsed_answer, examples, sequence_parity)
        assert edited_problems != [], "empty problems"
        return total_correct / len(self.problems), edited_problems

    def extract_locals(self, code) -> Tuple[str, str]:
        itf = ProgramChatInterface(answer_expr="solution()")
        return itf.run(code)

    def each_record(self, q: Question, a, p, e, s):
        q = super().each_record(q, a, (p[0], p[1][0]), e, s)
        q.record.code_gen_err = p[1][1]  # bug
        return q


class Arm4(BaseArm):
    run_type: str = "sim"
    set_name: str = "controlsim"


class Arm1(BaseArm):
    run_type: str = "nl"
    set_name: str = "nl"
