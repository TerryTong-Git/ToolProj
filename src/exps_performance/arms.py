import asyncio
import copy
import json
import logging
from typing import Any, List, Tuple

from tqdm import tqdm

from src.exps_performance.core.executor import ProgramChatInterface
from src.exps_performance.core.rlm_executor import RecursiveLMExecutor
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
from src.exps_performance.rich_ui import setup_rich_logging
from src.exps_performance.utils import cast_float_to_int, clean_code_llm, remove_python_triple_quote

setup_rich_logging()
logger = logging.getLogger(__name__)  # Using __name__ is a common practice

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
    "tsp_d": TspdCheckAndFormat,
    "gcp": GcpCheckAndFormat,
    "gcp_d": GcpdCheckAndFormat,
    "msp": MspCheckAndFormat,
    "ksp": KspCheckAndFormat,
}

RERUN = 3


class BaseArm:
    run_type: str
    set_name: str

    def __init__(self, data_subset: List[Question], default_args: Any, client: Any):
        self.problems: List[Question] = data_subset
        self.default_args = default_args
        self.client = client

    def run(self) -> Tuple[float, List[Question]]:
        examples = [d.util_pointer(self.run_type).format_one(d) for d in self.problems]
        messages = [[{"role": "user", "content": e}] for e in examples]
        logger.info(f"Running batches for {self.set_name}")
        answers = run_batch(messages, self.default_args, self.client)
        logger.info(f"Running parsing for {self.set_name}")
        parsed_answer = self._parse(answers)
        actual_parsed = [p[0] for p in parsed_answer]
        acc, sequence_parity = self._count_correct(actual_parsed)
        logger.info(f"Setting Results for {self.set_name}")
        edited_problems = self.set_record(answers, parsed_answer, examples, sequence_parity)
        self.parsed_answer = actual_parsed  # for testing
        return acc, edited_problems

    def _count_correct(self, parsed_answer: List[Any]) -> Tuple[float, List[bool]]:
        total_correct = []
        count = 0
        for q, a in zip(self.problems, parsed_answer):
            pUtil = q.util_pointer(self.run_type)
            correct, reason = pUtil.decision_check(q, a)
            count += 1 if correct else 0
            total_correct.append(bool(correct))
        return count / len(self.problems), total_correct

    @staticmethod
    def _is_default_model(parsed_output: Any, default_model: Any) -> bool:
        try:
            if hasattr(parsed_output, "model_dump") and hasattr(default_model, "model_dump"):
                return bool(parsed_output.model_dump() == default_model.model_dump())
            if hasattr(parsed_output, "dict") and hasattr(default_model, "dict"):
                return bool(parsed_output.dict() == default_model.dict())  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001
            return False
        return False

    def _parse(self, answers: List[str]) -> List[Tuple[Any, str]]:
        self.parse_fail = 0
        all_parsed = []
        parse_failed = []

        for i, (q, a) in enumerate(tqdm(zip(self.problems, answers), desc="parsing")):
            pUtil = q.util_pointer(self.run_type)
            a = remove_python_triple_quote(a)
            parsed_output, err = pUtil.parse_output(a)
            default = pUtil.PROB_TYPES[self.run_type]()
            if self._is_default_model(parsed_output, default):
                self.parse_fail += 1
                parse_failed.append((i, q, parsed_output, pUtil, default))
            all_parsed.append((parsed_output, str(err)))

        reparsed = self.rerun(parse_failed)
        for i, reparsed_output, err, rerun_answer in reparsed:
            all_parsed[i] = copy.deepcopy((reparsed_output, str(err)))
            answers[i] = rerun_answer
        self.parsed_fail_ind = [p[0] for p in parse_failed]
        self.reparse_ind = [p[0] for p in reparsed]
        assert self.parsed_fail_ind == self.reparse_ind, "parse fail and reparse_inds not the same"
        return all_parsed

    def each_record(self, q: Question, a: Any, p: Any, e: str, s: bool) -> Question:
        setattr(q.record, self.set_name + "_question", e)
        if self.run_type != "code":
            setattr(q.record, self.set_name + "_reasoning", p[0].simulation)
        setattr(q.record, self.set_name + "_answer", a)
        setattr(q.record, self.set_name + "_parse_err", p[1] != "ok")
        setattr(q.record, self.set_name + "_err_msg", p[1])
        setattr(q.record, self.set_name + "_correct", s)
        return q

    def set_record(self, answers: List[Any], parsed: List[Tuple[Any, str]], examples: List[str], sequence_parity: List[bool]) -> List[Question]:
        edited_problems = []
        for q, a, p, e, s in zip(self.problems, answers, parsed, examples, sequence_parity):
            changed_q = self.each_record(q, a, p, e, s)
            copied_q = copy.deepcopy(changed_q)
            edited_problems.append(copied_q)
        assert edited_problems != [], "nothing added"
        self.edited_problems = edited_problems
        return edited_problems

    def rerun(self, to_reparse: List[Tuple[int, Question, Any, Any, Any]]) -> List[Tuple[int, Any, Any, str]]:
        if to_reparse == []:
            return []

        outs: List[Tuple[int, Any, Any, str]] = []
        to_run = []
        for reparse in to_reparse:
            _i, problem, _parsed, pUtil, _default = reparse
            to_run += [[{"role": "user", "content": pUtil.format_one(problem)}] for _ in range(RERUN)]
            # assert list of lists of dict
        llm_out = run_batch(to_run, self.default_args, self.client)
        logger.info(f"Rerunning parsing for {self.set_name}")

        for prob_index, reparse in enumerate(to_reparse):
            og_ind, _problem, _prev_parsed, pUtil, default = reparse
            last_raw = ""
            last_parsed = default
            last_err: Any = "parse_failed"

            for rerun_index in range(RERUN):
                llm_index = prob_index * RERUN + rerun_index
                if llm_index >= len(llm_out):
                    break
                raw_output = llm_out[llm_index]
                llm_o = remove_python_triple_quote(raw_output)  # not accepted by langchain
                parsed_output, err = pUtil.parse_output(llm_o)
                last_raw = raw_output
                last_parsed = parsed_output
                last_err = err
                if not self._is_default_model(parsed_output, default):
                    break

            outs.append((og_ind, last_parsed, last_err, last_raw))

        return outs


class Arm2(BaseArm):
    run_type: str = "code"
    set_name: str = "sim"

    def each_record(self, q: Question, a: Any, p: Any, e: str, s: bool) -> Question:
        q.record.question = str(q.question)
        q.record.answer = str(q.answer)
        q.code = p[0].code
        q.record.sim_code = q.code
        q.record.sim_reasoning = getattr(p[0], "simulation", "")
        q.record.kind = q.kind
        q.record.digit = q.digits
        q.record.model = self.default_args.model
        q.record.seed = self.default_args.seed
        q = super().each_record(q, a, p, e, s)
        return q


class Arm3(BaseArm):
    run_type: str = "code"
    set_name: str = "code"

    def run(self) -> Tuple[float, List[Question]]:
        logger.info("Running Code Execution")
        self.parse_fail = 0
        sequence_parity: List[bool] = [False] * len(self.problems)
        parsed_answer: List[Tuple[Any, str]] = [("", "")] * len(self.problems)
        answers: List[str] = [""] * len(self.problems)
        examples: List[str] = [""] * len(self.problems)

        def _run_one(idx: int, p: Question) -> Tuple[int, str, str, Tuple[Any, str], bool, str | None]:
            pUtil = p.util_pointer(self.run_type)
            parse_err = "ok"
            if p.code == "":
                default_parsed = pUtil.PROB_TYPES[self.run_type]()
                return idx, "", "", (default_parsed, "no_code"), False, "no_code"
            cleaned_code = clean_code_llm(p.code)
            assert "```" not in cleaned_code
            code, gen_err = self.extract_locals(cleaned_code)
            code = cast_float_to_int(code)
            type_class = pUtil.PROB_TYPES[self.run_type]
            parsed = type_class()
            if pUtil.type_check_code(str(code)):
                kwargs = pUtil.get_field_kwargs(code)
                parsed = type_class(**kwargs)
            else:
                parse_err = "type_check_failed"
            code = str(code)
            correct, reason = pUtil.decision_check(p, parsed)
            err_msg = f"{parse_err},{gen_err}"
            return idx, cleaned_code, code, (parsed, err_msg), bool(correct), None

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=getattr(self.default_args, "exec_workers", 4)) as ex:
            futures = [ex.submit(_run_one, i, p) for i, p in enumerate(self.problems)]
            for fut in as_completed(futures):
                idx, cleaned_code, code, parsed_tuple, is_correct, parse_err_flag = fut.result()
                if parsed_tuple is None:
                    default_parsed = self.problems[idx].util_pointer(self.run_type).PROB_TYPES[self.run_type]()
                    parsed_tuple = (default_parsed, ("unknown_err", ""))
                examples[idx] = cleaned_code
                answers[idx] = code
                parsed_answer[idx] = parsed_tuple
                sequence_parity[idx] = bool(is_correct)
                if parse_err_flag:
                    self.parse_fail += 1

        total_correct = sum(sequence_parity)
        actual_parsed = [p[0] for p in parsed_answer if p is not None]  # type: ignore[index]
        self.parsed_answer = actual_parsed
        logger.info(f"Setting Results for {self.set_name}")
        edited_problems = self.set_record(answers, parsed_answer, examples, sequence_parity)
        assert edited_problems != [], "empty problems"
        return total_correct / len(self.problems), edited_problems

    def extract_locals(self, code: str) -> Tuple[str, str]:
        itf = ProgramChatInterface(
            answer_expr="solution()",
            timeout_seconds=getattr(self.default_args, "exec_timeout_seconds", 5),
            max_attempts=getattr(self.default_args, "exec_max_attempts", 2),
        )
        return itf.run(code)

    def each_record(self, q: Question, a: Any, p: Tuple[Any, str], e: str, s: bool) -> Question:
        q = super().each_record(q, a, (p[0], p[1]), e, s)
        q.record.code_gen_err = p[1]  # bug
        return q


class Arm4(BaseArm):
    run_type: str = "sim"
    set_name: str = "controlsim"


class Arm1(BaseArm):
    run_type: str = "nl"
    set_name: str = "nl"


class _BaseRLMArm(BaseArm):
    def _rlm_workers(self) -> int:
        return max(1, int(getattr(self.default_args, "exec_workers", 4)))

    async def _run_prompts_async(self, prompts: List[str]) -> Tuple[List[str], List[str], List[float], List[str]]:
        if hasattr(self.executor, "arun_many"):
            results = await self.executor.arun_many(
                prompts,
                max_concurrent=self._rlm_workers(),
                progress_desc=self.set_name,
                parent_task_id=getattr(self.default_args, "_rich_stage_task_id", None),
            )
        elif hasattr(self.executor, "arun"):
            semaphore = asyncio.Semaphore(self._rlm_workers())
            ordered_results: list[Any | None] = [None] * len(prompts)

            async def _one(index: int, prompt: str) -> None:
                async with semaphore:
                    ordered_results[index] = await self.executor.arun(prompt)

            await asyncio.gather(*[asyncio.create_task(_one(idx, prompt)) for idx, prompt in enumerate(prompts)])
            results = [r for r in ordered_results if r is not None]
        else:
            results = [self.executor.run(prompt) for prompt in prompts]

        answers: List[str] = []
        exec_errors: List[str] = []
        exec_times: List[float] = []
        metadata_jsons: List[str] = []
        for result in results:
            answers.append(result.response)
            exec_errors.append(result.err)
            exec_times.append(float(getattr(result, "execution_time", 0.0) or 0.0))
            metadata = getattr(result, "metadata", None)
            metadata_jsons.append(json.dumps(metadata, ensure_ascii=False, sort_keys=True) if metadata else "")
        return answers, exec_errors, exec_times, metadata_jsons

    def run(self) -> Tuple[float, List[Question]]:
        logger.info(f"Running RLM batches for {self.set_name}")
        self.executor = RecursiveLMExecutor(self.default_args)
        prompts = [q.util_pointer(self.run_type).format_one(q) for q in self.problems]
        examples: List[str] = list(prompts)
        answers, self.exec_errors, self.exec_times, self.exec_metadata_json = asyncio.run(
            self._run_prompts_async(prompts)
        )

        logger.info(f"Running parsing for {self.set_name}")
        parsed_answer = self._parse(answers)
        actual_parsed = [p[0] for p in parsed_answer]
        acc, sequence_parity = self._count_correct(actual_parsed)
        logger.info(f"Setting Results for {self.set_name}")
        edited_problems = self.set_record(answers, parsed_answer, examples, sequence_parity)
        self.parsed_answer = actual_parsed
        return acc, edited_problems

    def rerun(self, to_reparse: List[Tuple[int, Question, Any, Any, Any]]) -> List[Tuple[int, Any, Any, str]]:
        if to_reparse == []:
            return []

        logger.info(f"Rerunning parsing for {self.set_name}")
        prompts = [pUtil.format_one(problem) for _og_ind, problem, _prev_parsed, pUtil, _default in to_reparse]
        raw_answers, raw_exec_errors, _raw_exec_times, _raw_metadata_jsons = asyncio.run(
            self._run_prompts_async(prompts)
        )

        outs: List[Tuple[int, Any, Any, str]] = []
        for (og_ind, problem, _prev_parsed, pUtil, default), raw_response, exec_err in zip(
            to_reparse, raw_answers, raw_exec_errors
        ):
            last_parsed = default
            last_err: Any = "parse_failed"
            last_raw_response = raw_response
            last_exec_err = exec_err

            llm_o = remove_python_triple_quote(raw_response)
            parsed_output, err = pUtil.parse_output(llm_o)
            last_parsed = parsed_output
            last_err = err

            if self._is_default_model(parsed_output, default):
                retry_prompt = pUtil.format_one(problem)
                for _attempt in range(RERUN - 1):
                    retry_answers, retry_exec_errors, _retry_exec_times, _retry_metadata_jsons = asyncio.run(
                        self._run_prompts_async([retry_prompt])
                    )
                    retry_result = retry_answers[0]
                    last_raw_response = retry_result
                    last_exec_err = retry_exec_errors[0]
                    llm_o = remove_python_triple_quote(retry_result)
                    parsed_output, err = pUtil.parse_output(llm_o)
                    last_parsed = parsed_output
                    last_err = err
                    if not self._is_default_model(parsed_output, default):
                        break

            merged_err = str(last_err) if last_exec_err == "ok" else f"{last_err},{last_exec_err}"
            outs.append((og_ind, last_parsed, merged_err, last_raw_response))

        return outs

    def each_record(self, q: Question, a: Any, p: Any, e: str, s: bool) -> Question:
        q.record.question = str(q.question)
        q.record.answer = str(q.answer)
        q.record.kind = q.kind
        q.record.digit = q.digits
        q.record.model = self.default_args.model
        q.record.seed = self.default_args.seed
        return super().each_record(q, a, p, e, s)

    def set_record(self, answers: List[Any], parsed: List[Tuple[Any, str]], examples: List[str], sequence_parity: List[bool]) -> List[Question]:
        merged_parsed: List[Tuple[Any, str]] = []
        for idx, parsed_tuple in enumerate(parsed):
            parse_err = parsed_tuple[1]
            exec_err = self.exec_errors[idx] if idx < len(self.exec_errors) else "ok"
            if exec_err == "ok":
                merged_parsed.append(parsed_tuple)
            elif parse_err:
                merged_parsed.append((parsed_tuple[0], f"{parse_err},{exec_err}"))
            else:
                merged_parsed.append((parsed_tuple[0], exec_err))
        edited = super().set_record(answers, merged_parsed, examples, sequence_parity)
        for idx, q in enumerate(edited):
            setattr(q.record, f"{self.set_name}_execution_time", self.exec_times[idx])
            setattr(q.record, f"{self.set_name}_metadata_json", self.exec_metadata_json[idx])
        return edited


class ArmRLMNL(_BaseRLMArm):
    run_type: str = "nl"
    set_name: str = "rlmnl"


class ArmRLMCode(_BaseRLMArm):
    run_type: str = "code"
    set_name: str = "rlmcode"

    def each_record(self, q: Question, a: Any, p: Any, e: str, s: bool) -> Question:
        q = super().each_record(q, a, p, e, s)
        q.record.rlmcode_reasoning = getattr(p[0], "simulation", "")
        q.record.rlmcode_code = getattr(p[0], "code", "")
        return q
