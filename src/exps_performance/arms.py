import asyncio
import copy
import json
import logging
from typing import Any, List, Tuple

from src.exps_performance.core.executor import ProgramChatInterface
from src.exps_performance.core.rlm_executor import RecursiveLMExecutor
from src.exps_performance.llm import run_batch
from src.exps_performance.logger import CheckpointManager
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
from src.exps_performance.rich_ui import progress_manager, setup_rich_logging
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
        answers = run_batch(messages, self.default_args, self.client, advance_stage=True)
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

        for i, (q, a) in enumerate(zip(self.problems, answers)):
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
        question_field = self.set_name + "_question"
        reasoning_field = self.set_name + "_reasoning"
        answer_field = self.set_name + "_answer"
        parse_err_field = self.set_name + "_parse_err"
        err_msg_field = self.set_name + "_err_msg"
        correct_field = self.set_name + "_correct"

        prev_question = getattr(q.record, question_field, "")
        prev_reasoning = getattr(q.record, reasoning_field, "")
        prev_answer = getattr(q.record, answer_field, "")
        prev_parse_err = bool(getattr(q.record, parse_err_field, False))
        prev_err_msg = getattr(q.record, err_msg_field, "")
        prev_correct = bool(getattr(q.record, correct_field, False))

        new_parse_err = p[1] != "ok"
        new_reasoning = getattr(p[0], "simulation", "") if self.run_type != "code" else ""
        retry_mode = bool(getattr(self.default_args, "retry_failed_records", False))

        # In retry mode, do not let a failed retry erase a previously good result.
        preserve_existing_success = retry_mode and bool(prev_answer) and not prev_parse_err and new_parse_err

        setattr(q.record, question_field, e or prev_question)
        if self.run_type != "code":
            if preserve_existing_success:
                setattr(q.record, reasoning_field, prev_reasoning or new_reasoning)
            else:
                setattr(q.record, reasoning_field, new_reasoning or prev_reasoning)
        if preserve_existing_success:
            setattr(q.record, answer_field, prev_answer)
            setattr(q.record, parse_err_field, prev_parse_err)
            setattr(q.record, err_msg_field, prev_err_msg or p[1])
            setattr(q.record, correct_field, prev_correct)
        else:
            setattr(q.record, answer_field, a or prev_answer)
            setattr(q.record, parse_err_field, new_parse_err)
            setattr(q.record, err_msg_field, p[1])
            setattr(q.record, correct_field, s)
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
        llm_out = run_batch(to_run, self.default_args, self.client, advance_stage=False)
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
        q.code = getattr(p[0], "code", "") or q.code
        q.record.sim_code = q.code or q.record.sim_code
        q.record.sim_reasoning = getattr(p[0], "simulation", "") or q.record.sim_reasoning
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
        stage_task_id = getattr(self.default_args, "_rich_stage_task_id", None)
        if stage_task_id is not None:
            progress_manager.update(stage_task_id, stats="done=0 cost=$0.0000")

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
                if stage_task_id is not None:
                    progress_manager.update(stage_task_id, advance=1, stats="cost=$0.0000")

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

    async def _rerun_single_prompt_async(self, prompt: str) -> tuple[str, str, float, str]:
        if hasattr(self.executor, "arun"):
            result = await self.executor.arun(prompt)
        else:
            result = self.executor.run(prompt)
        metadata = getattr(result, "metadata", None)
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True) if metadata else ""
        return (
            str(getattr(result, "response", "") or ""),
            str(getattr(result, "err", "ok") or "ok"),
            float(getattr(result, "execution_time", 0.0) or 0.0),
            metadata_json,
        )

    async def _process_single_result_async(
        self,
        index: int,
        raw_answer: str,
        exec_err: str,
        exec_time: float,
        metadata_json: str,
        *,
        prompt_override: str | None = None,
    ) -> tuple[Question, Any, bool]:
        q = self.problems[index]
        pUtil = q.util_pointer(self.run_type)
        prompt_text = prompt_override or pUtil.format_one(q)
        response_text = raw_answer
        last_exec_err = exec_err
        last_exec_time = exec_time
        last_metadata_json = metadata_json

        llm_o = remove_python_triple_quote(response_text)
        parsed_output, err = pUtil.parse_output(llm_o)
        default = pUtil.PROB_TYPES[self.run_type]()

        parse_failed = self._is_default_model(parsed_output, default)
        if parse_failed:
            self.parse_fail += 1

        if parse_failed:
            for _attempt in range(RERUN - 1):
                retry_answer, retry_exec_err, retry_exec_time, retry_metadata_json = await self._rerun_single_prompt_async(prompt_text)
                response_text = retry_answer
                last_exec_err = retry_exec_err
                last_exec_time = retry_exec_time
                last_metadata_json = retry_metadata_json
                llm_o = remove_python_triple_quote(retry_answer)
                parsed_output, err = pUtil.parse_output(llm_o)
                if not self._is_default_model(parsed_output, default):
                    break

        merged_err = str(err) if last_exec_err == "ok" else f"{err},{last_exec_err}"
        correct, _reason = pUtil.decision_check(q, parsed_output)
        updated_q = self.each_record(q, response_text, (parsed_output, merged_err), prompt_text, bool(correct))
        setattr(updated_q.record, f"{self.set_name}_execution_time", float(last_exec_time))
        setattr(updated_q.record, f"{self.set_name}_metadata_json", last_metadata_json)
        return copy.deepcopy(updated_q), parsed_output, bool(correct)

    def run(self) -> Tuple[float, List[Question]]:
        logger.info(f"Running RLM batches for {self.set_name}")
        self.executor = RecursiveLMExecutor(self.default_args)
        prompts = [q.util_pointer(self.run_type).format_one(q) for q in self.problems]
        examples: List[str] = list(prompts)
        answers, self.exec_errors, self.exec_times, self.exec_metadata_json = asyncio.run(self._run_prompts_async(prompts))

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
        raw_answers, raw_exec_errors, _raw_exec_times, _raw_metadata_jsons = asyncio.run(self._run_prompts_async(prompts))

        outs: List[Tuple[int, Any, Any, str]] = []
        for (og_ind, problem, _prev_parsed, pUtil, default), raw_response, exec_err in zip(to_reparse, raw_answers, raw_exec_errors):
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
                    retry_answers, retry_exec_errors, _retry_exec_times, _retry_metadata_jsons = asyncio.run(self._run_prompts_async([retry_prompt]))
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

    def run_checkpointed(self, checkpoint: CheckpointManager, flush_every: int = 1) -> Tuple[float, List[Question]]:
        logger.info(f"Running eager checkpointed RLM batches for {self.set_name}")
        self.executor = RecursiveLMExecutor(self.default_args)
        prompts = [q.util_pointer(self.run_type).format_one(q) for q in self.problems]
        total = len(prompts)
        if total == 0:
            self.parsed_answer = []
            self.edited_problems = []
            return 0.0, []

        self.parse_fail = 0
        parsed_outputs: List[Any | None] = [None] * total
        correctness: List[bool] = [False] * total
        edited: List[Question | None] = [None] * total
        completed = 0
        flush_every = max(1, int(flush_every))

        async def _on_completed(index: int, result: Any) -> None:
            nonlocal completed
            edited_q, parsed_output, is_correct = await self._process_single_result_async(
                index,
                str(getattr(result, "response", "") or ""),
                str(getattr(result, "err", "ok") or "ok"),
                float(getattr(result, "execution_time", 0.0) or 0.0),
                json.dumps(getattr(result, "metadata", None), ensure_ascii=False, sort_keys=True) if getattr(result, "metadata", None) else "",
                prompt_override=prompts[index],
            )
            edited[index] = edited_q
            parsed_outputs[index] = parsed_output
            correctness[index] = is_correct
            completed += 1
            checkpoint.upsert(edited_q.record, flush=(completed % flush_every == 0 or completed == total))

        async def _run_all() -> None:
            if hasattr(self.executor, "arun_many"):
                await self.executor.arun_many(
                    prompts,
                    max_concurrent=self._rlm_workers(),
                    progress_desc=self.set_name,
                    parent_task_id=getattr(self.default_args, "_rich_stage_task_id", None),
                    on_completed=_on_completed,
                )
                return

            if hasattr(self.executor, "arun"):
                semaphore = asyncio.Semaphore(self._rlm_workers())

                async def _one(index: int, prompt: str) -> None:
                    async with semaphore:
                        result = await self.executor.arun(prompt)
                        await _on_completed(index, result)

                await asyncio.gather(*[asyncio.create_task(_one(index, prompt)) for index, prompt in enumerate(prompts)])
                return

            for index, prompt in enumerate(prompts):
                result = self.executor.run(prompt)
                await _on_completed(index, result)

        asyncio.run(_run_all())

        finalized = [q for q in edited if q is not None]
        self.parsed_answer = [p for p in parsed_outputs if p is not None]
        self.edited_problems = finalized
        accuracy = (sum(1 for flag in correctness if flag) / total) if total else 0.0
        return accuracy, finalized


class ArmRLMNL(_BaseRLMArm):
    run_type: str = "nl"
    set_name: str = "rlmnl"


class ArmRLMCode(_BaseRLMArm):
    run_type: str = "code"
    set_name: str = "rlmcode"

    def each_record(self, q: Question, a: Any, p: Any, e: str, s: bool) -> Question:
        q = super().each_record(q, a, p, e, s)
        q.record.rlmcode_reasoning = getattr(p[0], "simulation", "") or q.record.rlmcode_reasoning
        q.record.rlmcode_code = getattr(p[0], "code", "") or q.record.rlmcode_code
        return q
