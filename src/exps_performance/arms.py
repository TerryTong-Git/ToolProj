import copy
import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

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
from src.exps_performance.structured_outputs import STAGE_SYSTEM_INSTRUCTION, structured_output_request, validate_nonempty_fields
from src.exps_performance.utils import cast_float_to_int, clean_code_llm, remove_python_triple_quote

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        self.llm_responses: List[Any] = []

    def _run_batch_responses(
        self,
        messages: List[List[Dict[str, str]]],
        request_options_list: List[Optional[Dict[str, Any]]],
        on_response: Optional[Callable[[int, Any], None]] = None,
    ) -> List[Any]:
        try:
            responses = run_batch(
                messages,
                self.default_args,
                self.client,
                request_options_list=request_options_list,
                return_responses=True,
                on_response=on_response,
            )
            return list(responses)
        except TypeError as exc:
            err_text = str(exc)
            if "request_options_list" not in err_text and "return_responses" not in err_text and "on_response" not in err_text:
                raise
            answers = run_batch(messages, self.default_args, self.client)
            return [SimpleNamespace(text=str(answer or "")) for answer in answers]

    def _structured_outputs_enabled(self) -> bool:
        return bool(getattr(self.default_args, "openrouter_structured_outputs", False) and getattr(self.default_args, "backend", "") == "openrouter")

    def _reasoning_extra_body(self) -> Dict[str, Any]:
        if not getattr(self.default_args, "openrouter_reasoning_enabled", False):
            return {}
        reasoning: Dict[str, Any] = {"enabled": True}
        effort = getattr(self.default_args, "openrouter_reasoning_effort", None)
        max_tokens = getattr(self.default_args, "openrouter_reasoning_max_tokens", None)
        if effort:
            reasoning["effort"] = effort
        if max_tokens:
            reasoning["max_tokens"] = int(max_tokens)
        reasoning["exclude"] = bool(getattr(self.default_args, "openrouter_reasoning_exclude", False))
        return {"reasoning": reasoning}

    def _message_bundle(self, q: Question, prompt: str) -> tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]:
        if not self._structured_outputs_enabled():
            return [{"role": "user", "content": prompt}], None

        parser = q.util_pointer(self.run_type)
        model_cls = parser.PROB_TYPES[self.run_type]
        extra_body = self._reasoning_extra_body()
        verbosity = getattr(self.default_args, "openrouter_verbosity", None)
        if verbosity:
            extra_body["verbosity"] = verbosity
        request_options: Dict[str, Any] = {
            "response_format": structured_output_request(model_cls, strict=bool(getattr(self.default_args, "openrouter_structured_strict", True))),
            "retry_attempts": int(getattr(self.default_args, "openrouter_retry_attempts", 3)),
            "enable_response_healing": bool(getattr(self.default_args, "openrouter_response_healing", True)),
            "timeout": float(getattr(self.default_args, "request_timeout", 120.0)),
        }
        max_concurrency = getattr(self.default_args, "openrouter_max_concurrency", None)
        if max_concurrency:
            request_options["max_concurrency"] = int(max_concurrency)
        if extra_body:
            request_options["extra_body"] = extra_body
        messages: List[Dict[str, str]] = []
        system_instruction = STAGE_SYSTEM_INSTRUCTION.get(self.set_name)
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages, request_options

    def _parse_one(self, q: Question, answer: str, *, count_fail: bool = True) -> Tuple[Any, str]:
        pUtil = q.util_pointer(self.run_type)
        cleaned_answer = remove_python_triple_quote(answer)
        parsed_output, err = pUtil.parse_output(cleaned_answer)
        if err == "ok" and self._structured_outputs_enabled():
            err = validate_nonempty_fields(parsed_output)
        default = pUtil.PROB_TYPES[self.run_type]()
        if count_fail and self._is_default_model(parsed_output, default):
            self.parse_fail += 1
        return parsed_output, str(err)

    def _build_completed_question(self, idx: int, answer: str, example: str) -> Question:
        q = copy.deepcopy(self.problems[idx])
        parsed = self._parse_one(q, answer, count_fail=False)
        pUtil = q.util_pointer(self.run_type)
        correct, _ = pUtil.decision_check(q, parsed[0])
        return self.each_record(q, answer, parsed, example, bool(correct))

    def run(self, on_question_complete: Optional[Callable[[Question], None]] = None) -> Tuple[float, List[Question]]:
        examples = [d.util_pointer(self.run_type).format_one(d) for d in self.problems]
        bundled = [self._message_bundle(d, e) for d, e in zip(self.problems, examples)]
        messages = [bundle[0] for bundle in bundled]
        request_options = [bundle[1] for bundle in bundled]
        logger.info(f"Running batches for {self.set_name}")

        def _handle_response(idx: int, response: Any) -> None:
            if on_question_complete is None:
                return
            try:
                completed_q = self._build_completed_question(idx, str(response.text or ""), examples[idx])
                on_question_complete(completed_q)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"incremental checkpoint failed for {self.set_name} idx={idx}: {exc}")

        responses = self._run_batch_responses(messages, request_options, on_response=_handle_response if on_question_complete else None)
        self.llm_responses = list(responses)
        answers = [response.text for response in responses]
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
            if err == "ok" and self._structured_outputs_enabled():
                err = validate_nonempty_fields(parsed_output)
            default = pUtil.PROB_TYPES[self.run_type]()
            if self._is_default_model(parsed_output, default):
                self.parse_fail += 1
                parse_failed.append((i, q, parsed_output, pUtil, default))
            all_parsed.append((parsed_output, str(err)))

        self.parsed_fail_ind = [p[0] for p in parse_failed]
        if self._structured_outputs_enabled():
            self.reparse_ind = []
            return all_parsed

        reparsed = self.rerun(parse_failed)
        for i, reparsed_output, err in reparsed:
            all_parsed[i] = copy.deepcopy((reparsed_output, str(err)))
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

    def rerun(self, to_reparse: List[Tuple[int, Question, Any, Any, Any]]) -> List[Tuple[int, Any, Any]]:
        if to_reparse == []:
            return []
        outs = []
        to_run = []
        request_options_list: List[Optional[Dict[str, Any]]] = []
        for reparse in to_reparse:
            i, problem, parsed, pUtil, default = reparse
            for _ in range(RERUN):
                messages, request_options = self._message_bundle(problem, pUtil.format_one(problem))
                to_run.append(messages)
                request_options_list.append(request_options)
            # assert list of lists of dict
        rerun_responses = self._run_batch_responses(to_run, request_options_list)
        self.llm_responses.extend(rerun_responses)
        llm_out = [response.text for response in rerun_responses]
        i = 0
        logger.info(f"Rerunning parsing for {self.set_name}")
        while i < len(llm_out):
            llm_o = llm_out[i]
            prob_index = i // RERUN  # i w.r.t. to given list
            rerun_index = i % RERUN  # 443 -> 3
            og_ind, problem, _prev_parsed, pUtil, default = to_reparse[prob_index]
            llm_o = remove_python_triple_quote(llm_o)  # not accepted by langchain
            parsed_output, err = pUtil.parse_output(llm_o)
            if err == "ok" and self._structured_outputs_enabled():
                err = validate_nonempty_fields(parsed_output)
            if not self._is_default_model(parsed_output, default) or rerun_index == (RERUN - 1):
                outs.append((og_ind, parsed_output, err))
                i += RERUN - rerun_index
            else:
                i += 1
        if len(to_reparse) != len(outs):
            outs.append((og_ind, parsed_output, err))
        return outs

    def response_summary(self) -> Dict[str, Any]:
        if not self.llm_responses:
            return {}
        total = len(self.llm_responses)
        return {
            "llm_requests": total,
            "llm_total_attempts": sum(int(getattr(resp, "attempts", 1) or 1) for resp in self.llm_responses),
            "llm_errors": sum(1 for resp in self.llm_responses if getattr(resp, "error", "")),
            "structured_requested": sum(1 for resp in self.llm_responses if getattr(resp, "structured_requested", False)),
            "reasoning_requested": sum(1 for resp in self.llm_responses if getattr(resp, "reasoning_requested", False)),
            "reasoning_visible": sum(1 for resp in self.llm_responses if bool(getattr(resp, "reasoning", ""))),
            "reasoning_details_visible": sum(1 for resp in self.llm_responses if int(getattr(resp, "reasoning_details_count", 0) or 0) > 0),
            "reasoning_tokens_visible": sum(1 for resp in self.llm_responses if int(getattr(resp, "reasoning_tokens", 0) or 0) > 0),
            "empty_text_responses": sum(1 for resp in self.llm_responses if not str(getattr(resp, "text", "") or "").strip()),
        }


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

    def run(self, on_question_complete: Optional[Callable[[Question], None]] = None) -> Tuple[float, List[Question]]:
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
                if on_question_complete is not None:
                    try:
                        completed_q = copy.deepcopy(self.problems[idx])
                        completed_q = self.each_record(completed_q, code, parsed_tuple, cleaned_code, bool(is_correct))
                        on_question_complete(completed_q)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"incremental checkpoint failed for {self.set_name} idx={idx}: {exc}")

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
