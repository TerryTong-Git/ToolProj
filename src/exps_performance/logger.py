# define how to serialize here.
import hashlib
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

# class AggregateRecord(BaseModel):
#     model: str
#     digits: int
#     kinds: str
#     code_acc: float
#     nl_acc: float
#     sim_acc: float
#     controlled_acc: float
#     seed: int
#     n: int
#     exp_id: str


class Record(BaseModel):
    request_id: str = ""
    index_in_kind: int = -1
    model: str = ""  # answers depend on this
    seed: int = -1  # answers depend on this
    exp_id: str = ""
    digit: int = -1
    kind: str = ""
    question: str = ""
    answer: str = ""

    nl_question: str = ""
    nl_answer: str = ""
    nl_reasoning: str = ""
    nl_correct: bool = False
    nl_parse_err: bool = False
    nl_err_msg: str = ""  # defaults to "" if not err

    code_question: str = ""
    code_answer: str = ""  # (or err message)
    code_correct: bool = False
    code_parse_err: bool = False
    code_gen_err: str = ""
    code_err_msg: str = ""
    sim_code: str = ""

    sim_question: str = ""
    sim_reasoning: str = ""  # attempted reasoning
    sim_answer: str = ""
    sim_correct: bool = False
    sim_parse_err: bool = False
    sim_err_msg: str = ""

    controlsim_question: str = ""
    controlsim_reasoning: str = ""
    controlsim_answer: str = ""
    controlsim_correct: bool = False
    controlsim_parse_err: bool = False
    controlsim_err_msg: str = ""


# results/model_seed/exp_id/res.csv
def _latest_run_dir(tb_root: str) -> Optional[str]:
    if not os.path.isdir(tb_root):
        return None
    runs = [d for d in os.listdir(tb_root) if os.path.isdir(os.path.join(tb_root, d))]
    if not runs:
        return None
    # sort by mtime newest first
    runs = sorted(runs, key=lambda d: os.path.getmtime(os.path.join(tb_root, d)), reverse=True)
    return runs[0]


def create_dir(args: Any, base: Path) -> str:  # root would be like ./results
    outdir: str = args.model.split("/")[1] + f"_seed{args.seed}"
    abs_outdir = os.path.join(base, "results", outdir)
    os.makedirs(abs_outdir, exist_ok=True)
    tb_root = os.path.join(abs_outdir, "tb")
    os.makedirs(tb_root, exist_ok=True)
    exp_id = getattr(args, "exp_id", None)
    if not exp_id and getattr(args, "resume", False):
        exp_id = _latest_run_dir(tb_root)
    if not exp_id:
        exp_id = time.strftime("run_%Y%m%d_%H%M%S")
    actual_logdir = os.path.join(tb_root, exp_id)
    os.makedirs(actual_logdir, exist_ok=True)
    return actual_logdir


def init_tensorboard(args: Any, exp_dir: str) -> SummaryWriter:  # use the logdir to specify tmp for testing, later switch to results dir
    return SummaryWriter(log_dir=exp_dir)


def make_request_id(kind: str, digit: int, idx: int, seed: int, model: str) -> str:
    payload = f"{model}::{seed}::{kind}::{digit}::{idx}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def tb_text(
    tag: str,
    title: str,
    body: str,
    tb: SummaryWriter,
    args: Any,
    step: int = 0,
) -> None:
    if tb is None:
        return
    body = body or ""
    n = args.tb_text_chars
    body_show = body if len(body) <= n else (body[: max(0, n - 3)] + "...")
    tb.add_text(tag, f"**{title}**\n\n```\n{body_show}\n```", global_step=step)


# tb for fine-grained, csv for coarse grained / direct loading into dataframes


def write_text_to_tensorboard(records: List[Record], tb: SummaryWriter, args: Any) -> None:
    write_tb = partial(tb_text, tb=tb, args=args)
    for i, record in enumerate(records):  # assumes ordered according to the questions
        base_nl = f"{record.model}/nl/d{record.digit}/{record.kind}/i{i}"
        base_code = f"{record.model}/code/d{record.digit}/{record.kind}/i{i}"
        base_sim = f"{record.model}/sim/d{record.digit}/{record.kind}/i{i}"
        base_controlsim = f"{record.model}/controlsim/d{record.digit}/{record.kind}/i{i}"

        write_tb(f"{base_nl}/question", "Original Question Data", str(record.question))
        write_tb(f"{base_nl}/answer", "Gold Answer", str(record.answer))

        write_tb(f"{base_nl}/nl_question", "NL Prompt", str(record.nl_question))
        write_tb(f"{base_nl}/nl_answer", "Final Answer (NL)", str(record.nl_answer))
        write_tb(f"{base_nl}/nl_correct", "Is the answer correct?", str(record.nl_correct))
        write_tb(f"{base_nl}/nl_parse_err", "Is there an error?", str(record.nl_parse_err))
        write_tb(f"{base_nl}/nl_err_msg", "The err message if there is one", str(record.nl_err_msg))

        write_tb(f"{base_code}/code_question", "code Prompt", str(record.code_question))
        write_tb(f"{base_code}/code_answer", "Final Answer (code)", str(record.code_answer))
        write_tb(f"{base_code}/code_correct", "Is the answer correct?", str(record.code_correct))
        write_tb(f"{base_code}/code_parse_err", "Is there a parse error?", str(record.code_parse_err))
        write_tb(f"{base_code}/code_gen_err", "Is there a generation / formatting error?", str(record.code_parse_err))
        write_tb(f"{base_code}/code_err_msg", "The err message if there is one", str(record.code_err_msg))

        write_tb(f"{base_sim}/sim_question", "sim Prompt", str(record.sim_question))
        write_tb(f"{base_sim}/sim_answer", "Final Answer (sim)", str(record.sim_answer))
        write_tb(f"{base_sim}/sim_correct", "Is the answer correct?", str(record.sim_correct))
        write_tb(f"{base_sim}/sim_parse_err", "Is there an error?", str(record.sim_parse_err))
        write_tb(f"{base_sim}/sim_err_msg", "The err message if there is one", str(record.sim_err_msg))

        write_tb(f"{base_controlsim}/controlsim_question", "controlsim Prompt", str(record.controlsim_question))
        write_tb(f"{base_controlsim}/controlsim_answer", "Final Answer (controlsim)", str(record.controlsim_answer))
        write_tb(f"{base_controlsim}/controlsim_correct", "Is the answer correct?", str(record.controlsim_correct))
        write_tb(f"{base_controlsim}/controlsim_parse_err", "Is there an error?", str(record.controlsim_parse_err))
        write_tb(f"{base_controlsim}/controlsim_err_msg", "The err message if there is one", str(record.controlsim_err_msg))
    tb.flush()
    tb.close()


def write_to_csv(logdir: str, records: List[Record]) -> None:
    result: List[Dict[str, Any]] = []
    for record in records:
        result.append(record.model_dump())
    df = pd.DataFrame(result)
    df.to_csv(logdir, index=False)


def read_from_csv(logdir: str) -> List[Record]:
    result: List[Record] = []
    if not os.path.exists(logdir):
        return result

    df = pd.read_csv(logdir)
    total_rows = len(df)
    dicts = df.to_dict("records")

    # Drop duplicate request_ids while keeping the last occurrence on disk.
    records_by_id: Dict[str, Record] = {}
    insertion_order: List[str] = []
    for idx, r in enumerate(reversed(dicts)):
        clean: Dict = {}
        for name, field in Record.model_fields.items():  # type: ignore[attr-defined]
            val = r.get(name, None)
            if pd.isna(val):
                if field.annotation is str:
                    val = ""
                elif field.annotation is bool:
                    val = False
                elif field.annotation is int:
                    val = -1
                else:
                    val = None
            if field.annotation is str and not isinstance(val, str):
                val = str(val)
            clean[name] = val
        rec = Record(**clean)  # type: ignore
        # Preserve existing request_id/index from disk; do not rewrite hashes here.
        key = rec.request_id if rec.request_id else f"__row_{idx}"
        if key not in records_by_id:
            records_by_id[key] = rec
            insertion_order.append(key)

    insertion_order.reverse()
    result = [records_by_id[rid] for rid in insertion_order]
    # Assign missing per-kind indices only for legacy rows that have request_ids.
    per_kind_counter: Dict[str, int] = {}
    for rec in result:
        per_kind_counter.setdefault(rec.kind, 0)
        if rec.index_in_kind is None or rec.index_in_kind <= 0:
            if rec.request_id:
                per_kind_counter[rec.kind] += 1
                rec.index_in_kind = per_kind_counter[rec.kind]
    dropped = total_rows - len(result)
    if dropped > 0:
        try:
            df_clean = pd.DataFrame([r.model_dump() for r in result])
            df_clean.to_csv(logdir, index=False)
            with open(logdir, "rb") as f:
                os.fsync(f.fileno())
            print(f"[checkpoint load] compacted checkpoint: kept {len(result)} unique rows " f"(dropped {dropped})")
        except Exception:
            print(f"[checkpoint load] detected {dropped} duplicate rows but failed to rewrite checkpoint")

    # Debug summaries after load (no rewriting of ids/indices beyond legacy index fill).
    seen: set[str] = set()
    dups: set[str] = set()
    per_kind_counts: dict[str, int] = {}
    uniq_per_kind: dict[str, set[str]] = {}
    for rec in result:
        per_kind_counts[rec.kind] = per_kind_counts.get(rec.kind, 0) + 1
        uniq_per_kind.setdefault(rec.kind, set()).add(rec.request_id)
        if rec.request_id in seen:
            dups.add(rec.request_id)
        seen.add(rec.request_id)
    if dups:
        print(f"[checkpoint load] duplicate request_ids detected after compaction: {dups}")
    if per_kind_counts:
        print(f"[checkpoint load] rows per kind: {per_kind_counts}")
        uniq_counts = {k: len(v) for k, v in uniq_per_kind.items()}
        print(f"[checkpoint load] unique ids per kind: {uniq_counts}")
        max_idx = {k: max([r.index_in_kind for r in result if r.kind == k] or [0]) for k in uniq_per_kind}
        print(f"[checkpoint load] max index_in_kind per kind: {max_idx}")
    return result


def walk_results_folder(csv_folder: str) -> List[str]:
    csv_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(csv_folder):
        for filename in filenames:
            if filename.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files


def create_big_df(csv_files: Sequence[Union[str, Path]]) -> pd.DataFrame:
    big_df: List[pd.DataFrame] = []
    for csv_file in csv_files:
        path_obj = Path(csv_file)
        df = pd.read_csv(path_obj)
        if "sim_err_msg" in df.columns:
            df["sim_parse_err"] = df["sim_err_msg"]
        big_df.append(df)
    return pd.concat(big_df, axis=0, ignore_index=True)


class CheckpointManager:
    """
    Handles incremental logging of records to CSV so runs can resume.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._records: Dict[str, Record] = {}
        self._defaults = Record().model_dump()
        self._pending: List[Record] = []
        if os.path.exists(csv_path):
            for idx, rec in enumerate(read_from_csv(csv_path)):
                key = rec.request_id if rec.request_id else f"__row_{idx}"
                self._records[key] = rec
        self._flushed_ids = set(self._records.keys())

    def _should_override(self, key: str, value: Any) -> bool:
        if isinstance(value, str):
            return value != ""
        if isinstance(value, bool):
            return True  # booleans are meaningful even if False
        if isinstance(value, int):
            return value != -1
        return value is not None

    def get(self, request_id: str) -> Optional[Record]:
        return self._records.get(request_id)

    def upsert(self, record: Record, flush: bool = True) -> None:
        key = record.request_id if record.request_id else f"__row_{len(self._records)}"
        existing = self._records.get(key)
        to_store = record
        if existing is None:
            self._records[key] = record
        else:
            merged = existing.model_dump()
            incoming = record.model_dump()
            for k, v in incoming.items():
                if self._should_override(k, v):
                    merged[k] = v
            to_store = Record(**merged)
            # If nothing changed, skip appending to avoid duplicate rows on resume.
            if merged == existing.model_dump():
                self._records[key] = existing
                if flush:
                    self.flush()
                return
            self._records[key] = to_store
        # Only append if this request_id has not been flushed yet.
        if key not in self._flushed_ids:
            self._pending.append(to_store)
        if flush:
            self.flush()

    def flush(self) -> None:
        if not self._records:
            return
        df = pd.DataFrame([r.model_dump() for r in self._records.values()])
        df.to_csv(self.csv_path, mode="w", index=False, header=True)
        try:
            with open(self.csv_path, "rb") as f:
                os.fsync(f.fileno())
        except Exception:
            pass
        self._flushed_ids = set(self._records.keys())
        self._pending = []

    def all_records(self) -> List[Record]:
        return list(self._records.values())

    def get_by_question(self, kind: str, digit: int, question: str) -> Optional[Record]:
        """
        Look up a record by kind/digit/question text, supporting legacy rows without index_in_kind.
        """
        for rec in self._records.values():
            if rec.kind == kind and rec.digit == digit and rec.question == question:
                return rec
        return None
