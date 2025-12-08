#!/usr/bin/env python3
"""
NL-CoT vs Code-CoT on arithmetic, DP (LCS/Knapsack/Rod Cutting), and ILP tasks.

New kinds:
- lcs           : LCS length of two strings
- knap          : 0/1 knapsack max value
- rod           : rod-cutting max revenue
- ilp_assign    : assignment min cost (n x n)
- ilp_prod      : production planning (max profit with resource caps)
- ilp_partition : 2-way partition minimal difference

ILPs use PuLP if available; otherwise safe brute-force fallbacks (small sizes).
Code-CoT subprocess is constrained but allows imports.

Usage examples:
  python cot_general.py --backend hf --model google/gemma-2-9b-it \
    --n 60 --digits 8 9 10 --kinds add sub mul lcs knap rod ilp_assign ilp_prod ilp_partition \
    --exec_code --outdir out_hf
  tensorboard --logdir out_hf/tb
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, cast

from simple_parsing import parse

from src.exps_performance.arms import Arm1, Arm2, Arm3, Arm4, BaseArm
from src.exps_performance.dataset import make_dataset
from src.exps_performance.llm import llm
from src.exps_performance.logger import (
    CheckpointManager,
    create_dir,
    init_tensorboard,
    make_request_id,
    write_text_to_tensorboard,
)
from src.exps_performance.metrics import accuracy
from src.exps_performance.problems import Question
from src.exps_performance.utils import seed_all_and_setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Using __name__ is a common practice


def run(args: Any) -> None:
    seed_all_and_setup(args)
    client = llm(args)
    exp_dir = create_dir(args, Path(args.root))
    exp_id = Path(exp_dir).name
    checkpoint_path = os.path.join(exp_dir, "res.csv")
    checkpoint = CheckpointManager(checkpoint_path)
    logger.info(f"Using exp_dir={exp_dir} (exp_id={exp_id}) for model={args.model} seed={args.seed}")
    logger.info(f"Restored {len(checkpoint._records)} unique records from checkpoint " f"({checkpoint_path})")
    if os.path.exists(checkpoint_path):
        try:
            import pandas as pd

            df_counts = pd.read_csv(checkpoint_path)
            logger.info(f"Checkpoint rows in file: {len(df_counts)}; " f"unique request_ids: {len(checkpoint._records)}")
        except Exception:
            logger.info("Could not read checkpoint CSV for counts.")
    # main workflow
    logger.info("Making Dataset")
    data = make_dataset(args.kinds, args.n, args.digits_list)  # choose the data

    def _done_sim(rec: Any) -> bool:
        return bool(rec.sim_question) or bool(rec.sim_answer) or bool(rec.sim_parse_err) or bool(rec.sim_err_msg)

    def _done_code(rec: Any) -> bool:
        return bool(rec.code_question) or bool(rec.code_answer) or bool(rec.code_parse_err) or bool(rec.code_err_msg)

    def _done_control(rec: Any) -> bool:
        return bool(rec.controlsim_question) or bool(rec.controlsim_answer) or bool(rec.controlsim_parse_err) or bool(rec.controlsim_err_msg)

    def _done_nl(rec: Any) -> bool:
        return bool(rec.nl_question) or bool(rec.nl_answer) or bool(rec.nl_parse_err) or bool(rec.nl_err_msg)

    def _fully_done(rec: Any) -> bool:
        return _done_sim(rec) and _done_code(rec) and _done_control(rec) and _done_nl(rec)

    # Populate identifiers and restore from checkpoint using stable per-kind indexing (digit, original_pos)
    per_kind_counter: dict[str, int] = {}
    restored_idx = 0
    restored_by_kind: dict[str, int] = {}
    # Build stable index map per kind
    kind_groups: dict[str, list[Question]] = {}
    for q in data:
        kind_groups.setdefault(q.kind, []).append(q)
    for k, qs in kind_groups.items():
        qs_sorted = sorted(qs, key=lambda x: (x.digits, getattr(x, "original_pos", 0)))
        for idx, q in enumerate(qs_sorted, start=1):
            setattr(q, "_stable_index_in_kind", idx)
    # Enforce per-kind cap (args.n) to guarantee exactly n indices per kind at most.
    capped_data: list[Question] = []
    dropped_by_kind: dict[str, int] = {}
    for q in data:
        idx_in_kind = getattr(q, "_stable_index_in_kind", 0)
        if idx_in_kind <= args.n:
            capped_data.append(q)
        else:
            dropped_by_kind[q.kind] = dropped_by_kind.get(q.kind, 0) + 1
    if dropped_by_kind:
        logger.warning(f"Dropped items exceeding n per kind: {dropped_by_kind}")
    data = capped_data
    # Debug: show first few assigned IDs and track consistency against checkpoint.
    debug_assigned: list[tuple[str, str, int, int, int]] = []
    for q in data:
        idx_in_kind = getattr(q, "_stable_index_in_kind", 0)
        per_kind_counter[q.kind] = idx_in_kind
        q.record.model = args.model
        q.record.seed = args.seed
        q.record.exp_id = exp_id
        q.record.kind = q.kind
        q.record.digit = q.digits
        q.record.index_in_kind = idx_in_kind
        q.record.request_id = make_request_id(q.kind, q.digits, idx_in_kind, args.seed, args.model)
        existing = checkpoint.get(q.record.request_id)
        if existing:
            restored_by_kind[q.kind] = restored_by_kind.get(q.kind, 0) + 1
            restored_idx += 1  # request_id match implies index match
            q.record = existing
            if existing.sim_code:
                q.code = existing.sim_code
        if len(debug_assigned) < 10:
            debug_assigned.append((q.record.request_id, q.kind, q.digits, idx_in_kind, getattr(q, "original_pos", -1)))
    rows_in_file = len(checkpoint._records)
    unique_req_ids = len(set(checkpoint._records.keys()))
    pending_expected = len(data) - rows_in_file
    logger.info(f"Sample assigned IDs (first 10): {debug_assigned}")
    logger.info(f"Restore summary: total={len(data)}, restored_by_index={restored_idx}, " f"pending={len(data) - restored_idx}")
    logger.info(
        f"Checkpoint consistency: rows={rows_in_file}, unique_request_ids={unique_req_ids}, "
        f"restored_by_index={restored_idx}, pending_expected={pending_expected}"
    )
    if rows_in_file != unique_req_ids or unique_req_ids != restored_idx:
        logger.warning("Mismatch detected: rows, unique request IDs, and restored_by_index differ; " "dedupe/checkpoint compaction recommended.")
    logger.info(f"Restored per kind: {restored_by_kind}")
    # Debug: detect duplicate request_ids assigned in this dataset build
    seen_ids = set()
    dup_ids = []
    for q in data:
        if q.record.request_id in seen_ids:
            dup_ids.append(q.record.request_id)
        seen_ids.add(q.record.request_id)
    if dup_ids:
        logger.warning(f"Duplicate request_ids assigned in dataset build: {set(dup_ids)}")

    def run_stage(pending: List[Question], ArmCls: type[BaseArm], stage_name: str) -> List[Question]:
        if not pending:
            return pending
        logger.info(f"Running {stage_name} for {len(pending)} questions")
        chunk_size = max(1, args.checkpoint_every)
        updated_all = []
        for start in range(0, len(pending), chunk_size):
            batch = pending[start : start + chunk_size]
            arm = ArmCls(batch, args, client)
            _, updated = arm.run()
            for q in updated:
                checkpoint.upsert(q.record, flush=False)
            checkpoint.flush()  # flush every batch
            updated_all.extend(updated)
        return updated_all

    # Arm2 (sim/code generation)
    arm2_pending = [q for q in data if not _fully_done(q.record) and not _done_sim(q.record)]
    logger.info(f"Arm2 pending {len(arm2_pending)} / total {len(data)}")
    updated_arm2 = run_stage(arm2_pending, Arm2, "Arm2")
    # propagate code back to main list
    if updated_arm2:
        updated_map = {q.record.request_id: q for q in updated_arm2}
        for i, q in enumerate(data):
            if q.record.request_id in updated_map:
                data[i] = updated_map[q.record.request_id]
                data[i].code = data[i].record.sim_code or data[i].code

    # Arm3 (code execution)
    arm3_pending = [q for q in data if not _fully_done(q.record) and not _done_code(q.record)]
    logger.info(f"Arm3 pending {len(arm3_pending)} / total {len(data)}")
    updated_arm3 = run_stage(arm3_pending, Arm3, "Arm3")
    if updated_arm3:
        updated_map = {q.record.request_id: q for q in updated_arm3}
        for i, q in enumerate(data):
            if q.record.request_id in updated_map:
                data[i] = updated_map[q.record.request_id]

    # Arm4 (control simulation)
    arm4_pending = [q for q in data if not _fully_done(q.record) and not _done_control(q.record)]
    logger.info(f"Arm4 pending {len(arm4_pending)} / total {len(data)}")
    updated_arm4 = run_stage(arm4_pending, Arm4, "Arm4")
    if updated_arm4:
        updated_map = {q.record.request_id: q for q in updated_arm4}
        for i, q in enumerate(data):
            if q.record.request_id in updated_map:
                data[i] = updated_map[q.record.request_id]

    # Arm1 (nl)
    arm1_pending = [q for q in data if not _fully_done(q.record) and not _done_nl(q.record)]
    logger.info(f"Arm1 pending {len(arm1_pending)} / total {len(data)}")
    updated_arm1 = run_stage(arm1_pending, Arm1, "Arm1")
    if updated_arm1:
        updated_map = {q.record.request_id: q for q in updated_arm1}
        for i, q in enumerate(data):
            if q.record.request_id in updated_map:
                data[i] = updated_map[q.record.request_id]
    completed = data

    logger.info("Saving Results")
    records = [d.record for d in completed]
    df = accuracy(records)
    # summarize here
    for kind in df.values:
        for val in kind:
            continue

    # serialize results
    writer = init_tensorboard(args, exp_dir)
    write_text_to_tensorboard(records, writer, args)
    checkpoint.flush()


@dataclass
class Args:
    root: str = "."
    n: int = 1
    digits_list: List[int] = field(default_factory=lambda: [2, 4, 8])
    kinds: List[str] = field(
        default_factory=lambda: [
            # fine-grained arithmetic / DP / ILP
            "add",
            "sub",
            "mul",
            "lcs",
            "knap",
            "rod",
            "ilp_assign",
            "ilp_prod",
            "ilp_partition",
            # CLRS
            "clrs",
            # GSM8K
            "gsm8k",
            # NP-hard suite
            "spp",
            "tsp",
            "tsp_d",
            "msp",
            "ksp",
            "gcp",
            "gcp_d",
            "bsp",
            "edp",
        ],
        metadata={
            "choices": [
                # fine-grained
                "add",
                "sub",
                "mul",
                "lcs",
                "knap",
                "rod",
                "ilp_assign",
                "ilp_prod",
                "ilp_partition",
                # CLRS
                "clrs30",
                # GSM8K
                "gsm8k",
                # NP-hard
                "spp",
                "tsp",
                "tspd",
                "msp",
                "ksp",
                "gcp",
                "gcpd",
                "bsp",
                "edp",
            ]
        },
    )
    seed: int = 1
    backend: str = field(
        default="dummy",
        metadata={
            "choices": ["dummy", "openai", "openrouter", "running", "vllm"],
            "help": "LLM backend to use. 'running' hits an already-running OpenAI-compatible server; 'openrouter' uses https://openrouter.ai via the OpenAI SDK.",
        },
    )
    model: str = "gpt-4o"
    hf_dtype: str = field(default="bfloat16", metadata={"choices": ["auto", "float16", "bfloat16", "float32"]})
    hf_device_map: str = "auto"
    hf_trust_remote_code: bool = False
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 4192
    temperature: float = 0.1
    top_p: float = 0.90
    sim_code_only: bool = False
    exec_code: bool = False
    exec_workers: int = 4
    controlled_sim: bool = False
    log_every: int = 50
    checkpoint_every: int = 1
    tb_text_chars: int = 10000
    tb_disable: bool = False
    exp_id: Optional[str] = None
    resume: bool = False
    batch_size: int = 8
    vllm_dtype: str = field(default="float32", metadata={"choices": ["auto", "float16", "bfloat16", "float32"]})
    vllm_tensor_parallel: int = 8
    vllm_gpu_mem_util: float = 0.90
    vllm_max_model_len: int = 8192
    vllm_download_dir: str = "/nlpgpu/data/terry/ToolProj/src/models"


def parse_args() -> Args:
    return cast(Args, parse(Args))


if __name__ == "__main__":
    start_time = time.perf_counter()
    args = parse_args()
    run(args)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")
