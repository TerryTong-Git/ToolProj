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

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, cast

from simple_parsing import parse

from src.exps_performance.arms import Arm1, Arm2, Arm3, Arm4, BaseArm
from src.exps_performance.dataset import make_dataset
from src.exps_performance.llm import llm
from src.exps_performance.logger import (
    CheckpointManager,
    create_dir,
    generate_unique_tag,
    make_request_id,
)
from src.exps_performance.metrics import accuracy
from src.exps_performance.problems import Question
from src.exps_performance.utils import seed_all_and_setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Using __name__ is a common practice


def dump_args(args: Args, output_path: Path) -> None:
    """
    Serialize parsed arguments to JSON for reproducibility and debugging.
    """
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(args), f, indent=2, sort_keys=True)
    except Exception as exc:
        logger.warning(f"Failed to write args to {output_path}: {exc}")


def resolve_sample_count(arg_value: int, env_var: str) -> int:
    env_val = os.getenv(env_var)
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            logger.warning(f"Could not parse {env_var}={env_val} as int; falling back to CLI/default value.")
    return arg_value


def compute_effective_samples(
    checkpoint: CheckpointManager,
    model: str,
    seed: int,
    target_gsm_samples: int,
    target_clrs_samples: int,
) -> tuple[int, int, dict[str, int]]:
    """
    Ensure we do not drop previously generated samples when targets increase.
    Returns (gsm_samples, clrs_samples, existing_counts_by_kind).
    """
    existing_counts: dict[str, int] = {}
    for rec in checkpoint.all_records():
        if rec.model != model or rec.seed != seed:
            continue
        existing_counts[rec.kind] = existing_counts.get(rec.kind, 0) + 1

    existing_gsm = existing_counts.get("gsm8k", 0)
    existing_clrs = sum(count for kind, count in existing_counts.items() if kind.startswith("clrs"))

    gsm_samples = max(target_gsm_samples, existing_gsm)
    clrs_samples = max(target_clrs_samples, existing_clrs)
    return gsm_samples, clrs_samples, existing_counts


def assign_sequential_indices(
    questions: List[Question],
    n: int,
    seed: int,
    model: str,
    exp_id: str,
    checkpoint: CheckpointManager,
    per_kind_limits: Optional[dict[str, int]] = None,
) -> tuple[list[Question], dict[str, int], dict[str, int], list[tuple[str, str, int, int, int]]]:
    """
    Assign stable per-kind indices ordered by digit then original position.
    Returns (assigned_questions, dropped_by_kind, restored_by_kind, debug_samples).
    """
    kind_groups: dict[str, list[Question]] = {}
    for q_item in questions:
        kind_groups.setdefault(q_item.kind, []).append(q_item)

    assigned: list[Question] = []
    dropped_by_kind: dict[str, int] = {}
    restored_by_kind: dict[str, int] = {}
    debug_assigned: list[tuple[str, str, int, int, int]] = []

    per_kind_limits = per_kind_limits or {}

    for kind, qs in kind_groups.items():
        qs_sorted = sorted(qs, key=lambda x: (x.digits, getattr(x, "original_pos", 0)))
        limit = per_kind_limits.get(kind, n)
        for idx, q_item in enumerate(qs_sorted, start=1):
            if idx > limit:
                dropped_by_kind[kind] = dropped_by_kind.get(kind, 0) + 1
                continue
            rec = q_item.record
            rec.model = model
            rec.seed = seed
            rec.exp_id = exp_id
            rec.kind = q_item.kind
            rec.digit = q_item.digits
            rec.index_in_kind = idx
            rec.unique_tag = rec.unique_tag or generate_unique_tag(q_item.kind, q_item.digits, idx, seed, model)
            rec.request_id = rec.request_id or make_request_id(q_item.kind, q_item.digits, idx, seed, model)
            existing = checkpoint.get(rec.unique_tag) or checkpoint.get(rec.request_id)
            if existing:
                restored_by_kind[kind] = restored_by_kind.get(kind, 0) + 1
                if not existing.unique_tag:
                    existing.unique_tag = rec.unique_tag
                rec = existing
                if existing.sim_code:
                    q_item.code = existing.sim_code
            q_item.record = rec
            assigned.append(q_item)
            if len(debug_assigned) < 10:
                debug_assigned.append((rec.unique_tag or rec.request_id, q_item.kind, q_item.digits, idx, getattr(q_item, "original_pos", -1)))
    return assigned, dropped_by_kind, restored_by_kind, debug_assigned


def _stage_complete(stage_name: str, rec: Any) -> bool:
    """
    Conservative completion checks per stage; missing required fields means the stage failed.
    """
    if stage_name == "Arm2":  # sim generation
        return bool(rec.sim_question) and (bool(rec.sim_answer) or bool(rec.sim_err_msg) or bool(rec.sim_parse_err) or bool(rec.sim_correct))
    if stage_name == "Arm3":  # code execution
        return bool(rec.code_question) and (bool(rec.code_answer) or bool(rec.code_err_msg) or bool(rec.code_parse_err) or bool(rec.code_gen_err))
    if stage_name == "Arm4":  # control sim
        return bool(rec.controlsim_question) and (
            bool(rec.controlsim_answer) or bool(rec.controlsim_err_msg) or bool(rec.controlsim_parse_err) or bool(rec.controlsim_correct)
        )
    if stage_name == "Arm1":  # natural language reasoning
        return bool(rec.nl_question) and (bool(rec.nl_answer) or bool(rec.nl_err_msg) or bool(rec.nl_parse_err) or bool(rec.nl_correct))
    return True


def run_stage_batch(
    pending: List[Question],
    ArmCls: type[BaseArm],
    stage_name: str,
    args: Any,
    client: Any,
    checkpoint: CheckpointManager,
) -> List[Question]:
    """
    Run a single stage with batch-level completeness checks. Batches that fail completeness are not checkpointed.
    """
    if not pending:
        return pending
    logger.info(f"Running {stage_name} for {len(pending)} questions")
    chunk_size = max(1, args.checkpoint_every)
    updated_all: List[Question] = []
    for start in range(0, len(pending), chunk_size):
        batch = pending[start : start + chunk_size]
        arm = ArmCls(batch, args, client)
        _, updated = arm.run()
        batch_complete = all(_stage_complete(stage_name, q.record) for q in updated)
        if batch_complete:
            checkpoint.save_batch([q.record for q in updated], flush=True)
            updated_all.extend(updated)
        else:
            logger.warning(f"[checkpoint guard] Skipping checkpoint for {stage_name} batch starting at {start}: {len(batch)} items incomplete")
    return updated_all


def run(args: Any) -> None:
    seed_all_and_setup(args)
    client = llm(args)
    exp_dir = create_dir(args, Path(args.root))
    exp_id = Path(exp_dir).name
    checkpoint_path = os.path.join(exp_dir, "res.jsonl")
    checkpoint = CheckpointManager(checkpoint_path)
    dump_args(args, Path(exp_dir) / "args.json")
    logger.info(f"Using exp_dir={exp_dir} (exp_id={exp_id}) for model={args.model} seed={args.seed}")
    logger.info(f"Restored {len(checkpoint._records)} unique records from checkpoint ({checkpoint_path})")
    if os.path.exists(checkpoint_path):
        try:
            import pandas as pd

            df_counts = pd.read_json(checkpoint_path, lines=True)
            logger.info(f"Checkpoint rows in file: {len(df_counts)}; unique request_ids: {len(checkpoint._records)}")
        except Exception:
            logger.info("Could not read checkpoint JSONL for counts.")
    # main workflow
    logger.info("Making Dataset")
    target_gsm_samples = resolve_sample_count(args.gsm_samples, "GSM8K_SAMPLES")
    target_clrs_samples = resolve_sample_count(args.clrs_samples, "CLRS30_SAMPLES")
    gsm_samples, clrs_samples, existing_counts = compute_effective_samples(checkpoint, args.model, args.seed, target_gsm_samples, target_clrs_samples)
    logger.info(f"Existing samples by kind for model={args.model}, seed={args.seed}: {existing_counts}")
    if gsm_samples != target_gsm_samples or clrs_samples != target_clrs_samples:
        logger.info(f"Adjusted sample caps to preserve prior runs: gsm8k={gsm_samples}, clrs={clrs_samples}")
    data = make_dataset(args.kinds, args.n, args.digits_list, gsm_samples=gsm_samples, clrs_samples=clrs_samples)  # choose the data
    per_kind_limits: dict[str, int] = {}
    if "gsm8k" in args.kinds:
        per_kind_limits["gsm8k"] = gsm_samples
    ClrsQuestionType: Optional[type[Question]] = None
    try:
        from src.exps_performance.problems.clrs import ClrsQuestion as ImportedClrsQuestion

        ClrsQuestionType = ImportedClrsQuestion
    except Exception:
        ClrsQuestionType = None

    if ClrsQuestionType is not None:
        for q in data:
            if isinstance(q, ClrsQuestionType):
                per_kind_limits[q.kind] = clrs_samples

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
    data, dropped_by_kind, restored_by_kind, debug_assigned = assign_sequential_indices(
        list(data), args.n, args.seed, args.model, exp_id, checkpoint, per_kind_limits=per_kind_limits
    )
    if dropped_by_kind:
        logger.warning(f"Dropped items exceeding n per kind: {dropped_by_kind}")

    rows_in_file = len(checkpoint._records)
    total_by_kind: dict[str, int] = {}
    for q in data:
        total_by_kind[q.kind] = total_by_kind.get(q.kind, 0) + 1
    # Pending should reflect only the current dataset, not any out-of-scope checkpoint rows.
    pending_by_kind = {k: max(total_by_kind.get(k, 0) - restored_by_kind.get(k, 0), 0) for k in args.kinds}

    logger.info(f"Restore summary: total={len(data)}, restored_by_kind={restored_by_kind}, pending_by_kind={pending_by_kind}")
    logger.info(f"Checkpoint rows restored into memory: {rows_in_file}")
    # Debug: detect duplicate unique tags assigned in this dataset build
    seen_ids = set()
    dup_ids = []
    for q in data:
        tag = q.record.unique_tag or q.record.request_id
        if tag in seen_ids:
            dup_ids.append(tag)
        seen_ids.add(tag)
    if dup_ids:
        logger.warning(f"Duplicate identifiers assigned in dataset build: {set(dup_ids)}")

    # Arm2 (sim/code generation)
    arm2_pending = [q for q in data if not _fully_done(q.record) and not _done_sim(q.record)]
    logger.info(f"Arm2 pending {len(arm2_pending)} / total {len(data)}")
    updated_arm2 = run_stage_batch(arm2_pending, Arm2, "Arm2", args, client, checkpoint)
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
    updated_arm3 = run_stage_batch(arm3_pending, Arm3, "Arm3", args, client, checkpoint)
    if updated_arm3:
        updated_map = {q.record.request_id: q for q in updated_arm3}
        for i, q in enumerate(data):
            if q.record.request_id in updated_map:
                data[i] = updated_map[q.record.request_id]

    # Arm4 (control simulation)
    arm4_pending = [q for q in data if not _fully_done(q.record) and not _done_control(q.record)]
    logger.info(f"Arm4 pending {len(arm4_pending)} / total {len(data)}")
    updated_arm4 = run_stage_batch(arm4_pending, Arm4, "Arm4", args, client, checkpoint)
    if updated_arm4:
        updated_map = {q.record.request_id: q for q in updated_arm4}
        for i, q in enumerate(data):
            if q.record.request_id in updated_map:
                data[i] = updated_map[q.record.request_id]

    # Arm1 (nl)
    arm1_pending = [q for q in data if not _fully_done(q.record) and not _done_nl(q.record)]
    logger.info(f"Arm1 pending {len(arm1_pending)} / total {len(data)}")
    updated_arm1 = run_stage_batch(arm1_pending, Arm1, "Arm1", args, client, checkpoint)
    if updated_arm1:
        updated_map = {q.record.request_id: q for q in updated_arm1}
        for i, q in enumerate(data):
            if q.record.request_id in updated_map:
                data[i] = updated_map[q.record.request_id]
    completed = data

    logger.info("Saving Results")
    records = [d.record for d in completed]
    _ = accuracy(records)  # Compute accuracy (result logged internally)

    # serialize results
    # writer = init_tensorboard(args, exp_dir)
    # write_text_to_tensorboard(records, writer, args)
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
                "tsp_d",
                "msp",
                "ksp",
                "gcp",
                "gcp_d",
                "bsp",
                "edp",
            ]
        },
    )
    gsm_samples: int = field(default=500, metadata={"help": "Samples to use for GSM8K (override env: GSM8K_SAMPLES)."})
    clrs_samples: int = field(default=500, metadata={"help": "Samples to use for CLRS30 (override env: CLRS30_SAMPLES)."})
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
    vllm_download_dir: str = os.getenv("VLLM_DOWNLOAD_DIR", str(Path(__file__).resolve().parent / "models"))


def parse_args() -> Args:
    return cast(Args, parse(Args))


if __name__ == "__main__":
    start_time = time.perf_counter()
    args = parse_args()

    run(args)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")
