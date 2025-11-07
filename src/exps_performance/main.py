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

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from code_exec import exec_from_rationale
from dataset import make_dataset
from llm import DummyClient, HFLocalClient, LLMClient, OpenAIChatClient, VLLMClient
from parsing import parse_response
from prompts import CODE_PROMPT, NL_PROMPT, SIM_PROMPT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ------------------------------- Utilities ----------------------------------


@dataclass
class Record:
    idx: int
    problem: str
    kind: str
    digits: int
    truth: int
    answer_nl: Optional[int]
    correct_nl: int
    answer_code: Optional[int]
    correct_code: int
    answer_code_exec: Optional[int]
    correct_code_exec: int
    raw_nl: str
    raw_code: str
    raw_sim: str
    answer_sim: Optional[int]
    correct_sim_ans: int

    exec_ok: Optional[int] = None
    exec_retcode: Optional[int] = None
    exec_timeout: Optional[int] = None
    exec_stdout: Optional[str] = None
    exec_stderr: Optional[str] = None


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    def binom_cdf_leq(n, k):
        s = 0.0
        for i in range(0, k + 1):
            s += math.comb(n, i)
        return s * (0.5**n)

    p = 2.0 * binom_cdf_leq(n, k)
    return min(1.0, p)


# ------------------------------- Runner -------------------------------------


def run(args):
    # backend
    random.seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.backend == "dummy":
        client: LLMClient = DummyClient()
    elif args.backend == "openai":
        client = OpenAIChatClient(seed=args.seed)
    elif args.backend == "hf":
        client = HFLocalClient(
            model_name=args.model,
            dtype=args.hf_dtype,
            device_map=args.hf_device_map,
            trust_remote_code=args.hf_trust_remote_code,
        )
    elif args.backend == "vllm":
        print("Instantiating VLLM")
        client = VLLMClient(
            model_name=args.model,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.vllm_tensor_parallel,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            max_model_len=args.vllm_max_model_len,
            download_dir=args.vllm_download_dir,
            trust_remote_code=args.hf_trust_remote_code,
            seed=args.seed,
        )
    else:
        raise ValueError("backend must be one of {dummy, openai, hf}")

    problems = make_dataset(args.n, args.digits, args.kinds, seed=args.seed)

    # TensorBoard
    outdir: str = args.model.split("/")[1]
    if args.kinds[0] == "gsm8k":
        outdir += "_gsm8k"

    os.makedirs(outdir, exist_ok=True)
    exp_id = time.strftime("run_%Y%m%d_%H%M%S")
    tb = None if args.tb_disable else SummaryWriter(log_dir=os.path.join(outdir, "tb", exp_id))

    def tb_text(tag: str, title: str, body: str, step: int = 0):
        if tb is None:
            return
        body = body or ""
        n = args.tb_text_chars
        body_show = body if len(body) <= n else (body[: max(0, n - 3)] + "...")
        tb.add_text(tag, f"**{title}**\n\n```\n{body_show}\n```", global_step=step)

    nl_msgs = [[{"role": "user", "content": NL_PROMPT.format(problem=pb.text())}] for pb in problems]
    code_msgs = [[{"role": "user", "content": CODE_PROMPT.format(problem=pb.text())}] for pb in problems]

    def run_batch(messages_list):
        if hasattr(client, "chat_many") and callable(getattr(client, "chat_many")) and args.batch_size > 1:
            outs = []
            for start in tqdm(range(0, len(messages_list), args.batch_size), desc="Chatting"):
                chunk = messages_list[start : start + args.batch_size]
                outs.extend(
                    client.chat_many(
                        args.model,
                        chunk,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        stop=None,
                    )
                )
            return outs
        else:
            return [client.chat(args.model, m, max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, stop=None) for m in tqdm(messages_list)]

    # === Generate all NL outputs, then all Code outputs (order preserved) ===
    nl_raw_all = run_batch(nl_msgs)
    code_raw_all = run_batch(code_msgs)

    records: List[Record] = []
    for i, pb in enumerate(tqdm(problems, total=len(problems), desc="eval")):
        problem_text = pb.text()
        truth = pb.ground_truth()

        nl_raw = nl_raw_all[i]
        nl_parsed = parse_response(nl_raw)
        ans_nl = nl_parsed.answer
        correct_nl = int(ans_nl == truth)

        base_nl = f"{args.model}/nl/d{pb.digits}/{pb.kind}/i{i}"
        tb_text(f"{base_nl}/prompt", "Prompt (NL-CoT)", NL_PROMPT.format(problem=problem_text))
        tb_text(f"{base_nl}/rationale", "NL Rationale", nl_parsed.rationale or "")
        tb_text(f"{base_nl}/raw_json", "Raw NL JSON", nl_parsed.raw)
        tb_text(f"{base_nl}/answer", "Final Answer (NL)", "" if ans_nl is None else str(ans_nl))

        code_raw = code_raw_all[i]
        code_parsed = parse_response(code_raw)
        ans_code = code_parsed.answer
        correct_code = int(ans_code == truth)

        base_code = f"{args.model}/code/d{pb.digits}/{pb.kind}/i{i}"
        tb_text(f"{base_code}/prompt", "Prompt (Code-CoT)", CODE_PROMPT.format(problem=problem_text))
        tb_text(f"{base_code}/rationale", "Code Rationale (fenced)", code_parsed.rationale or "")
        tb_text(f"{base_code}/raw_json", "Raw Code JSON", code_parsed.raw)
        tb_text(f"{base_code}/answer", "Final Answer (Code)", "" if ans_code is None else str(ans_code))

        ans_code_exec = {
            "ok": False,
            "value": None,
            "stdout": "",
            "stderr": "",
            "retcode": None,
            "timeout": False,
        }

        if args.exec_code:
            ans_code_exec = exec_from_rationale(code_parsed.rationale, allow_imports=True)
            tb_text(
                f"{base_code}/exec_answer",
                "Executed Answer (subprocess)",
                "" if ans_code_exec is None else str(ans_code_exec),
            )
            tb_text(f"{base_code}/exec_stdout", "Executed STDOUT", ans_code_exec.get("stdout", ""))
            tb_text(f"{base_code}/exec_stderr", "Executed STDERR", ans_code_exec.get("stderr", ""))
            tb_text(
                f"{base_code}/exec_meta",
                "Exec Meta",
                f"retcode={ans_code_exec.get('retcode')} timeout={ans_code_exec.get('timeout')} \
                ok={ans_code_exec.get('ok')} value={ans_code_exec.get('value')}",
            )
        correct_code_exec = int(ans_code_exec.get("value") == truth) if ans_code_exec is not None else 0
        correct_sim_ans = 0
        answer_sim = 0
        sim_raw = ""
        if args.controlled_sim:
            message = [{"role": "user", "content": SIM_PROMPT.format(problem=code_parsed.rationale)}]
            llm_output = client.chat(args.model, message, max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, stop=None)
            parsed_output = parse_response(llm_output)
            # import pdb; pdb.set_trace()
            answer_sim = parsed_output.answer
            sim_raw = parsed_output.raw
            correct_sim_ans = int(answer_sim == truth)

        records.append(
            Record(
                idx=i,
                problem=problem_text,
                kind=pb.kind,
                digits=pb.digits,
                truth=truth,
                answer_nl=ans_nl,
                correct_nl=correct_nl,
                answer_code=ans_code,
                correct_code=correct_code,
                answer_code_exec=ans_code_exec,
                correct_code_exec=correct_code_exec,
                raw_nl=nl_raw,
                raw_code=code_raw,
                raw_sim=sim_raw,
                answer_sim=answer_sim,
                correct_sim_ans=correct_sim_ans,
            )
        )

    # CSV
    import csv

    csv_path = os.path.join(outdir, f"{exp_id}_results_seed_{args.seed}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "kind",
                "digits",
                "truth",
                "answer_nl",
                "correct_nl",
                "answer_code",
                "correct_code",
                "answer_code_exec",
                "correct_code_exec",
                "answer_sim",
                "correct_sim",
                "problem",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.idx,
                    r.kind,
                    r.digits,
                    r.truth,
                    r.answer_nl,
                    r.correct_nl,
                    r.answer_code,
                    r.correct_code,
                    r.answer_code_exec,
                    r.correct_code_exec,
                    r.answer_sim,
                    r.correct_sim_ans,
                    r.problem,
                ]
            )

    # Summary (overall + per kind)
    def acc(xs):
        return sum(xs) / max(1, len(xs))

    acc_nl = acc([r.correct_nl for r in records])
    acc_code = acc([r.correct_code for r in records])
    has_exec = any(r.answer_code_exec is not None for r in records)
    acc_exec = acc([r.correct_code_exec for r in records if r.answer_code_exec is not None]) if has_exec else float("nan")
    acc_sim = acc([r.correct_sim_ans for r in records])

    # b = sum(1 for r in records if r.correct_code == 1 and r.correct_nl == 0)
    # c = sum(1 for r in records if r.correct_code == 0 and r.correct_nl == 1)
    # p_mc = mcnemar_exact_p(b, c)

    by_kind: Dict[str, List[Record]] = {}
    for r in records:
        by_kind.setdefault(r.kind, []).append(r)

    lines: List[str] = []
    lines.append(f"N={len(records)}  exp_id={exp_id}")
    lines.append(f"Accuracy NL-CoT (overall):   {acc_nl:.4f}")
    lines.append(f"Accuracy Code-CoT (overall): {acc_code:.4f}")
    if has_exec:
        lines.append(f"Execution condition (subprocess): acc={acc_exec:.4f}")
    lines.append(f"Accuracy Sim (overall): {acc_sim:.4f}")

    # lines.append(f"Discordant pairs b=code>nl: {b}, c=nl>code: {c}, McNemar exact p={p_mc:.4g}")
    lines.append("Per-kind bins:")
    # --- Per-kind × digit breakdown (printed) ---
    by_kd: Dict[Tuple[str, int], List[Record]] = {}
    for r in records:
        by_kd.setdefault((r.kind, r.digits), []).append(r)

    def _acc(lst):
        return sum(lst) / max(1, len(lst))

    lines.append("")
    lines.append("Per-kind × digit bins:")
    for kind in sorted({k for (k, _) in by_kd.keys()}):
        lines.append(f"  kind={kind}")
        for d in sorted({d for (k, d) in by_kd.keys() if k == kind}):
            grp = by_kd[(kind, d)]
            N = len(grp)
            acc_nl = _acc([x.correct_nl for x in grp])
            acc_code = _acc([x.correct_code for x in grp])
            acc_sim = _acc([x.correct_sim_ans for x in grp])

            # Exec accuracy only for items that actually executed (and if exec requested)
            if args.exec_code:
                exec_vals = [x.correct_code_exec for x in grp if x.answer_code_exec is not None]
                acc_exec = (sum(exec_vals) / len(exec_vals)) if exec_vals else float("nan")
            else:
                acc_exec = float("nan")

            lines.append(f"    digits={d:2d}: N={N:3d}  NL={acc_nl:.4f}  Code={acc_code:.4f}  Exec={acc_exec:.4f} Sim={acc_sim}")

            # Optional: log to TensorBoard as kind/digit tags
            if tb is not None:
                tb.add_scalar(f"{args.model}/acc_nl/{kind}/d{d}", acc_nl)
                tb.add_scalar(f"{args.model}/acc_code/{kind}/d{d}", acc_code)
                tb.add_scalar(f"{args.model}/acc_sim/{kind}/d{d}", acc_sim)
                if args.exec_code and not math.isnan(acc_exec):
                    tb.add_scalar(f"{args.model}/acc_exec/{kind}/d{d}", acc_exec)
    csv_kd_path = os.path.join(outdir, f"{exp_id}_by_kind_digit.csv")
    with open(csv_kd_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "digits", "N", "acc_nl", "acc_code", "acc_exec", "acc_sim"])
        for kind in sorted({k for (k, _) in by_kd.keys()}):
            for d in sorted({d for (k, d) in by_kd.keys() if k == kind}):
                grp = by_kd[(kind, d)]
                N = len(grp)
                acc_nl = sum(x.correct_nl for x in grp) / N if N else float("nan")
                acc_code = sum(x.correct_code for x in grp) / N if N else float("nan")
                acc_sim = sum(x.correct_sim_ans for x in grp) / N if N else float("nan")

                if args.exec_code:
                    exec_vals = [x.correct_code_exec for x in grp if x.answer_code_exec is not None]
                    acc_exec = (sum(exec_vals) / len(exec_vals)) if exec_vals else float("nan")
                else:
                    acc_exec = float("nan")
                w.writerow([kind, d, N, f"{acc_nl:.6f}", f"{acc_code:.6f}", "" if math.isnan(acc_exec) else f"{acc_exec:.6f}", f"{acc_sim:.6f}"])

    summary_path = os.path.join(outdir, f"{exp_id}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote: {csv_path}\nWrote: {summary_path}")
    if tb is not None:
        tb.flush()
        tb.close()


# ------------------------------- CLI ----------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="total problems (balanced over kinds)")
    p.add_argument(
        "--digits",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Global hardness levels. For arithmetic: number magnitude. "
        "LCS: string length; knap: #items; rod: rod length; "
        "ilp_assign: n×n size; ilp_prod: scales products/resources/bounds; "
        "ilp_partition: #items.",
    )
    p.add_argument(
        "--kinds",
        type=str,
        nargs="+",
        default=[
            "add",
            "sub",
            "mul",
            "lcs",
            "knap",
            "rod",
            "ilp_assign",
            "ilp_prod",
            "ilp_partition",
        ],
        choices=[
            "add",
            "sub",
            "mul",
            "mix",
            "lcs",
            "knap",
            "rod",
            "ilp_assign",
            "ilp_prod",
            "ilp_partition",
            "gsm8k",
            "nphardeval",
            "clrs30",
        ],
    )
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--backend", type=str, default="dummy", choices=["dummy", "openai", "hf", "vllm"])
    p.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name or HF repo/path when --backend=hf",
    )
    p.add_argument(
        "--hf_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    p.add_argument("--hf_device_map", type=str, default="auto")
    p.add_argument("--hf_trust_remote_code", action="store_true")

    p.add_argument("--max_tokens", type=int, default=4192)
    p.add_argument("--temperature", type=int, default=0.1)
    p.add_argument("--top_p", type=int, default=0.90)

    p.add_argument("--sim_code_only", action="store_true", help="Simulate only the generated code, not any NL input for fair comparison with arm 3")
    p.add_argument(
        "--exec_code",
        action="store_true",
        help="execute code-CoT in sandboxed subprocess (imports allowed)",
    )
    p.add_argument(
        "--controlled_sim",
        action="store_true",
        help="do fair controlled simulation w/o prompt",
    )
    p.add_argument("--log_every", type=int, default=50)

    # TensorBoard text limits
    p.add_argument("--tb_text_chars", type=int, default=10000)
    p.add_argument("--tb_disable", action="store_true")

    # vLLM options (kept minimal; defaults are conservative)
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for backends that support chat_many (vLLM).",
    )
    p.add_argument("--vllm_dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16"])
    p.add_argument("--vllm_tensor_parallel", type=int, default=8)
    p.add_argument("--vllm_gpu_mem_util", type=float, default=0.90)
    p.add_argument("--vllm_max_model_len", type=int, default=None)
    p.add_argument("--vllm_download_dir", type=str, default="../models")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
