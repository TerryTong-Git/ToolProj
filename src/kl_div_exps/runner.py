# cot_jointprob_exp/runner.py
import os, math, argparse, random, json, hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

from cox.store import Store
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .tasks import make_taskset, Task
from .prompts import nl_instruction, code_instruction, STOP_STR
from .metrics import summarize_pairwise_kl, dist_over_sequences_from_scores
from .plotting import save_summary_plot

# ----------------------------- config -----------------------------
@dataclass
class GenConfig:
    model_name: str
    representation: str         # "code" or "nl"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    device: str = "auto"        # "cuda", "cpu", "auto"
    dtype: str = "auto"         # "auto", "bfloat16", "float16", "float32"
    stop_str: str = STOP_STR
    do_path_logprob: bool = False
    include_prompt_ll: bool = True  # include log p(x) in 'joint' approx
    seed: int = 0

def pick_device(flag: str) -> str:
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag

def pick_dtype(flag: str):
    if flag == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[flag]

def to_chat_ids(tokenizer, user_text: str) -> Tuple[str, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": user_text}]
        prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        prompt_str = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    else:
        prompt_str = f"USER: {user_text}\nASSISTANT:"
        prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=True).input_ids
    return prompt_str, prompt_ids

@torch.no_grad()
def per_token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns per-token log p(next_token | prefix) aligned to labels position.
    For sequence ids s = [t0, t1, ..., tN], returns logprobs for tokens t1..tN.
    """
    out = model(input_ids=input_ids)
    logits = out.logits  # [1, T, V]
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)   # [1, T-1, V]
    tgt = input_ids[:, 1:]                            # [1, T-1]
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return token_logp[0]  # [T-1]

def teacher_forced_logp(model, tokenizer, device, prompt_ids: torch.Tensor, target_text: str,
                        include_prompt_ll: bool) -> Dict[str, Any]:
    """
    Exact per-token logprobs for prompt (optional) + target continuation.
    Returns: dict with arrays and sums.
    """
    # tokenize target (no specials)
    tgt_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prompt_ids = prompt_ids.to(device)
    full_ids = torch.cat([prompt_ids, tgt_ids], dim=1)  # [1, P+K]

    # (A) prompt part log p(x) under LM (approx joint). We compute logp of prompt tokens
    # as if model generated them from BOS.
    logp_prompt_sum = 0.0
    n_prompt_tokens = 0
    if include_prompt_ll:
        lp_prompt = per_token_logprobs(model, prompt_ids)   # length P-1
        logp_prompt_sum = float(lp_prompt.sum().item())
        n_prompt_tokens = int(lp_prompt.numel())

    # (B) continuation part log p(z | prompt)
    lp_full = per_token_logprobs(model, full_ids)           # length P+K-1
    K = tgt_ids.shape[1]
    cont_logps = lp_full[-K:]                                # last K positions correspond to target
    logp_cont_sum = float(cont_logps.sum().item())
    n_cont = int(cont_logps.numel())

    # combined
    total_sum = logp_prompt_sum + logp_cont_sum
    total_tok = n_prompt_tokens + n_cont
    avg = total_sum / max(1, total_tok)
    ppl = math.exp(-avg) if total_tok > 0 else float("inf")
    return dict(
        sum_logp_joint=total_sum, n_tokens_joint=total_tok, avg_logp_joint=avg, ppl_joint=ppl,
        sum_logp_prompt=logp_prompt_sum, n_tokens_prompt=n_prompt_tokens,
        sum_logp_cont=logp_cont_sum, n_tokens_cont=n_cont,
        cont_logps=cont_logps.detach().cpu().numpy().tolist()
    )

def build_instruction(task: Task, representation: str) -> str:
    return code_instruction(task) if representation == "code" else nl_instruction(task)

def seq_fingerprint(text: str) -> str:
    # Stable id for a z sequence across seeds/models (on same task+rep)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

@torch.no_grad()
def sample_text(model, tokenizer, device, prompt_ids: torch.Tensor,
                max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                repetition_penalty: float, seed: int) -> str:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = model.generate(
        prompt_ids.to(device),
        max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
        use_cache=True,
        return_dict_in_generate=False
    )[0]
    return tokenizer.decode(gen[prompt_ids.shape[-1]:], skip_special_tokens=True)

# ----------------------------- DDP helpers -----------------------------
def setup_ddp_if_needed(distributed: bool):
    import torch.distributed as dist
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if distributed:
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1

def shard_indices(n: int, rank: int, world: int):
    idx = list(range(n))
    return idx[rank::world]

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--exp_id", type=str, default=None)

    # models / reps
    ap.add_argument("--models", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--representations", type=str, default="nl,code")  # compare both

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    # seeds / tasks
    ap.add_argument("--num_seeds", type=int, default=12)
    ap.add_argument("--task_seed", type=int, default=7)
    ap.add_argument("--n_per_theta", type=int, default=20)
    ap.add_argument("--thetas", type=str, default="add,sub,mul")

    # joint prob control
    ap.add_argument("--include_prompt_ll", action="store_true",
                    help="Include log p(prompt) when computing log p(z,x|θ,r).")

    # perf
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--distributed", action="store_true",
                    help="Use torch.distributed with torchrun.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    console = Console()

    # DDP init
    rank, world = setup_ddp_if_needed(args.distributed)

    # Store + TB
    store = Store(args.out_dir, exp_id=args.exp_id)
    tb = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))

    # Taskset
    thetas = tuple([t.strip() for t in args.thetas.split(",") if t.strip()])
    tasks = make_taskset(n_per_theta=args.n_per_theta, seed=args.task_seed, thetas=thetas)
    task_indices = shard_indices(len(tasks), rank, world)

    models = [m.strip() for m in args.models.split(",")]
    reps = [r.strip() for r in args.representations.split(",")]
    include_prompt = args.include_prompt_ll

    # Cox schemas
    if "config" not in store.tables:
        store.add_table("config", {
            "models": str, "representations": str, "max_new_tokens": int, "temperature": float,
            "top_p": float, "top_k": int, "repetition_penalty": float, "num_seeds": int,
            "n_per_theta": int, "thetas": str, "include_prompt_ll": bool, "distributed_world": int
        })
    store["config"].append_row({
        "models": ",".join(models), "representations": ",".join(reps),
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
        "num_seeds": args.num_seeds, "n_per_theta": args.n_per_theta, "thetas": ",".join(thetas),
        "include_prompt_ll": include_prompt, "distributed_world": world
    })

    if "samples" not in store.tables:
        
        store.add_table("samples", {
            "rank": int, "model_name": str, "rep": str, "seed": int,
            "theta": str, "a": int, "b": int,
            "prompt_text": str, "cot_text": str, "answer_text": str, "full_text": str,
            "seq_id": str,
            "sum_logp_joint": float, "n_tokens_joint": int, "avg_logp_joint": float, "ppl_joint": float,
            "sum_logp_prompt": float, "n_tokens_prompt": int,
            "sum_logp_cont": float, "n_tokens_cont": int
        })

    if "kl_stats" not in store.tables:
        store.add_table("kl_stats", {
            "model_name": str, "rep": str, "theta": str,
            "num_pairs": int, "max_kl": float, "avg_kl": float, "var_kl": float
        })

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(), transient=False, console=console
    ) as progress:

        # main loop per (model, rep)
        for model_name in models:
            # Load once per (rank, model)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype if device != "cpu" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            if device == "cpu": model.to("cpu")

            for rep in reps:
                desc = f"[rank {rank}] {model_name} / {rep}"
                t_id = progress.add_task(desc, total=len(task_indices) * args.num_seeds)

                # These hold, per theta, the map: seq_id -> sum_logp_joint (aggregated over seeds)
                # We will convert to distributions over the *union* of sequences for KL.
                per_theta_seq2lp: Dict[str, Dict[str, float]] = {}  # theta -> {seq_id: logp}
                per_theta_support: Dict[str, set] = {}              # theta -> set(seq_id)

                for ti in task_indices:
                    task = tasks[ti]
                    instr = build_instruction(task, rep)
                    prompt_str, prompt_ids = to_chat_ids(tokenizer, instr)

                    for seed in range(args.num_seeds):
                        text = sample_text(
                            model, tokenizer, device, prompt_ids,
                            args.max_new_tokens, args.temperature, args.top_p,
                            args.top_k, args.repetition_penalty, seed=(seed + 1337*rank)
                        )
                        # split CoT and final answer
                        idx = text.find(STOP_STR)
                        if idx == -1:
                            cot_text, ans_text = text.strip(), ""
                        else:
                            cot_text = text[:idx].strip()
                            ans_text = text[idx+len(STOP_STR):].strip()

                        # joint logprob (optionally including prompt)
                        tf = teacher_forced_logp(model, tokenizer, device, prompt_ids, text, include_prompt)
                        seq_id = seq_fingerprint(text)

                        # record
                        CAPS = {
                            "prompt_text": 580,
                            "cot_text":    580,
                            "answer_text": 580,
                            "full_text":   580,
                        }

                        def _clip(s: str, limit: int) -> str:
                            if s is None: return ""
                            return s if len(s) <= limit else (s[:limit-3] + "...")

                
                        store["samples"].append_row({
                            "rank": rank, "model_name": model_name, "rep": rep, "seed": seed,
                            "theta": task.theta, "a": task.a, "b": task.b,
                            "prompt_text": _clip(prompt_str, CAPS["prompt_text"]),
                            "cot_text":    _clip(cot_text, CAPS["cot_text"]),
                            "answer_text": _clip(ans_text, CAPS["answer_text"]),
                            "full_text":   _clip(text, CAPS["full_text"]), "seq_id": seq_id,
                            "sum_logp_joint": tf["sum_logp_joint"], "n_tokens_joint": tf["n_tokens_joint"],
                            "avg_logp_joint": tf["avg_logp_joint"], "ppl_joint": tf["ppl_joint"],
                            "sum_logp_prompt": tf["sum_logp_prompt"], "n_tokens_prompt": tf["n_tokens_prompt"],
                            "sum_logp_cont": tf["sum_logp_cont"], "n_tokens_cont": tf["n_tokens_cont"],
                        })

                        # aggregate for sequence distribution (take max over duplicates for stability)
                        d = per_theta_seq2lp.setdefault(task.theta, {})
                        s = d.get(seq_id, -float("inf"))
                        d[seq_id] = max(s, tf["sum_logp_joint"])
                        per_theta_support.setdefault(task.theta, set()).add(seq_id)

                        progress.update(t_id, advance=1)

                # Compute KL across seeds via *empirical distribution over sequences*
                # For each seed, build a distribution over the union support using that seed's sequences.
                import pandas as pd
                df = store["samples"].df
                df = df[(df["rank"] == rank) & (df["model_name"] == model_name) & (df["rep"] == rep)]

                for theta in thetas:
                    df_t = df[df["theta"] == theta]

                    # support = union of sequences for (model,rep,theta) across *all seeds on this rank*
                    support = sorted(list(per_theta_support.get(theta, set())))

                    # Build one logprob vector per seed by mapping seq_id->sum_logp_joint
                    seed_vecs: List[np.ndarray] = []
                    for sd in sorted(df_t["seed"].unique()):
                        rows = df_t[df_t["seed"] == sd]
                        m = {r.seq_id: r.sum_logp_joint for r in rows.itertuples()}
                        seed_vecs.append(dist_over_sequences_from_scores(m, support))

                    stats = summarize_pairwise_kl(seed_vecs)
                    store["kl_stats"].append_row({
                        "model_name": model_name, "rep": rep, "theta": theta,
                        **stats
                    })

                    # TB scalars (average across seeds)
                    avg_logps = [float(x) for x in df_t["avg_logp_joint"].values.tolist()]
                    if len(avg_logps) > 0:
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/mean_avg_logp_joint", float(np.mean(avg_logps)), rank)
                    if stats["num_pairs"] > 0:
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/kl_avg", stats["avg_kl"], rank)
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/kl_max", stats["max_kl"], rank)

                # quick console line
                console.log(f"[rank {rank}] Finished KL for {model_name} / {rep}")

        # After loops (rank 0): make a small summary plot comparing NL vs Code.
        if rank == 0:
            df = store["samples"].df
            kl = store["kl_stats"].df

            def collect(rep):
                avg = df[df["rep"] == rep]["avg_logp_joint"].values.tolist()
                d = kl[kl["rep"] == rep]
                # average the KL stats across thetas and models for the picture
                if len(d) == 0:
                    return avg, dict(max_kl=np.nan, avg_kl=np.nan, var_kl=np.nan)
                return avg, dict(
                    max_kl=float(d["max_kl"].mean()),
                    avg_kl=float(d["avg_kl"].mean()),
                    var_kl=float(d["var_kl"].mean()),
                )

            avg_code, kl_code = collect("code")
            avg_nl, kl_nl     = collect("nl")
            tag = "rank0"
            png = save_summary_plot(args.out_dir, tag, avg_code, avg_nl, kl_code, kl_nl)
            console.log(f"Saved plot to {png}")

            # Also dump a JSON summary for quick grep
            with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
                json.dump({
                    "avg_logp_joint": {"code": float(np.mean(avg_code) if len(avg_code)>0 else np.nan),
                                       "nl": float(np.mean(avg_nl) if len(avg_nl)>0 else np.nan)},
                    "kl": {"code": kl_code, "nl": kl_nl},
                }, f, indent=2)

    tb.close()
    console.rule("[bold green]Done")

if __name__ == "__main__":
    main()
