# cot_jointprob_exp/metrics.py
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import logsumexp

def kl_divergence(p_log: np.ndarray, q_log: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL(p||q) for two log-prob vectors over the same support.
    """
    p = np.exp(p_log - logsumexp(p_log))
    q = np.exp(q_log - logsumexp(q_log))
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def summarize_pairwise_kl(logprob_vectors: List[np.ndarray]) -> Dict[str, float]:
    """
    Given a list of log-prob vectors (same length, same support), compute pairwise KL stats.
    """
    if len(logprob_vectors) < 2:
        return dict(num_pairs=0, max_kl=np.nan, avg_kl=np.nan, var_kl=np.nan)
    vals = []
    for i in range(len(logprob_vectors)):
        for j in range(len(logprob_vectors)):
            if i != j:
                vals.append(kl_divergence(logprob_vectors[i], logprob_vectors[j]))
    vals = np.array(vals, dtype=float)
    return dict(num_pairs=int(vals.size), max_kl=float(vals.max()),
                avg_kl=float(vals.mean()), var_kl=float(vals.var()))

def dist_over_sequences_from_scores(seq_id_to_logp: Dict[str, float], support: List[str]) -> np.ndarray:
    """
    Convert a dict of {sequence_id: logprob} to a logprob vector over a fixed 'support' list.
    Missing sequences get logprob = -inf (handled via small epsilon in KL).
    """
    vec = np.full(len(support), -np.inf, dtype=float)
    for idx, sid in enumerate(support):
        lp = seq_id_to_logp.get(sid, None)
        if lp is not None:
            vec[idx] = lp
    return vec

# cot_jointprob_exp/plotting.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def save_summary_plot(out_dir: str, tag: str,
                      avg_logps_code: list, avg_logps_nl: list,
                      kl_code: dict, kl_nl: dict):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(8,4.5))
    # Left: avg logp box
    ax1 = fig.add_subplot(1,2,1)
    ax1.boxplot([avg_logps_code, avg_logps_nl], labels=["code z", "NL z"])
    ax1.set_title("Avg log-prob of (z,x|θ,r)")
    ax1.set_ylabel("avg log p per token")
    # Right: KL stats
    ax2 = fig.add_subplot(1,2,2)
    x = np.arange(3)
    ax2.bar(x - 0.15, [kl_code["max_kl"], kl_code["avg_kl"], kl_code["var_kl"]], width=0.3, label="code z")
    ax2.bar(x + 0.15, [kl_nl["max_kl"], kl_nl["avg_kl"], kl_nl["var_kl"]], width=0.3, label="NL z")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["max KL", "avg KL", "var KL"])
    ax2.legend()
    fig.suptitle(f"NL vs Code intermediaries — {tag}")
    out_png = os.path.join(out_dir, f"summary_{tag}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    return out_png

# cot_jointprob_exp/prompts.py
SYSTEM = "You are a helpful math tutor. Always show detailed reasoning before the final answer."
STOP_STR = "Final answer:"

def nl_instruction(task) -> str:
    return (
        f"{SYSTEM}\n"
        f"Task: {task.x_nl}\n"
        f"Answer the question. Show your step-by-step reasoning, "
        f"then put \"{STOP_STR}\" on its own line followed by the final number."
    )

def code_instruction(task) -> str:
    # ask for a small code-like spec as z, then final answer
    return (
        f"{SYSTEM}\n"
        f"Task (produce a minimal Python-like plan first):\n"
        f"- Define variables a={task.a}, b={task.b}\n"
        f"- Operation: {task.theta}\n"
        f"- Compute result r\n"
        f"Then show the computation lines explicitly, and finally put "
        f"\"{STOP_STR}\" on its own line followed by the final number."
    )

# cot_jointprob_exp/runner.py
import os, math, argparse, random, json, hashlib
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Set
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

from cox.store import Store

from .tasks import make_taskset, Task
from .prompts import nl_instruction, code_instruction, STOP_STR
from .metrics import summarize_pairwise_kl, dist_over_sequences_from_scores
from .plotting import save_summary_plot

from tqdm.auto import tqdm

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
    include_prompt_ll: bool = True
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
    out = model(input_ids=input_ids)
    logits = out.logits
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:]
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return token_logp[0]

def teacher_forced_logp(model, tokenizer, device, prompt_ids: torch.Tensor, target_text: str,
                        include_prompt_ll: bool) -> Dict[str, Any]:
    tgt_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prompt_ids = prompt_ids.to(device)
    full_ids = torch.cat([prompt_ids, tgt_ids], dim=1)

    logp_prompt_sum = 0.0
    n_prompt_tokens = 0
    if include_prompt_ll:
        lp_prompt = per_token_logprobs(model, prompt_ids)
        logp_prompt_sum = float(lp_prompt.sum().item())
        n_prompt_tokens = int(lp_prompt.numel())

    lp_full = per_token_logprobs(model, full_ids)
    K = tgt_ids.shape[1]
    cont_logps = lp_full[-K:]
    logp_cont_sum = float(cont_logps.sum().item())
    n_cont = int(cont_logps.numel())

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

class GlobalTQDM:
    def __init__(self, total_global: int, distributed: bool, rank: int, world: int, sync_every: int = 64):
        self.distributed = distributed
        self.rank = rank
        self.world = world
        self.sync_every = max(1, sync_every)
        self.local_count = 0
        self.last_shown = 0
        self.total_global = total_global
        self.bar = tqdm(total=total_global, dynamic_ncols=True, leave=True) if rank == 0 else None
        if distributed:
            import torch.distributed as dist
            assert dist.is_initialized()
            self.buf = torch.tensor([0], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")

    def update_local(self, n: int = 1):
        self.local_count += n
        if not self.distributed:
            if self.bar is not None:
                self.bar.update(n)
            return
        if self.local_count % self.sync_every == 0:
            import torch.distributed as dist
            self.buf.fill_(self.local_count)
            dist.all_reduce(self.buf, op=dist.ReduceOp.SUM)
            global_done = int(self.buf.item())
            if self.rank == 0 and self.bar is not None:
                delta = max(0, global_done - self.last_shown)
                if delta:
                    self.bar.update(delta)
                    self.last_shown = global_done

    def finalize(self):
        if self.distributed:
            import torch.distributed as dist
            self.buf.fill_(self.local_count)
            dist.all_reduce(self.buf, op=dist.ReduceOp.SUM)
            global_done = int(self.buf.item())
            if self.rank == 0 and self.bar is not None:
                delta = max(0, global_done - self.last_shown)
                if delta:
                    self.bar.update(delta)
                    self.last_shown = global_done
        if self.bar is not None:
            self.bar.close()

# ----------------------------- resume helpers -----------------------------
def make_unit_key(rank:int, model:str, rep:str, theta:str, a:int, b:int, seed:int) -> Tuple:
    return (int(rank), str(model), str(rep), str(theta), int(a), int(b), int(seed))

def load_done_units(store: Store) -> Set[Tuple]:
    """
    Build a set of completed unit keys from the existing 'samples' table.
    Safe to call when the table does not exist.
    """
    done: Set[Tuple] = set()
    try:
        if "samples" in store.tables:
            df = store["samples"].df
            # tolerate older runs that may not have all columns
            cols = set(df.columns)
            needed = {"rank","model_name","rep","theta","a","b","seed"}
            if needed.issubset(cols):
                for r in df.itertuples(index=False):
                    try:
                        done.add(make_unit_key(
                            getattr(r,"rank"), getattr(r,"model_name"),
                            getattr(r,"rep"), getattr(r,"theta"),
                            getattr(r,"a"), getattr(r,"b"), getattr(r,"seed")
                        ))
                    except Exception:
                        continue
    except Exception:
        pass
    return done

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--exp_id", type=str, default=None)

    # models / reps
    ap.add_argument("--models", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--representations", type=str, default="nl,code")

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

    # previews & TB text
    ap.add_argument("--preview_chars", type=int, default=128)
    ap.add_argument("--tb_text_chars", type=int, default=4000)

    # resume control
    ap.add_argument("--no_resume", action="store_true",
                    help="Disable resume; process everything regardless of existing rows.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # DDP init
    rank, world = setup_ddp_if_needed(args.distributed)
    if rank == 0:
        print(f"[runner] world={world}  device={args.device}  dtype={args.dtype}")
        print(f"[runner] models={args.models}  reps={args.representations}  thetas={args.thetas}")
        print(f"[runner] seeds={args.num_seeds}  n_per_theta={args.n_per_theta}  include_prompt_ll={args.include_prompt_ll}")
        if args.no_resume:
            print("[runner] Resume disabled: will recompute all units.")

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
            "n_per_theta": int, "thetas": str, "include_prompt_ll": bool, "distributed_world": int,
            "preview_chars": int, "tb_text_chars": int
        })
    store["config"].append_row({
        "models": ",".join(models), "representations": ",".join(reps),
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
        "num_seeds": args.num_seeds, "n_per_theta": args.n_per_theta, "thetas": ",".join(thetas),
        "include_prompt_ll": include_prompt, "distributed_world": world,
        "preview_chars": args.preview_chars, "tb_text_chars": args.tb_text_chars
    })

    if "samples" not in store.tables:
        store.add_table("samples", {
            "rank": int, "model_name": str, "rep": str, "seed": int,
            "theta": str, "a": int, "b": int,

            "prompt_preview": str, "prompt_len": int,
            "cot_preview": str,    "cot_len": int,
            "answer_preview": str, "answer_len": int,
            "full_preview": str,   "full_len": int,

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

    # --- build resume set (what's already done) ---
    done_units: Set[Tuple] = set()
    if not args.no_resume:
        done_units = load_done_units(store)
        if rank == 0:
            print(f"[runner] Resume: found {len(done_units)} completed units in existing store.")

    # --- tqdm global ETA setup ---
    total_global = len(tasks) * args.num_seeds * len(models) * len(reps)
    tracker = GlobalTQDM(total_global=total_global, distributed=args.distributed, rank=rank, world=world, sync_every=64)

    def clip(s: str, n: int) -> str:
        if s is None: return ""
        return s if len(s) <= n else (s[:max(0, n-3)] + "...")

    # -------------- main training/eval with graceful interrupt --------------
    try:
        for model_name in models:
            if rank == 0:
                print(f"[runner] Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
            cache_dir = "./models"

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype if device != "cpu" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                cache_dir=cache_dir
            )
            if device == "cpu": model.to("cpu")

            for rep in reps:
                if rank == 0:
                    print(f"[runner] Start pass: model={model_name} rep={rep}")

                per_theta_seq2lp: Dict[str, Dict[str, float]] = {}
                per_theta_support: Dict[str, set] = {}

                for ti in task_indices:
                    task = tasks[ti]
                    instr = build_instruction(task, rep)
                    prompt_str, prompt_ids = to_chat_ids(tokenizer, instr)

                    for seed in range(args.num_seeds):
                        unit_key = make_unit_key(rank, model_name, rep, task.theta, task.a, task.b, seed)
                        if (not args.no_resume) and (unit_key in done_units):
                            # already completed—skip, but still advance tracker for ETA
                            tracker.update_local(1)
                            continue

                        text = sample_text(
                            model, tokenizer, device, prompt_ids,
                            args.max_new_tokens, args.temperature, args.top_p,
                            args.top_k, args.repetition_penalty, seed=(seed + 1337*rank)
                        )
                        idx = text.find(STOP_STR)
                        if idx == -1:
                            cot_text, ans_text = text.strip(), ""
                        else:
                            cot_text = text[:idx].strip()
                            ans_text = text[idx+len(STOP_STR):].strip()

                        tf = teacher_forced_logp(model, tokenizer, device, prompt_ids, text, include_prompt)
                        seq_id = seq_fingerprint(text)

                        store["samples"].append_row({
                            "rank": rank, "model_name": model_name, "rep": rep, "seed": seed,
                            "theta": task.theta, "a": task.a, "b": task.b,

                            "prompt_preview": clip(instr, args.preview_chars), "prompt_len": len(instr),
                            "cot_preview":    clip(cot_text, args.preview_chars), "cot_len": len(cot_text),
                            "answer_preview": clip(ans_text, args.preview_chars), "answer_len": len(ans_text),
                            "full_preview":   clip(text, args.preview_chars), "full_len": len(text),

                            "seq_id": seq_id,
                            "sum_logp_joint": tf["sum_logp_joint"], "n_tokens_joint": tf["n_tokens_joint"],
                            "avg_logp_joint": tf["avg_logp_joint"], "ppl_joint": tf["ppl_joint"],
                            "sum_logp_prompt": tf["sum_logp_prompt"], "n_tokens_prompt": tf["n_tokens_prompt"],
                            "sum_logp_cont": tf["sum_logp_cont"], "n_tokens_cont": tf["n_tokens_cont"],
                        })

                        # log full texts to TB (with truncation to keep files small)
                        task_id = f"a{task.a}_b{task.b}"
                        base = f"{model_name}/{rep}/{task.theta}/seed_{seed}/{task_id}"

                        def tb_block(title, body):
                            if args.tb_text_chars and len(body) > args.tb_text_chars:
                                body_show = body[:args.tb_text_chars - 3] + "..."
                            else:
                                body_show = body
                            return f"**{title}**\n\n```\n{body_show}\n```"

                        step = seed
                        tb.add_text(f"{base}/prompt", tb_block("Prompt", instr), global_step=step)
                        tb.add_text(f"{base}/cot", tb_block("Chain of Thought", cot_text), global_step=step)
                        tb.add_text(f"{base}/answer", tb_block("Final Answer", ans_text), global_step=step)
                        tb.add_text(f"{base}/full", tb_block("Full Raw Output", text), global_step=step)

                        # update resume set & occasionally flush TB
                        done_units.add(unit_key)
                        if (len(done_units) % 200) == 0 and rank == 0:
                            tb.flush()

                        d = per_theta_seq2lp.setdefault(task.theta, {})
                        s = d.get(seq_id, -float("inf"))
                        d[seq_id] = max(s, tf["sum_logp_joint"])
                        per_theta_support.setdefault(task.theta, set()).add(seq_id)

                        tracker.update_local(1)

                    if (ti % max(1, len(task_indices)//5) == 0) and rank == 0:
                        print(f"[runner] …model={model_name} rep={rep} progress task_index={ti}/{len(task_indices)}")

                # KL across seeds for this (model,rep)
                import pandas as pd
                df = store["samples"].df
                df = df[(df["rank"] == rank) & (df["model_name"] == model_name) & (df["rep"] == rep)]

                for theta in thetas:
                    df_t = df[df["theta"] == theta]
                    support = sorted(list(per_theta_support.get(theta, set())))
                    seed_vecs: List[np.ndarray] = []
                    for sd in sorted(df_t["seed"].unique()):
                        rows = df_t[df_t["seed"] == sd]
                        m = {r.seq_id: r.sum_logp_joint for r in rows.itertuples()}
                        seed_vecs.append(dist_over_sequences_from_scores(m, support))
                    stats = summarize_pairwise_kl(seed_vecs)
                    store["kl_stats"].append_row({
                        "model_name": model_name, "rep": rep, "theta": theta, **stats
                    })

                    avg_logps = [float(x) for x in df_t["avg_logp_joint"].values.tolist()]
                    if len(avg_logps) > 0:
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/mean_avg_logp_joint", float(np.mean(avg_logps)), rank)
                    if stats["num_pairs"] > 0:
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/kl_avg", stats["avg_kl"], rank)
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/kl_max", stats["max_kl"], rank)

                if rank == 0:
                    print(f"[runner] Finished KL: model={model_name} rep={rep}")

    except KeyboardInterrupt:
        # graceful save & exit
        if rank == 0:
            print("\n[runner] Caught KeyboardInterrupt: saving state & closing writers…")
        try:
            tracker.finalize()
        except Exception:
            pass
        try:
            tb.flush()
            tb.close()
        except Exception:
            pass
        # Store rows are already persisted on append_row; nothing else to do.
        if rank == 0:
            print("[runner] Safe-to-resume state persisted. Bye.")
        return

    # finalize bar
    tracker.finalize()

    # After loops (rank 0): small summary plot comparing NL vs Code.
    if rank == 0:
        df = store["samples"].df
        kl = store["kl_stats"].df

        def collect(rep):
            avg = df[df["rep"] == rep]["avg_logp_joint"].values.tolist()
            d = kl[kl["rep"] == rep]
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
        print(f"[runner] Saved plot to {png}")

        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump({
                "avg_logp_joint": {"code": float(np.mean(avg_code) if len(avg_code)>0 else np.nan),
                                   "nl": float(np.mean(avg_nl) if len(avg_nl)>0 else np.nan)},
                "kl": {"code": kl_code, "nl": kl_nl},
            }, f, indent=2)
        print("[runner] Wrote JSON summary")

    tb.close()
    if rank == 0:
        print("[runner] Done.")

if __name__ == "__main__":
    main()

# cot_jointprob_exp/tasks.py
from dataclasses import dataclass
from typing import List, Dict, Any
import random

@dataclass
class Task:
    theta: str     # "add" | "sub" | "mul"
    a: int
    b: int

    @property
    def x_nl(self) -> str:
        if self.theta == "add":
            return f"Add {self.a} and {self.b}."
        if self.theta == "sub":
            return f"Subtract {self.b} from {self.a}."
        if self.theta == "mul":
            return f"Multiply {self.a} by {self.b}."
        raise ValueError(self.theta)

    @property
    def y(self) -> int:
        if self.theta == "add":
            return self.a + self.b
        if self.theta == "sub":
            return self.a - self.b
        if self.theta == "mul":
            return self.a * self.b
        raise ValueError(self.theta)

def make_taskset(n_per_theta: int, seed: int = 0,
                 thetas=("add","sub","mul"),
                 a_range=(2,99), b_range=(2,99)) -> List[Task]:
    rng = random.Random(seed)
    tasks: List[Task] = []
    for theta in thetas:
        for _ in range(n_per_theta):
            a = rng.randint(*a_range)
            b = rng.randint(*b_range)
            # avoid negatives for sub if you want; feel free to remove
            if theta == "sub" and b > a:
                a, b = b, a
            tasks.append(Task(theta, a, b))
    return tasks
