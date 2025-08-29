# cot_jointprob_exp/runner.py
import os, math, argparse, random, json, hashlib, warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Set
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

from cox.store import Store

from .tasks import make_taskset, make_paired_taskset, Task
from .prompts import nl_instruction, code_instruction, STOP_STR
from .metrics import (
    summarize_pairwise_kl,
    dist_over_sequences_from_scores,
    build_pooled_dist,
    symmetric_kl,
    js_divergence,
    kl_divergence,
)
from .plotting import save_summary_plot

from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

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
            return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
        else:
            return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[flag]

def pick_eos_pad_ids(tokenizer):
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if eos is None and pad is not None:
        eos = pad
    if pad is None and eos is not None:
        pad = eos
    if eos is None and pad is None:
        eos = pad = 0
    return int(eos), int(pad)

def _has_chat_template(tokenizer) -> bool:
    return hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)

def to_chat_ids(tokenizer, user_text: str) -> Tuple[str, torch.Tensor]:
    if _has_chat_template(tokenizer):
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
    if logits.size(1) <= 1:
        return torch.empty((0,), dtype=logits.dtype, device=logits.device)
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
        logp_prompt_sum = float(lp_prompt.sum().item()) if lp_prompt.numel() > 0 else 0.0
        n_prompt_tokens = int(lp_prompt.numel())

    lp_full = per_token_logprobs(model, full_ids)
    K = tgt_ids.shape[1]
    cont_logps = lp_full[-K:] if lp_full.numel() >= K else lp_full.new_empty((0,))
    logp_cont_sum = float(cont_logps.sum().item()) if cont_logps.numel() > 0 else 0.0
    n_cont = int(cont_logps.numel())

    total_sum = logp_prompt_sum + logp_cont_sum
    total_tok = n_prompt_tokens + n_cont
    avg = total_sum / max(1, total_tok)
    ppl = math.exp(-avg) if total_tok > 0 else float("inf")
    return dict(
        sum_logp_joint=total_sum, n_tokens_joint=total_tok, avg_logp_joint=avg, ppl_joint=ppl,
        sum_logp_prompt=logp_prompt_sum, n_tokens_prompt=n_prompt_tokens,
        sum_logp_cont=logp_cont_sum, n_tokens_cont=n_cont,
        cont_logps=cont_logps.detach().cpu().numpy().tolist() if cont_logps.numel() > 0 else []
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
    eos_id, pad_id = pick_eos_pad_ids(tokenizer)
    top_k = int(top_k) if top_k and top_k > 0 else 0
    gen = model.generate(
        prompt_ids.to(device),
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=top_k,
        repetition_penalty=float(repetition_penalty),
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=True,
        return_dict_in_generate=False
    )[0]
    return tokenizer.decode(gen[prompt_ids.shape[-1]:], skip_special_tokens=True)

# ----------- NEW: chain-rule token-level KL helpers -----------
@torch.no_grad()
def _logsoftmax_rows_for_path(model, full_ids: torch.Tensor, prompt_len: int, K: int) -> torch.Tensor:
    out = model(input_ids=full_ids)
    logps = torch.log_softmax(out.logits[:, :-1, :], dim=-1)[0]  # [L-1, V]
    start = max(0, prompt_len - 1)
    end = start + K
    return logps[start:end, :]  # [K, V]

@torch.no_grad()
def chainrule_kl_between_prompts(model, tokenizer, device,
                                 prompt_ids_p: torch.Tensor, prompt_ids_q: torch.Tensor,
                                 continuation_text: str) -> Tuple[float, int]:
    tgt_ids = tokenizer(continuation_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    K = int(tgt_ids.shape[1])
    if K == 0:
        return 0.0, 0
    p_ids = torch.cat([prompt_ids_p.to(device), tgt_ids], dim=1)
    q_ids = torch.cat([prompt_ids_q.to(device), tgt_ids], dim=1)
    P = int(prompt_ids_p.shape[1])
    Q = int(prompt_ids_q.shape[1])
    logp_rows = _logsoftmax_rows_for_path(model, p_ids, prompt_len=P, K=K)  # [K, V]
    logq_rows = _logsoftmax_rows_for_path(model, q_ids, prompt_len=Q, K=K)  # [K, V]
    p_probs = logp_rows.double().exp()
    kl_steps = (p_probs * (logp_rows.double() - logq_rows.double())).sum(dim=-1)
    kl_sum = float(kl_steps.sum().item())
    return kl_sum, K

# ----------------------------- DDP helpers -----------------------------
def setup_ddp_if_needed(distributed: bool):
    import torch.distributed as dist
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
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
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.buf = torch.tensor([0], dtype=torch.long, device=dev)

    def update_local(self, n: int = 1):
        self.local_count += n
        if not self.distributed:
            if self.bar is not None:
                self.bar.update(n)
            return
        if (self.local_count % self.sync_every) == 0:
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
    done: Set[Tuple] = set()
    try:
        if "samples" in store.tables:
            df = store["samples"].df
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

    # concept experiment controls
    ap.add_argument("--paired_inputs", action="store_true",
                    help="If set, all thetas share the same (a,b) inputs.")
    ap.add_argument("--n_pairs", type=int, default=50,
                    help="How many (a,b) inputs when --paired_inputs is set.")

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
        if args.paired_inputs:
            print(f"[runner] paired_inputs=True  n_pairs={args.n_pairs}")
        else:
            print(f"[runner] paired_inputs=False  n_per_theta={args.n_per_theta}")
        print(f"[runner] seeds={args.num_seeds}  include_prompt_ll={args.include_prompt_ll}")
        if args.no_resume:
            print("[runner] Resume disabled: will recompute all units.")

    # Store + TB
    store = Store(args.out_dir, exp_id=args.exp_id)
    tb = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))

    # Taskset
    thetas = tuple([t.strip() for t in args.thetas.split(",") if t.strip()])
    if args.paired_inputs:
        tasks = make_paired_taskset(n_pairs=args.n_pairs, seed=args.task_seed, thetas=thetas)
    else:
        tasks = make_taskset(n_per_theta=args.n_per_theta, seed=args.task_seed, thetas=thetas)
    task_indices = shard_indices(len(tasks), rank, world)

    models = [m.strip() for m in args.models.split(",")]
    reps = [r.strip() for r in args.representations.split(",")]
    include_prompt = args.include_prompt_ll

    # Cox schemas (only short strings; no full_text to avoid HDF5 string limits)
    if "config" not in store.tables:
        store.add_table("config", {
            "models": str, "representations": str, "max_new_tokens": int, "temperature": float,
            "top_p": float, "top_k": int, "repetition_penalty": float, "num_seeds": int,
            "thetas": str, "include_prompt_ll": bool, "distributed_world": int,
            "preview_chars": int, "tb_text_chars": int, "paired_inputs": bool, "n_pairs": int
        })
    store["config"].append_row({
        "models": ",".join(models), "representations": ",".join(reps),
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
        "num_seeds": args.num_seeds, "thetas": ",".join(thetas),
        "include_prompt_ll": include_prompt, "distributed_world": world,
        "preview_chars": args.preview_chars, "tb_text_chars": args.tb_text_chars,
        "paired_inputs": bool(args.paired_inputs), "n_pairs": int(args.n_pairs)
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

    if "kl_concept" not in store.tables:
        store.add_table("kl_concept", {
            "model_name": str, "rep": str,
            "a": int, "b": int,
            "theta_i": str, "theta_j": str,
            "kl_ij": float, "kl_ji": float,
            "skl": float, "jsd": float
        })

    if "kl_token_chain" not in store.tables:
        store.add_table("kl_token_chain", {
            "model_name": str, "rep": str,
            "a": int, "b": int, "seed": int,
            "theta_i": str, "theta_j": str,
            "kl_chain_ij": float, "kl_chain_ji": float,
            "len_i": int, "len_j": int
        })

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    done_units: Set[Tuple] = set()
    if not args.no_resume:
        done_units = load_done_units(store)
        if rank == 0:
            print(f"[runner] Resume: found {len(done_units)} completed units in existing store.")

    total_global = len(tasks) * args.num_seeds * len(models) * len(reps)
    tracker = GlobalTQDM(total_global=total_global, distributed=args.distributed, rank=rank, world=world, sync_every=64)

    def clip(s: str, n: int) -> str:
        if s is None: return ""
        return s if len(s) <= n else (s[:max(0, n-3)] + "...")

    try:
        for model_name in models:
            if rank == 0:
                print(f"[runner] Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            eos_id, pad_id = pick_eos_pad_ids(tokenizer)
            tokenizer.pad_token_id = pad_id

            cache_dir = "./models"
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype if device != "cpu" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                cache_dir=cache_dir
            ).eval()
            if device == "cpu":
                model.to("cpu")

            for rep in reps:
                if rank == 0:
                    print(f"[runner] Start pass: model={model_name} rep={rep}")

                per_theta_support: Dict[str, set] = {}
                # NEW: in-memory cache for full continuations (avoid storing long strings in Cox)
                texts_cache: Dict[Tuple[int,int,int,str], str] = {}

                for ti in task_indices:
                    task = tasks[ti]
                    instr = build_instruction(task, rep)
                    prompt_str, prompt_ids = to_chat_ids(tokenizer, instr)

                    for seed in range(args.num_seeds):
                        unit_key = make_unit_key(rank, model_name, rep, task.theta, task.a, task.b, seed)
                        if (not args.no_resume) and (unit_key in done_units):
                            tracker.update_local(1)
                            continue

                        text = sample_text(
                            model, tokenizer, device, prompt_ids,
                            args.max_new_tokens, args.temperature, args.top_p,
                            args.top_k, args.repetition_penalty, seed=(seed + 1337*rank)
                        )

                        idx = text.find(STOP_STR) if STOP_STR else -1
                        if idx == -1:
                            cot_text, ans_text = text.strip(), ""
                        else:
                            cot_text = text[:idx].strip()
                            ans_text = text[idx+len(STOP_STR):].strip()

                        tf = teacher_forced_logp(model, tokenizer, device, prompt_ids, text, include_prompt)
                        seq_id = seq_fingerprint(text)

                        # Store only short previews + numbers (safe for HDF5 fixed strings)
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

                        # Keep full text only in memory for chain-rule KL across concepts
                        texts_cache[(int(task.a), int(task.b), int(seed), str(task.theta))] = text

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

                        done_units.add(unit_key)
                        per_theta_support.setdefault(task.theta, set()).add(seq_id)
                        tracker.update_local(1)

                    if (ti % max(1, len(task_indices)//5) == 0) and rank == 0:
                        print(f"[runner] …model={model_name} rep={rep} progress task_index={ti}/{len(task_indices)}")

                # KL across seeds (unchanged)
                import pandas as pd
                df = store["samples"].df
                df = df[(df["rank"] == rank) & (df["model_name"] == model_name) & (df["rep"] == rep)]

                for theta in thetas:
                    df_t = df[df["theta"] == theta]
                    support = sorted(list(per_theta_support.get(theta, set())))
                    seed_vecs: List[np.ndarray] = []
                    for sd in sorted(df_t["seed"].unique()):
                        rows = df_t[df_t["seed"] == sd]
                        if len(rows) == 0:
                            continue
                        m = {r.seq_id: r.sum_logp_joint for r in rows.itertuples()}
                        vec = dist_over_sequences_from_scores(m, support)
                        if vec is not None and np.isfinite(vec).any():
                            seed_vecs.append(vec)
                    if len(seed_vecs) >= 2:
                        stats = summarize_pairwise_kl(seed_vecs)
                    else:
                        stats = {"num_pairs": 0, "max_kl": float("nan"), "avg_kl": float("nan"), "var_kl": float("nan")}
                    store["kl_stats"].append_row({
                        "model_name": model_name, "rep": rep, "theta": theta, **stats
                    })

                    avg_logps = [float(x) for x in df_t["avg_logp_joint"].values.tolist()]
                    if len(avg_logps) > 0:
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/mean_avg_logp_joint", float(np.mean(avg_logps)), rank)
                    if stats["num_pairs"] > 0:
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/kl_avg", stats["avg_kl"], rank)
                        tb.add_scalar(f"{model_name}/{rep}/{theta}/kl_max", stats["max_kl"], rank)

                # --- Sequence-level across-concept (existing)
                df_all = store["samples"].df
                df_mr = df_all[(df_all["rank"] == rank) & (df_all["model_name"] == model_name) & (df_all["rep"] == rep)]
                if len(df_mr) > 0:
                    ab_groups = df_mr.groupby(["a","b"])
                    for (aa, bb), df_ab in ab_groups:
                        support = sorted(df_ab["seq_id"].unique().tolist())
                        theta_vecs = {}
                        for th in thetas:
                            rows_th = list(df_ab[df_ab["theta"] == th].itertuples())
                            if not rows_th:
                                continue
                            vec = build_pooled_dist(rows_th, support)
                            if vec is None or not np.isfinite(vec).any():
                                continue
                            theta_vecs[th] = vec
                        th_list = sorted(theta_vecs.keys())
                        for i in range(len(th_list)):
                            for j in range(i+1, len(th_list)):
                                ti, tj = th_list[i], th_list[j]
                                pi, pj = theta_vecs[ti], theta_vecs[tj]
                                kl_ij = kl_divergence(pi, pj)
                                kl_ji = kl_divergence(pj, pi)
                                skl   = symmetric_kl(pi, pj)
                                jsd   = js_divergence(pi, pj)
                                vals = [kl_ij, kl_ji, skl, jsd]
                                if not all(np.isfinite(v) for v in vals):
                                    continue
                                store["kl_concept"].append_row({
                                    "model_name": model_name, "rep": rep,
                                    "a": int(aa), "b": int(bb),
                                    "theta_i": ti, "theta_j": tj,
                                    "kl_ij": float(kl_ij), "kl_ji": float(kl_ji),
                                    "skl": float(skl), "jsd": float(jsd)
                                })

                # --- NEW: Token-level chain-rule KL across concepts (θ vs θ*) per (a,b,seed), using in-memory texts
                # Build set of (a,b,seed) triples we have for more than one theta
                triples = {}
                for (aa, bb, sd, th) in texts_cache.keys():
                    triples.setdefault((aa, bb, sd), set()).add(th)
                for (aa, bb, sd), ths in triples.items():
                    th_list = sorted([th for th in thetas if th in ths])
                    if len(th_list) < 2:
                        continue
                    # Build prompts for each theta
                    prompt_ids_by_theta = {}
                    for th in th_list:
                        fake_task = Task(theta=th, a=int(aa), b=int(bb))
                        instr_th = build_instruction(fake_task, rep)
                        _, pids_th = to_chat_ids(tokenizer, instr_th)
                        prompt_ids_by_theta[th] = pids_th
                    # Compute KL both directions for each pair
                    for i in range(len(th_list)):
                        for j in range(i+1, len(th_list)):
                            ti, tj = th_list[i], th_list[j]
                            text_i = texts_cache[(aa, bb, sd, ti)]
                            text_j = texts_cache[(aa, bb, sd, tj)]
                            pids_i = prompt_ids_by_theta[ti]
                            pids_j = prompt_ids_by_theta[tj]
                            kl_ij, K_i = chainrule_kl_between_prompts(model, tokenizer, device, pids_i, pids_j, text_i)
                            kl_ji, K_j = chainrule_kl_between_prompts(model, tokenizer, device, pids_j, pids_i, text_j)
                            store["kl_token_chain"].append_row({
                                "model_name": model_name, "rep": rep,
                                "a": int(aa), "b": int(bb), "seed": int(sd),
                                "theta_i": ti, "theta_j": tj,
                                "kl_chain_ij": float(kl_ij), "kl_chain_ji": float(kl_ji),
                                "len_i": int(K_i), "len_j": int(K_j)
                            })

                # free memory for this rep before next rep/model
                texts_cache.clear()

                if rank == 0:
                    print(f"[runner] Finished KL (seeds, concepts, token-chain): model={model_name} rep={rep}")

    except KeyboardInterrupt:
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
        return

    tracker.finalize()

    # After loops (rank 0): summary + plot
    if rank == 0:
        df = store["samples"].df
        kl = store["kl_stats"].df

        def collect(rep):
            avg = df[df["rep"] == rep]["avg_logp_joint"].values.tolist()
            d = kl[kl["rep"] == rep]
            if len(d) == 0:
                return avg, dict(max_kl=np.nan, avg_kl=np.nan, var_kl=np.nan)
            return avg, dict(
                max_kl=float(np.nanmean(d["max_kl"].replace([np.inf, -np.inf], np.nan).astype(float))) if len(d["max_kl"]) else np.nan,
                avg_kl=float(np.nanmean(d["avg_kl"].replace([np.inf, -np.inf], np.nan).astype(float))) if len(d["avg_kl"]) else np.nan,
                var_kl=float(np.nanmean(d["var_kl"].replace([np.inf, -np.inf], np.nan).astype(float))) if len(d["var_kl"]) else np.nan,
            )

        avg_code, kl_code = collect("code")
        avg_nl, kl_nl     = collect("nl")
        tag = "rank0"
        try:
            png = save_summary_plot(args.out_dir, tag, avg_code, avg_nl, kl_code, kl_nl)
            print(f"[runner] Saved plot to {png}")
        except Exception as e:
            print(f"[runner] Plotting failed (continuing): {e}")

        summary = {
            "avg_logp_joint": {"code": float(np.mean(avg_code) if len(avg_code)>0 else np.nan),
                               "nl":   float(np.mean(avg_nl) if len(avg_nl)>0 else np.nan)},
            "kl_across_seeds": {"code": kl_code, "nl": kl_nl},
        }

        if "kl_concept" in store.tables and len(store["kl_concept"].df) > 0:
            kc = store["kl_concept"].df.copy()
            for col in ["kl_ij","kl_ji","skl","jsd"]:
                if col in kc:
                    kc[col] = kc[col].replace([np.inf, -np.inf], np.nan).astype(float)
            def _m(series):
                arr = series.values.astype(float)
                return float(np.nanmean(arr)) if arr.size else np.nan
            summary["kl_across_concepts"] = {
                "sKL": {"code": _m(kc[kc["rep"]=="code"]["skl"]) if "skl" in kc else np.nan,
                        "nl":   _m(kc[kc["rep"]=="nl"]["skl"])   if "skl" in kc else np.nan},
                "JSD": {"code": _m(kc[kc["rep"]=="code"]["jsd"]) if "jsd" in kc else np.nan,
                        "nl":   _m(kc[kc["rep"]=="nl"]["jsd"])   if "jsd" in kc else np.nan},
                "KL_ij_mean": {"code": _m(kc[kc["rep"]=="code"]["kl_ij"]) if "kl_ij" in kc else np.nan,
                               "nl":   _m(kc[kc["rep"]=="nl"]["kl_ij"])   if "kl_ij" in kc else np.nan},
                "KL_ji_mean": {"code": _m(kc[kc["rep"]=="code"]["kl_ji"]) if "kl_ji" in kc else np.nan,
                               "nl":   _m(kc[kc["rep"]=="nl"]["kl_ji"])   if "kl_ji" in kc else np.nan},
            }

        if "kl_token_chain" in store.tables and len(store["kl_token_chain"].df) > 0:
            kt = store["kl_token_chain"].df.copy()
            for col in ["kl_chain_ij","kl_chain_ji"]:
                if col in kt:
                    kt[col] = kt[col].replace([np.inf, -np.inf], np.nan).astype(float)
            def _m(series):
                arr = series.values.astype(float)
                return float(np.nanmean(arr)) if arr.size else np.nan
            summary["kl_token_chain_across_concepts"] = {
                "KL_chain_ij_mean": {"code": _m(kt[(kt["rep"]=="code")]["kl_chain_ij"]),
                                     "nl":   _m(kt[(kt["rep"]=="nl")]["kl_chain_ij"])},
                "KL_chain_ji_mean": {"code": _m(kt[(kt["rep"]=="code")]["kl_chain_ji"]),
                                     "nl":   _m(kt[(kt["rep"]=="nl")]["kl_chain_ji"])},
                "avg_path_len": {"code": _m(kt[(kt["rep"]=="code")]["len_i"]),
                                 "nl":   _m(kt[(kt["rep"]=="nl")]["len_i"])},
            }

        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("[runner] Wrote JSON summary")

    tb.close()
    if rank == 0:
        print("[runner] Done.")

if __name__ == "__main__":
    main()
