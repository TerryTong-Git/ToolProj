#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I(Z; S_R | bucket, r) with auto-generated multi-task probes.
Added: detailed logging, assertions, and progress tracking. Fast-path cache for APPLY(Z) semantics.

Key instrumentation:
- --log_level, --log_mem to emit periodic RAM/VRAM stats
- Dataset summary per (rep,bucket)
- Timing + throughput for APPLY and SCORE steps with tqdm postfix
- Assertions on all array lengths and shapes
- Optional limits: --limit_pairs, --limit_groups, --limit_probes for debug
- Persistent on-disk cache of semantics: --sem_cache_path (default out_dir/sem_cache.jsonl)
"""

from __future__ import annotations
import os, re, json, math, argparse, hashlib, time, sys, random, shutil
from typing import Dict, Any, List, Tuple, Optional
from contextlib import contextmanager

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------- Logging ----------------

import logging
LOG = logging.getLogger("mi_semantics")

def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

def try_mem_stats(enable: bool) -> str:
    if not enable:
        return ""
    try:
        import psutil
        vmem = psutil.virtual_memory()
        ram = f"RAM {vmem.used/1e9:.1f}G/{vmem.total/1e9:.1f}G"
    except Exception:
        ram = "RAM n/a"
    try:
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            used = torch.cuda.memory_allocated(dev) / 1e9
            reserved = torch.cuda.memory_reserved(dev) / 1e9
            total = torch.cuda.get_device_properties(dev).total_memory / 1e9
            vram = f"VRAM used {used:.1f}G res {reserved:.1f}G / {total:.1f}G"
        else:
            vram = "VRAM n/a"
    except Exception:
        vram = "VRAM n/a"
    return f"[{ram}; {vram}]"

@contextmanager
def timed(msg: str, log_mem: bool):
    t0 = time.perf_counter()
    LOG.info("%s ... %s", msg, try_mem_stats(log_mem))
    yield
    dt = time.perf_counter() - t0
    LOG.info("%s done in %.3fs %s", msg, dt, try_mem_stats(log_mem))

# ---------------- TB reading (same schema) ----------------

_THETA = r"(add|sub|mul|mix|lcs|knap|rod|ilp_assign|ilp_prod|ilp_partition)"
_END   = r"(prompt|raw_json|full)"
TAG_RE = re.compile(
    rf'^(?:text_summary/)?'
    rf'(?P<model>.+?)/(?P<rep>nl|code)/'
    rf'(?:d(?P<digits>\d+)/)?'
    rf'(?P<theta>{_THETA})/i(?P<idx>\d+)/'
    rf'(?P<leaf>{_END})(?:/text_summary)?$'
)

def _strip_md_block(s: str) -> str:
    m = re.search(r"```(?:[a-zA-Z0-9]+)?\s*\n([\s\S]*?)\n```", s, flags=re.S)
    return m.group(1).strip() if m else s.strip()

def _tb_text_from_event(ev) -> str:
    tp = ev.tensor_proto
    if tp.string_val:
        val = tp.string_val[0]
        return val.decode("utf-8", "replace") if isinstance(val, (bytes, bytearray)) else str(val)
    if tp.tensor_content:
        try:
            return tp.tensor_content.decode("utf-8", "replace")
        except Exception:
            pass
    return ""

def _iter_event_accumulators(root: str):
    if os.path.isdir(root) and any(fn.startswith("events") for fn in os.listdir(root)):
        yield os.path.basename(root), EventAccumulator(root, size_guidance={'tensors': 10**7})
    else:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p) and any(fn.startswith("events") for fn in os.listdir(p)):
                yield name, EventAccumulator(p, size_guidance={'tensors': 10**7})

def load_tb_pairs(tb_dir: str) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for run_name, ea in _iter_event_accumulators(tb_dir):
        ea.Reload()
        tags = set(ea.Tags().get('tensors', []))
        prompts, raws, meta = {}, {}, {}
        for tag in tags:
            m = TAG_RE.match(tag)
            if not m:
                continue
            digits = m['digits']
            digits_int = int(digits) if digits is not None else None
            key = (m['model'], m['rep'], digits_int, m['theta'], int(m['idx']))
            evs = ea.Tensors(tag)
            if not evs:
                continue
            txt = _tb_text_from_event(evs[-1])
            leaf = m['leaf']
            if leaf == "prompt":
                prompts[key] = _strip_md_block(txt)
            elif leaf in ("raw_json", "full"):
                raws[key] = _strip_md_block(txt)
            meta[key] = dict(model=m['model'], rep=m['rep'], digits=digits_int, theta=m['theta'], idx=int(m['idx']))
        for k in sorted(set(prompts) & set(raws)):
            md = meta[k]
            pairs.append(dict(
                run=run_name,
                model=md["model"], rep=md["rep"], digits=md["digits"], theta=md["theta"], idx=md["idx"],
                prompt=prompts[k], raw_json=raws[k]
            ))
    return pairs

# ---------------- Bucketing ----------------

def assign_bucket(row: Dict[str, Any], kind: str, prompt_len_bins: int = 5) -> str:
    if kind == "digits":
        d = row.get("digits", None)
        return f"d{d}" if isinstance(d, int) and d is not None else "dNA"
    elif kind == "promptlen":
        L = int(row.get("prompt_len", 0))
        width = 64
        idx = min(prompt_len_bins-1, max(0, L // width))
        return f"plen_bin{idx}"
    else:
        return "all"

# ---------------- Engines ----------------

def _vllm_load(model_name: str, tensor_parallel_size: int, dtype: Optional[str], gpu_mem_util: float, max_model_len: Optional[int]):
    from vllm import LLM
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, dtype=dtype,
              gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len, download_dir='../models', enforce_eager=True, swap_space=8)
    tok = llm.get_tokenizer()
    return tok, llm

def _hf_load(model_name: str, device: str, dtype_flag: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eos = tok.eos_token_id; pad = tok.pad_token_id
    if eos is None and pad is not None: eos = pad
    if pad is None and eos is not None: pad = eos
    if eos is None and pad is None: eos = pad = 0
    tok.pad_token_id = pad
    def pick_dtype(flag: str):
        if flag == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
            return torch.float32
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[flag]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=pick_dtype(dtype_flag) if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).eval()
    if device == "cpu": model.to("cpu")
    return tok, model

# ---------------- Teacher-forced Z scoring ----------------

@torch.inference_mode()
def _hf_per_token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids)
    logits = out.logits
    if logits.size(1) <= 1: return torch.empty((0,), dtype=logits.dtype, device=logits.device)
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:]
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return token_logp[0]

@torch.inference_mode()
def _score_z_hf(tok, model, device: str, prompt_str: str, target_text: str, include_prompt_ll: bool):
    prompt_ids = tok(prompt_str, return_tensors="pt", add_special_tokens=True).input_ids.to(device).long()
    target_ids = tok(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device).long()
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    sum_prompt = 0.0; n_prompt = 0
    if include_prompt_ll:
        lp_prompt = _hf_per_token_logprobs(model, prompt_ids)
        sum_prompt = float(lp_prompt.sum().item()); n_prompt = int(lp_prompt.numel())
    lp_full = _hf_per_token_logprobs(model, full_ids)
    K = target_ids.shape[1]
    cont_logps = lp_full[-K:] if lp_full.numel() >= K else lp_full.new_empty((0,))
    sum_cont = float(cont_logps.sum().item()) if cont_logps.numel() > 0 else 0.0
    n_cont = int(cont_logps.numel())
    total_sum = sum_prompt + sum_cont; total_tok = n_prompt + n_cont
    return dict(sum_logp_joint=total_sum, n_tokens_joint=total_tok,
                sum_logp_prompt=sum_prompt, n_tokens_prompt=n_prompt,
                sum_logp_cont=sum_cont, n_tokens_cont=n_cont)

def has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None)

def to_chat_prompt(tok, user_text: str) -> str:
    if has_chat_template(tok):
        msgs = [{"role":"user","content":user_text}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return f"USER: {user_text}\nASSISTANT:"

def _vllm_score_batched(llm, tok, prompts, targets, include_prompt_ll, bs, desc):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0,
                        detokenize=True, skip_special_tokens=False)
    rows = []
    for i in tqdm(range(0, len(prompts), bs), desc='scoring_batched'):
        P = prompts[i:i+bs]
        T = targets[i:i+bs]
        fulls  = [p + t for p, t in zip(P, T)]
        splits = [len(tok(p, add_special_tokens=True)["input_ids"]) for p in P]
        outs = llm.generate(fulls, sp, use_tqdm=False)

        # align helper
        def _align(pl_ids, pl_obj):
            L = len(pl_ids); pl = list(pl_obj) if pl_obj is not None else []
            if len(pl)==L: return pl
            if len(pl)==L-1: return [None]+pl
            if len(pl)<L: return [None]*(L-len(pl))+pl
            return pl[:L]
        def _chosen(entry, tok_id):
            if entry is None: return None
            if isinstance(entry, (int,float)): return float(entry)
            if isinstance(entry, dict):
                v = entry.get(tok_id); 
                if v is None:
                    try: v = entry.get(tok.decode([tok_id], skip_special_tokens=False))
                    except Exception: v = None
                if isinstance(v, dict) and "logprob" in v: return float(v["logprob"])
                if hasattr(v, "logprob"): return float(v.logprob)
                return None
            if hasattr(entry, "logprob"): return float(entry.logprob)
            if isinstance(entry, (list, tuple)):
                for v in entry:
                    vid = (v.get("id") if isinstance(v, dict) else getattr(v, "id", None)) \
                          or (v.get("token_id") if isinstance(v, dict) else getattr(v, "token_id", None))
                    lp  = (v.get("logprob") if isinstance(v, dict) else getattr(v, "logprob", None))
                    if vid == tok_id and lp is not None: return float(lp)
            return None

        for o, split in zip(outs, splits):
            ids = list(o.prompt_token_ids)
            pl  = _align(ids, o.prompt_logprobs)
            chosen = [_chosen(pl[j], ids[j]) for j in range(len(ids))]

            def span(lo, hi):
                s = 0.0; n = 0
                for j in range(max(0, lo), min(hi, len(chosen))):
                    if chosen[j] is not None: s += float(chosen[j]); n += 1
                return s, n

            sum_prompt = n_prompt = 0.0
            if include_prompt_ll:
                sum_prompt, n_prompt = span(0, split)
            sum_cont, n_cont = span(split, len(chosen))
            rows.append(dict(
                sum_logp_joint = sum_prompt + sum_cont,
                n_tokens_joint = int(n_prompt + n_cont),
                sum_logp_prompt= sum_prompt,
                n_tokens_prompt= int(n_prompt),
                sum_logp_cont  = sum_cont,
                n_tokens_cont  = int(n_cont),
            ))
    return rows


# ---------------- APPLY(Z, x) → y for semantics ----------------

def build_apply_prompt(tok, z_text: str, x_text: str) -> str:
    task = (
        "Apply the following procedure exactly to the given input. "
        "Return only the final numeric or string answer on one line, with no extra words.\n\n"
        "PROCEDURE:\n"
        f"{z_text}\n\nINPUT:\n{x_text}\n\nANSWER:"
    )
    if has_chat_template(tok):
        msgs = [{"role":"user","content":task}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return f"USER: {task}\nASSISTANT:"

def _normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.splitlines()[-1].strip() if s else ""
    s = re.sub(r"^(answer|final|result)\s*[:=-]\s*", "", s, flags=re.I)
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return m.group(0) if m else (s if s else "⊥")

def apply_batch(llm_or_model, tok, engine: str, z_list: List[str], x_list: List[str],
                apply_max_new_tokens: int, bs: int=128, desc: str="APPLY") -> List[str]:
    assert len(z_list)==len(x_list), "Z and X length mismatch in APPLY"
    outs=[]
    total=len(z_list)
    if engine=="vllm":
        from vllm import SamplingParams
        sp=SamplingParams(temperature=0.0, max_tokens=apply_max_new_tokens, stop=[], detokenize=True)
        # build prompts lazily per chunk to reduce peak RAM
        with timed(f"{desc} vLLM total={total}", log_mem=False):
            pbar = tqdm(total=total, desc=desc, leave=False)
            for i in range(0,total,bs):
                chunkN=min(bs, total-i)
                prompts=[build_apply_prompt(tok, z_list[i+j], x_list[i+j]) for j in range(chunkN)]
                t0=time.perf_counter()
                gens=llm_or_model.generate(prompts, sp, use_tqdm=False)
                dt=time.perf_counter()-t0
                for g in gens:
                    text=g.outputs[0].text if g.outputs else ""
                    outs.append(_normalize_answer(text))
                pbar.set_postfix_str(f"chunk={chunkN} thr={chunkN/max(dt,1e-6):.1f}/s")
                pbar.update(chunkN)
            pbar.close()
    else:
        model=llm_or_model
        device="cuda" if torch.cuda.is_available() else "cpu"
        with timed(f"{desc} HF total={total}", log_mem=False):
            pbar = tqdm(total=total, desc=desc, leave=False)
            for i in range(0,total,bs):
                chunkN=min(bs, total-i)
                chunk=[build_apply_prompt(tok, z_list[i+j], x_list[i+j]) for j in range(chunkN)]
                enc=tok(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
                t0=time.perf_counter()
                with torch.no_grad():
                    gen_ids=model.generate(**enc, max_new_tokens=apply_max_new_tokens, do_sample=False)
                texts=tok.batch_decode(gen_ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                dt=time.perf_counter()-t0
                outs.extend([_normalize_answer(t) for t in texts])
                pbar.set_postfix_str(f"chunk={chunkN} thr={chunkN/max(dt,1e-6):.1f}/s")
                pbar.update(chunkN)
            pbar.close()
    assert len(outs)==total, f"APPLY produced {len(outs)} outputs but expected {total}"
    return outs

# ---------------- Probe generation (AUTO) ----------------

def _rand_int_with_digits(d: int) -> int:
    lo = 10**(d-1)
    hi = 10**d - 1
    return random.randint(lo, hi)

def gen_probe(kind: str, digits: int) -> str:
    if kind=="add":
        a=_rand_int_with_digits(digits); b=_rand_int_with_digits(digits)
        return f"Add {a} and {b}."
    if kind=="sub":
        a=_rand_int_with_digits(digits); b=_rand_int_with_digits(digits)
        if b>a: a,b=b,a
        return f"Subtract {b} from {a}."
    if kind=="mul":
        a=_rand_int_with_digits(digits); b=_rand_int_with_digits(digits)
        return f"Multiply {a} and {b}."
    if kind=="lcs":
        L=max(2, digits)
        import string
        s1="".join(random.choice(string.ascii_uppercase[:6]) for _ in range(L))
        s2="".join(random.choice(string.ascii_uppercase[:6]) for _ in range(L))
        return f"What is the length of the LCS of {s1} and {s2}?"
    if kind=="knap":
        n=max(4, min(10, digits//2 + 2))
        weights=[random.randint(1, 9) for _ in range(n)]
        values=[random.randint(1, 20) for _ in range(n)]
        cap=max(5, sum(weights)//2)
        items=", ".join(f"(w={w}, v={v})" for w,v in zip(weights,values))
        return f"0/1 knapSack. Items: {items}. Capacity={cap}. Output the maximum total value."
    if kind=="rod":
        n=max(5, min(12, digits + 3))
        prices=[random.randint(1, 10)+i for i in range(n)]
        plist=", ".join(str(p) for p in prices)
        return f"Rod cutting. Prices for lengths 1..{n}: {plist}. Compute max revenue for rod length {n}."
    if kind=="ilp_assign":
        m=max(3, min(6, digits//2+2))
        n=max(3, min(6, digits//2+2))
        C=[[random.randint(1,9) for _ in range(n)] for _ in range(m)]
        lines=[" ".join(map(str,row)) for row in C]
        mat="; ".join(lines)
        return f"Assignment ILP: cost matrix {m}x{n} = [{mat}]. Min total cost. Return the minimum cost."
    if kind=="ilp_prod":
        m=max(2, min(5, digits//4+2))
        a=[random.randint(1,9) for _ in range(m)]
        b=[random.randint(10,30) for _ in range(m)]
        profit=[random.randint(5,20) for _ in range(m)]
        return ("Production planning ILP: maximize sum p_i x_i s.t. a_i x_i <= b_i, x_i integer>=0. "
                f"a={a}, b={b}, p={profit}. Return the optimal objective.")
    if kind=="ilp_partition":
        n=max(6, min(14, digits+4))
        arr=[random.randint(1, 20) for _ in range(n)]
        return f"Partition ILP: Given set {arr}, return the minimum difference between two subset sums."
    return f"Task={kind}, digits={digits}. Provide the final numeric answer."

def build_probes_auto(kinds: List[str], digits_list: List[int], total_per_bucket: int,
                      per_kind_cap: int, seed: int=0) -> Dict[str, List[str]]:
    random.seed(seed)
    probes_all=[]
    per_kind_counts={k:0 for k in kinds}
    combos=[(k,d) for k in kinds for d in digits_list]
    i=0
    while len(probes_all) < total_per_bucket:
        k,d = combos[i % len(combos)]
        if per_kind_counts[k] < per_kind_cap:
            probes_all.append(gen_probe(k,d))
            per_kind_counts[k]+=1
        i+=1
        if all(per_kind_counts[k]>=per_kind_cap for k in kinds): break
    return {"all": probes_all}

# ---------------- MI helpers ----------------

def _normalize_vec(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64); s = v.sum()
    if not np.isfinite(s) or s <= 0: return np.full_like(v, 1.0 / max(1, v.size), dtype=np.float64)
    return v / (s + eps)

def _logsumexp(a: np.ndarray, axis=None):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def pz_given_sem_logweights(df_bucket_sem: pd.DataFrame, support: List[str], alpha=1e-3) -> np.ndarray:
    per_seq_logs: Dict[str, List[float]] = {}
    for r in df_bucket_sem.itertuples(index=False):
        per_seq_logs.setdefault(r.seq_id, []).append(float(r.sum_logp_joint))
    logvec = np.full(len(support), -np.inf, dtype=np.float64)
    idx = {sid: i for i, sid in enumerate(support)}
    for sid, lst in per_seq_logs.items():
        if sid in idx:
            logvec[idx[sid]] = _logsumexp(np.array(lst, dtype=np.float64))
    m = np.max(logvec)
    exps = np.exp(logvec - m, where=np.isfinite(logvec))
    exps = np.where(np.isfinite(exps), exps, 0.0)
    exps += float(alpha)
    return _normalize_vec(exps)

def mi_for_bucket_semantics(df_b: pd.DataFrame, sem_ids_all: List[str], alpha=1e-3, prior="uniform"):
    support = sorted(df_b["seq_id"].unique().tolist())
    if not support: return float("nan"), float("nan"), float("nan"), 0, 0
    present = sorted(df_b["sem_id"].unique().tolist())
    valid = [s for s in sem_ids_all if s in present]
    if len(valid) < 2: return float("nan"), float("nan"), float("nan"), len(support), len(valid)
    P=[]; counts=[]
    for sid in valid:
        df_s = df_b[df_b["sem_id"]==sid]
        P.append(pz_given_sem_logweights(df_s, support, alpha=alpha))
        counts.append(len(df_s))
    P=np.stack(P, axis=0)
    pS = _normalize_vec(np.asarray(counts, dtype=np.float64)) if prior=="weighted" else np.full(len(valid), 1.0/len(valid))
    pz = _normalize_vec((pS[:,None]*P).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(P, pz[None,:], out=np.ones_like(P), where=(pz[None,:]>0))
        terms = pS[:,None]*P*np.log(np.clip(ratio, 1e-300, None), dtype=np.float64)
        mi_nats=float(terms.sum())
    H_S = -float((pS*np.log(np.clip(pS,1e-300,None), dtype=np.float64)).sum())
    H_c = H_S - mi_nats
    log2=math.log(2.0)
    return mi_nats/log2, H_S/log2, H_c/log2, len(support), len(valid)

# ---------------- Semantics cache ----------------

class SemCache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.mem: Dict[Tuple[str,str], str] = {}  # (z_sha, R_sha) -> sem_id
        if path and os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    for line in f:
                        zsha, rsha, sem = json.loads(line)
                        self.mem[(zsha, rsha)] = sem
                LOG.info("Loaded semantics cache entries: %d", len(self.mem))
            except Exception as e:
                LOG.warning("Failed to read cache %s: %s", path, e)

    def get(self, z_sha: str, r_sha: str) -> Optional[str]:
        return self.mem.get((z_sha, r_sha))

    def put(self, z_sha: str, r_sha: str, sem_id: str):
        self.mem[(z_sha, r_sha)] = sem_id
        if self.path:
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps([z_sha, r_sha, sem_id])+"\n")
            except Exception as e:
                LOG.warning("Failed to write cache line: %s", e)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--engine", choices=["vllm","hf"], default="vllm")
    ap.add_argument("--score_batch_size", type=int, default=64)

    # logging/debug
    ap.add_argument("--log_level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    ap.add_argument("--log_mem", action="store_true")
    ap.add_argument("--limit_pairs", type=int, default=0, help="Debug: cap number of pairs processed overall")
    ap.add_argument("--limit_groups", type=int, default=0, help="Debug: cap number of (rep,bucket) groups")
    ap.add_argument("--limit_probes", type=int, default=0, help="Debug: cap number of probes used")

    # bucket + MI
    ap.add_argument("--bucket_kind", choices=["digits","promptlen","all"], default="digits")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--sem_prior", choices=["uniform","weighted"], default="uniform")
    ap.add_argument("--log_tb", action="store_true")

    # engines
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_dtype", type=str, default=None)
    ap.add_argument("--gpu_memory_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto")

    # scoring Z
    ap.add_argument("--include_prompt_ll", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)

    # APPLY
    ap.add_argument("--apply_max_new_tokens", type=int, default=48)

    # PROBES (AUTO)
    ap.add_argument("--kinds", type=str, default="")
    ap.add_argument("--digits", type=str, default="")
    ap.add_argument("--probes_total_per_bucket", type=int, default=1024)
    ap.add_argument("--probes_per_kind", type=int, default=128)
    ap.add_argument("--probe_seed", type=int, default=0)
    ap.add_argument("--limit_semantics", type=int, default=0,
                help="Cap number of rationales per (rep,bucket) for APPLY/semantics")


    # cache
    ap.add_argument("--sem_cache_path", type=str, default=None, help="jsonl cache; default out_dir/sem_cache.jsonl")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.sem_cache_path is None:
        args.sem_cache_path = os.path.join(args.out_dir, "sem_cache.jsonl")

    setup_logging(args.log_level)
    LOG.info("Args: %s", vars(args))

    # load pairs
    with timed("Load TensorBoard pairs", args.log_mem):
        pairs = load_tb_pairs(args.tb_dir)
    if not pairs:
        raise RuntimeError("No (prompt, raw_json/full) pairs parsed from TB.")
    if args.limit_pairs > 0:
        pairs = pairs[:args.limit_pairs]
    LOG.info("Total paired examples: %d %s", len(pairs), try_mem_stats(args.log_mem))

    # engine
    if args.engine=="vllm":
        tok, llm = _vllm_load(args.model, args.tensor_parallel_size, args.vllm_dtype,
                              args.gpu_memory_util, args.max_model_len)
        engine=("vllm", llm)
    else:
        tok, hf_model = _hf_load(args.model, args.device, args.dtype)
        engine=("hf", hf_model)

    # work table
    work=[]
    for p in pairs:
        prompt_str = to_chat_prompt(tok, p["prompt"])
        z_text = p["raw_json"]
        seq_id = hashlib.sha1(z_text.encode("utf-8")).hexdigest()[:16]
        work.append({
            "model": p["model"], "rep": p["rep"], "digits": int(p["digits"] if p["digits"] is not None else 0),
            "theta": p["theta"], "idx": int(p["idx"]),
            "prompt_str": prompt_str, "z_text": z_text,
            "z_sha": hashlib.sha1(z_text.encode("utf-8")).hexdigest(),
            "seq_id": seq_id, "prompt_len": len(prompt_str)
        })
    df0=pd.DataFrame(work)
    df0["bucket"]=[assign_bucket(r, args.bucket_kind) for r in df0.to_dict(orient="records")]

    # dataset summary
    LOG.info("Rep counts: %s", df0["rep"].value_counts().to_dict())
    LOG.info("Bucket counts: %s", df0["bucket"].value_counts().to_dict())
    LOG.info("Theta counts (top 10): %s", df0["theta"].value_counts().head(10).to_dict())

    # infer kinds/digits
    present_kinds=sorted(set(df0["theta"].tolist()))
    present_digits=sorted(set([d for d in df0["digits"].tolist() if d is not None]))
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()] or present_kinds
    digits_list = [int(x) for x in args.digits.split(",") if x.strip()] or present_digits or [2,4,8,16,32]
    LOG.info("Probe kinds: %s", kinds)
    LOG.info("Probe digits: %s", digits_list)

    # build probes
    with timed("Build probes", args.log_mem):
        probes_map = build_probes_auto(kinds, digits_list, args.probes_total_per_bucket,
                                       args.probes_per_kind, seed=args.probe_seed)
    R_all = probes_map["all"]
    if args.limit_probes > 0:
        R_all = R_all[:args.limit_probes]
    assert len(R_all)>0, "Empty probe set"
    R_sha = hashlib.sha1(("\x1f".join(R_all)).encode("utf-8")).hexdigest()
    LOG.info("Probes total=%d R_sha=%s", len(R_all), R_sha)

    # semantics cache
    cache = SemCache(args.sem_cache_path)

    # semantics via APPLY (with caching) and scoring
    kind, eng = engine
    rows=[]
    grouped = list(df0.groupby(["rep","bucket"]))
    if args.limit_groups>0:
        grouped = grouped[:args.limit_groups]
    LOG.info("Groups to process: %d", len(grouped))

    for (rep, bucket), df_grp in tqdm(grouped, total=len(grouped), desc="groups"):
        LOG.info("Group rep=%s bucket=%s size=%d", rep, bucket, len(df_grp))
        if args.limit_semantics > 0 and len(df_grp) > args.limit_semantics:
            df_grp = df_grp.sample(n=args.limit_semantics, random_state=0)
            LOG.info("Subsampled to %d rationales for semantics", len(df_grp))
        # Build per-z semantics, using cache
        z_list = df_grp["z_text"].tolist()
        z_sha_list = df_grp["z_sha"].tolist()

        # Determine which z need APPLY
        need_idx = []
        cached_sem = {}
        for i, zsha in enumerate(z_sha_list):
            sem_cached = cache.get(zsha, R_sha)
            if sem_cached is None:
                need_idx.append(i)
            else:
                cached_sem[i] = sem_cached

        LOG.info("Semantics cache hits=%d miss=%d", len(cached_sem), len(need_idx))

        sem_ids = [None] * len(z_list)
        # Fill cached
        for i, sem in cached_sem.items():
            sem_ids[i] = sem

        # Compute missing via APPLY in large batch
        if need_idx:
            Z_need = [z_list[i] for i in need_idx]
            X_need = [x for _ in range(len(Z_need)) for x in R_all]
            Z_expand = [z for z in Z_need for _ in range(len(R_all))]
            assert len(Z_expand)==len(X_need), "Expansion mismatch"
            desc = f"APPLY[{rep}|{bucket}|need={len(Z_need)}|R={len(R_all)}]"
            with timed(desc, args.log_mem):
                ys = apply_batch(eng, tok, kind, Z_expand, X_need, args.apply_max_new_tokens,
                                 bs=max(32, args.batch_size), desc=desc)
            assert len(ys)==len(Z_expand), "APPLY outputs size mismatch"
            # collapse
            for j, i in enumerate(need_idx):
                vec = ys[j*len(R_all):(j+1)*len(R_all)]
                assert len(vec)==len(R_all), "Collapse slice mismatch"
                sig = "\x1f".join([v if v!="" else "⊥" for v in vec])
                sem = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
                sem_ids[i] = sem
                cache.put(z_sha_list[i], R_sha, sem)

        assert all(s is not None for s in sem_ids), "Unset sem_id after APPLY"
        df_grp = df_grp.copy()
        df_grp["sem_id"] = sem_ids

        # score Z log-likelihoods
        batch_prompts=df_grp["prompt_str"].tolist()
        batch_targets=df_grp["z_text"].tolist()
        assert len(batch_prompts)==len(batch_targets)==len(sem_ids)

        if kind=="vllm":
            scored = _vllm_score_batched(eng, tok, batch_prompts, batch_targets,
                             args.include_prompt_ll, args.score_batch_size,
                             desc=f"{rep}|{bucket}")
            assert len(scored)==len(batch_prompts)
            df_grp["sum_logp_joint"]  =[s["sum_logp_joint"]  for s in scored]
            df_grp["n_tokens_joint"]  =[s["n_tokens_joint"]  for s in scored]
            df_grp["sum_logp_prompt"] =[s["sum_logp_prompt"] for s in scored]
            df_grp["n_tokens_prompt"] =[s["n_tokens_prompt"] for s in scored]
            df_grp["sum_logp_cont"]   =[s["sum_logp_cont"]   for s in scored]
            df_grp["n_tokens_cont"]   =[s["n_tokens_cont"]   for s in scored]
        else:
            LOG.info("SCORE HF rep=%s bucket=%s N=%d", rep, bucket, len(batch_prompts))
            sums=[]; toks=[]; sp=[]; np_=[]; sc=[]; nc=[]
            pbar = tqdm(total=len(batch_prompts), desc=f"SCORE[{rep}|{bucket}]", leave=False)
            for p_str, tgt in zip(batch_prompts, batch_targets):
                o=_score_z_hf(tok, eng, args.device, p_str, tgt, args.include_prompt_ll)
                sums.append(o["sum_logp_joint"]); toks.append(o["n_tokens_joint"])
                sp.append(o["sum_logp_prompt"]); np_.append(o["n_tokens_prompt"])
                sc.append(o["sum_logp_cont"]);   nc.append(o["n_tokens_cont"])
                if len(sums)%32==0: pbar.update(32)
            left = len(batch_prompts) - (len(sums)//32)*32
            if left>0: pbar.update(left)
            pbar.close()
            df_grp["sum_logp_joint"]=sums; df_grp["n_tokens_joint"]=toks
            df_grp["sum_logp_prompt"]=sp;  df_grp["n_tokens_prompt"]=np_
            df_grp["sum_logp_cont"]=sc;    df_grp["n_tokens_cont"]=nc

        rows.append(df_grp)

    df=pd.concat(rows, ignore_index=True)
    LOG.info("Completed semantics+scoring for %d items", len(df))

    # MI per (model, rep, bucket)
    tb = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb_mi_semantics")) if args.log_tb else None
    out=[]
    for (model_name, rep), df_mr in df.groupby(["model","rep"]):
        sem_all=sorted(df_mr["sem_id"].unique().tolist())
        for bucket, df_b in df_mr.groupby("bucket"):
            n_b=len(df_b)
            with timed(f"MI rep={rep} bucket={bucket} N={n_b}", args.log_mem):
                mi_bits, H_bits, Hc_bits, supp, S = mi_for_bucket_semantics(df_b, sem_all, alpha=args.alpha, prior=args.sem_prior)
            rec={"model":model_name, "rep":rep, "bucket":bucket,
                 "MI_bits_Z_semantics":float(mi_bits),
                 "H_S_bits":float(H_bits), "H_S_given_Z_bits":float(Hc_bits),
                 "support_size":int(supp), "num_semantics_present":int(S),
                 "include_prompt_ll":bool(args.include_prompt_ll),
                 "bucket_kind":args.bucket_kind, "engine":args.engine,
                 "N_bucket":int(n_b)}
            out.append(rec)
            if tb is not None and np.isfinite(mi_bits):
                tb.add_scalar(f"{model_name}/{rep}/MI_bits_semantics/{bucket}", mi_bits)
                tb.add_scalar(f"{model_name}/{rep}/H_S_bits/{bucket}", H_bits)
                tb.add_scalar(f"{model_name}/{rep}/H_S_given_Z_bits/{bucket}", Hc_bits)
        df_rep=[r for r in out if r["model"]==model_name and r["rep"]==rep and np.isfinite(r["MI_bits_Z_semantics"])]
        avg_mi=float(np.mean([r["MI_bits_Z_semantics"] for r in df_rep])) if df_rep else float("nan")
        LOG.info("[MI] %30s | %4s | avg I(Z;S_R|%s) = %.4f bits over %d buckets",
                 model_name, rep, args.bucket_kind, avg_mi, len(df_rep))
        if tb is not None and np.isfinite(avg_mi):
            tb.add_scalar(f"{model_name}/{rep}/MI_bits_semantics/avg_over_buckets", avg_mi)

    with open(os.path.join(args.out_dir,"mi_semantics_summary.json"),"w") as f:
        json.dump(out,f,indent=2)
    LOG.info("Wrote summary json with %d records", len(out))

    # dump per-example table
    try:
        path = os.path.join(args.out_dir,"mi_semantics_items.parquet")
        df.to_parquet(path, index=False)
    except Exception:
        path = os.path.join(args.out_dir,"mi_semantics_items.csv")
        df.to_csv(path, index=False)
    LOG.info("Wrote items table: %s", path)

    if tb is not None: tb.flush(); tb.close()

if __name__=="__main__":
    main()
