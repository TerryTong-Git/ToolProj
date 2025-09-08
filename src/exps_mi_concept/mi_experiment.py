#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mutual Information I(Z; θ | bucket, r) with vLLM batching.

Key changes:
- --engine {vllm,hf} (default: vllm)
- Batched prefill scoring with vLLM via prompt_logprobs on (prompt+target).
- Token-exact split to isolate continuation log-likelihood.

Requires:
  pip install vllm>=0.8  transformers tensorboard numpy pandas tqdm

Notes:
- We set SamplingParams(max_tokens=1, prompt_logprobs=1). vLLM does not
  universally support max_tokens=0; one token decode is negligible.
"""
from __future__ import annotations
import os, re, json, math, argparse, hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ----------------------------- TB reading -----------------------------

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
        prompts, raws = {}, {}
        for tag in tags:
            m = TAG_RE.match(tag); 
            if not m: 
                continue
            digits = m['digits']; digits_int = int(digits) if digits is not None else None
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

        for k in sorted(set(prompts) & set(raws)):
            model, rep, digits_int, theta, idx = k
            pairs.append(dict(
                run=run_name, model=model, rep=rep, digits=digits_int, theta=theta, idx=idx,
                prompt=prompts[k], raw_json=raws[k]
            ))
    return pairs

# ----------------------------- Bucketing -----------------------------

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

# ----------------------------- Prob helpers & MI -----------------------------

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(v, 1.0 / max(1, v.size), dtype=np.float64)
    return v / (s + eps)

def _logsumexp(a: np.ndarray, axis=None):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def pz_given_theta_logweights(df_bucket_theta: pd.DataFrame, support: List[str], alpha=1e-3) -> np.ndarray:
    per_seq_logs: Dict[str, List[float]] = {}
    for r in df_bucket_theta.itertuples(index=False):
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
    return _normalize(exps)

def mi_for_bucket(df_b: pd.DataFrame, thetas_all: List[str], alpha=1e-3, prior="uniform") -> Tuple[float,float,float,int,int]:
    support = sorted(df_b["seq_id"].unique().tolist())
    if not support:
        return float("nan"), float("nan"), float("nan"), 0, 0
    present = sorted(df_b["theta"].unique().tolist())
    valid = [t for t in thetas_all if t in present]
    if len(valid) < 2:
        return float("nan"), float("nan"), float("nan"), len(support), len(valid)

    P = []
    counts = []
    for th in valid:
        df_th = df_b[df_b["theta"] == th]
        P.append(pz_given_theta_logweights(df_th, support, alpha=alpha))
        counts.append(len(df_th))
    P = np.stack(P, axis=0)  # [T, Z]

    if prior == "weighted":
        pth = _normalize(np.asarray(counts, dtype=np.float64))
    else:
        pth = np.full(len(valid), 1.0 / len(valid), dtype=np.float64)

    pz = _normalize((pth[:, None] * P).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(P, pz[None, :], out=np.ones_like(P), where=(pz[None, :] > 0))
        terms = pth[:, None] * P * np.log(np.clip(ratio, 1e-300, None), dtype=np.float64)
        mi_nats = float(terms.sum())
    H_theta = -float((pth * np.log(np.clip(pth, 1e-300, None), dtype=np.float64)).sum())
    H_c = H_theta - mi_nats
    log2 = math.log(2.0)
    return mi_nats/log2, H_theta/log2, H_c/log2, len(support), len(valid)

# ----------------------------- Prompt encoding -----------------------------

def has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None)

def to_chat_prompt(tok, user_text: str) -> str:
    if has_chat_template(tok):
        msgs = [{"role": "user", "content": user_text}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return f"USER: {user_text}\nASSISTANT:"

# ----------------------------- HF path (fallback) -----------------------------

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

@torch.no_grad()
def _hf_per_token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    input_ids = input_ids.long()
    out = model(input_ids=input_ids)
    logits = out.logits  # [1, L, V]
    if logits.size(1) <= 1:
        return torch.empty((0,), dtype=logits.dtype, device=logits.device)
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, L-1, V]
    tgt = input_ids[:, 1:]  # [1, L-1]
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
    return token_logp[0]  # [L-1]

@torch.no_grad()
def _hf_score_example(tok, model, prompt_str: str, target_text: str, include_prompt_ll: bool, device: str) -> Dict[str, Any]:
    # Tokenize with HF for HF path only
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

# ----------------------------- vLLM path -----------------------------

def _vllm_load(model_name: str, tensor_parallel_size: int, dtype: Optional[str], gpu_mem_util: float, max_model_len: Optional[int], seed):
    from vllm import LLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,  # None => auto
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        download_dir='../models',
        seed=seed
    )
    tok = llm.get_tokenizer()
    return tok, llm

def _safe_len(x):
    try: return len(x)
    except Exception: return 0

def _chosen_logprob_from_entry(entry, token_id: int, tok) -> Optional[float]:
    # vLLM may return: dict(str->obj), dict[int]->obj, list[obj], where obj has fields {id|token_id, logprob} or is float
    if entry is None: 
        return None
    # direct numeric
    if isinstance(entry, (float, int)):
        return float(entry)
    # mapping
    if isinstance(entry, dict):
        # try by id
        if token_id in entry:
            v = entry[token_id]; 
            if isinstance(v, dict): 
                return float(v.get("logprob")) if "logprob" in v else None
            lp = getattr(v, "logprob", None)
            return float(lp) if lp is not None else None
        # try by token string
        try:
            s = tok.decode([token_id], skip_special_tokens=False)
            if s in entry:
                v = entry[s]
                if isinstance(v, dict): 
                    return float(v.get("logprob")) if "logprob" in v else None
                lp = getattr(v, "logprob", None)
                return float(lp) if lp is not None else None
        except Exception:
            pass
        # scan values
        for v in entry.values():
            vid = None
            if isinstance(v, dict):
                vid = v.get("id") or v.get("token_id")
                lp = v.get("logprob")
                if vid == token_id and lp is not None:
                    return float(lp)
            else:
                vid = getattr(v, "id", None) or getattr(v, "token_id", None)
                lp = getattr(v, "logprob", None)
                if vid == token_id and lp is not None:
                    return float(lp)
        return None
    # list of candidates
    if isinstance(entry, (list, tuple)):
        for v in entry:
            if isinstance(v, dict):
                vid = v.get("id") or v.get("token_id")
                if vid == token_id:
                    lp = v.get("logprob")
                    return float(lp) if lp is not None else None
            else:
                vid = getattr(v, "id", None) or getattr(v, "token_id", None)
                lp = getattr(v, "logprob", None)
                if vid == token_id and lp is not None:
                    return float(lp)
        return None
    # object with fields
    lp = getattr(entry, "logprob", None)
    return float(lp) if lp is not None else None

def _align_prompt_logprobs(prompt_token_ids: List[int], prompt_logprobs_obj) -> List[Optional[Any]]:
    """
    Align per-token logprobs to token positions.
    Some vLLM versions return len=Len, others Len-1 (missing first).
    Normalize to length==len(prompt_token_ids) with [None] at index 0 if needed.
    """
    L = len(prompt_token_ids)
    pl = list(prompt_logprobs_obj) if prompt_logprobs_obj is not None else []
    if _safe_len(pl) == L: 
        return pl
    if _safe_len(pl) == L - 1:
        return [None] + list(pl)
    # fallback: pad/truncate
    if _safe_len(pl) < L:
        return [None] * (L - _safe_len(pl)) + list(pl)
    return list(pl)[:L]

def _vllm_score_batch(llm, tok, batch: List[Tuple[str, str]], include_prompt_ll: bool) -> List[Dict[str, Any]]:
    # Prepare full texts and token-length splits under SAME tokenizer
    prompts = []
    fulls   = []
    split_idxs = []  # token-index where continuation starts
    prompt_char_lens = []
    for prompt_str, target_text in batch:
        full_text = prompt_str + target_text
        # token counts
        prompt_tok_ids = tok(prompt_str, add_special_tokens=True)["input_ids"]
        full_tok_ids   = tok(full_text, add_special_tokens=True)["input_ids"]
        split = len(prompt_tok_ids)
        prompts.append(prompt_str); fulls.append(full_text)
        split_idxs.append(split)
        prompt_char_lens.append(len(prompt_str))

    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,            # trigger prefill; generation ignored
        logprobs=None,           # not needed
        prompt_logprobs=1,       # chosen token logprob included by API
        detokenize=True,
        skip_special_tokens=False,
    )
    outputs = llm.generate(fulls, sp, use_tqdm=False)

    out_rows: List[Dict[str, Any]] = []
    for i, out in enumerate(outputs):
        token_ids: List[int] = list(out.prompt_token_ids)
        pl_raw = out.prompt_logprobs  # shape ~ [len] or [len-1]
        pl_aligned = _align_prompt_logprobs(token_ids, pl_raw)

        chosen_lp: List[Optional[float]] = []
        for pos, entry in enumerate(pl_aligned):
            # chosen token is token_ids[pos]
            lp = _chosen_logprob_from_entry(entry, token_ids[pos], tok)
            chosen_lp.append(lp)

        # Sum over spans
        split = split_idxs[i]
        # skip positions with None (first token typically)
        def _span_sum(lo, hi):
            s = 0.0; n = 0
            for j in range(max(lo, 0), min(hi, len(chosen_lp))):
                lp = chosen_lp[j]
                if lp is None: 
                    continue
                s += float(lp); n += 1
            return s, n

        sum_prompt = 0.0; n_prompt = 0
        if include_prompt_ll:
            sum_prompt, n_prompt = _span_sum(0, split)

        sum_cont, n_cont = _span_sum(split, len(chosen_lp))
        total_sum = sum_prompt + sum_cont
        total_tok = n_prompt + n_cont

        out_rows.append(dict(
            sum_logp_joint=total_sum, n_tokens_joint=total_tok,
            sum_logp_prompt=sum_prompt, n_tokens_prompt=n_prompt,
            sum_logp_cont=sum_cont,   n_tokens_cont=n_cont,
            prompt_len_chars=prompt_char_lens[i],
        ))
    return out_rows

# ----------------------------- Main -----------------------------

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--seed", required=True, type=int)


    # Engine selection
    ap.add_argument("--engine", choices=["vllm","hf"], default="vllm")

    # Shared
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--include_prompt_ll", action="store_true")
    ap.add_argument("--bucket_kind", choices=["digits","promptlen","all"], default="digits")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--theta_prior", choices=["uniform","weighted"], default="uniform")
    ap.add_argument("--log_tb", action="store_true")

    # HF flags
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto")

    # vLLM flags
    ap.add_argument("--tensor_parallel_size", type=int, default=8)
    ap.add_argument("--vllm_dtype", type=str, default="float16", help="e.g., float16, bfloat16, float32, auto(None)")
    ap.add_argument("--gpu_memory_util", type=float, default=0.5)
    ap.add_argument("--max_model_len", type=int, default=None)

    args = ap.parse_args()
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[mi] reading TB from: {args.tb_dir}")
    pairs = load_tb_pairs(args.tb_dir)
    if not pairs:
        raise RuntimeError("No (prompt, raw_json/full) pairs parsed from TB.")
    print(f"[mi] Total paired examples: {len(pairs)}")

    # Load engine + tokenizer
    if args.engine == "hf":
        tok, hf_model = _hf_load(args.model, args.device, args.dtype)
        engine = ("hf", hf_model)
    else:
        tok, vllm_llm = _vllm_load(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.vllm_dtype,
            gpu_mem_util=args.gpu_memory_util,
            max_model_len=args.max_model_len,
            seed=args.seed
        )
        engine = ("vllm", vllm_llm)

    # Build work items
    work: List[Dict[str, Any]] = []
    for p in pairs:
        prompt_str = to_chat_prompt(tok, p["prompt"])
        target_text = p["raw_json"]
        seq_id = hashlib.sha1(target_text.encode("utf-8")).hexdigest()[:16]
        work.append({
            "model": p["model"], "rep": p["rep"],
            "digits": int(p["digits"] if p["digits"] is not None else 0),
            "theta": p["theta"], "idx": int(p["idx"]),
            "prompt_str": prompt_str, "target_text": target_text,
            "seq_id": seq_id
        })

    # Scoring in batches
    rows = []
    B = max(1, int(args.batch_size))
    kind, eng = engine
    for i in tqdm(range(0, len(work), B), desc=f"scoring[{kind}]"):
        chunk = work[i:i+B]
        if kind == "hf":
            for ex in chunk:
                tf = _hf_score_example(tok, eng, ex["prompt_str"], ex["target_text"], args.include_prompt_ll, args.device)
                rows.append({
                    "model": ex["model"], "rep": ex["rep"], "digits": ex["digits"], "theta": ex["theta"], "idx": ex["idx"],
                    "prompt_len": len(ex["prompt_str"]), "seq_id": ex["seq_id"],
                    **tf
                })
        else:
            batch_in = [(ex["prompt_str"], ex["target_text"]) for ex in chunk]
            scored = _vllm_score_batch(eng, tok, batch_in, args.include_prompt_ll)
            for ex, tf in zip(chunk, scored):
                rows.append({
                    "model": ex["model"], "rep": ex["rep"], "digits": ex["digits"], "theta": ex["theta"], "idx": ex["idx"],
                    "prompt_len": tf.get("prompt_len_chars", len(ex["prompt_str"])),
                    "seq_id": ex["seq_id"],
                    "sum_logp_joint": tf["sum_logp_joint"], "n_tokens_joint": tf["n_tokens_joint"],
                    "sum_logp_prompt": tf["sum_logp_prompt"], "n_tokens_prompt": tf["n_tokens_prompt"],
                    "sum_logp_cont": tf["sum_logp_cont"], "n_tokens_cont": tf["n_tokens_cont"],
                })

    df = pd.DataFrame(rows)
    df["bucket"] = [assign_bucket(r, args.bucket_kind) for r in df.to_dict(orient="records")]

    tb = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb_mi")) if args.log_tb else None
    out_records = []
    for (model_name, rep), df_mr in df.groupby(["model", "rep"]):
        thetas_all = sorted(df_mr["theta"].unique().tolist())
        for bucket, df_b in df_mr.groupby("bucket"):
            mi_bits, H_bits, Hc_bits, supp, T = mi_for_bucket(
                df_b, thetas_all, alpha=args.alpha, prior=args.theta_prior
            )
            rec = {
                "model": model_name, "rep": rep, "bucket": bucket,
                "MI_bits": float(mi_bits), "H_theta_bits": float(H_bits), "H_theta_given_Z_bits": float(Hc_bits),
                "support_size": int(supp), "num_thetas_present": int(T),
                "include_prompt_ll": bool(args.include_prompt_ll),
                "bucket_kind": args.bucket_kind,
                "engine": args.engine
            }
            out_records.append(rec)
            if tb is not None and np.isfinite(mi_bits):
                tb.add_scalar(f"{model_name}/{rep}/MI_bits/{bucket}", mi_bits)
                tb.add_scalar(f"{model_name}/{rep}/H_theta_bits/{bucket}", H_bits)
                tb.add_scalar(f"{model_name}/{rep}/H_theta_given_Z_bits/{bucket}", Hc_bits)

        df_rep = [r for r in out_records if r["model"]==model_name and r["rep"]==rep and np.isfinite(r["MI_bits"])]
        avg_mi = float(np.mean([r["MI_bits"] for r in df_rep])) if df_rep else float("nan")
        print(f"[mi] {model_name:>30s} | {rep:>4s} | avg I(Z;theta | {args.bucket_kind}) = {avg_mi:.4f} bits over {len(df_rep)} buckets")
        if tb is not None and np.isfinite(avg_mi):
            tb.add_scalar(f"{model_name}/{rep}/MI_bits/avg_over_buckets", avg_mi)

    out_json = os.path.join(args.out_dir, "mi_bucket_summary.json")
    with open(out_json, "w") as f:
        json.dump(out_records, f, indent=2)
    print(f"[mi] wrote {out_json}")
    if tb is not None:
        tb.flush(); tb.close()

if __name__ == "__main__":
    main()
