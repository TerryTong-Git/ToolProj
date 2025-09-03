#!/usr/bin/env python3
"""
Compute I(Z; theta | bucket, r) with Z = CoT text (JSON), theta ∈ {add,sub,mul,lcs,knap,rod,ilp_assign,ilp_prod,ilp_partition},
bucket = relaxed X (digits | prompt length bins | all), r ∈ {nl, code}.

Reads prompt/raw_json from your performance TensorBoard run(s), re-scores them with a HF
model (teacher forcing), then estimates MI per bucket for NL vs Code.

Examples:
  # point to a single run directory (e.g., out_hf/tb/run_20250902_122649)
  python mi_bucket_experiment.py \
    --tb_dir ../out_hf/tb/run_20250902_122649 \
    --out_dir ../out_hf_mi_buckets \
    --model google/gemma-2-9b-it \
    --bucket_kind digits --batch_size 8 --include_prompt_ll --log_tb

  # or point to the *root* tb directory (it will recurse all runs)
  python mi_bucket_experiment.py --tb_dir ../out_hf/tb ...
"""

from __future__ import annotations
import os, re, json, math, argparse, hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto import tensor_pb2

# ----------------------------- TB reading -----------------------------

# --- replace your TAG_RE + helpers with this ---

# allow both with/without digits, both prefix/suffix "text_summary", and slashed model names
_THETA = r"(add|sub|mul|mix|lcs|knap|rod|ilp_assign|ilp_prod|ilp_partition)"
_END   = r"(prompt|raw_json|full)"
TAG_RE = re.compile(
    rf'^(?:text_summary/)?'                           # optional TF text prefix
    rf'(?P<model>.+?)/(?P<rep>nl|code)/'              # model (greedy), rep
    rf'(?:d(?P<digits>\d+)/)?'                        # OPTIONAL digits segment
    rf'(?P<theta>{_THETA})/i(?P<idx>\d+)/'            # theta + idx
    rf'(?P<leaf>{_END})(?:/text_summary)?$'           # leaf + optional suffix
)

def _strip_md_block(s: str) -> str:
    # Pull content out of the fenced block we wrote to TB
    m = re.search(r"```(?:[a-zA-Z0-9]+)?\s*\n([\s\S]*?)\n```", s, flags=re.S)
    return m.group(1).strip() if m else s.strip()

def _tb_text_from_event(ev) -> str:
    tp = ev.tensor_proto
    if tp.string_val:  # preferred path
        val = tp.string_val[0]
        return val.decode("utf-8", "replace") if isinstance(val, (bytes, bytearray)) else str(val)
    if tp.tensor_content:  # fallback
        try:
            return tp.tensor_content.decode("utf-8", "replace")
        except Exception:
            pass
    return ""

# --- replace your load_tb_pairs with this ---

def _iter_event_accumulators(root: str):
    # accept a single run dir or a TB root containing runs
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
            m = TAG_RE.match(tag)
            if not m:
                continue
            # digits may be missing -> None
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

        for k in sorted(set(prompts) & set(raws)):
            model, rep, digits_int, theta, idx = k
            pairs.append(dict(
                run=run_name, model=model, rep=rep, digits=digits_int, theta=theta, idx=idx,
                prompt=prompts[k], raw_json=raws[k]
            ))
    return pairs

# ----------------------------- Tokenization / Scoring -----------------------------

def pick_device(flag: str) -> str:
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag

def pick_dtype(flag: str):
    if flag == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
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

def has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None)

def to_chat_prompt_ids(tok, user_text: str) -> Tuple[str, torch.Tensor]:
    if has_chat_template(tok):
        msgs = [{"role": "user", "content": user_text}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        prompt_str = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    else:
        prompt_str = f"USER: {user_text}\nASSISTANT:"
        prompt_ids = tok(prompt_str, return_tensors="pt", add_special_tokens=True).input_ids
    return prompt_str, prompt_ids

@torch.no_grad()
def per_token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    # input_ids: [1, L] (long)
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
def teacher_forced_sums(
    model, tokenizer, device,
    prompt_ids: torch.Tensor, target_text: str,
    include_prompt_ll: bool
) -> Dict[str, Any]:
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device).long()
    prompt_ids = prompt_ids.to(device).long()
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)

    sum_prompt = 0.0; n_prompt = 0
    if include_prompt_ll:
        lp_prompt = per_token_logprobs(model, prompt_ids)
        sum_prompt = float(lp_prompt.sum().item())
        n_prompt = int(lp_prompt.numel())

    lp_full = per_token_logprobs(model, full_ids)
    K = target_ids.shape[1]
    cont_logps = lp_full[-K:] if lp_full.numel() >= K else lp_full.new_empty((0,))
    sum_cont = float(cont_logps.sum().item()) if cont_logps.numel() > 0 else 0.0
    n_cont = int(cont_logps.numel())

    total_sum = sum_prompt + sum_cont
    total_tok = n_prompt + n_cont
    avg = total_sum / max(1, total_tok)
    ppl = math.exp(-avg) if total_tok > 0 else float("inf")
    return dict(
        sum_logp_joint=total_sum, n_tokens_joint=total_tok,
        sum_logp_prompt=sum_prompt, n_tokens_prompt=n_prompt,
        sum_logp_cont=sum_cont, n_tokens_cont=n_cont,
        avg_logp_joint=avg, ppl_joint=ppl
    )

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
    # accumulate per-seq log scores with logsumexp
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
    """
    Returns (mi_bits, H_theta_bits, H_theta_given_Z_bits, support_size, num_thetas_present)
    """
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

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True, type=str, help="Path to a TB run dir OR the tb root; we recurse.")
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=8)  # (kept for API symmetry; scoring is per-example here)
    ap.add_argument("--include_prompt_ll", action="store_true")
    ap.add_argument("--bucket_kind", choices=["digits","promptlen","all"], default="digits")
    ap.add_argument("--alpha", type=float, default=1e-3, help="Dirichlet smoothing for p(z|theta,bucket)")
    ap.add_argument("--theta_prior", choices=["uniform","weighted"], default="uniform")
    ap.add_argument("--log_tb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[mi] reading TB from: {args.tb_dir}")
    pairs = load_tb_pairs(args.tb_dir)
    if not pairs:
        raise RuntimeError("No (prompt, raw_json/full) pairs parsed from TB. "
                           "Point --tb_dir at a specific run dir or the tb root that contains event files.")
    print(f"[mi] Total paired examples: {len(pairs)}")

    # Model
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)
    print(f"[mi] Loading model on {device} (dtype={dtype}) …")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    eos_id, pad_id = pick_eos_pad_ids(tok)
    tok.pad_token_id = pad_id
    cache_dir = "../models"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        cache_dir=cache_dir
    ).eval()
    if device == "cpu":
        model.to("cpu")

    # Score all pairs
    rows = []
    for p in tqdm(pairs, desc="scoring"):
        prompt_str, prompt_ids = to_chat_prompt_ids(tok, p["prompt"])
        tf = teacher_forced_sums(model, tok, device, prompt_ids, p["raw_json"], include_prompt_ll=args.include_prompt_ll)
        seq_id = hashlib.sha1(p["raw_json"].encode("utf-8")).hexdigest()[:16]
        rows.append({
            "model": p["model"], "rep": p["rep"], "digits": int(p["digits"] if p["digits"] else 0), "theta": p["theta"], "idx": int(p["idx"]),
            "prompt_len": len(prompt_str), "seq_id": seq_id,
            "sum_logp_joint": tf["sum_logp_joint"], "n_tokens_joint": tf["n_tokens_joint"],
            "sum_logp_prompt": tf["sum_logp_prompt"], "n_tokens_prompt": tf["n_tokens_prompt"],
            "sum_logp_cont": tf["sum_logp_cont"], "n_tokens_cont": tf["n_tokens_cont"],
        })

    df = pd.DataFrame(rows)
    # Attach buckets
    df["bucket"] = [assign_bucket(r, args.bucket_kind) for r in df.to_dict(orient="records")]

    # MI per (model, rep, bucket)
    tb = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb_mi")) if args.log_tb else None
    out_records = []
    for (model_name, rep), df_mr in df.groupby(["model", "rep"]):
        thetas_all = sorted(df_mr["theta"].unique().tolist())
        for bucket, df_b in df_mr.groupby("bucket"):
            mi_bits, H_bits, Hc_bits, supp, T = mi_for_bucket(
                df_b, thetas_all, alpha=args.alpha, prior=args.theta_prior
            )
            out_records.append({
                "model": model_name, "rep": rep, "bucket": bucket,
                "MI_bits": float(mi_bits), "H_theta_bits": float(H_bits), "H_theta_given_Z_bits": float(Hc_bits),
                "support_size": int(supp), "num_thetas_present": int(T),
                "include_prompt_ll": bool(args.include_prompt_ll),
                "bucket_kind": args.bucket_kind,
            })
            if tb is not None and np.isfinite(mi_bits):
                tb.add_scalar(f"{model_name}/{rep}/MI_bits/{bucket}", mi_bits)
                tb.add_scalar(f"{model_name}/{rep}/H_theta_bits/{bucket}", H_bits)
                tb.add_scalar(f"{model_name}/{rep}/H_theta_given_Z_bits/{bucket}", Hc_bits)

        # summary over buckets
        df_rep = [r for r in out_records if r["model"]==model_name and r["rep"]==rep and np.isfinite(r["MI_bits"])]
        avg_mi = float(np.mean([r["MI_bits"] for r in df_rep])) if df_rep else float("nan")
        print(f"[mi] {model_name:>30s} | {rep:>4s} | avg I(Z;theta | {args.bucket_kind}) = {avg_mi:.4f} bits over {len(df_rep)} buckets")
        if tb is not None and np.isfinite(avg_mi):
            tb.add_scalar(f"{model_name}/{rep}/MI_bits/avg_over_buckets", avg_mi)

    # Write summary JSON
    out_json = os.path.join(args.out_dir, "mi_bucket_summary.json")
    with open(out_json, "w") as f:
        json.dump(out_records, f, indent=2)
    print(f"[mi] wrote {out_json}")

    if tb is not None:
        tb.flush(); tb.close()

if __name__ == "__main__":
    main()
