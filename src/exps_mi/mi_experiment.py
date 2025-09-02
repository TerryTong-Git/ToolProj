#!/usr/bin/env python3
from __future__ import annotations
import os, re, json, math, time, argparse, hashlib
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util
from cox.store import Store
from tqdm import tqdm

# --------------------- regex & helpers ---------------------

PROMPT_PROB_LINE = re.compile(r"Problem:\s*Compute:\s*(\d+)\s*([+\-*])\s*(\d+)", re.I)
FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n([\s\S]*?)\n```", re.M)
JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.M)  # first {...}
INT_RE = re.compile(r"[-+]?\d+")
CANON_THETA = {"+": "add", "-": "sub", "*": "mul"}

def sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def clip(s: str, n: int) -> str:
    if s is None: return ""
    return s if len(s) <= n else s[:max(0, n-3)] + "..."

def extract_codeblock(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    m = FENCE_RE.search(text)
    return m.group(1).strip() if m else text

def find_tb_runs(tb_root: str) -> List[str]:
    if not os.path.isdir(tb_root): return []
    has_events = any(fn.startswith("events.out") for fn in os.listdir(tb_root))
    if has_events: return [tb_root]
    out = []
    for name in sorted(os.listdir(tb_root)):
        p = os.path.join(tb_root, name)
        if os.path.isdir(p) and any(fn.startswith("events.out") for fn in os.listdir(p)):
            out.append(p)
    return out

# --------------------- HF scoring (teacher-forced) ---------------------

def pick_device(flag: str) -> str:
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag

def pick_dtype(flag: str):
    if flag == "auto":
        if torch.cuda.is_available():
            sup = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if sup else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[flag]

def _has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None)

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
    # Make absolutely sure indices are integer type for embedding lookup
    if input_ids.dtype != torch.long and input_ids.dtype != torch.int:
        input_ids = input_ids.long()
    out = model(input_ids=input_ids)
    logits = out.logits
    if logits.size(1) <= 1:
        return torch.empty((0,), dtype=logits.dtype, device=logits.device)
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:]
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return token_logp[0]


@torch.no_grad()
def teacher_forced_sums(model, tokenizer, device,
                        prompt_text: str, continuation_text: str,
                        include_prompt_ll: bool) -> Dict[str, Any]:
    # Tokenize
    _, prompt_ids = to_chat_ids(tokenizer, prompt_text)
    tgt_ids = tokenizer(continuation_text, add_special_tokens=False, return_tensors="pt").input_ids

    # Move to device and force integer dtype
    prompt_ids = prompt_ids.to(device)
    tgt_ids = tgt_ids.to(device)
    if prompt_ids.dtype != torch.long:
        prompt_ids = prompt_ids.long()
    if tgt_ids.dtype != torch.long:
        tgt_ids = tgt_ids.long()

    # Full sequence for teacher forcing
    full_ids = torch.cat([prompt_ids, tgt_ids], dim=1)
    if full_ids.dtype != torch.long:
        full_ids = full_ids.long()

    # Optional prompt log-prob
    sum_prompt = 0.0
    n_prompt = 0
    if include_prompt_ll:
        lp_prompt = per_token_logprobs(model, prompt_ids)
        sum_prompt = float(lp_prompt.sum().item()) if lp_prompt.numel() > 0 else 0.0
        n_prompt = int(lp_prompt.numel())

    # Continuation log-prob (conditioned on prompt)
    lp_full = per_token_logprobs(model, full_ids)
    K = tgt_ids.shape[1]
    cont_logps = lp_full[-K:] if lp_full.numel() >= K else lp_full.new_empty((0,))
    sum_cont = float(cont_logps.sum().item()) if cont_logps.numel() > 0 else 0.0
    n_cont = int(cont_logps.numel())

    total_sum = sum_prompt + sum_cont
    total_tok = n_prompt + n_cont
    return dict(
        sum_logp_prompt=sum_prompt, n_tokens_prompt=n_prompt,
        sum_logp_cot=sum_cont,     n_tokens_cot=n_cont,
        sum_logp_joint=total_sum,  n_tokens_joint=total_tok,
        avg_logp_joint=(total_sum / max(1, total_tok)) if total_tok > 0 else float("-inf"),
    )

# --------------------- MI builders ---------------------

def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(v, 1.0 / max(1, v.size), dtype=np.float64)
    return v / (s + eps)

def build_pz_given_theta_counts(df_ab_theta, support, alpha=1e-3):
    idx = {sid: i for i, sid in enumerate(support)}
    vec = np.full(len(support), float(alpha), dtype=np.float64)
    for r in df_ab_theta.itertuples(index=False):
        vec[idx[getattr(r, "seq_id")]] += 1.0
    return _normalize(vec)

def build_pz_given_theta_logweights(df_ab_theta, support, alpha=1e-3):
    per_seq_logs: Dict[str, List[float]] = {}
    for r in df_ab_theta.itertuples(index=False):
        sid = getattr(r, "seq_id")
        lp = float(getattr(r, "sum_logp_joint"))
        per_seq_logs.setdefault(sid, []).append(lp)
    logvec = np.full(len(support), -np.inf, dtype=np.float64)
    idx = {sid: i for i, sid in enumerate(support)}
    for sid, lst in per_seq_logs.items():
        logvec[idx[sid]] = _logsumexp(np.array(lst, dtype=np.float64))
    m = np.max(logvec)
    exps = np.exp(logvec - m, where=np.isfinite(logvec))
    exps = np.where(np.isfinite(exps), exps, 0.0)
    exps += float(alpha)
    return _normalize(exps)

def mi_for_one_x(df_x: pd.DataFrame, thetas_all, alpha=1e-3, use_weighted=True,
                 prior_mode="uniform") -> Tuple[float, float, float, int, int]:
    support = sorted(df_x["seq_id"].unique().tolist())
    if len(support) == 0:
        return 0.0, 0.0, 0.0, 0, 0
    present = sorted(df_x["theta"].unique().tolist())
    valid_thetas = [th for th in thetas_all if th in present]
    if len(valid_thetas) < 2:
        return 0.0, 0.0, 0.0, len(support), len(valid_thetas)

    build = build_pz_given_theta_logweights if use_weighted else build_pz_given_theta_counts
    P = []
    theta_weights = []
    for th in valid_thetas:
        df_th = df_x[df_x["theta"] == th]
        P.append(build(df_th, support, alpha=alpha))
        theta_weights.append(len(df_th))
    P = np.stack(P, axis=0)

    if prior_mode == "weighted":
        pth = _normalize(np.array(theta_weights, dtype=np.float64))
    else:
        pth = np.full(len(valid_thetas), 1.0/len(valid_thetas), dtype=np.float64)

    pz = _normalize((pth[:, None] * P).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(P, pz[None, :], out=np.ones_like(P), where=(pz[None, :] > 0))
        terms = pth[:, None] * P * np.log(np.clip(ratio, 1e-300, None), dtype=np.float64)
        mi_nats = float(terms.sum())

    H_theta = -float((pth * np.log(np.clip(pth, 1e-300, None), dtype=np.float64)).sum())
    H_theta_given_Z = H_theta - mi_nats
    if mi_nats < 0 and mi_nats > -1e-9:
        mi_nats = 0.0
    bits = float(mi_nats / math.log(2.0))
    H_bits = float(H_theta / math.log(2.0))
    Hc_bits = float(H_theta_given_Z / math.log(2.0))
    return bits, H_bits, Hc_bits, len(support), len(valid_thetas)

# --------------------- TB ingestion ---------------------

@dataclass
class TBRow:
    model_name: str
    rep: str          # "nl"|"code"
    digits: int
    kind: str         # "add"|"sub"|"mul"
    idx: int
    prompt: str
    raw_json_text: str
    a: int
    b: int
    theta: str        # "add"|"sub"|"mul"
    truncated: int

def _decode_tb_text(event):
    val = tensor_util.make_ndarray(event.tensor_proto)
    try:
        x = val.item()
    except Exception:
        x = val
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    return str(x)

def load_tb_rows(tb_run_dir: str, tb_text_chars: int,
                 prompt_suffixes=("prompt",),
                 json_suffixes=("raw_json","full"),
                 verbose_if_empty=True) -> List[TBRow]:
    ea = EventAccumulator(tb_run_dir, size_guidance={"tensors": 0})
    ea.Reload()
    tensor_tags = ea.Tags().get("tensors", [])
    if not tensor_tags and verbose_if_empty:
        print(f"[mi] No 'tensors' tags in TB run: {tb_run_dir}")
        return []

    # Build map: base -> {"prompt": str, "raw_json": str}
    seen_bases: Dict[str, Dict[str, str]] = {}
    for tag in tensor_tags:
        parts = tag.split("/")
        if not parts:
            continue
        # Text summaries are logged under ".../<suffix>/text_summary"
        if parts[-1].lower() == "text_summary" and len(parts) >= 2:
            suffix = parts[-2].lower()
            base = "/".join(parts[:-2])
        else:
            suffix = parts[-1].lower()
            base = "/".join(parts[:-1])

        if suffix not in set(s.lower() for s in prompt_suffixes + json_suffixes):
            continue

        events = ea.Tensors(tag)
        if not events:
            continue
        text = _decode_tb_text(events[-1])
        seen_bases.setdefault(base, {})[suffix] = text

    if not seen_bases and verbose_if_empty:
        print(f"[mi] Found 0 usable tags. Here are a few tags in this run:")
        for t in tensor_tags[:20]:
            print("   -", t)
        return []

    rows: List[TBRow] = []
    for base, kv in seen_bases.items():
        # require prompt and at least one of raw_json/full
        keys = {k.lower() for k in kv.keys()}
        if ("prompt" not in keys) or (("raw_json" not in keys) and ("full" not in keys)):
            continue

        parts = base.split("/")
        # Expect: model / rep / d{digits} / {kind} / i{idx}
        model_name = parts[0] if len(parts) > 0 else "model"
        rep = parts[1] if len(parts) > 1 else "nl"
        digits = -1
        kind = "add"
        idx = -1
        if len(parts) >= 5:
            dpart, kind, ipart = parts[2], parts[3], parts[4]
            try:
                digits = int(dpart[1:]) if dpart.startswith("d") else int(dpart)
            except Exception:
                pass
            try:
                idx = int(ipart[1:]) if ipart.startswith("i") else int(ipart)
            except Exception:
                pass

        prompt_text_wrapped = kv.get("prompt", "")
        raw_json_text_wrapped = kv.get("raw_json", kv.get("full", ""))

        # Extract fenced bodies
        prompt_body = extract_codeblock(prompt_text_wrapped)
        json_body = extract_codeblock(raw_json_text_wrapped)

        # Parse a,b,theta from prompt body
        m = PROMPT_PROB_LINE.search(prompt_body)
        if not m:
            continue
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        theta = CANON_THETA.get(op, None)
        if theta is None:
            continue

        truncated = 1 if json_body.endswith("...") else 0

        rows.append(TBRow(
            model_name=model_name, rep=rep, digits=digits, kind=kind, idx=idx,
            prompt=prompt_body, raw_json_text=json_body, a=a, b=b, theta=theta, truncated=truncated
        ))
    return rows

def parse_rationale_from_rawjson_text(raw_json_text: str) -> str:
    """
    raw_json_text is (ideally) the JSON string; but we’re robust to extra markdown.
    Extract the JSON object and return obj["rationale"] as str (or "" if missing).
    """
    m = JSON_OBJ_RE.search(raw_json_text)
    frag = m.group(0) if m else raw_json_text
    obj = None
    try:
        obj = json.loads(frag)
    except Exception:
        m2 = re.search(r'"rationale"\s*:\s*"([\s\S]*?)"', frag)
        if m2:
            val = m2.group(1)
            return val.encode("utf-8").decode("unicode_escape", errors="ignore")
        return ""
    val = obj.get("rationale", "")
    return val if isinstance(val, str) else (str(val) if val is not None else "")

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Same out_dir used by cot_tb.py")
    ap.add_argument("--model", type=str, required=True, help="HF model name/path for teacher-forced scoring")
    ap.add_argument("--tb_dir", type=str, default=None, help="Specific TB run dir (defaults to latest under out_dir/tb)")
    ap.add_argument("--include_prompt_ll", action="store_true", help="Include log p(prompt) in joint score")
    ap.add_argument("--use_weighted", action="store_true", help="Weighted p(z|theta,x) via exp(sum_logp_joint)")
    ap.add_argument("--theta_prior", choices=["uniform","weighted"], default="uniform")
    ap.add_argument("--tb_text_chars", type=int, default=4000, help="Char budget used when logging text in cot_tb.py")
    ap.add_argument("--skip_truncated", action="store_true", help="Skip entries whose TB text was truncated")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--mi_tb", action="store_true", help="Also log MI scalars to TensorBoard (out_dir/tb_mi)")
    args = ap.parse_args()

    # resolve TB run
    tb_root = os.path.join(args.out_dir, "tb") if args.tb_dir is None else args.tb_dir
    run_dirs = [tb_root] if (args.tb_dir and os.path.isdir(tb_root)) else find_tb_runs(tb_root)
    if not run_dirs:
        raise RuntimeError(f"No TensorBoard event files found under: {tb_root}")
    run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    tb_run = run_dirs[0]
    print(f"[mi] reading TB run: {tb_run}")

    # load TB rows
    tb_rows = load_tb_rows(
        tb_run,
        tb_text_chars=args.tb_text_chars,
        prompt_suffixes=("prompt",),
        json_suffixes=("raw_json","full"),
        verbose_if_empty=True
    )
    if not tb_rows:
        raise RuntimeError("No (prompt, raw_json/full) pairs parsed from TB. "
                           "Run with --tb_dir <specific-run> and check printed tag names.")

    if args.skip_truncated:
        tb_rows = [r for r in tb_rows if not r.truncated]
        if not tb_rows:
            raise RuntimeError("All entries were truncated and skipped. Re-run perf with higher --tb_text_chars.")

    # model
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cache_dir="../models"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        cache_dir = cache_dir
    ).eval()
    if device == "cpu":
        model.to("cpu")

    # Cox + TB writers
    exp_id = time.strftime("mi_%Y%m%d_%H%M%S")
    store = Store(args.out_dir, exp_id=exp_id)
    tb = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb_mi", exp_id)) if args.mi_tb else None

    # tables
    if "samples" not in store.tables:
        store.add_table("samples", {
            "model_name": str, "rep": str, "theta": str,
            "a": int, "b": int, "idx": int,
            "seq_id": str,
            "prompt_preview": str, "cot_preview": str, "raw_json_preview": str,
            "sum_logp_prompt": float, "n_tokens_prompt": int,
            "sum_logp_cot": float, "n_tokens_cot": int,
            "sum_logp_joint": float, "n_tokens_joint": int,
            "avg_logp_joint": float,
            "truncated": int
        })
    if "mi_stats" not in store.tables:
        store.add_table("mi_stats", {
            "model_name": str, "rep": str, "a": int, "b": int,
            "support_size": int, "num_thetas_present": int,
            "mi_bits": float, "H_theta_bits": float, "H_theta_given_Z_bits": float
        })

    # score each example
    print(f"[mi] scoring {len(tb_rows)} examples…")
    for row in tqdm(tb_rows):
        rationale = parse_rationale_from_rawjson_text(row.raw_json_text)
        tf = teacher_forced_sums(
            model, tokenizer, device,
            prompt_text=row.prompt,
            continuation_text=rationale,
            include_prompt_ll=args.include_prompt_ll
        )
        seq_id = sha1_16(rationale)

        store["samples"].append_row({
            "model_name": row.model_name, "rep": row.rep, "theta": row.theta,
            "a": int(row.a), "b": int(row.b), "idx": int(row.idx),
            "seq_id": seq_id,
            "prompt_preview": clip(row.prompt, 256),
            "cot_preview": clip(rationale, 256),
            "raw_json_preview": clip(row.raw_json_text, 256),
            "sum_logp_prompt": float(tf["sum_logp_prompt"]),
            "n_tokens_prompt": int(tf["n_tokens_prompt"]),
            "sum_logp_cot": float(tf["sum_logp_cot"]),
            "n_tokens_cot": int(tf["n_tokens_cot"]),
            "sum_logp_joint": float(tf["sum_logp_joint"]),
            "n_tokens_joint": int(tf["n_tokens_joint"]),
            "avg_logp_joint": float(tf["avg_logp_joint"]),
            "truncated": int(row.truncated),
        })

        if tb is not None:
            base = f"{row.model_name}/{row.rep}/a{row.a}_b{row.b}/i{row.idx}"
            tb.add_scalars(f"{base}/logps", {
                "sum_logp_prompt": tf["sum_logp_prompt"],
                "sum_logp_cot": tf["sum_logp_cot"],
                "sum_logp_joint": tf["sum_logp_joint"],
                "avg_logp_joint": tf["avg_logp_joint"],
            }, global_step=0)

    # compute MI per (model, rep) and per input (a,b)
    df = store["samples"].df.copy()
    results = []
    for (model_name, rep), df_mr in df.groupby(["model_name", "rep"]):
        thetas_all = sorted(df_mr["theta"].unique().tolist())
        mi_vals = []
        for (aa, bb), df_x in df_mr.groupby(["a", "b"]):
            mi_bits, H_bits, Hc_bits, support_size, t_present = mi_for_one_x(
                df_x, thetas_all, alpha=1e-3,
                use_weighted=bool(args.use_weighted),
                prior_mode=args.theta_prior
            )
            store["mi_stats"].append_row({
                "model_name": model_name, "rep": rep, "a": int(aa), "b": int(bb),
                "support_size": int(support_size), "num_thetas_present": int(t_present),
                "mi_bits": float(mi_bits),
                "H_theta_bits": float(H_bits),
                "H_theta_given_Z_bits": float(Hc_bits),
            })
            if t_present >= 2:
                mi_vals.append(mi_bits)
            if tb is not None:
                tag = f"{model_name}/{rep}/a{aa}_b{bb}"
                tb.add_scalar(f"{tag}/MI_bits", mi_bits, global_step=0)
                tb.add_scalar(f"{tag}/H_theta_bits", H_bits, global_step=0)
                tb.add_scalar(f"{tag}/H_theta_given_Z_bits", Hc_bits, global_step=0)

        avg_mi = float(np.mean(mi_vals)) if mi_vals else float("nan")
        results.append((model_name, rep, avg_mi, len(mi_vals)))
        if tb is not None:
            tb.add_scalar(f"{model_name}/{rep}/MI_bits/avg_over_inputs", avg_mi, global_step=0)
        print(f"[mi] {model_name:>30s} | {rep:>4s} | avg I(Z;θ|X,r) = {avg_mi:.4f} bits over {len(mi_vals)} inputs")

    out_json = os.path.join(args.out_dir, f"{exp_id}_mi_summary.json")
    with open(out_json, "w") as f:
        json.dump([
            {"model_name": m, "rep": r, "avg_mi_bits": mi, "num_inputs": n}
            for (m, r, mi, n) in results
        ], f, indent=2)
    print(f"[mi] wrote {out_json}")

    if tb is not None:
        tb.flush(); tb.close()

if __name__ == "__main__":
    main()
