#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end pipeline (corpus-only, no finetuning) to test coupling:
1) Load one or more datasets (JSONL/CSV), each row has: id, text, rep ∈ {code,nl}.
2) Proportionally subsample ~1% from each dataset (stratified by rep).
3) Label each sampled document with a semantic task in {add, sub, mul, lcs, knap, rod, ilp_assign, ilp_prod, ilp_partition}
   by executing the SAME probe/evaluator procedure used in the MI pipeline:
   - Build fixed probe sets R_k for each task k
   - For each doc z, APPLY(z, x) for all x ∈ R_k using a frozen LM (HF generate, T=0)
   - Compare answers to the task’s evaluator f_k(x); assign label = argmax_k match rate (abstain if below threshold)
4) Train multinomial logistic regression (bag-of-words) to predict the semantic labels from text.
5) Report accuracy and cross-entropy overall and by channel (code vs nl).

Notes:
- Uses only HuggingFace generate for APPLY; set a small max_new_tokens for speed.
- Probes are light but discriminative; evaluators are exact DP/greedy ILP surrogates (integer objectives).
- For large runs, increase --batch-size and use a GPU-enabled HF model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------- I/O helpers -----------------------------


def read_any(path: str) -> List[dict]:
    rows = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    elif path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)  # type: ignore
            for row in r:
                rows.append(row)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    # normalize fields
    out: List = []
    for r in rows:
        rid = str(r.get("id") or r.get("doc_id") or r.get("uid") or len(out))  # type: ignore
        text = r.get("text") or r.get("content") or r.get("body") or ""  # type: ignore
        rep = (r.get("rep") or r.get("channel") or "").strip().lower()  # type: ignore
        if rep not in {"code", "nl"}:
            rep = "code" if looks_like_code(text) else "nl"
        if not text.strip():
            continue
        out.append({"id": rid, "text": text, "rep": rep})
    return out


CODE_MARKERS = re.compile(r"(\bdef\b|\bclass\b|;|{|}|\breturn\b|\bfor\b|\bwhile\b)")


def looks_like_code(text: str) -> bool:
    s = text.strip()
    return bool(CODE_MARKERS.search(s)) or ("def " in s or "class " in s or "{" in s or "}" in s)


# ----------------------------- Probes + Evaluators -----------------------------
# Nine tasks: add, sub, mul, lcs, knap, rod, ilp_assign, ilp_prod, ilp_partition


def _rand_int_with_digits(d: int) -> int:
    lo = 10 ** (d - 1)
    hi = 10**d - 1
    return random.randint(lo, hi)


def gen_probes(num_per_kind: int, seed: int = 0) -> Dict[str, List[dict]]:
    random.seed(seed)
    kinds = ["add", "sub", "mul", "lcs", "knap", "rod", "ilp_assign", "ilp_prod", "ilp_partition"]
    R: Dict = {k: [] for k in kinds}
    for _ in range(num_per_kind):
        # add / sub / mul
        a = _rand_int_with_digits(3)
        b = _rand_int_with_digits(3)
        R["add"].append({"a": a, "b": b})
        a = _rand_int_with_digits(3)
        b = _rand_int_with_digits(3)
        if b > a:
            a, b = b, a
        R["sub"].append({"a": a, "b": b})
        a = _rand_int_with_digits(3)
        b = _rand_int_with_digits(3)
        R["mul"].append({"a": a, "b": b})
        # lcs (strings)
        import string

        L = random.randint(6, 10)
        s1 = "".join(random.choice(string.ascii_uppercase[:6]) for _ in range(L))
        s2 = "".join(random.choice(string.ascii_uppercase[:6]) for _ in range(L))
        R["lcs"].append({"s1": s1, "s2": s2})
        # knapsack
        n = random.randint(5, 8)
        weights = [random.randint(1, 9) for _ in range(n)]
        values = [random.randint(1, 20) for _ in range(n)]
        cap = max(5, sum(weights) // 2)
        R["knap"].append({"w": weights, "v": values, "cap": cap})
        # rod cutting
        n = random.randint(6, 10)
        prices = [random.randint(1, 12) + i for i in range(n)]
        R["rod"].append({"prices": prices, "n": n})
        # ILP assign
        m = random.randint(3, 5)
        nn = random.randint(3, 5)
        C = [[random.randint(1, 12) for _ in range(nn)] for _ in range(m)]
        R["ilp_assign"].append({"C": C})
        # ILP prod (univariate constraints)
        m = random.randint(2, 4)
        a1 = [random.randint(1, 9) for _ in range(m)]
        b1 = [random.randint(8, 25) for _ in range(m)]
        p1 = [random.randint(5, 20) for _ in range(m)]
        R["ilp_prod"].append({"a": a1, "b": b1, "p": p1})
        # ILP partition
        n = random.randint(8, 12)
        arr = [random.randint(1, 25) for _ in range(n)]
        R["ilp_partition"].append({"arr": arr})
    return R


# Evaluators (deterministic)


def eval_add(x):
    return x["a"] + x["b"]


def eval_sub(x):
    return x["a"] - x["b"]


def eval_mul(x):
    return x["a"] * x["b"]


def eval_lcs(x):
    s1, s2 = x["s1"], x["s2"]
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if s1[i] == s2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def eval_knap(x):
    w, v, cap = x["w"], x["v"], x["cap"]
    n = len(w)
    dp = [0] * (cap + 1)
    for i in range(n):
        for c in range(cap, w[i] - 1, -1):
            dp[c] = max(dp[c], dp[c - w[i]] + v[i])
    return dp[cap]


def eval_rod(x):
    prices = x["prices"]
    n = x["n"]
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        best = 0
        for cut in range(1, i + 1):
            best = max(best, prices[cut - 1] + dp[i - cut])
        dp[i] = best
    return dp[n]


def eval_ilp_assign(x):
    # min cost perfect assignment (Hungarian via simple DP/bitmask for small sizes)
    C = x["C"]
    m = len(C)
    n = len(C[0])
    assert m == n, "square cost matrix expected"
    N = 1 << n
    dp = [math.inf] * N
    dp[0] = 0
    for i in range(m):
        ndp = [math.inf] * N
        for mask in range(N):
            if dp[mask] == math.inf:
                continue
            for j in range(n):
                if (mask >> j) & 1:
                    continue
                nm = mask | (1 << j)
                ndp[nm] = min(ndp[nm], dp[mask] + C[i][j])
        dp = ndp
    return dp[N - 1]


def eval_ilp_prod(x):
    # maximize sum p_i * x_i s.t. a_i x_i <= b_i, x_i integer>=0 (independent knapsacks)
    a, b, p = x["a"], x["b"], x["p"]
    tot = 0
    for ai, bi, pi in zip(a, b, p):
        xi = bi // ai
        tot += pi * xi
    return tot


def eval_ilp_partition(x):
    arr = x["arr"]
    S = sum(arr)
    T = S // 2
    dp = [False] * (T + 1)
    dp[0] = True
    for v in arr:
        for s in range(T, v - 1, -1):
            dp[s] = dp[s] or dp[s - v]
    best = max(s for s in range(T + 1) if dp[s])
    return S - 2 * best


EVALS = {
    "add": eval_add,
    "sub": eval_sub,
    "mul": eval_mul,
    "lcs": eval_lcs,
    "knap": eval_knap,
    "rod": eval_rod,
    "ilp_assign": eval_ilp_assign,
    "ilp_prod": eval_ilp_prod,
    "ilp_partition": eval_ilp_partition,
}


# Build prompts to APPLY the procedure text to an input instance
def apply_prompt(z_text: str, x: dict) -> str:
    # Serialize probe x compactly
    def ser(d):
        if isinstance(d, dict):
            items = []
            for k, v in d.items():
                if isinstance(v, list):
                    items.append(f"{k}=[{','.join(map(str,v))}]")
                else:
                    items.append(f"{k}={v}")
            return "; ".join(items)
        return str(d)

    task = (
        "Apply the following procedure exactly to the input. "
        "Return only the final numeric answer on one line.\n\n"
        f"PROCEDURE:\n{z_text}\n\nINPUT:\n{ser(x)}\n\nANSWER:"
    )
    return task


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.splitlines()[-1].strip() if s else ""
    s = re.sub(r"^(answer|final|result)\s*[:=-]\s*", "", s, flags=re.I)
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return m.group(0) if m else "⊥"


# ----------------------------- HF LM for APPLY -----------------------------


def build_hf(model_name: str, dtype: str = "auto", device: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def pick_dtype(flag):
        if flag == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
            return torch.float32
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[flag]

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    torch_dtype = pick_dtype(dtype) if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto" if device != "cpu" else None).eval()
    if device == "cpu":
        model.to("cpu")
    return tok, model


def hf_generate_batch(tok, model, prompts: List[str], max_new_tokens: int = 32) -> List[str]:
    import torch

    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    for k in enc:
        enc[k] = enc[k].to(model.device)
    with torch.inference_mode():
        gen = model.generate(**enc, do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens)
    outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return [normalize_answer(o) for o in outs]


# ----------------------------- Label by denotation against references -----------------------------


def precompute_reference_vectors(R_by_kind: Dict[str, List[dict]]) -> Dict[str, List[str]]:
    """For each kind k, compute string answers for each probe using the gold evaluator."""
    ref = {}
    for k, probes in R_by_kind.items():
        f = EVALS[k]
        ref[k] = [str(f(x)) for x in probes]
    return ref


def assign_semantic_label(
    z_text: str,
    tok,
    model,
    R_by_kind: Dict[str, List[dict]],
    ref_vecs: Dict[str, List[str]],
    max_new_tokens: int = 32,
    batch_size: int = 64,
) -> Tuple[Optional[str], float, Dict[str, float]]:
    """APPLY z_text to all probes for every kind; compute match rate vs ref; pick best."""
    scores = {}
    for k, probes in R_by_kind.items():
        prompts = [apply_prompt(z_text, x) for x in probes]
        outs = []
        for i in range(0, len(prompts), batch_size):
            outs.extend(hf_generate_batch(tok, model, prompts[i : i + batch_size], max_new_tokens=max_new_tokens))
        ref = ref_vecs[k]
        match = sum(1 for a, b in zip(outs, ref) if a == b)
        scores[k] = match / max(1, len(ref))
    # choose best with confidence margin
    lab = max(scores.items(), key=lambda kv: kv[1])[0]
    conf = scores[lab]
    return (lab if conf >= 0.6 else None), conf, scores  # threshold configurable


# ----------------------------- Subsampling -----------------------------


def proportional_subsample(datasets: List[List[dict]], rate: float, seed: int = 0) -> List[dict]:
    random.seed(seed)
    out = []
    for ds in datasets:
        n = len(ds)
        k = max(1, int(round(n * rate)))
        # stratify by rep for stability
        by = defaultdict(list)
        for r in ds:
            by[r["rep"]].append(r)
        take = []
        for rep, lst in by.items():
            kk = max(1, int(round(len(lst) * rate)))
            take.extend(random.sample(lst, min(kk, len(lst))))
        # adjust to target k if needed
        if len(take) < k:
            pool = [r for r in ds if r not in take]
            take.extend(random.sample(pool, min(k - len(take), len(pool))))
        out.extend(take[:k])
    return out


# ----------------------------- BoW + Multinomial LR -----------------------------


def train_eval_bow_lr(
    rows_labeled: List[dict],
    min_df: int = 20,
    max_features: int = 200000,
    svd_dim: Optional[int] = None,
):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import train_test_split

    texts = [r["text"] for r in rows_labeled]
    labels = [r["label"] for r in rows_labeled]
    reps = [r["rep"] for r in rows_labeled]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=max_features)
    X = vec.fit_transform(texts)
    if svd_dim:
        svd = TruncatedSVD(n_components=svd_dim, random_state=0)
        X = svd.fit_transform(X)

    Xtr, Xte, ytr, yte, rtr, rte = train_test_split(X, labels, reps, test_size=0.2, stratify=labels, random_state=0)

    clf = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)
    acc = accuracy_score(yte, prob.argmax(1))
    xent = log_loss(yte, prob, labels=clf.classes_)

    # per-channel
    def subset_metrics(mask):
        from sklearn.metrics import accuracy_score, log_loss

        idx = np.where(mask)[0]
        if len(idx) == 0:
            return float("nan"), float("nan")
        pa = prob[idx]
        ya = np.array(yte)[idx]
        return accuracy_score(ya, pa.argmax(1)), log_loss(ya, pa, labels=clf.classes_)

    mask_code = np.array([r == "code" for r in rte])
    mask_nl = ~mask_code
    acc_code, xent_code = subset_metrics(mask_code)
    acc_nl, xent_nl = subset_metrics(mask_nl)

    return {
        "overall": {"acc": float(acc), "xent": float(xent)},
        "code": {"acc": float(acc_code), "xent": float(xent_code)},
        "nl": {"acc": float(acc_nl), "xent": float(xent_nl)},
        "classes": list(clf.classes_),
    }


# ----------------------------- Main -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Paths to JSONL/CSV datasets with fields {id,text,rep?}",
    )
    ap.add_argument("--subsample-rate", type=float, default=0.01, help="Rate per dataset, e.g., 0.01 for 1%")
    ap.add_argument("--hf-model", type=str, default="google/gemma-2-9b-it", help="HF causal LM for APPLY")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu"])
    ap.add_argument("--apply-max-new", type=int, default=32)
    ap.add_argument("--apply-batch-size", type=int, default=64)
    ap.add_argument("--probes-per-kind", type=int, default=32, help="Number of probes per task kind")
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--label-conf-thresh", type=float, default=0.6)
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--svd-dim", type=int, default=0)
    ap.add_argument("--out-labels", type=str, default="labels_labeled.csv")
    args = ap.parse_args()

    # Load datasets
    corpora = [read_any(p) for p in args.datasets]
    total = sum(len(c) for c in corpora)
    print(f"[info] loaded {len(corpora)} datasets, total rows={total}")

    # Subsample proportionally
    sample = proportional_subsample(corpora, rate=args.subsample_rate, seed=args.probe_seed)
    print(f"[info] subsampled {len(sample)} rows (~{args.subsample_rate*100:.2f}%)")

    # Build probes and reference vectors
    R_by_kind = gen_probes(args.probes_per_kind, seed=args.probe_seed)
    ref_vecs = precompute_reference_vectors(R_by_kind)

    # HF model
    tok, model = build_hf(args.hf_model, dtype=args.dtype, device=args.device)

    # Label each sampled document by denotation match against references
    labeled = []
    t0 = time.time()
    for i, r in enumerate(sample, 1):
        lab, conf, scores = assign_semantic_label(
            r["text"],
            tok,
            model,
            R_by_kind,
            ref_vecs,
            max_new_tokens=args.apply_max_new,
            batch_size=args.apply_batch_size,
        )
        if lab is None or conf < args.label_conf_thresh:
            continue
        labeled.append({"id": r["id"], "rep": r["rep"], "text": r["text"], "label": lab, "conf": conf})
        if i % 250 == 0:
            dt = time.time() - t0
            print(f"[label] processed={i}/{len(sample)} kept={len(labeled)} rate={i/max(dt,1e-6):.1f}/s")
    print(f"[label] done. kept {len(labeled)} labeled docs")

    # Persist labels
    if args.out_labels:
        os.makedirs(os.path.dirname(args.out_labels), exist_ok=True) if os.path.dirname(args.out_labels) else None
        with open(args.out_labels, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "rep", "label", "conf", "text"])
            w.writeheader()
            for row in labeled:
                w.writerow(row)
        print(f"[write] labels -> {args.out_labels}")

    # Train multinomial LR with bag-of-words
    if len(labeled) < 100:
        print("[warn] too few labeled samples for stable LR; exiting.")
        return
    metrics = train_eval_bow_lr(
        labeled,
        min_df=args.min_df,
        max_features=args.max_features,
        svd_dim=(args.svd_dim if args.svd_dim > 0 else None),
    )

    print("\n=== Multinomial Logistic Regression (BoW) ===")
    print(f"Overall: acc={metrics['overall']['acc']:.4f}  xent={metrics['overall']['xent']:.4f}")
    print(f"Code:    acc={metrics['code']['acc']:.4f}     xent={metrics['code']['xent']:.4f}")
    print(f"NL:      acc={metrics['nl']['acc']:.4f}       xent={metrics['nl']['xent']:.4f}")
    print(f"Classes: {metrics['classes']}")


if __name__ == "__main__":
    main()
