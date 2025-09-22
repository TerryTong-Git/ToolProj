#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept-vs-Answer correlations *conditioned on stored CoT* (fixed/robust).

Key changes vs prior version
  • Use .iloc (not .loc) when copying correctness columns to avoid index drift.
  • For Code-CoT context, extract ONLY the fenced code (```...```) for conditioning.
  • Add --exec-viable-only flag to restrict correlations to rows with a code fence AND valid exec label.
  • Enable letter prior de-biasing by default (can turn off with --no-deprior-letters).
  • Print correlations for ALL rows and EXEC-VIABLE subset.
"""

from __future__ import annotations
import os, re, math, argparse, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------- Concepts & letters -----------------
CONCEPTS = ["add","sub","mul","ilp_assign","ilp_partition","ilp_prod","lcs","knap","rod"]
LETTER_MAP = dict(zip(list("ABCDEFGHI"), CONCEPTS))
INV_LETTER_MAP = {v:k for k,v in LETTER_MAP.items()}
LETTERS = list(LETTER_MAP.keys())

# ----------------- TensorBoard loader -----------------
TAG_RE = re.compile(
    r'^(?:text_summary/)?(?P<model>.+?)/(?P<rep>nl|code)/d(?P<digits>\d+)/(?P<kind>add|sub|mul|lcs|knap|rod|ilp_assign|ilp_prod|ilp_partition)/i(?P<idx>\d+)/(?:rationale|raw_json)(?:/text_summary)?$'
)

FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n([\s\S]*?)\n```", re.MULTILINE)

def _tb_text_from_event(ev) -> str:
    tp = ev.tensor_proto
    if tp.string_val:
        val = tp.string_val[0]
        return val.decode("utf-8","replace") if isinstance(val,(bytes,bytearray)) else str(val)
    if tp.tensor_content:
        try: return tp.tensor_content.decode("utf-8","replace")
        except Exception: pass
    return ""

def load_tb_rationales(tb_dir: str, model_name: str) -> Dict[Tuple[str,int,str,int], str]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out: Dict[Tuple[str,int,str,int], str] = {}
    # iterate event directories
    dirs = []
    if os.path.isdir(tb_dir) and any(fn.startswith("events") for fn in os.listdir(tb_dir)):
        dirs = [tb_dir]
    else:
        for name in os.listdir(tb_dir):
            p = os.path.join(tb_dir, name)
            if os.path.isdir(p) and any(fn.startswith("events") for fn in os.listdir(p)):
                dirs.append(p)
    for d in dirs:
        ea = EventAccumulator(d, size_guidance={'tensors': 10**7})
        ea.Reload()
        for tag in ea.Tags().get('tensors', []):
            m = TAG_RE.match(tag)
            if not m or m['model'] != model_name:
                continue
            rep    = m['rep']
            digits = int(m['digits'])
            kind   = m['kind']
            idx    = int(m['idx'])
            evs = ea.Tensors(tag)
            if not evs:
                continue
            out[(rep, digits, kind, idx)] = _tb_text_from_event(evs[-1])
    return out

def extract_fenced_code(txt: Optional[str]) -> str:
    if not txt:
        return ""
    m = FENCE_RE.search(txt)
    if not m:
        return ""
    return m.group(1).strip()

# ----------------- Prompt builders -----------------
def concept_mc_prompt(task_text: str, options: List[Tuple[str,str]]) -> str:
    opts = "\n".join([f"{L}) {name}" for (L, name) in options])
    return (
        "Identify the single best concept for solving the task.\n"
        f"{opts}\n"
        "Task:\n"
        f"{task_text}\n"
        "Answer only with one capital letter A..I (no punctuation):"
    )

def build_task_text_with_cot(problem_text: str, cot_text: Optional[str], rep: str) -> str:
    if rep == "nl":
        if cot_text and cot_text.strip():
            return f"{problem_text}\n\nRATIONALE (natural language):\n{cot_text}"
        return problem_text
    else:
        code = extract_fenced_code(cot_text)
        if code:
            return f"{problem_text}\n\nRATIONALE (code):\n```python\n{code}\n```"
        # if no fenced code, fall back to problem only (prevents noisy blobs)
        return problem_text

# ----------------- vLLM prefill scorer -----------------
class VLLMScorer:
    def __init__(self, model_name: str, tensor_parallel_size: int=1,
                 gpu_memory_util: float=0.95, max_model_len: Optional[int]=None,
                 seed: int=0, vllm_dtype: Optional[str]="float16",
                 download_dir: Optional[str] = "../models"):
        from vllm import LLM
        self.llm = LLM(model=model_name,
                       tensor_parallel_size=int(tensor_parallel_size),
                       gpu_memory_utilization=float(gpu_memory_util),
                       max_model_len=(int(max_model_len) if max_model_len else None),
                       dtype=vllm_dtype,
                       download_dir=download_dir,
                       seed=int(seed))
        self.tok = self.llm.get_tokenizer()

    def _align(self, token_ids, prompt_logprobs):
        L = len(token_ids)
        pl = list(prompt_logprobs) if prompt_logprobs is not None else []
        if len(pl)==L: return pl
        if len(pl)==L-1: return [None]+pl
        if len(pl)<L: return [None]*(L-len(pl))+pl
        return pl[:L]

    def score_letters(self, prompts: List[str], targets: List[str], batch_size: int=256) -> List[float]:
        from vllm import SamplingParams
        assert len(prompts)==len(targets)
        t_lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in targets]
        texts  = [p+t for p,t in zip(prompts, targets)]
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1,
                            detokenize=True, skip_special_tokens=False)
        out_vals=[]
        for i in tqdm(range(0, len(texts), batch_size), desc="prefill"):
            outs = self.llm.generate(texts[i:i+batch_size], sp, use_tqdm=False)
            for j, o in enumerate(outs):
                ids = list(o.prompt_token_ids)
                pl  = self._align(ids, o.prompt_logprobs)
                T   = t_lens[i+j]
                s = 0.0; n = 0
                for pos in range(len(pl)-T, len(pl)):
                    entry = pl[pos]
                    if entry is None: 
                        continue
                    tokid = ids[pos]
                    lp = None
                    if isinstance(entry, dict):
                        v = entry.get(tokid, None)
                        if isinstance(v, dict) and "logprob" in v: lp = v["logprob"]
                        elif hasattr(v, "logprob"): lp = v.logprob
                        if lp is None:
                            for cand in entry.values():
                                lid = cand.get("id") if isinstance(cand, dict) else getattr(cand,"id",None)
                                lpp = cand.get("logprob") if isinstance(cand, dict) else getattr(cand,"logprob",None)
                                if lid == tokid and lpp is not None:
                                    lp = lpp; break
                    else:
                        lp = getattr(entry, "logprob", None)
                    if lp is not None:
                        s += float(lp); n += 1
                out_vals.append((s/max(1,n)) if n>0 else float("-inf"))
        return out_vals

# ----------------- Stats helpers -----------------
def softmax_from_logprobs(logps: List[float]) -> List[float]:
    m = max(logps)
    ex = [math.exp(z - m) for z in logps]
    Z = sum(ex) if ex else 1.0
    return [e/Z for e in ex]

def pointbiserial(binary, continuous):
    a = np.asarray(binary, dtype=float); b = np.asarray(continuous, dtype=float)
    ma, mb = a.mean(), b.mean()
    num = float(((a-ma)*(b-mb)).sum())
    den = math.sqrt(float(((a-ma)**2).sum()) * float(((b-mb)**2).sum()))
    return num/den if den>0 else float("nan")

def phi_from_counts(a,b,c,d):
    den = math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    return 0.0 if den==0 else (a*d - b*c)/den

def phi_binary(x, y):
    x = np.asarray(x, dtype=int).ravel()
    y = np.asarray(y, dtype=int).ravel()
    a = int(((x==1)&(y==1)).sum())
    b = int(((x==1)&(y==0)).sum())
    c = int(((x==0)&(y==1)).sum())
    d = int(((x==0)&(y==0)).sum())
    return phi_from_counts(a,b,c,d), (a,b,c,d)

# ----------------- Concept scorer -----------------
def score_concepts_for_context(scorer: VLLMScorer,
                               task_texts: List[str],
                               true_kinds: List[str],
                               rng: random.Random,
                               batch_size: int,
                               prior_debias: bool) -> Tuple[List[float], List[int], List[str]]:
    prompts, targets, layouts = [], [], []

    for _ in range(len(task_texts)):
        perm = LETTERS[:]
        rng.shuffle(perm)
        options = [(L, LETTER_MAP[L]) for L in perm]
        layouts.append(options)

    for task, opts in zip(task_texts, layouts):
        prompt = concept_mc_prompt(task, opts) + " "
        for (L, _) in opts:
            prompts.append(prompt); targets.append(L)

    logps = scorer.score_letters(prompts, targets, batch_size=batch_size)
    assert len(logps) == len(task_texts) * len(LETTERS)

    # empirical letter priors (optional)
    if prior_debias:
        letter2mass = {L: 0.0 for L in LETTERS}
        for i, opts in enumerate(layouts):
            row = logps[i*len(LETTERS):(i+1)*len(LETTERS)]
            probs = np.array(softmax_from_logprobs(row), dtype=np.float64)
            for j, (L, _) in enumerate(opts):
                letter2mass[L] += float(probs[j])
        total = sum(letter2mass.values()) or 1.0
        for L in LETTERS:
            letter2mass[L] = max(1e-6, letter2mass[L] / total)
        log_prior_by_letter = {L: math.log(letter2mass[L]) for L in LETTERS}
    else:
        log_prior_by_letter = {L: 0.0 for L in LETTERS}

    s_true, top1_correct, top1_letter = [], [], []
    for i, opts in enumerate(layouts):
        row = logps[i*len(LETTERS):(i+1)*len(LETTERS)]
        row_adj = [row[j] - float(log_prior_by_letter[opts[j][0]]) for j in range(len(LETTERS))]
        probs = softmax_from_logprobs(row_adj)
        true_kind = true_kinds[i]
        idx_true = [j for j,(_, name) in enumerate(opts) if name == true_kind][0]
        j_hat = int(np.argmax(probs))
        top1_letter.append(opts[j_hat][0])
        top1_correct.append(int(j_hat == idx_true))
        s_true.append(float(probs[idx_true]))
    return s_true, top1_correct, top1_letter

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-csv", required=True, type=str)
    ap.add_argument("--tb-dir", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--vllm-tensor-parallel", type=int, default=8)
    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.95)
    ap.add_argument("--vllm-max-model-len", type=int, default=None)
    ap.add_argument("--vllm-dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--vllm-download-dir", type=str, default="../models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-deprior-letters", action="store_true",
                    help="Disable empirical letter prior subtraction")
    ap.add_argument("--exec-viable-only", action="store_true",
                    help="Restrict correlations to rows with a code fence AND valid exec label")
    ap.add_argument("--out", type=str, default="corr_with_cot.csv")
    args = ap.parse_args()

    # Load perf CSV
    dfp = pd.read_csv(args.perf_csv)
    dfp = dfp[:500]
    needed = ["idx","kind","digits","problem","correct_nl","correct_code","correct_code_exec"]
    miss = [c for c in needed if c not in dfp.columns]
    if miss:
        raise ValueError(f"perf-csv missing columns: {miss}")

    # Load TB rationales for this model
    model_key = args.model if "/" in args.model else os.path.basename(args.model)
    tb_map = load_tb_rationales(args.tb_dir, model_name=model_key)

    # Build contexts
    idxs   = dfp["idx"].astype(int).tolist()
    kinds  = dfp["kind"].astype(str).tolist()
    digits = dfp["digits"].astype(int).tolist()
    probs  = dfp["problem"].astype(str).tolist()

    cot_nl   = [tb_map.get(("nl",   d, k, i), "") for i,k,d in zip(idxs, kinds, digits)]
    cot_code = [tb_map.get(("code", d, k, i), "") for i,k,d in zip(idxs, kinds, digits)]

    task_nl   = [build_task_text_with_cot(p, c, "nl")   for p,c in zip(probs, cot_nl)]
    task_code = [build_task_text_with_cot(p, c, "code") for p,c in zip(probs, cot_code)]

    # vLLM scorer
    scorer = VLLMScorer(
        model_name=args.model,
        tensor_parallel_size=args.vllm_tensor_parallel,
        gpu_memory_util=float(args.vllm_gpu_mem_util),
        max_model_len=args.vllm_max_model_len,
        seed=int(args.seed),
        vllm_dtype=args.vllm_dtype,
        download_dir=args.vllm_download_dir
    )
    rng = random.Random(args.seed)

    # Concept probabilities conditioned on CoTs
    s_true_nl,   top1_nl,   predL_nl   = score_concepts_for_context(
        scorer, task_nl, kinds, rng, batch_size=args.batch_size, prior_debias=not args.no_deprior_letters
    )
    s_true_code, top1_code, predL_code = score_concepts_for_context(
        scorer, task_code, kinds, rng, batch_size=args.batch_size, prior_debias=not args.no_deprior_letters
    )

    # Pack rows (use iloc to avoid index drift)
    out_rows = []
    for i in range(len(idxs)):
        row = dfp.iloc[i]
        out_rows.append(dict(
            idx=int(row["idx"]),
            kind=str(row["kind"]),
            digits=int(row["digits"]),
            instance_text=str(row["problem"]),
            correct_nl=int(row["correct_nl"]),
            correct_code=int(row["correct_code"]),
            correct_code_exec=int(row["correct_code_exec"]),
            s_concept_true_given_nl=float(s_true_nl[i]),
            s_concept_true_given_code=float(s_true_code[i]),
            concept_top1_correct_nl=int(top1_nl[i]),
            concept_top1_correct_code=int(top1_code[i]),
            pred_letter_nl=predL_nl[i],
            pred_letter_code=predL_code[i],
            has_code_fence=int(bool(extract_fenced_code(cot_code[i]))),
        ))
    df = pd.DataFrame(out_rows)
    df.to_csv(args.out, index=False)

    # ---- Correlations (All rows) ----
    def corr_pb(y, s): return pointbiserial(np.asarray(y, int), np.asarray(s, float))
    def median_phi(s, y):
        s = np.asarray(s, float); y = np.asarray(y, int)
        thr = float(np.median(s)); C = (s >= thr).astype(int)
        phi, tbl = phi_binary(C, y)
        return phi, tbl, thr

    r_pb_nl_all   = corr_pb(df["correct_nl"],        df["s_concept_true_given_nl"])
    r_pb_code_all = corr_pb(df["correct_code"],      df["s_concept_true_given_code"])
    r_pb_exec_all = corr_pb(df["correct_code_exec"], df["s_concept_true_given_code"])

    phi_nl_all,   tbl_nl_all,   thr_nl   = median_phi(df["s_concept_true_given_nl"],   df["correct_nl"])
    phi_code_all, tbl_code_all, thr_code = median_phi(df["s_concept_true_given_code"], df["correct_code"])
    phi_exec_all, tbl_exec_all, thr_exec = median_phi(df["s_concept_true_given_code"], df["correct_code_exec"])

    # ---- Exec-viable subset (this is what should pop) ----
    mask_exec = (df["has_code_fence"] == 1) & df["correct_code_exec"].isin([0,1])
    dfE = df.loc[mask_exec].copy()

    r_pb_exec_viable = corr_pb(dfE["correct_code_exec"], dfE["s_concept_true_given_code"]) if len(dfE) else float('nan')
    phi_exec_viable, tbl_exec_viable, thr_exec_viable = median_phi(
        dfE["s_concept_true_given_code"], dfE["correct_code_exec"]) if len(dfE) else (float('nan'), (0,0,0,0), float('nan'))

    # ---- Print ----
    print("=== DIAGNOSTICS ===")
    print(f"N(all)={len(df)} | N(exec-viable)={int(mask_exec.sum())}")
    print("answer counts all:",
          "nl",   dict(df["correct_nl"].value_counts()),
          "| code", dict(df["correct_code"].value_counts()),
          "| exec", dict(df["correct_code_exec"].value_counts()))
    print("top1 (NL-context) letter dist:", dict(pd.Series(df["pred_letter_nl"]).value_counts(normalize=True).round(3)))
    print("top1 (Code-context) letter dist:", dict(pd.Series(df["pred_letter_code"]).value_counts(normalize=True).round(3)))

    print("\n=== CORRELATIONS (All rows, point-biserial) ===")
    print(f"r( correct_nl  , s_true | NL-CoT   ) = {r_pb_nl_all: .4f}")
    print(f"r( correct_code, s_true | Code-CoT ) = {r_pb_code_all: .4f}")
    print(f"r( correct_exec, s_true | Code-CoT ) = {r_pb_exec_all: .4f}")

    print("\n=== φ after median split (All rows) ===")
    print(f"phi NL   = {phi_nl_all: .4f}  thr={thr_nl:.4f}   table a,b,c,d={tbl_nl_all}")
    print(f"phi Code = {phi_code_all: .4f}  thr={thr_code:.4f} table a,b,c,d={tbl_code_all}")
    print(f"phi Exec = {phi_exec_all: .4f}  thr={thr_exec:.4f} table a,b,c,d={tbl_exec_all}")

    print("\n=== EXEC-VIABLE ONLY (code fence & valid exec label) ===")
    print(f"N = {len(dfE)}")
    print(f"r( correct_exec, s_true | Code-CoT ) = {r_pb_exec_viable: .4f}")
    print(f"phi Exec = {phi_exec_viable: .4f}  thr={thr_exec_viable:.4f} table a,b,c,d={tbl_exec_viable}")

    if args.exec_viable_only and len(dfE):
        # optional: also dump a restricted CSV if requested
        base = os.path.splitext(args.out)[0]
        dfE.to_csv(base + "_exec_viable.csv", index=False)
        print(f"\nSaved exec-viable slice -> {base}_exec_viable.csv")

if __name__ == "__main__":
    main()
