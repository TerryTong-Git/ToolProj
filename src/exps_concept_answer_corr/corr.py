#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept-vs-Answer correlations *conditioned on stored CoT*.

What this script does
  • Load your performance CSV (needs: idx, kind, digits, problem, correct_nl, correct_code, correct_code_exec).
  • Load the per-item CoT from TensorBoard for BOTH channels:
        {model}/nl/d{digits}/{kind}/i{idx}/rationale
        {model}/code/d{digits}/{kind}/i{idx}/rationale
  • For each item, build two TASK texts:
        - problem + NL-CoT
        - problem + Code-CoT   (we use the fenced code text as *context*, no execution)
  • Score a 9-way multiple choice concept head (A..I) with vLLM prompt_logprobs (no generation).
      - randomize A..I order per item (controlled by --seed)
      - optionally de-bias letter priors (subtract log priors and renormalize)
  • Report correlations:
        - point-biserial( correct_nl , s_concept_true | problem+nl_cot )
        - point-biserial( correct_code , s_concept_true | problem+code_cot )
        - point-biserial( correct_code_exec , s_concept_true | problem+code_cot )
    Plus diagnostic histograms and hard–hard φ with median splits.

Usage
  python concept_answer_corr_with_cot.py \
    --perf-csv out/run_..._results_seed_1.csv \
    --tb-dir out/tb \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 256 \
    --out corr_with_cot.csv

Requires: vllm, transformers, pandas, numpy, tqdm, scipy, tensorboard
"""

from __future__ import annotations
import os, re, math, argparse, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------- Concepts and letter maps -----------------
CONCEPTS = [
    "add","sub","mul",
    "ilp_assign","ilp_partition","ilp_prod",
    "lcs","knap","rod",
]
LETTER_MAP = dict(zip(list("ABCDEFGHI"), CONCEPTS))
INV_LETTER_MAP = {v:k for k,v in LETTER_MAP.items()}
LETTERS = list(LETTER_MAP.keys())

# ----------------- TensorBoard loader (rationales) -----------------
TAG_RE = re.compile(
    r'^(?:text_summary/)?(?P<model>.+?)/(?P<rep>nl|code)/d(?P<digits>\d+)/(?P<kind>add|sub|mul|lcs|knap|rod|ilp_assign|ilp_prod|ilp_partition)/i(?P<idx>\d+)/(?:rationale|raw_json)(?:/text_summary)?$'
)

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
    """
    Returns dict keyed by (rep, digits, kind, idx) -> rationale_text
    """
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
        tags = ea.Tags().get('tensors', [])
        for tag in tags:
            m = TAG_RE.match(tag)
            if not m: 
                continue
            if m['model'] != model_name:
                # different model run
                continue
            rep    = m['rep']         # "nl" | "code"
            digits = int(m['digits'])
            kind   = m['kind']
            idx    = int(m['idx'])
            evs = ea.Tensors(tag)
            if not evs: 
                continue
            txt = _tb_text_from_event(evs[-1])
            # Prefer 'rationale' tags; if raw_json also matched, this regex keeps both.
            key = (rep, digits, kind, idx)
            # Keep last seen; TB often has single latest entry per tag.
            out[key] = txt
    return out

# ----------------- Prompt builders -----------------
def concept_mc_prompt(task_text: str, options: List[Tuple[str,str]]) -> str:
    # options is a list of (letter, concept_name) in randomized order
    opts = "\n".join([f"{L}) {name}" for (L, name) in options])
    return (
        "Identify the single best concept for solving the task.\n"
        f"{opts}\n"
        "Task:\n"
        f"{task_text}\n"
        "Answer only with one capital letter A..I (no punctuation):"
    )

def build_task_text_with_cot(problem_text: str, cot_text: Optional[str], rep: str) -> str:
    if not cot_text or not cot_text.strip():
        return problem_text
    if rep == "nl":
        # NL-CoT is likely natural language; include verbatim under a header
        return f"{problem_text}\n\nRATIONALE (natural language):\n{cot_text}"
    else:
        # Code-CoT: include fenced block for clarity; we are *conditioning*, not executing
        return f"{problem_text}\n\nRATIONALE (code):\n{cot_text}"

# ----------------- vLLM prefill scorer -----------------
class VLLMScorer:
    """vLLM prompt_logprobs scorer (no generation)."""
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
        """
        For each (prompt, single-letter target) pair, return avg logprob of the letter token.
        Returns a flat list aligned with input pairs.
        """
        from vllm import SamplingParams
        assert len(prompts)==len(targets)
        # Pre-tokenize target lengths (should be 1 token per capital letter)
        t_lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in targets]
        texts  = [p+t for p,t in zip(prompts, targets)]
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1,
                            detokenize=True, skip_special_tokens=False)
        out_vals=[]
        for i in tqdm(range(0, len(texts), batch_size), desc="prefill"):
            chunk = texts[i:i+batch_size]
            outs = self.llm.generate(chunk, sp, use_tqdm=False)
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

# ----------------- Utilities: stats & transforms -----------------
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

# ----------------- Main scoring routine -----------------
def score_concepts_for_context(scorer: VLLMScorer,
                               task_texts: List[str],
                               true_kinds: List[str],
                               rng: random.Random,
                               batch_size: int,
                               prior_debias: bool) -> Tuple[List[float], List[int], List[str]]:
    """
    Returns:
      s_true: probability assigned to the TRUE concept (after softmax; prior-debiased if requested)
      top1_correct: 1 if argmax letter matches true letter else 0
      top1_letter: the predicted letter for each item
    """
    prompts = []
    targets = []
    layouts = []   # per item: the randomized [(letter, concept)] order

    for _ in range(len(task_texts)):
        perm = LETTERS[:]  # A..I
        rng.shuffle(perm)
        options = [(L, LETTER_MAP[L]) for L in perm]
        layouts.append(options)

    # Build batched pairs for vLLM scorer
    for task, opts in zip(task_texts, layouts):
        prompt = concept_mc_prompt(task, opts) + " "
        for (L, _) in opts:
            prompts.append(prompt)
            targets.append(L)

    # avg logprob per target letter token
    logps = scorer.score_letters(prompts, targets, batch_size=batch_size)  # flat
    assert len(logps) == len(task_texts) * len(LETTERS)

    s_true = []
    top1_correct = []
    top1_letter = []

    # Optional: estimate empirical letter priors (by LETTER identity, not slot index)
    if prior_debias:
        # Accumulate probability mass for each LETTER across all rows
        letter2mass = {L: 0.0 for L in LETTERS}
        for i, opts in enumerate(layouts):
            row = logps[i*len(LETTERS):(i+1)*len(LETTERS)]
            probs = np.array(softmax_from_logprobs(row), dtype=np.float64)
            for j, (L, _name) in enumerate(opts):
                letter2mass[L] += float(probs[j])
        # Normalize to an empirical prior over letters
        total = sum(letter2mass.values())
        for L in LETTERS:
            letter2mass[L] = max(1e-6, letter2mass[L] / max(1.0, total))
        log_prior_by_letter = {L: math.log(letter2mass[L]) for L in LETTERS}
    else:
        log_prior_by_letter = {L: 0.0 for L in LETTERS}


    for i, (task, opts) in enumerate(zip(task_texts, layouts)):
        row = logps[i*len(LETTERS):(i+1)*len(LETTERS)]
        # subtract log priors aligned to this shuffled order
        row_adj = [row[j] - float(log_prior_by_letter[opts[j][0]]) for j in range(len(LETTERS))]
        probs = softmax_from_logprobs(row_adj)

        # map true kind -> letter index under this permutation
        true_kind = true_kinds[i]
        true_letter = INV_LETTER_MAP[true_kind]
        idx_true = [j for j,(L,name) in enumerate(opts) if name == true_kind][0]

        # take top1
        j_hat = int(np.argmax(probs))
        L_hat = opts[j_hat][0]
        top1_letter.append(L_hat)
        top1_correct.append(int(j_hat == idx_true))
        s_true.append(float(probs[idx_true]))
    return s_true, top1_correct, top1_letter

# ----------------- Plot helpers -----------------
def make_hist_by_label(vals, labels, title, out_png):
    import matplotlib.pyplot as plt
    v = np.asarray(vals, dtype=float)
    y = np.asarray(labels, dtype=int)
    plt.figure(figsize=(7,6))
    plt.hist(v[y==0], bins=20, alpha=0.6, label='label=0')
    plt.hist(v[y==1], bins=20, alpha=0.6, label='label=1')
    plt.title(title); plt.xlabel('prob(true concept)'); plt.ylabel('count'); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-csv", required=True, type=str,
                    help="CSV from your generator with idx, kind, digits, problem, correct_nl, correct_code, correct_code_exec")
    ap.add_argument("--tb-dir", required=True, type=str,
                    help="Root of TensorBoard logs (where events.* live)")
    ap.add_argument("--model", required=True, type=str,
                    help="Model name/path for vLLM")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--vllm-tensor-parallel", type=int, default=8)
    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.95)
    ap.add_argument("--vllm-max-model-len", type=int, default=None)
    ap.add_argument("--vllm-dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--vllm-download-dir", type=str, default="../models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deprior-letters", action="store_true",
                    help="Subtract empirical letter priors before softmax")
    ap.add_argument("--out", type=str, default="corr_with_cot.csv")
    args = ap.parse_args()

    # Load perf CSV
    dfp = pd.read_csv(args.perf_csv)
    dfp = dfp[:100]
    needed = ["idx","kind","digits","problem","correct_nl","correct_code","correct_code_exec"]
    miss = [c for c in needed if c not in dfp.columns]
    if miss:
        raise ValueError(f"perf-csv missing columns: {miss}")

    # Load TB rationales for this model
    tb_map = load_tb_rationales(args.tb_dir, model_name=os.path.basename(args.model) if "/" not in args.model else args.model)

    # Build contexts
    idxs   = dfp["idx"].astype(int).tolist()
    kinds  = dfp["kind"].astype(str).tolist()
    digits = dfp["digits"].astype(int).tolist()
    probs  = dfp["problem"].astype(str).tolist()

    cot_nl  = []
    cot_code= []
    for i, k, d in zip(idxs, kinds, digits):
        cot_nl.append(tb_map.get(("nl", d, k, i), ""))
        cot_code.append(tb_map.get(("code", d, k, i), ""))

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

    # Score concept probs *conditioned on NL-CoT* and *Code-CoT*
    s_true_nl,   top1_nl,   predL_nl   = score_concepts_for_context(
        scorer, task_nl, kinds, rng, batch_size=args.batch_size, prior_debias=args.deprior_letters
    )
    s_true_code, top1_code, predL_code = score_concepts_for_context(
        scorer, task_code, kinds, rng, batch_size=args.batch_size, prior_debias=args.deprior_letters
    )

    # Pack output rows
    out_rows = []
    for i in range(len(idxs)):
        out_rows.append(dict(
            idx=int(idxs[i]),
            kind=kinds[i],
            digits=int(digits[i]),
            instance_text=probs[i],
            # hard answers from CSV
            correct_nl=int(dfp.loc[i,"correct_nl"]),
            correct_code=int(dfp.loc[i,"correct_code"]),
            correct_code_exec=int(dfp.loc[i,"correct_code_exec"]),
            # soft concept probs (conditioned on CoT)
            s_concept_true_given_nl=float(s_true_nl[i]),
            s_concept_true_given_code=float(s_true_code[i]),
            # hard concept correctness (top-1) for debugging
            concept_top1_correct_nl=int(top1_nl[i]),
            concept_top1_correct_code=int(top1_code[i]),
            pred_letter_nl=predL_nl[i],
            pred_letter_code=predL_code[i],
        ))
    df = pd.DataFrame(out_rows)
    df.to_csv(args.out, index=False)

    # -------- Correlations (main results) --------
    # NL chain: correlate to NL correctness
    r_pb_nl  = pointbiserial(df["correct_nl"], df["s_concept_true_given_nl"])
    # Code chain: correlate to Code correctness & Exec correctness
    r_pb_code = pointbiserial(df["correct_code"], df["s_concept_true_given_code"])
    r_pb_exec = pointbiserial(df["correct_code_exec"], df["s_concept_true_given_code"])

    # Median-split φ as sanity
    def median_phi(s, y):
        s = np.asarray(s, dtype=float)
        thr = float(np.median(s))
        C = (s >= thr).astype(int)
        phi, (a,b,c,d) = phi_binary(C, np.asarray(y, dtype=int))
        return phi, (a,b,c,d), thr

    phi_nl,   tbl_nl,   thr_nl   = median_phi(df["s_concept_true_given_nl"],   df["correct_nl"])
    phi_code, tbl_code, thr_code = median_phi(df["s_concept_true_given_code"], df["correct_code"])
    phi_exec, tbl_exec, thr_exec = median_phi(df["s_concept_true_given_code"], df["correct_code_exec"])

    # Diagnostics
    print("=== DIAGNOSTICS ===")
    print("N =", len(df))
    print("answer value counts: nl", dict(df["correct_nl"].value_counts()),
          "| code", dict(df["correct_code"].value_counts()),
          "| exec", dict(df["correct_code_exec"].value_counts()))
    print("top1 (NL-context) letter dist:", dict(pd.Series(df["pred_letter_nl"]).value_counts(normalize=True).round(3)))
    print("top1 (Code-context) letter dist:", dict(pd.Series(df["pred_letter_code"]).value_counts(normalize=True).round(3)))

    print("\n=== CORRELATIONS (point-biserial) ===")
    print(f"r( correct_nl ,   s_concept_true | problem+NL-CoT )   = {r_pb_nl: .4f}")
    print(f"r( correct_code , s_concept_true | problem+Code-CoT ) = {r_pb_code: .4f}")
    print(f"r( correct_exec , s_concept_true | problem+Code-CoT ) = {r_pb_exec: .4f}")

    print("\n=== φ after median split (sanity) ===")
    print(f"phi NL   = {phi_nl: .4f}  thr={thr_nl:.4f}  table a,b,c,d={tbl_nl}")
    print(f"phi Code = {phi_code: .4f}  thr={thr_code:.4f}  table a,b,c,d={tbl_code}")
    print(f"phi Exec = {phi_exec: .4f}  thr={thr_exec:.4f}  table a,b,c,d={tbl_exec}")

    # Optional histograms (by channel)
    base = os.path.splitext(args.out)[0]
    try:
        make_hist_by_label(df["s_concept_true_given_nl"],   df["correct_nl"],        "s_true | problem+NL-CoT",   base+"_nl_hist.png")
        make_hist_by_label(df["s_concept_true_given_code"], df["correct_code"],      "s_true | problem+Code-CoT", base+"_code_hist.png")
        make_hist_by_label(df["s_concept_true_given_code"], df["correct_code_exec"], "s_true | problem+Code-CoT vs Exec", base+"_exec_hist.png")
        print(f"\nSaved table:  {args.out}")
        print(f"Saved plots:  {base}_nl_hist.png, {base}_code_hist.png, {base}_exec_hist.png")
    except Exception as e:
        print("[warn] plotting failed:", e)

if __name__ == "__main__":
    main()
