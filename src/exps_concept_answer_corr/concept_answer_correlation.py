#!/usr/bin/env python3
"""
Soft-Score Concept–Answer Correlation

Objective
  Estimate correlation between:
    s_concept = P_model(true_concept | x)   (soft probability, multi-class)
    s_answer  = P_model(target_answer | x)  (soft, via normalized token-average likelihood)
  and the binary answer correctness A ∈ {0,1} from an actual generation.

Design
  • Generation engine: mock | hf | vllm | openai   (produces an answer to score A)
  • Scoring engine:    hf                         (computes s_concept and s_answer via teacher forcing)
  • Concepts are mapped to letters A..I and scored as multiple-choice next-token probabilities.
  • s_answer uses per-token average log-prob exp(avg_logprob) in [0,1], comparable across lengths.

Usage
  python concept_answer_softcorr.py \
    --samples-per-concept 40 \
    --gen-engine vllm \
    --gen-model meta-llama/Llama-3.1-8B-Instruct \
    --scorer-model meta-llama/Llama-3.1-8B-Instruct \
    --out softcorr_results.csv

Deps
  pip install torch transformers vllm pandas numpy
"""

from __future__ import annotations
import os, re, json, math, argparse, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Tasks and ground-truth solvers
# ------------------------------------------------------------
CONCEPTS = [
    "add","sub","mul",
    "ilp_assign","ilp_partition","ilp_prod",
    "lcs","knap","rod",
]
LETTER_MAP = dict(zip(list("ABCDEFGHI"), CONCEPTS))
INV_LETTER_MAP = {v:k for k,v in LETTER_MAP.items()}

def set_seed(s: int):
    random.seed(s); np.random.seed(s)

@dataclass
class Instance:
    concept: str
    text: str
    meta: Dict[str, Any]
    solution: Any

def gen_add() -> Instance:
    a,b = random.randint(-99,99), random.randint(-99,99)
    return Instance("add", f"Compute {a} + {b}. Return only the integer.", {"a":a,"b":b}, a+b)

def gen_sub() -> Instance:
    a,b = random.randint(-99,99), random.randint(-99,99)
    return Instance("sub", f"Compute {a} - {b}. Return only the integer.", {"a":a,"b":b}, a-b)

def gen_mul() -> Instance:
    a,b = random.randint(-20,20), random.randint(-20,20)
    return Instance("mul", f"Compute {a} * {b}. Return only the integer.", {"a":a,"b":b}, a*b)

def lcs_len(s1: str, s2: str) -> int:
    n,m=len(s1),len(s2)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1]=dp[i][j]+1 if s1[i]==s2[j] else max(dp[i][j+1],dp[i+1][j])
    return dp[n][m]

def gen_lcs() -> Instance:
    import string
    s1="".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(4,8)))
    s2="".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(4,8)))
    return Instance("lcs", f"Compute the length of the Longest Common Subsequence of '{s1}' and '{s2}'. Return only the integer length.", {"s1":s1,"s2":s2}, lcs_len(s1,s2))

def knap_01(values, weights, W):
    dp=[0]*(W+1)
    for v,w in zip(values,weights):
        for c in range(W,w-1,-1):
            dp[c]=max(dp[c], dp[c-w]+v)
    return dp[W]

def gen_knap() -> Instance:
    n=random.randint(4,7)
    values=[random.randint(1,20) for _ in range(n)]
    weights=[random.randint(1,15) for _ in range(n)]
    W=random.randint(sum(weights)//3, sum(weights)//2)
    return Instance("knap",
        "0/1 Knapsack. Given item values and weights and capacity W, return the maximum total value.\n"
        f"values={values}\nweights={weights}\nW={W}\nReturn only the integer maximum value.",
        {"values":values,"weights":weights,"W":W}, knap_01(values,weights,W))

def rod_cut(prices, n):
    dp=[0]*(n+1)
    for i in range(1,n+1):
        best=prices[i-1]
        for c in range(1,i):
            best=max(best, dp[c]+dp[i-c])
        dp[i]=best
    return dp[n]

def gen_rod() -> Instance:
    n=random.randint(4,8)
    prices=[random.randint(1,15) for _ in range(n)]
    return Instance("rod",
        "Rod cutting. Given prices for lengths 1..n and rod length n, return the maximum revenue.\n"
        f"prices={prices} (prices[i-1] is price of length i)\n"
        f"n={n}\nReturn only the integer maximum revenue.",
        {"prices":prices,"n":n}, rod_cut(prices,n))

def assign_min_cost(cost):
    from itertools import permutations
    n=len(cost); best=None; arg=None
    for perm in permutations(range(n)):
        tot=sum(cost[i][perm[i]] for i in range(n))
        if best is None or tot<best: best, arg = tot, list(perm)
    return best,arg

def gen_ilp_assign() -> Instance:
    n=random.randint(3,4)
    cost=[[random.randint(1,20) for _ in range(n)] for _ in range(n)]
    _,perm=assign_min_cost(cost)
    return Instance("ilp_assign",
        "Assignment Problem. Given a square cost matrix, choose a bijection row->col minimizing total cost.\n"
        f"cost={cost}\nReturn only the assignment as a zero-based list like [2,0,1].",
        {"cost":cost}, perm)

def min_partition_diff(arr):
    total=sum(arr); target=total//2
    dp={0}
    for x in arr:
        nxt={s+x for s in dp}
        dp |= nxt
        dp={s for s in dp if s<=target}
    best=max(dp)
    return abs(total-2*best)

def gen_ilp_partition() -> Instance:
    n=random.randint(6,10)
    arr=[random.randint(1,20) for _ in range(n)]
    return Instance("ilp_partition",
        "Partition ILP. Given a list of positive integers, split into two subsets minimizing absolute difference of sums.\n"
        f"arr={arr}\nReturn only the minimal difference as an integer.",
        {"arr":arr}, min_partition_diff(arr))

def gen_ilp_prod() -> Instance:
    from itertools import product
    m=random.randint(2,3)
    A_cap=random.randint(20,40); B_cap=random.randint(20,40)
    a=[random.randint(1,6) for _ in range(m)]
    b=[random.randint(1,6) for _ in range(m)]
    p=[random.randint(3,15) for _ in range(m)]
    ub=[min(A_cap//max(1,a[i]), B_cap//max(1,b[i])) for i in range(m)]
    best=-1
    for xs in product(*[range(u+1) for u in ub]):
        Au=sum(a[i]*xs[i] for i in range(m))
        Bu=sum(b[i]*xs[i] for i in range(m))
        if Au<=A_cap and Bu<=B_cap:
            val=sum(p[i]*xs[i] for i in range(m))
            best=max(best,val)
    return Instance("ilp_prod",
        "Production ILP. Choose nonnegative integers x_i to maximize profit p·x subject to resource constraints:\n"
        f"profit p={p}\nA usage a={a}, capacity A_cap={A_cap}\nB usage b={b}, capacity B_cap={B_cap}\n"
        "Return only the optimal profit as an integer.",
        {"p":p,"a":a,"b":b,"A_cap":A_cap,"B_cap":B_cap}, best)

GEN = {
    "add": gen_add, "sub": gen_sub, "mul": gen_mul,
    "lcs": gen_lcs, "knap": gen_knap, "rod": gen_rod,
    "ilp_assign": gen_ilp_assign, "ilp_partition": gen_ilp_partition, "ilp_prod": gen_ilp_prod,
}

def parse_list_of_ints(s: str) -> Optional[List[int]]:
    m=re.search(r"\[([^\]]+)\]", s)
    if not m: return None
    toks=re.split(r"[,\s]+", m.group(1).strip())
    out=[]
    for t in toks:
        t=t.strip()
        if t=="":
            continue
        try:
            out.append(int(t))
        except:
            return None
    return out

def check_answer_correct(inst: Instance, out: str) -> bool:
    t=out.strip()
    if inst.concept=="ilp_assign":
        lst=parse_list_of_ints(t)
        if not isinstance(lst,list): return False
        n=len(inst.meta["cost"])
        return len(lst)==n and sorted(lst)==list(range(n)) and lst==inst.solution
    else:
        m=re.findall(r"[-+]?\d+", t)
        v=int(m[0]) if m else None
        return v==inst.solution

# ------------------------------------------------------------
# Prompts (generation and scoring)
# ------------------------------------------------------------
def concept_mc_prompt(inst: Instance) -> str:
    # Multiple-choice with letters A..I, used for scoring s_concept
    opts="\n".join([f"{L}) {name}" for L,name in LETTER_MAP.items()])
    return (
        f"""
        You are selecting the best underlying algorithmic concept for the task.

        OPTIONS (exactly one is correct; order is arbitrary):
        A) add
        B) sub
        C) mul
        D) ilp_assign
        E) ilp_partition
        F) ilp_prod
        G) lcs
        H) knap
        I) rod

        TASK:
        {inst.text}

        INSTRUCTIONS:
        - Read the task carefully.
        - Consider all options before choosing.
        - Reply with only a single capital letter A..I.

        Answer:

        """
    )

def answer_prompt(inst: Instance) -> str:
    if inst.concept=="ilp_assign":
        fmt="Return only the assignment as a zero-based list like [2,0,1]. No extra text."
    else:
        fmt="Return only the final integer. No extra text."
    return f"Solve the task.\nTask:\n{inst.text}\n{fmt}"

def canonical_answer_string(inst: Instance) -> str:
    if inst.concept=="ilp_assign":
        return json.dumps(inst.solution)
    return str(int(inst.solution))

# ------------------------------------------------------------
# Generation engines (to produce an answer for A)
# ------------------------------------------------------------
class GenEngine:
    def generate(self, prompt: str, max_new_tokens=128) -> str: raise NotImplementedError

class MockGen(GenEngine):
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        # naive fake
        return "0"

class HFGen(GenEngine):
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tk=AutoTokenizer.from_pretrained(model_name)
        self.m=AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=(__import__("torch").float16 if __import__("torch").cuda.is_available() else None))
        self.device=self.m.device
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        import torch
        toks=self.tk(prompt, return_tensors="pt").to(self.device)
        out=self.m.generate(**toks, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tk.eos_token_id)
        txt=self.tk.decode(out[0], skip_special_tokens=True)
        return txt[len(self.tk.decode(toks["input_ids"][0], skip_special_tokens=True)):].strip()

class VLLMGen(GenEngine):
    def __init__(self, model_name: str):
        from vllm import LLM, SamplingParams
        self.llm=LLM(model=model_name, download_dir = '../models', seed=0)
        self.sp=SamplingParams(temperature=0.0, max_tokens=128)
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        outs=self.llm.generate([prompt], self.sp)
        return outs[0].outputs[0].text.strip()

class OpenAIGen(GenEngine):
    def __init__(self, model_name: str):
        import openai
        self.c=openai.OpenAI(); self.model=model_name
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        r=self.c.chat.completions.create(model=self.model, messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=max_new_tokens)
        return r.choices[0].message.content.strip()

def make_gen_engine(kind: str, model: Optional[str]) -> GenEngine:
    kind=kind.lower()
    if kind=="mock": return MockGen()
    if kind=="hf":   return HFGen(model)
    if kind=="vllm": return VLLMGen(model)
    if kind=="openai": return OpenAIGen(model)
    raise ValueError(kind)

# ------------------------------------------------------------
# HF Scorer (teacher forcing log-likelihoods)
# ------------------------------------------------------------
class HFScorer:
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tk=AutoTokenizer.from_pretrained(model_name)
        self.m=AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=(__import__("torch").float16 if __import__("torch").cuda.is_available() else None))
        self.device=self.m.device
        self.m.eval()

    @torch.no_grad()
    def next_token_logprobs(self, prompt: str) -> Dict[int,float]:
        import torch, torch.nn.functional as F
        enc=self.tk(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        out=self.m(**enc)
        logits=out.logits[:, -1, :]                       # [1, V]
        logps=F.log_softmax(logits, dim=-1)[0]            # [V]
        return {i: float(logps[i].item()) for i in range(logps.shape[0])}

    @torch.no_grad()
    def seq_logprob(self, prompt: str, target: str) -> Tuple[float,int]:
        """
        Sum of log-probs of target tokens conditioned on prompt+prefix; also returns token count.
        """
        import torch, torch.nn.functional as F
        p_ids=self.tk(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)  # [1, P]
        t_ids=self.tk(target, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)  # [1, T]
        inp=torch.cat([p_ids, t_ids], dim=1)                                                   # [1, P+T]
        out=self.m(input_ids=inp)
        logits=out.logits[:, :-1, :]                                                           # shift
        # We want positions corresponding to target tokens: indices P..P+T-1 predict tokens at P+1..P+T
        target_positions = torch.arange(p_ids.shape[1]-1, p_ids.shape[1]-1 + t_ids.shape[1], device=inp.device)
        # Gather logits at those positions
        sel_logits=logits[:, target_positions, :]                                              # [1, T, V]
        logps=F.log_softmax(sel_logits, dim=-1)                                                # [1, T, V]
        lp=logps[0, torch.arange(t_ids.shape[1], device=inp.device), t_ids[0]]                 # [T]
        total=float(lp.sum().item()); T=int(t_ids.shape[1])
        return total, T

# ------------------------------------------------------------
# Soft scores
# ------------------------------------------------------------
def concept_softprob(scorer: HFScorer, inst: Instance) -> float:
    """
    s_concept = probability mass assigned to the true concept letter among A..I.
    Implemented via log-sum-exp over per-option next-token log-likelihoods for the option string.
    """
    prompt = concept_mc_prompt(inst) + " "
    true_letter = INV_LETTER_MAP[inst.concept]
    # Score each letter as a short target (allow tokenization to decide)
    letters = list(LETTER_MAP.keys())
    scores=[]
    for L in letters:
        # Use seq_logprob over the letter token(s)
        lp, T = scorer.seq_logprob(prompt, L)
        # Normalize by tokens to avoid penalizing multi-token letter encodings
        scores.append(lp / max(1,T))
    # Softmax over normalized scores
    m=max(scores); exps=[math.exp(s-m) for s in scores]; Z=sum(exps)
    probs=[e/Z for e in exps]
    return probs[letters.index(true_letter)]

def answer_softprob(scorer: HFScorer, inst: Instance) -> float:
    """
    s_answer = exp(avg_logprob_per_token) of the canonical target answer string.
    """
    prompt = answer_prompt(inst) + "\n"
    target = canonical_answer_string(inst)
    lp, T = scorer.seq_logprob(prompt, target)
    return math.exp(lp / max(1,T))  # in (0,1]

# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
def phi_from_counts(a,b,c,d):
    den = math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    return 0.0 if den==0 else (a*d - b*c)/den

def run(samples_per_concept: int, gen_engine: str, gen_model: Optional[str], scorer_model: str, seed: int, out_csv: str):
    set_seed(seed)
    gen = make_gen_engine(gen_engine, gen_model)
    scorer = HFScorer(scorer_model)

    rows=[]
    for concept in CONCEPTS:
        make = GEN[concept]
        for _ in range(samples_per_concept):
            inst = make()

            # Generate an answer (for A)
            pred_answer = gen.generate(answer_prompt(inst))
            answer_correct = int(check_answer_correct(inst, pred_answer))

            # Soft probabilities
            s_c = concept_softprob(scorer, inst)
            s_a = answer_softprob(scorer, inst)

            rows.append({
                "concept": concept,
                "instance_text": inst.text,
                "answer_pred_text": pred_answer,
                "answer_correct": answer_correct,
                "s_concept_true": s_c,
                "s_answer_target": s_a,
            })

    df=pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Correlations
    # 1) Point-biserial: Pearson between s_concept_true and A
    x=df["s_concept_true"].to_numpy()
    y=df["answer_correct"].to_numpy()
    mx, my = float(x.mean()), float(y.mean())
    num = float(((x-mx)*(y-my)).sum())
    den = math.sqrt(float(((x-mx)**2).sum()) * float(((y-my)**2).sum()))
    r_pointbiserial = (num/den) if den>0 else float("nan")

    # 2) Pearson between s_concept_true and s_answer_target
    z=df["s_answer_target"].to_numpy()
    mz = float(z.mean())
    num2 = float(((x-mx)*(z-mz)).sum())
    den2 = math.sqrt(float(((x-mx)**2).sum()) * float(((z-mz)**2).sum()))
    r_soft_soft = (num2/den2) if den2>0 else float("nan")

    # 3) Optional: calibration by binning s_concept_true
    bins=np.linspace(0,1,11)
    df["bin"]=np.digitize(df["s_concept_true"], bins, right=True)
    cal=df.groupby("bin").agg(n=("answer_correct","size"),
                              acc=("answer_correct","mean"),
                              s_mean=("s_concept_true","mean")).reset_index()

    # 4) Also report 2×2 φ using a threshold on s_concept_true (median split)
    thr=float(np.median(x))
    C=(x>=thr).astype(int)
    A=y.astype(int)
    a=int(((C==1)&(A==1)).sum()); b=int(((C==1)&(A==0)).sum())
    c=int(((C==0)&(A==1)).sum()); d=int(((C==0)&(A==0)).sum())
    phi=phi_from_counts(a,b,c,d)

    print("N:", len(df))
    print(f"Point-biserial r( s_concept_true , A ): {r_pointbiserial:.4f}")
    print(f"Pearson r( s_concept_true , s_answer_target ): {r_soft_soft:.4f}")
    print(f"Phi( A vs [s_concept_true ≥ median] ): {phi:.4f}")
    print("\nCalibration (by s_concept_true bins):\n", cal.to_string(index=False))
    print(f"\nSaved: {out_csv}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples-per-concept", type=int, default=30)
    ap.add_argument("--gen-engine", type=str, default="vllm", choices=["mock","hf","vllm","openai"])
    ap.add_argument("--gen-model", type=str, default=None, help="Model name for generation engine.")
    ap.add_argument("--scorer-model", type=str, required=True, help="HF model name for teacher-forced scoring.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="softcorr_results.csv")
    args=ap.parse_args()
    run(args.samples_per_concept, args.gen_engine, args.gen_model, args.scorer_model, args.seed, args.out)

if __name__=="__main__":
    main()
