#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoT -> joint label classification with LM embeddings + multinomial logistic regression.

Targets (choose with --label):
  - kind         : concept only
  - theta_new    : (kind, digits)
  - gamma        : (kind, digits, value-bin from problem text)   <-- NEW default

Embedding backends (choose with --feats):
  - tfidf              : word+char TF-IDF baseline
  - hf-cls             : HuggingFace Transformer encoder (mean/CLS pooling)
  - st                 : Sentence-Transformers encode (mean pooling)
  - openai             : OpenAI embeddings API (batch)

Input
  * Either --tbdir to read directly from TensorBoard logs (your existing schema),
    or --csv for a flat file with columns: rationale, kind, digits, [prompt], [rep], [split].

Outputs
  * Prints CE/acc/F1 and a variational MI lower bound H(label) - CE.
  * Optional: --save-preds path to save per-example probs & predictions.

Examples
  python cot_ce_logreg_theta_new_embed.py --tbdir out/tb --rep code \
    --label gamma --feats st --embed-model sentence-transformers/all-MiniLM-L6-v2 --bits

  python cot_ce_logreg_theta_new_embed.py --csv cot_db.csv --rep nl \
    --label theta_new --feats hf-cls --embed-model google-bert/bert-base-uncased --pool mean

  python cot_ce_logreg_theta_new_embed.py --tbdir out/tb --feats openai \
    --embed-model text-embedding-3-large --batch 128 --label gamma --value-bins 8
"""

import argparse, os, math, sys, re, json
from typing import Dict, Any, Tuple, Optional, List
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------- TB parsing ----------------

_TAG_RE = re.compile(
    r'^(?:text_summary/)?'
    r'(?P<model>.+?)/(?P<rep>nl|code)/'
    r'd(?P<digits>\d+)/(?P<kind>[^/]+)/i(?P<idx>\d+)/'
    r'(?P<leaf>prompt|rationale|raw_json|full|answer)'
    r'(?:/text_summary)?$'
)

def _strip_md_block(s: str) -> str:
    m = re.search(r"```(?:[a-zA-Z0-9]+)?\s*\n([\s\S]*?)\n```", s, flags=re.S)
    return m.group(1).strip() if m else (s or "").strip()

def _tb_text_from_event(ev) -> str:
    tp = ev.tensor_proto
    if tp.string_val:
        val = tp.string_val[0]
        return val.decode("utf-8","replace") if isinstance(val,(bytes,bytearray)) else str(val)
    if tp.tensor_content:
        try:    return tp.tensor_content.decode("utf-8","replace")
        except: pass
    return ""

def _iter_event_dirs(root: str):
    if os.path.isdir(root) and any(fn.startswith("events") for fn in os.listdir(root)):
        yield root
    else:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p) and any(fn.startswith("events") for fn in os.listdir(p)):
                yield p

def _parse_raw_json(s: str):
    try:
        start = s.index('{'); end = s.rindex('}')
        frag = s[start:end+1]
        obj = json.loads(frag)
        rat = obj.get("rationale")
        ans = obj.get("answer")
        try:
            ans = int(ans) if ans is not None else None
        except Exception:
            nums = re.findall(r"[-+]?\d+", str(ans))
            ans = int(nums[-1]) if nums else None
        return rat, ans
    except Exception:
        return None, None

def load_from_tb(tbdir: str, prefer_tag_rationale: bool=False) -> pd.DataFrame:
    rows: Dict[Tuple[str,str,int,str,int], Dict[str,Any]] = {}
    for ed in _iter_event_dirs(tbdir):
        ea = EventAccumulator(ed, size_guidance={'tensors': 10**7})
        ea.Reload()
        tags = list(ea.Tags().get('tensors', []))
        tags += [t for t in ea.Tags().get('text', []) if t not in tags]

        cache: Dict[Tuple[str,str,int,str,int,str], str] = {}
        for tag in tags:
            m = _TAG_RE.match(tag)
            if not m: 
                continue
            model = m['model']; rep = m['rep']
            digits = int(m['digits']); kind = m['kind']; idx = int(m['idx'])
            leaf = m['leaf']
            evs = ea.Tensors(tag) or []
            if not evs: 
                continue
            txt = _strip_md_block(_tb_text_from_event(evs[-1]))
            cache[(model,rep,digits,kind,idx,leaf)] = txt

        for (model,rep,digits,kind,idx,_) in list(cache.keys()):
            key = (model,rep,digits,kind,idx)
            if key not in rows:
                rows[key] = dict(model=model,rep=rep,kind=kind,digits=digits,idx=idx,
                                 prompt="",rationale="",answer=None,raw_json="")
            prompt = cache.get((model,rep,digits,kind,idx,"prompt"))
            if prompt: rows[key]["prompt"] = prompt

            raw_json = cache.get((model,rep,digits,kind,idx,"raw_json")) or cache.get((model,rep,digits,kind,idx,"full"))
            if raw_json:
                rows[key]["raw_json"] = raw_json
                rj_rat, rj_ans = _parse_raw_json(raw_json)
                if rj_ans is not None: rows[key]["answer"] = rj_ans
                if rj_rat and not prefer_tag_rationale:
                    rows[key]["rationale"] = rj_rat

            tag_rat = cache.get((model,rep,digits,kind,idx,"rationale"))
            if tag_rat and (prefer_tag_rationale or not rows[key]["rationale"]):
                rows[key]["rationale"] = tag_rat

            tag_ans = cache.get((model,rep,digits,kind,idx,"answer"))
            if tag_ans and rows[key]["answer"] is None:
                nums = re.findall(r"[-+]?\d+", tag_ans)
                if nums: rows[key]["answer"] = int(nums[-1])

    df = pd.DataFrame([rows[k] for k in sorted(rows.keys(), key=lambda t:(t[0],t[1],t[2],t[3],t[4]))])
    df = df[df["rationale"].astype(str).str.len() > 0].reset_index(drop=True)
    return df

# ---------------- gamma construction ----------------

def _ints_in_text(s: str):
    return [int(x) for x in re.findall(r'\d+', s or "")]

def _bin_equal_width(x: int, U: int, K: int) -> int:
    base = 10 ** max(1, int(U))
    x = max(0, min(base-1, int(x)))
    return min(K-1, (x * K) // base)
# ---------------- gamma construction (kind, digits, value-bin) ----------------
import re

_INT_RE = re.compile(r'[-+]?\d+')

def _ints_in_text(s: str):
    s = s or ""
    return [int(x) for x in _INT_RE.findall(s)]

def _ew_bin_lohi(x: int, lo: int, hi: int, K: int) -> int:
    """
    Equal-width binning for integer x over the CLOSED interval [lo, hi].
    Returns an index in {0, ..., K-1}.
    """
    if K <= 1:
        return 0
    lo = int(lo); hi = int(hi)
    if lo > hi:
        lo, hi = hi, lo
    x = int(x)
    if x < lo: x = lo
    if x > hi: x = hi
    span = (hi - lo + 1)
    # floor( (x - lo) * K / span ), capped at K-1
    idx = ( (x - lo) * K ) // span
    return min(K-1, max(0, int(idx)))

def _safe_len(lst):
    try: return len(lst)
    except: return 0

def make_gamma_label(kind: str,
                     digits: int,
                     problem_text: str,
                     K_bins: int = 8,
                     use_joint_id: bool = True) -> str:
    """
    γ = (kind, digits, value-bin). For arithmetic kinds we parse the first
    two integers (operands A,B) from the problem text and bin them with
    equal-width bins over the *correct* magnitude interval:
        A,B ∈ [10^{d-1}, 10^d - 1]

    For non-arithmetic kinds we construct simple bins from problem size/shape
    stats so γ is always finer than (kind,digits) but not exploding.

    Returns a string label "kind|d{digits}|b{bin_id}".
    """
    k = str(kind)
    d = int(digits)
    t = problem_text or ""

    # ---------- Arithmetic: bin A and B jointly ----------
    if k in {"add","sub","mul","mix"}:
        nums = _ints_in_text(t)
        # We expect A and B to be the first two integers in the formatted prompt.
        A = nums[0] if _safe_len(nums) >= 1 else 10**(d-1)
        B = nums[1] if _safe_len(nums) >= 2 else 10**(d-1)
        lo = 10**(d-1)
        hi = 10**d - 1
        ba = _ew_bin_lohi(A, lo, hi, K_bins)
        bb = _ew_bin_lohi(B, lo, hi, K_bins)
        if use_joint_id:
            bin_id = ba * K_bins + bb           # K^2 total bins per (kind,d)
        else:
            bin_id = (ba, bb)                   # tuple if you prefer
        return f"{k}|d{d}|b{bin_id}"

    # ---------- LCS: bin by (|S|, |T|) ----------
    if k == "lcs":
        # Problem text template puts strings on lines like S="..." T="..."
        # Fallback: derive rough lengths from the prompt if exact parsing fails.
        # Here we just extract quoted strings as S,T if possible:
        # (works with your current prompt formatting)
        S_match = re.search(r'S\s*=\s*"([^"]*)"', t)
        T_match = re.search(r'T\s*=\s*"([^"]*)"', t)
        Ls = len(S_match.group(1)) if S_match else max(2, d)
        Lt = len(T_match.group(1)) if T_match else max(2, d)
        bLs = _ew_bin_lohi(Ls, 1, max(2, 2*d), K_bins)
        bLt = _ew_bin_lohi(Lt, 1, max(2, 2*d), K_bins)
        return f"{k}|d{d}|b{bLs*K_bins + bLt}"

    # ---------- Knapsack: bin by (#items, capacity-ratio) ----------
    if k == "knap":
        # Extract W, V, C lists if present in the prompt (they are printed as Python lists)
        # If parsing is brittle, you can log structured fields to TB and read them directly.
        # Here we just count items and derive a rough cap_ratio from numbers.
        # Better: store these in TB and read back.
        # Try to parse "W = [...]", "V = [...]", "C = x"
        Wm = re.search(r'W\s*=\s*\[([^\]]*)\]', t)
        Cm = re.search(r'C\s*=\s*([0-9]+)', t)
        if Wm:
            W = [int(x) for x in _INT_RE.findall(Wm.group(1))]
            n_items = len(W)
            C = int(Cm.group(1)) if Cm else max(1, int(0.5 * sum(W)))
            cap_ratio = C / max(1, sum(W))
        else:
            n_items = max(3, d)
            cap_ratio = 0.5
        bN = _ew_bin_lohi(n_items, 1, max(3, 2*d), K_bins)
        bR = _ew_bin_lohi(int(cap_ratio * 1000), 0, 1000, K_bins)  # 0..1 scaled to 0..1000
        return f"{k}|d{d}|b{bN*K_bins + bR}"

    # ---------- Rod cutting: bin by N (#prices) ----------
    if k == "rod":
        Nm = re.search(r'N\s*=\s*([0-9]+)', t)
        N = int(Nm.group(1)) if Nm else max(2, d)
        bN = _ew_bin_lohi(N, 1, max(2, 2*d), K_bins)
        return f"{k}|d{d}|b{bN}"

    # ---------- ILP assignment: bin by n (matrix size) ----------
    if k == "ilp_assign":
        # Prompt prints "C = [[...],[...],...]"; approximate n by counting '[' at top level
        # but safer (and simple): estimate n from digits: n≈min(d,7)
        n = min(max(2, d), 7)
        bN = _ew_bin_lohi(n, 2, 7, K_bins)
        return f"{k}|d{d}|b{bN}"

    # ---------- ILP production: bin by (#products, #resources) ----------
    if k == "ilp_prod":
        # From make_problem: P≈min(2 + d//3, 6), R≈min(2 + d//4, 4)
        P = min(2 + d // 3, 6)
        R = min(2 + d // 4, 4)
        bP = _ew_bin_lohi(P, 2, 6, K_bins)
        bR = _ew_bin_lohi(R, 2, 4, K_bins)
        return f"{k}|d{d}|b{bP*K_bins + bR}"

    # ---------- ILP partition: bin by #items ----------
    if k == "ilp_partition":
        # From make_problem: n_items≈min(d,24)
        n_items = min(max(4, d), 24)
        bN = _ew_bin_lohi(n_items, 4, 24, K_bins)
        return f"{k}|d{d}|b{bN}"

    # Fallback
    return f"{k}|d{d}|bNA"


# ---------------- utilities ----------------

def empirical_entropy_bits(labels: List[str]) -> float:
    cnt = Counter(labels)
    n = sum(cnt.values())
    ps = np.array([c/n for c in cnt.values() if c > 0], dtype=float)
    return float(-(ps * (np.log(ps)/np.log(2))).sum())

def train_test_split_or_use_column(df, y_col="label", split_col="split", test_size=0.2, seed=0):
    if split_col in df.columns:
        tr = df[df[split_col].astype(str).str.lower()=="train"]
        te = df[df[split_col].astype(str).str.lower()=="test"]
        if len(tr) and len(te): return tr, te
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[y_col])

def maybe_strip_fences(text: str) -> str:
    return (text or "").replace("```python", " ").replace("```", " ")

# ---------------- featurizers ----------------

class Featurizer:
    def fit(self, texts: List[str]): return self
    def transform(self, texts: List[str]) -> np.ndarray: raise NotImplementedError

class TfidfFeaturizer(Featurizer):
    def __init__(self, strip_fences: bool):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9, max_features=200_000,
            lowercase=True, token_pattern=r"(?u)\b\w+\b",
            preprocessor=maybe_strip_fences if strip_fences else None,
        )
        char_vec = TfidfVectorizer(
            analyzer="char", ngram_range=(3,5), min_df=3, max_df=0.98, lowercase=False,
            preprocessor=maybe_strip_fences if strip_fences else None,
        )
        from sklearn.pipeline import FeatureUnion as FU
        self.vec = FU([("w", word_vec), ("c", char_vec)])
    def fit(self, texts): self.vec.fit(texts); return self
    def transform(self, texts): return self.vec.transform(texts)

class HFCLSFeaturizer(Featurizer):
    def __init__(self, model_name: str, pool: str = "mean", device: Optional[str] = None, trust_remote_code: bool=False):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pool = pool
        self.torch = __import__("torch")

    def _pool(self, last_hidden_state, attention_mask):
        if self.pool == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    @torch.no_grad()
    def transform(self, texts: List[str]) -> np.ndarray:
        torch = self.torch
        OUT = []
        bs = 32
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            enc = {k: v.to(self.device) for k,v in enc.items()}
            out = self.model(**enc)
            hid = out.last_hidden_state
            pooled = self._pool(hid, enc["attention_mask"])
            OUT.append(pooled.cpu().numpy())
        return np.vstack(OUT)

class SentenceTransformersFeaturizer(Featurizer):
    def __init__(self, model_name: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        import torch
        self.model = SentenceTransformer(model_name, device=(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    def transform(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False))

class OpenAIEmbeddingFeaturizer(Featurizer):
    def __init__(self, model_name: str, batch: int = 128):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("pip install openai>=1.0 required for --feats openai") from e
        self.client = OpenAI()
        self.model = model_name
        self.batch = int(batch)
    def transform(self, texts: List[str]) -> np.ndarray:
        OUT = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i+self.batch]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            OUT.append(np.stack(vecs, axis=0))
        return np.vstack(OUT)

def build_featurizer(kind: str, embed_model: Optional[str], pool: str, strip_fences: bool, device: Optional[str], batch: int):
    kind = kind.lower()
    if kind == "tfidf":
        return TfidfFeaturizer(strip_fences=strip_fences)
    if kind == "hf-cls":
        if not embed_model:
            raise ValueError("--embed-model is required for --feats hf-cls")
        return HFCLSFeaturizer(embed_model, pool=pool, device=device)
    if kind == "st":
        if not embed_model:
            raise ValueError("--embed-model is required for --feats st")
        return SentenceTransformersFeaturizer(embed_model, device=device)
    if kind == "openai":
        if not embed_model:
            embed_model = "text-embedding-3-large"
        return OpenAIEmbeddingFeaturizer(embed_model, batch=batch)
    raise ValueError(f"Unknown --feats {kind}")

# ---------------- train/eval ----------------

def run(args):
    # Load
    if args.tbdir:
        df = load_from_tb(args.tbdir)
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        raise SystemExit("Provide either --tbdir or --csv")

    print("Loaded data")
    need = {"rationale","kind","digits"}
    if not need.issubset(df.columns):
        raise ValueError(f"Data must contain columns {need}. Got: {df.columns.tolist()}")

    # Filter by rep if present and requested
    if args.rep != "all" and "rep" in df.columns:
        df = df[df["rep"].astype(str).str.lower() == args.rep]
        if len(df) == 0:
            raise ValueError(f"No rows after filtering rep={args.rep}")

    # Make auxiliary columns
    df["digits"] = df["digits"].astype(int)
    if "prompt" not in df.columns:
        df["prompt"] = ""

    df["theta_new"] = df["kind"].astype(str) + "__d" + df["digits"].astype(int).astype(str)
    # Prefer an explicit 'prompt' column; else fall back to 'problem' if present; else empty.
    src_text = df["prompt"].astype(str) if "prompt" in df.columns else pd.Series([""]*len(df))
    if "problem" in df.columns:
        src_text = np.where(src_text.str.len() > 0, src_text, df["problem"].astype(str))

    df["gamma"] = [
        make_gamma_label(k, int(d), t, K_bins=args.value_bins)
        for k, d, t in zip(df["kind"].astype(str), df["digits"].astype(int), src_text.astype(str))
    ]
    print("Created Columns")
    
    # Choose label
    label_col = {"theta_new":"theta_new", "gamma":"gamma", "kind":"kind"}[args.label]
    df = df[df[label_col].astype(str).str.len() > 0].reset_index(drop=True)
    df = df[df["rationale"].astype(str).str.len() > 0].reset_index(drop=True)

    # Split
    df = df.copy()
    df["label"] = df[label_col].astype(str)
    train_df, test_df = train_test_split_or_use_column(df, y_col="label", test_size=args.test_size, seed=args.seed)

    # Features
    texts_tr = train_df["rationale"].astype(str).tolist()
    texts_te = test_df["rationale"].astype(str).tolist()
    feats = build_featurizer(args.feats, args.embed_model, args.pool, args.strip_fences, args.device, args.batch)
    if hasattr(feats, "fit"):
        feats.fit(texts_tr)
    Xtr = feats.transform(texts_tr)
    Xte = feats.transform(texts_te)
    print("Extracted Features")
    
    # Labels -> ids
    le = LabelEncoder()
    ytr = le.fit_transform(train_df["label"].astype(str).values)
    yte = le.transform(test_df["label"].astype(str).values)
    classes_idx = np.arange(len(le.classes_))

    print("Extracted Labels")
    # Classifier
    clf = LogisticRegression(
        penalty="l2", C=args.C, solver="saga", multi_class="multinomial",
        max_iter=args.max_iter, n_jobs=-1, verbose=0
    )
    clf.fit(Xtr, ytr)
    P = clf.predict_proba(Xte)
    yhat = P.argmax(1)
    print("Finish Prediction")
    
    # Metrics
    ce_nat = log_loss(yte, P, labels=classes_idx)
    acc = accuracy_score(yte, yhat)
    f1m = f1_score(yte, yhat, average="macro")

    to_bits = 1.0 / math.log(2.0) if args.bits else 1.0
    unit = "bits" if args.bits else "nats"
    H_bits = empirical_entropy_bits(test_df["label"].astype(str).values)
    CE_bits = ce_nat * to_bits
    I_lb_bits = H_bits - CE_bits

    print("\n=== Results (embedding features) ===")
    print(f"Target label:     {args.label}  (#classes={len(le.classes_)})")
    print(f"Feats:            {args.feats} | model={args.embed_model or 'n/a'} | pool={args.pool if args.feats=='hf-cls' else '-'}")
    print(f"Rep filter:       {args.rep}")
    print(f"N_train / N_test: {len(train_df)} / {len(test_df)}")
    print(f"Cross-entropy:    {CE_bits:.4f} {unit}")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Macro F1:         {f1m:.4f}")
    print(f"H({args.label}):  {H_bits:.4f} bits  (empirical on test)")
    print(f"I({args.label}; Z_r) ≥ {I_lb_bits:.4f} bits   (variational lower bound)")

    # Save predictions (optional)
    if args.save_preds:
        out = test_df[["rationale","kind","digits","prompt"]].copy()
        out["true_label"] = test_df["label"].values
        out["pred_label"] = le.inverse_transform(yhat)
        out["neglogp_true_nat"] = -np.log(P[np.arange(len(P)), yte] + 1e-15)
        out.to_csv(args.save_preds, index=False)
        print(f"Saved predictions: {args.save_preds}")

    # Optional K-fold CV
    if args.cv and args.cv > 1:
        print(f"\n=== {args.cv}-fold CV ({args.feats}, label={args.label}) ===")
        y_all = le.transform(df["label"].astype(str).values)
        ce_list, acc_list = [], []
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        X_all_text = df["rationale"].astype(str).values
        for k, (tr, te) in enumerate(skf.split(X_all_text, y_all), 1):
            fz = build_featurizer(args.feats, args.embed_model, args.pool, args.strip_fences, args.device, args.batch)
            tr_texts = X_all_text[tr].tolist(); te_texts = X_all_text[te].tolist()
            if hasattr(fz, "fit"): fz.fit(tr_texts)
            Xtr = fz.transform(tr_texts); Xte = fz.transform(te_texts)
            lr = LogisticRegression(penalty="l2", C=args.C, solver="saga", multi_class="multinomial",
                                    max_iter=args.max_iter, n_jobs=-1)
            lr.fit(Xtr, y_all[tr])
            Pte = lr.predict_proba(Xte)
            ce = log_loss(y_all[te], Pte, labels=np.arange(len(le.classes_)))
            ac = accuracy_score(y_all[te], Pte.argmax(1))
            ce_list.append(ce); acc_list.append(ac)
            print(f"  fold {k}: CE={(ce*to_bits):.4f} {unit} | acc={ac:.4f}")
        print(f"CV mean CE={(np.mean(ce_list)*to_bits):.4f} {unit} (±{np.std(ce_list)*to_bits:.4f}), "
              f"acc={np.mean(acc_list):.4f} (±{np.std(acc_list):.4f})")

# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tbdir", type=str, default=None, help="Read rationales directly from TensorBoard logs")
    p.add_argument("--csv", type=str, default=None, help="CSV with columns: rationale, kind, digits, [prompt], [rep], [split]")
    p.add_argument("--rep", choices=["nl","code","all"], default="all")

    # Labels
    p.add_argument("--label", choices=["theta_new","gamma","kind"], default="gamma",
                   help="Classification target: θ_new=(kind,digits) or γ=(kind,digits,value-bin) or concept only.")
    p.add_argument("--value-bins", type=int, default=8, help="Number of equal-width bins per operand for gamma.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cv", type=int, default=0)

    # Embeddings
    p.add_argument("--feats", choices=["tfidf","hf-cls","st","openai"], default="tfidf")
    p.add_argument("--embed-model", type=str, default=None,
                   help="hf-cls: HF repo; st: ST repo; openai: embedding model.")
    p.add_argument("--pool", choices=["mean","cls"], default="mean", help="Pooling for hf-cls.")
    p.add_argument("--device", type=str, default=None, help="Force device for hf-cls/st (e.g. cuda, cpu).")
    p.add_argument("--batch", type=int, default=128, help="Batch size for OpenAI embeddings.")
    p.add_argument("--strip-fences", action="store_true", help="Strip ``` code fences before embedding (tfidf & hf-cls/st).")

    # Classifier
    p.add_argument("--C", type=float, default=2.0)
    p.add_argument("--max_iter", type=int, default=400)

    # Reporting
    p.add_argument("--bits", action="store_true", help="Report CE in bits and print MI lower bound.")
    p.add_argument("--save-preds", type=str, default=None, help="Optional path to save test predictions CSV.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
