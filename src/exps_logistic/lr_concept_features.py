# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# CoT -> joint label classification with LM embeddings + multinomial logistic regression.

# Targets (choose with --label):
#   - kind         : concept only
#   - theta_new    : (kind, digits)
#   - gamma        : (kind, digits, value-bin from problem text)   <-- NEW default

# Embedding backends (choose with --feats):
#   - tfidf              : word+char TF-IDF baseline
#   - hf-cls             : HuggingFace Transformer encoder (mean/CLS pooling, robust chunking to 512)
#   - st                 : Sentence-Transformers encode (mean pooling)
#   - openai             : OpenAI embeddings API (batch)
# """

# import argparse
# import math
# import re
# from collections import Counter
# from typing import List, Optional

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score, log_loss
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.preprocessing import LabelEncoder

# # ------------------------------------------------------------------------------
# # Utilities
# # ------------------------------------------------------------------------------


# def empirical_entropy_bits(labels: List[str]) -> float:
#     cnt = Counter(labels)
#     n = sum(cnt.values())
#     ps = np.array([c / n for c in cnt.values() if c > 0], dtype=float)
#     return float(-(ps * (np.log(ps) / np.log(2))).sum())


# def maybe_strip_fences(text: str) -> str:
#     return (text or "").replace("```python", " ").replace("```", " ")


# def stratified_split_robust(df, y_col="label", test_size=0.2, seed=0, min_count=2, verbose=True):
#     y = df[y_col].astype(str).values
#     cnt = Counter(y)
#     keep_mask = df[y_col].map(cnt).ge(min_count)
#     dropped = int((~keep_mask).sum())
#     if dropped and verbose:
#         print(f"[split] Dropping {dropped} samples from classes with <{min_count} total examples.")
#     df = df[keep_mask].reset_index(drop=True)
#     if len(df) == 0:
#         raise ValueError("All samples dropped due to rare classes; try lowering bin granularity (--value-bins).")

#     ts = float(test_size)
#     for _ in range(6):
#         try:
#             tr, te = train_test_split(df, test_size=ts, random_state=seed, stratify=df[y_col])
#             return tr, te
#         except ValueError as e:
#             ts *= 0.5
#             if verbose:
#                 print(f"[split] Stratified split failed ({e}); retrying with test_size={ts:.4f}")
#             if ts < 0.02:
#                 break
#     if verbose:
#         print("[split] Falling back to non-stratified split.")
#     return train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)


# # ------------------------------------------------------------------------------
# # Robust problem slicing from NL/CODE prompts
# # ------------------------------------------------------------------------------

# # Primary wrapper markers used in your NL_PROMPT / CODE_PROMPT
# _PROBLEM_SLICE_RE = re.compile(
#     r"Here\s+is\s+the\s+actual\s+problem:\s*(.*?)\s*Give\s+the\s+solution:",
#     re.DOTALL | re.IGNORECASE,
# )

# # Code-fence stripper (keeps inner text)
# _FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n([\s\S]*?)\n```", re.MULTILINE)


# def _strip_md_code_fences(s: str) -> str:
#     if not s:
#         return ""
#     return _FENCE_RE.sub(lambda m: m.group(1), s)


# # Known per-kind headers emitted by Problem.text()
# _KNOWN_KIND_HEADERS = [
#     r"^Compute:\s*\(",  # mix
#     r"^Compute:\s*\d+\s*[\+\-\*]\s*\d+",  # add/sub/mul
#     r"^Compute the length of the Longest Common Subsequence",
#     r"^0/1 Knapsack:",
#     r"^Rod cutting:",
#     r"^Assignment problem:",
#     r"^Production planning:",
#     r"^Partition:",
# ]
# _KIND_START_RE = re.compile("|".join(_KNOWN_KIND_HEADERS), re.MULTILINE)


# def extract_problem_text(full_prompt: str) -> str:
#     """
#     Return ONLY the {problem} block printed by Problem.text().
#     Works with:
#       - Your NL/CODE templates (between 'Here is the actual problem:' and 'Give the solution:')
#       - Raw {problem} text (no wrapper)
#       - Prompts wrapped in code fences
#       - Extra text before/after (falls back to scanning for known headers)
#     """
#     s = full_prompt or ""
#     s = _strip_md_code_fences(s).strip()

#     # Ideal path: inside the NL/CODE wrapper.
#     m = _PROBLEM_SLICE_RE.search(s)
#     if m:
#         return m.group(1).strip()

#     # Fallback: find the first line that looks like a Problem.text() header and
#     # capture that paragraph (until the next blank line or end).
#     m2 = _KIND_START_RE.search(s)
#     if m2:
#         start = m2.start()
#         tail = s[start:].strip()
#         parts = re.split(r"\n\s*\n", tail, maxsplit=1)
#         return parts[0].strip()

#     # Last resort: if the prompt is already just the problem, return it.
#     return s


# # ------------------------------------------------------------------------------
# # TB parsing (unchanged except we prefer our slicing later)
# # ------------------------------------------------------------------------------

# _TAG_RE = re.compile(
#     r"^(?:text_summary/)?"
#     r"(?P<model>.+?)/(?P<rep>nl|code)/"
#     r"d(?P<digits>\d+)/(?P<kind>[^/]+)/i(?P<idx>\d+)/"
#     r"(?P<leaf>prompt|rationale|raw_json|full|answer)"
#     r"(?:/text_summary)?$"
# )
# # ------------------------------------------------------------------------------
# # gamma construction: strict parsers tied to Problem.text()
# # ------------------------------------------------------------------------------

# _INT_RE = re.compile(r"[-+]?\d+")


# def _ew_bin_lohi(x: int, lo: int, hi: int, K: int) -> int:
#     """Equal-width binning for CLOSED interval [lo, hi], returns in {0..K-1}."""
#     if K <= 1:
#         return 0
#     lo = int(lo)
#     hi = int(hi)
#     if lo > hi:
#         lo, hi = hi, lo
#     x = int(x)
#     if x < lo:
#         x = lo
#     if x > hi:
#         x = hi
#     span = hi - lo + 1
#     idx = ((x - lo) * K) // span
#     return max(0, min(K - 1, int(idx)))


# def _safe_len(x):
#     try:
#         return len(x)
#     except Exception:
#         return 0


# def _parse_list_of_ints(s: str) -> List[int]:
#     return [int(x) for x in _INT_RE.findall(s or "")]


# def _parse_list_of_list_ints(s: str) -> List[List[int]]:
#     rows = []
#     for row in re.findall(r"\[([^\[\]]*)\]", s or ""):
#         vals = [int(x) for x in _INT_RE.findall(row)]
#         if vals:
#             rows.append(vals)
#     return rows


# # Arithmetic lines (strictly on the sliced problem text)
# _AR_ADD = re.compile(r"^\s*Compute:\s*(\d+)\s*\+\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
# _AR_SUB = re.compile(r"^\s*Compute:\s*(\d+)\s*-\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
# _AR_MUL = re.compile(r"^\s*Compute:\s*(\d+)\s*\*\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
# _AR_MIX = re.compile(r"^\s*Compute:\s*\(\s*(\d+)\s*\+\s*(\d+)\s*\)\s*\*\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)


# def _parse_arithmetic_operands(kind: str, text: str, d: int) -> Optional[tuple]:
#     if kind == "add":
#         m = _AR_ADD.search(text)
#         return (int(m.group(1)), int(m.group(2))) if m else None
#     if kind == "sub":
#         m = _AR_SUB.search(text)
#         return (int(m.group(1)), int(m.group(2))) if m else None
#     if kind == "mul":
#         m = _AR_MUL.search(text)
#         return (int(m.group(1)), int(m.group(2))) if m else None
#     if kind == "mix":
#         m = _AR_MIX.search(text)
#         if m:
#             a, b, _a2 = map(int, m.groups())
#             return a, b
#     return None


# def _parse_lcs_lengths(text: str, d: int) -> tuple:
#     Sm = re.search(r'S\s*=\s*"([^"]*)"', text)
#     Tm = re.search(r'T\s*=\s*"([^"]*)"', text)
#     Ls = len(Sm.group(1)) if Sm else max(2, d)
#     Lt = len(Tm.group(1)) if Tm else max(2, d)
#     return Ls, Lt


# def _parse_knap_stats(text: str, d: int) -> tuple:
#     Wm = re.search(r"W\s*=\s*\[([^\]]*)\]", text)
#     Vm = re.search(r"V\s*=\s*\[([^\]]*)\]", text)
#     Cm = re.search(r"C\s*=\s*([0-9]+)", text)
#     if Wm and Vm:
#         W = _parse_list_of_ints(Wm.group(1))
#         n_items = len(W)
#         C = int(Cm.group(1)) if Cm else max(1, int(0.5 * sum(W)))
#         cap_ratio = C / max(1, sum(W))
#     else:
#         n_items = max(3, d)
#         cap_ratio = 0.5
#     return n_items, cap_ratio


# def _parse_rod_N(text: str, d: int) -> int:
#     Nm = re.search(r"\bN\s*=\s*([0-9]+)", text)
#     if Nm:
#         return int(Nm.group(1))
#     Pm = re.search(r"P\s*=\s*\[([^\]]*)\]", text)
#     if Pm:
#         return len(_parse_list_of_ints(Pm.group(1)))
#     return max(2, d)


# def _parse_ilp_assign_n(text: str, d: int) -> int:
#     Cm = re.search(r"C\s*=\s*(\[[\s\S]*\])\s*$", text, re.MULTILINE)
#     if Cm:
#         mat = _parse_list_of_list_ints(Cm.group(1))
#         if _safe_len(mat) > 0:
#             return len(mat)
#     return min(max(2, d), 7)


# def _parse_ilp_prod_PR(text: str, d: int) -> tuple:
#     Pm = re.search(r"profit\s*=\s*\[([^\]]*)\]", text)
#     Cm = re.search(r"consumption\s*\(rows=resources\)\s*=\s*(\[[\s\S]*\])", text)
#     if Pm:
#         P = len(_parse_list_of_ints(Pm.group(1)))
#     else:
#         P = min(2 + d // 3, 6)
#     if Cm:
#         cons = _parse_list_of_list_ints(Cm.group(1))
#         R = len(cons) if _safe_len(cons) > 0 else min(2 + d // 4, 4)
#     else:
#         R = min(2 + d // 4, 4)
#     return int(P), int(R)


# def _parse_ilp_partition_n(text: str, d: int) -> int:
#     Wm = re.search(r"weights\s*=\s*\[([^\]]*)\]", text)
#     if Wm:
#         return len(_parse_list_of_ints(Wm.group(1)))
#     return min(max(4, d), 24)


# def make_gamma_label(kind: str, digits: int, problem_text: str, K_bins: int = 8, use_joint_id: bool = True) -> str:
#     """
#     γ = (kind, digits, value-bin), where the value-bin is derived from fields
#     parsed *exactly* from Problem.text() for each kind.
#     """
#     k = str(kind)
#     d = int(digits)
#     t = problem_text or ""

#     # Arithmetic: operands in [10^(d-1), 10^d - 1]
#     if k in {"add", "sub", "mul", "mix"}:
#         parsed = _parse_arithmetic_operands(k, t, d)
#         lo = 10 ** (d - 1)
#         hi = 10**d - 1
#         if parsed is None:
#             return f"{k}|d{d}|bNA"
#         A, B = parsed
#         ba = _ew_bin_lohi(A, lo, hi, K_bins)
#         bb = _ew_bin_lohi(B, lo, hi, K_bins)
#         bin_id = ba * K_bins + bb if use_joint_id else (ba, bb)
#         return f"{k}|d{d}|b{bin_id}"

#     # LCS: (|S|, |T|)
#     if k == "lcs":
#         Ls, Lt = _parse_lcs_lengths(t, d)
#         bLs = _ew_bin_lohi(Ls, 1, max(2, 2 * d), K_bins)
#         bLt = _ew_bin_lohi(Lt, 1, max(2, 2 * d), K_bins)
#         return f"{k}|d{d}|b{bLs*K_bins + bLt}"

#     # Knapsack: (#items, capacity-ratio)
#     if k == "knap":
#         n_items, cap_ratio = _parse_knap_stats(t, d)
#         bN = _ew_bin_lohi(n_items, 1, max(3, 2 * d), K_bins)
#         bR = _ew_bin_lohi(int(round(cap_ratio * 1000)), 0, 1000, K_bins)  # 0..1 -> 0..1000
#         return f"{k}|d{d}|b{bN*K_bins + bR}"

#     # Rod: N
#     if k == "rod":
#         N = _parse_rod_N(t, d)
#         bN = _ew_bin_lohi(N, 1, max(2, 2 * d), K_bins)
#         return f"{k}|d{d}|b{bN}"

#     # ILP assignment: n
#     if k == "ilp_assign":
#         n = _parse_ilp_assign_n(t, d)
#         bN = _ew_bin_lohi(n, 2, 7, K_bins)
#         return f"{k}|d{d}|b{bN}"

#     # ILP production: (P, R)
#     if k == "ilp_prod":
#         P, R = _parse_ilp_prod_PR(t, d)
#         bP = _ew_bin_lohi(P, 2, 6, K_bins)
#         bR = _ew_bin_lohi(R, 2, 4, K_bins)
#         return f"{k}|d{d}|b{bP*K_bins + bR}"

#     # ILP partition: #items
#     if k == "ilp_partition":
#         n_items = _parse_ilp_partition_n(t, d)
#         bN = _ew_bin_lohi(n_items, 4, 24, K_bins)
#         return f"{k}|d{d}|b{bN}"

#     return f"{k}|d{d}|bNA"


# def run(args):
#     # Load

#     df = pd.read_csv(args.csv)

#     print("Loaded data")
#     need = {"rationale", "kind", "digits"}  # different types of rationales
#     if not need.issubset(df.columns):
#         raise ValueError(f"Data must contain columns {need}. Got: {df.columns.tolist()}")

#     # Filter by rep if present and requested, should be sim_answer vs nl_answer, should only be our examples.
#     if args.rep != "all" and "rep" in df.columns:
#         df = df[df["rep"].astype(str).str.lower() == args.rep]
#         if len(df) == 0:
#             raise ValueError(f"No rows after filtering rep={args.rep}")

#     # Aux columns
#     df["digits"] = df["digits"].astype(int)
#     if "prompt" not in df.columns:
#         df["prompt"] = ""

#     # Prefer prompt; else 'problem' if present; else empty
#     src_text = df["prompt"].astype(str) if "prompt" in df.columns else pd.Series([""] * len(df))

#     # θ_new and γ
#     df["theta_new"] = df["kind"].astype(str) + "__d" + df["digits"].astype(int).astype(str)
#     df["gamma"] = [
#         make_gamma_label(k, int(d), t, K_bins=args.value_bins)
#         for k, d, t in zip(df["kind"].astype(str), df["digits"].astype(int), src_text.astype(str))
#     ]
#     print("Created Columns")

#     # Sanity: γ diversity per (kind,d)
#     gb = df.assign(kd=df["kind"].astype(str) + "|d" + df["digits"].astype(str)).groupby("kd")["gamma"].nunique().sort_values()
#     print("[gamma sanity] distinct bins per (kind,d):")
#     print(gb.value_counts().sort_index())
#     print(gb.head(20))
#     n_empty = (df["prompt"].astype(str).str.len() == 0).sum()
#     print(f"[gamma sanity] empty prompt rows: {n_empty} / {len(df)}")

#     # Choose label
#     label_col = {"theta_new": "theta_new", "gamma": "gamma", "kind": "kind"}[args.label]
#     df = df[df[label_col].astype(str).str.len() > 0].reset_index(drop=True)
#     df = df[df["rationale"].astype(str).str.len() > 0].reset_index(drop=True)

#     # Split
#     df = df.copy()
#     df["label"] = df[label_col].astype(str)
#     train_df, test_df = stratified_split_robust(df, y_col="label", test_size=args.test_size, seed=args.seed, min_count=2, verbose=True)

#     # Features
#     texts_tr = train_df["rationale"].astype(str).tolist()
#     texts_te = test_df["rationale"].astype(str).tolist()
#     feats = build_featurizer(args.feats, args.embed_model, args.pool, args.strip_fences, args.device, args.batch)
#     if hasattr(feats, "fit"):
#         feats.fit(texts_tr)
#     Xtr = feats.transform(texts_tr)
#     Xte = feats.transform(texts_te)
#     print("Extracted Features")

#     # Labels -> ids
#     le = LabelEncoder()
#     ytr = le.fit_transform(train_df["label"].astype(str).values)
#     yte = le.transform(test_df["label"].astype(str).values)
#     classes_idx = np.arange(len(le.classes_))

#     print("Extracted Labels")

#     # Classifier
#     clf = LogisticRegression(
#         penalty="l2",
#         C=args.C,
#         solver="saga",
#         multi_class="multinomial",
#         max_iter=args.max_iter,
#         n_jobs=-1,
#         verbose=1,
#     )
#     clf.fit(Xtr, ytr)
#     print("Finish Fit")
#     P = clf.predict_proba(Xte)
#     yhat = P.argmax(1)
#     print("Finish Prediction")

#     # Metrics
#     ce_nat = log_loss(yte, P, labels=classes_idx)
#     acc = accuracy_score(yte, yhat)
#     f1m = f1_score(yte, yhat, average="macro")

#     to_bits = 1.0 / math.log(2.0) if args.bits else 1.0
#     unit = "bits" if args.bits else "nats"
#     H_bits = empirical_entropy_bits(test_df["label"].astype(str).values)
#     CE_bits = ce_nat * to_bits
#     I_lb_bits = H_bits - CE_bits

#     print("\n=== Results (embedding features) ===")
#     print(f"Target label:     {args.label}  (#classes={len(le.classes_)})")
#     print(f"Feats:            {args.feats} | model={args.embed_model or 'n/a'} | pool={args.pool if args.feats=='hf-cls' else '-'}")
#     print(f"Rep filter:       {args.rep}")
#     print(f"N_train / N_test: {len(train_df)} / {len(test_df)}")
#     print(f"Cross-entropy:    {CE_bits:.4f} {unit}")
#     print(f"Accuracy:         {acc:.4f}")
#     print(f"Macro F1:         {f1m:.4f}")
#     print(f"H({args.label}):  {H_bits:.4f} bits  (empirical on test)")
#     print(f"I({args.label}; Z_r) ≥ {I_lb_bits:.4f} bits   (variational lower bound)")

#     # Save predictions (optional)
#     if args.save_preds:
#         out = test_df[["rationale", "kind", "digits", "prompt"]].copy()
#         out["true_label"] = test_df["label"].values
#         out["pred_label"] = le.inverse_transform(yhat)
#         out["neglogp_true_nat"] = -np.log(P[np.arange(len(P)), yte] + 1e-15)
#         out.to_csv(args.save_preds, index=False)
#         print(f"Saved predictions: {args.save_preds}")

#     # Optional K-fold CV
#     if args.cv and args.cv > 1:
#         print(f"\n=== {args.cv}-fold CV ({args.feats}, label={args.label}) ===")
#         y_all = le.transform(df["label"].astype(str).values)
#         ce_list, acc_list = [], []
#         skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
#         X_all_text = df["rationale"].astype(str).values
#         for k, (tr, te) in enumerate(skf.split(X_all_text, y_all), 1):
#             fz = build_featurizer(args.feats, args.embed_model, args.pool, args.strip_fences, args.device, args.batch)
#             tr_texts = X_all_text[tr].tolist()
#             te_texts = X_all_text[te].tolist()
#             if hasattr(fz, "fit"):
#                 fz.fit(tr_texts)
#             Xtr = fz.transform(tr_texts)
#             Xte = fz.transform(te_texts)
#             lr = LogisticRegression(
#                 penalty="l2",
#                 C=args.C,
#                 solver="saga",
#                 multi_class="multinomial",
#                 max_iter=args.max_iter,
#                 n_jobs=-1,
#             )
#             lr.fit(Xtr, y_all[tr])
#             Pte = lr.predict_proba(Xte)
#             ce = log_loss(y_all[te], Pte, labels=np.arange(len(le.classes_)))
#             ac = accuracy_score(y_all[te], Pte.argmax(1))
#             ce_list.append(ce)
#             acc_list.append(ac)
#             print(f"  fold {k}: CE={(ce*to_bits):.4f} {unit} | acc={ac:.4f}")
#         print(
#             f"CV mean CE={(np.mean(ce_list)*to_bits):.4f} {unit} (±{np.std(ce_list)*to_bits:.4f}), "
#             f"acc={np.mean(acc_list):.4f} (±{np.std(acc_list):.4f})"
#         )


# # ------------------------------------------------------------------------------
# # CLI
# # ------------------------------------------------------------------------------


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--tbdir", type=str, default=None, help="Read rationales directly from TensorBoard logs")
#     p.add_argument(
#         "--csv",
#         type=str,
#         default=None,
#         help="CSV with columns: rationale, kind, digits, [prompt], [rep], [split]",
#     )
#     p.add_argument("--rep", choices=["nl", "code", "all"], default="all")

#     # Labels
#     p.add_argument(
#         "--label",
#         choices=["theta_new", "gamma", "kind"],
#         default="gamma",
#         help="Classification target: θ_new=(kind,digits) or γ=(kind,digits,value-bin) or concept only.",
#     )
#     p.add_argument(
#         "--value-bins",
#         type=int,
#         default=8,
#         help="Number of equal-width bins per operand for gamma.",
#     )
#     p.add_argument("--test-size", type=float, default=0.2)
#     p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--cv", type=int, default=0)

#     # Embeddings
#     p.add_argument("--feats", choices=["tfidf", "hf-cls", "st", "openai"], default="tfidf")
#     p.add_argument(
#         "--embed-model",
#         type=str,
#         default=None,
#         help="hf-cls: HF repo; st: ST repo; openai: embedding model.",
#     )
#     p.add_argument("--pool", choices=["mean", "cls"], default="mean", help="Pooling for hf-cls.")
#     p.add_argument("--device", type=str, default=None, help="Force device for hf-cls/st (e.g. cuda, cpu).")
#     p.add_argument("--batch", type=int, default=128, help="Batch size for OpenAI embeddings.")
#     p.add_argument(
#         "--strip-fences",
#         action="store_true",
#         help="Strip ``` code fences before embedding (tfidf & hf-cls/st).",
#     )

#     # Classifier
#     p.add_argument("--C", type=float, default=2.0)
#     p.add_argument("--max_iter", type=int, default=400)

#     # Reporting
#     p.add_argument("--bits", action="store_true", help="Report CE in bits and print MI lower bound.")
#     p.add_argument("--save-preds", type=str, default=None, help="Optional path to save test predictions CSV.")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     run(args)
