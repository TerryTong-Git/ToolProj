#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_olmo_sample.py

Robust OLMo mix sampler without streaming/globbing issues.

- Lists parquet shards in a HF dataset repo.
- Loads them via the parquet builder (no streaming).
- Heuristically assigns rep ∈ {code,nl}.
- Emits a balanced sample of size --target as JSONL with fields {id, text, rep}.

Requires: datasets>=2.20, pyarrow>=15, fsspec>=2024.5, huggingface_hub.
"""

import os, sys, re, json, random, argparse
from typing import List, Tuple
from collections import defaultdict
from tqdm import tqdm

from datasets import load_dataset
from huggingface_hub import list_repo_files

CODE_SOURCES = {"starcoder", "the-stack", "stack", "github", "code"}
CODE_MARKERS = re.compile(r"(\bdef\b|\bclass\b|;|{|}|\breturn\b|\bfor\b|\bwhile\b|```)", re.I)

def looks_like_code(txt: str) -> bool:
    s = (txt or "").strip()
    return bool(CODE_MARKERS.search(s)) or ("def " in s or "class " in s or "{" in s or "}" in s)

def choose_text_column(colnames: List[str]) -> str:
    prefs = ["text", "content", "document", "text_content", "body"]
    for p in prefs:
        if p in colnames: return p
    # fallback: first column with string dtype will be recognized later at row time
    return prefs[0]  # placeholder; we’ll handle row-level fallback

from datasets import load_dataset
from huggingface_hub import list_repo_files

def _list_shards(repo_id: str, split: str):
    files = list_repo_files(repo_id, repo_type="dataset")
    # prefer split-scoped paths like data/*/<split>/*.json.gz or *.parquet
    cand = [f for f in files if f"/{split}/" in f or f.startswith(f"{split}/")]
    cand = [f for f in cand if f.endswith(".json.gz") or f.endswith(".jsonl") or f.endswith(".parquet")]
    if not cand:
        # fallback: any shard whose filename contains the split token
        cand = [f for f in files if split in f and (f.endswith(".json.gz") or f.endswith(".jsonl") or f.endswith(".parquet"))]
    if not cand:
        raise RuntimeError(f"No shards found in {repo_id} for split='{split}'. First few files: {files[:20]}")
    return sorted(cand)

def load_olmo_mix(repo_id: str, split: str):
    paths = _list_shards(repo_id, split)
    urls = [f"hf://datasets/{repo_id}/{p}" for p in paths]
    if paths[0].endswith(".parquet"):
        ds = load_dataset("parquet", data_files=urls, split="train", streaming=True, cache_dir="../../data/")
    else:
        ds = load_dataset("json",    data_files=urls, split="train", streaming=True, cache_dir="../../data/")

    # Build a lightweight materialized list of dicts with just needed fields.
    # No caching, no schema merge.
    rows = []
    for row in tqdm(ds, desc = 'retrieving'):
        txt = row.get("text") or row.get("content") or ""
        if not isinstance(txt, str) or not txt.strip():
            continue
        src = row.get("source") or row.get("dataset_name") or "unknown"
        rid = row.get("id")
        rows.append({"text": txt, "source": src, "id": str(rid) if rid is not None else None})
    from datasets import Dataset
    mat = Dataset.from_list(rows)  # small, clean schema: text/source/id
    return mat, "text", "source", "id"

def infer_rep(txt: str, src: str) -> str:
    ssrc = (src or "").lower()
    if any(tag in ssrc for tag in CODE_SOURCES): return "code"
    return "code" if looks_like_code(txt) else "nl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/olmo-mix-1124")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=int, default=20000, help="total rows to emit")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--max-per-source", type=int, default=2500)
    args = ap.parse_args()
    random.seed(args.seed)

    ds, TEXT_COL, SRC_COL, ID_COL = load_olmo_mix(args.dataset, args.split)

    k_total = int(args.target)
    k_code  = k_total // 2
    k_nl    = k_total - k_code

    kept = {"code": [], "nl": []}
    per_source = defaultdict(int)

    # Reservoir-like fill until targets met
    for row in tqdm(ds, desc='inferring'):
        # row is a dict-like (Arrow batch row); fetch text
        txt = None
        if TEXT_COL in row and isinstance(row[TEXT_COL], str) and row[TEXT_COL].strip():
            txt = row[TEXT_COL]
        else:
            # fallback: find any string-typed column
            for k, v in row.items():
                if isinstance(v, str) and v.strip():
                    txt = v; break
        if not txt: 
            continue

        src = row[SRC_COL] if SRC_COL and SRC_COL in row else "unknown"
        if per_source[src] >= args.max_per_source:
            continue

        rep = infer_rep(txt, src)
        tgt = k_code if rep == "code" else k_nl
        if len(kept[rep]) >= tgt:
            continue

        rid = str(row[ID_COL]) if ID_COL and ID_COL in row and row[ID_COL] is not None else f"{src}:{len(kept[rep])}"
        kept[rep].append({"id": rid, "text": txt, "rep": rep})
        per_source[src] += 1

        if len(kept["code"]) >= k_code and len(kept["nl"]) >= k_nl:
            break

    total = len(kept["code"]) + len(kept["nl"])
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rep in ("code", "nl"):
            for r in kept[rep]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[sample] wrote {total} rows -> {args.out} (code={len(kept['code'])}, nl={len(kept['nl'])})")

if __name__ == "__main__":
    main()
