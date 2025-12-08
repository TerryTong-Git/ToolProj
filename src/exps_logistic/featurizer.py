#!/usr/bin/env python3
"""Feature extraction backends for text classification."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from .parsers import maybe_strip_fences


class Featurizer(ABC):
    """Base class for text featurizers."""

    def fit(self, texts: List[str]) -> "Featurizer":
        """Fit the featurizer on training texts."""
        return self

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors."""
        raise NotImplementedError


class TfidfFeaturizer(Featurizer):
    """TF-IDF word + character n-gram featurizer."""

    def __init__(self, strip_fences: bool = False):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        preprocessor = maybe_strip_fences if strip_fences else None

        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            max_features=200_000,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            preprocessor=preprocessor,
        )
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.98,
            lowercase=False,
            preprocessor=preprocessor,
        )
        self.vec = FeatureUnion([("w", word_vec), ("c", char_vec)])

    def fit(self, texts: List[str]) -> "TfidfFeaturizer":
        self.vec.fit(texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.vec.transform(texts).toarray())


class HFCLSFeaturizer(Featurizer):
    """HuggingFace Transformer encoder featurizer."""

    def __init__(
        self,
        model_name: str,
        pool: str = "mean",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
        batch_size: int = 16,
        window_stride: int = 0,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        _dtype_map = {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = _dtype_map.get(torch_dtype, None)

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code, torch_dtype=dtype)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pool = pool
        self._torch = torch
        self.batch_size = int(batch_size)
        self.window_stride = int(max(0, window_stride))

    def _pool_output(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        if self.pool == "cls":
            return last_hidden_state[:, 0, :]
        # Mean pooling
        mask = attention_mask.unsqueeze(-1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    @property
    def _max_len(self) -> int:
        ml = getattr(self.tok, "model_max_length", 512)
        try:
            return int(min(ml, 512))
        except Exception:
            return 512

    def transform(self, texts: List[str]) -> np.ndarray:
        OUT = []
        bs = self.batch_size
        L = self._max_len

        with self._torch.no_grad():
            for i in range(0, len(texts), bs):
                chunk_texts = texts[i : i + bs]
                # Tokenize; default to truncation for minimal memory unless window_stride>0
                truncate_only = self.window_stride <= 0
                enc_full = self.tok(
                    chunk_texts,
                    padding=True,
                    truncation=truncate_only,
                    max_length=L if truncate_only else None,
                    return_tensors="pt",
                )
                seq_len = enc_full["input_ids"].shape[-1]

                if not truncate_only and seq_len >= L:
                    stride = max(1, self.window_stride)
                    enc_full["input_ids"] = enc_full["input_ids"].unfold(1, L, stride)
                    enc_full["attention_mask"] = enc_full["attention_mask"].unfold(1, L, stride)
                else:
                    enc_full["input_ids"] = enc_full["input_ids"][:, None, :]
                    enc_full["attention_mask"] = enc_full["attention_mask"][:, None, :]

                enc_full = {k: v.to(self.device) for k, v in enc_full.items()}
                reps = []

                for j in range(len(chunk_texts)):
                    enc = {"input_ids": enc_full["input_ids"][j, :, :], "attention_mask": enc_full["attention_mask"][j, :, :]}
                    out = self.model(**enc)
                    hid = out.last_hidden_state
                    rep = self._pool_output(hid, enc["attention_mask"])
                    reps.append(rep.mean(dim=0))

                pooled_batch = self._torch.stack(reps, dim=0)
                OUT.append(pooled_batch.cpu().numpy())

        return np.vstack(OUT)


class SentenceTransformersFeaturizer(Featurizer):
    """Sentence-Transformers featurizer."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        device = device or ("cuda" if self._has_cuda() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def transform(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False))


class OpenAIEmbeddingFeaturizer(Featurizer):
    """OpenAI Embeddings API featurizer."""

    def __init__(self, model_name: str = "text-embedding-3-large", batch: int = 128):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install openai>=1.0 required for --feats openai") from e

        self.client = OpenAI()
        self.model = model_name
        self.batch = int(batch)

    def transform(self, texts: List[str]) -> np.ndarray:
        OUT = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i : i + self.batch]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            OUT.append(np.stack(vecs, axis=0))
        return np.vstack(OUT)


def build_featurizer(
    kind: str,
    embed_model: Optional[str] = None,
    pool: str = "mean",
    strip_fences: bool = False,
    device: Optional[str] = None,
    batch: int = 128,
    hf_batch: int = 16,
    hf_dtype: str = "auto",
    hf_window_stride: int = 0,
) -> Featurizer:
    """Factory function to create the appropriate featurizer."""
    kind = kind.lower()

    if kind == "tfidf":
        return TfidfFeaturizer(strip_fences=strip_fences)

    if kind == "hf-cls":
        if not embed_model:
            raise ValueError("--embed-model is required for --feats hf-cls")
        return HFCLSFeaturizer(
            embed_model,
            pool=pool,
            device=device,
            torch_dtype=hf_dtype,
            batch_size=hf_batch,
            window_stride=hf_window_stride,
        )

    if kind == "st":
        if not embed_model:
            raise ValueError("--embed-model is required for --feats st")
        return SentenceTransformersFeaturizer(embed_model, device=device)

    if kind == "openai":
        return OpenAIEmbeddingFeaturizer(embed_model or "text-embedding-3-large", batch=batch)

    raise ValueError(f"Unknown --feats {kind}")
