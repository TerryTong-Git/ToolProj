# class Featurizer:
#     def fit(self, texts: List[str]):
#         return self

#     def transform(self, texts: List[str]) -> np.ndarray:
#         raise NotImplementedError


# class TfidfFeaturizer(Featurizer):
#     def __init__(self, strip_fences: bool):
#         from sklearn.feature_extraction.text import TfidfVectorizer
#         from sklearn.pipeline import FeatureUnion as FU

#         word_vec = TfidfVectorizer(
#             analyzer="word",
#             ngram_range=(1, 2),
#             min_df=3,
#             max_df=0.9,
#             max_features=200_000,
#             lowercase=True,
#             token_pattern=r"(?u)\b\w+\b",
#             preprocessor=maybe_strip_fences if strip_fences else None,
#         )
#         char_vec = TfidfVectorizer(
#             analyzer="char",
#             ngram_range=(3, 5),
#             min_df=3,
#             max_df=0.98,
#             lowercase=False,
#             preprocessor=maybe_strip_fences if strip_fences else None,
#         )
#         self.vec = FU([("w", word_vec), ("c", char_vec)])

#     def fit(self, texts):
#         self.vec.fit(texts)
#         return self

#     def transform(self, texts):
#         return self.vec.transform(texts)


# class HFCLSFeaturizer(Featurizer):
#     def __init__(
#         self,
#         model_name: str,
#         pool: str = "mean",
#         device: Optional[str] = None,
#         trust_remote_code: bool = False,
#     ):
#         import torch
#         from transformers import AutoModel, AutoTokenizer

#         self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
#         self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
#         self.model.eval()
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.pool = pool
#         self.torch = __import__("torch")

#     def _pool(self, last_hidden_state, attention_mask):
#         if self.pool == "cls":
#             return last_hidden_state[:, 0, :]
#         mask = attention_mask.unsqueeze(-1)
#         summed = (last_hidden_state * mask).sum(dim=1)
#         counts = mask.sum(dim=1).clamp(min=1)
#         return summed / counts

#     @property
#     def _max_len(self):
#         ml = getattr(self.tok, "model_max_length", 512)
#         try:
#             return int(min(ml, 512))
#         except Exception:
#             return 512

#     @torch.no_grad()
#     def transform(self, texts: List[str]) -> np.ndarray:
#         torch = self.torch
#         OUT = []
#         bs = 16
#         L = self._max_len
#         for i in range(0, len(texts), bs):
#             chunk_texts = texts[i : i + bs]
#             enc_full = self.tok(chunk_texts, padding=True, truncation=False, return_tensors="pt")
#             seq_len = enc_full["input_ids"].shape[-1]
#             if seq_len >= L:
#                 enc_full["input_ids"] = enc_full["input_ids"].unfold(1, L, 1)
#                 enc_full["attention_mask"] = enc_full["attention_mask"].unfold(1, L, 1)
#             else:
#                 enc_full["input_ids"] = enc_full["input_ids"][:, None, :]
#                 enc_full["attention_mask"] = enc_full["attention_mask"][:, None, :]
#             enc_full = {k: v.to(self.device) for k, v in enc_full.items()}
#             reps = []
#             enc = {}
#             for i in range(len(chunk_texts)):
#                 enc["input_ids"] = enc_full["input_ids"][i, :, :]
#                 enc["attention_mask"] = enc_full["attention_mask"][i, :, :]
#                 out = self.model(**enc)
#                 hid = out.last_hidden_state
#                 rep = self._pool(hid, enc["attention_mask"])
#                 reps.append(rep.mean(dim=0))  # Mean across seq len
#             pooled_batch = torch.stack(reps, dim=0)
#             OUT.append(pooled_batch.cpu().numpy())
#         return np.vstack(OUT)


# class SentenceTransformersFeaturizer(Featurizer):
#     def __init__(self, model_name: str, device: Optional[str] = None):
#         from sentence_transformers import SentenceTransformer

#         self.model = SentenceTransformer(model_name, device=(device or ("cuda" if self._has_cuda() else "cpu")))

#     def _has_cuda(self):
#         try:
#             import torch

#             return torch.cuda.is_available()
#         except Exception:
#             return False

#     def transform(self, texts: List[str]) -> np.ndarray:
#         return np.asarray(self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False))


# class OpenAIEmbeddingFeaturizer(Featurizer):
#     def __init__(self, model_name: str, batch: int = 128):
#         try:
#             from openai import OpenAI  # type: ignore
#         except Exception as e:
#             raise RuntimeError("pip install openai>=1.0 required for --feats openai") from e
#         self.client = OpenAI()
#         self.model = model_name or "text-embedding-3-large"
#         self.batch = int(batch)

#     def transform(self, texts: List[str]) -> np.ndarray:
#         OUT = []
#         for i in range(0, len(texts), self.batch):
#             chunk = texts[i : i + self.batch]
#             resp = self.client.embeddings.create(model=self.model, input=chunk)
#             vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
#             OUT.append(np.stack(vecs, axis=0))
#         return np.vstack(OUT)


# def build_featurizer(
#     kind: str,
#     embed_model: Optional[str],
#     pool: str,
#     strip_fences: bool,
#     device: Optional[str],
#     batch: int,
# ):
#     kind = kind.lower()
#     if kind == "tfidf":
#         return TfidfFeaturizer(strip_fences=strip_fences)
#     if kind == "hf-cls":
#         if not embed_model:
#             raise ValueError("--embed-model is required for --feats hf-cls")
#         return HFCLSFeaturizer(embed_model, pool=pool, device=device)
#     if kind == "st":
#         if not embed_model:
#             raise ValueError("--embed-model is required for --feats st")
#         return SentenceTransformersFeaturizer(embed_model, device=device)
#     if kind == "openai":
#         return OpenAIEmbeddingFeaturizer(embed_model or "text-embedding-3-large", batch=batch)
#     raise ValueError(f"Unknown --feats {kind}")
