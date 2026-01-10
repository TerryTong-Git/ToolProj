from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Reduce verbose HTTP logging from clients.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


class LLMClient:
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        raise NotImplementedError


class DummyClient(LLMClient):
    """Deterministic stub: returns correct integer for known templates; else 0."""

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        last = messages[-1]["content"]
        ans = 0
        # arithmetic quick parse
        m = re.search(r"Compute:\s*(\d+)\s*([+\-*])\s*(\d+)", last)
        if m:
            a, op, b = m.groups()
            a, b = int(a), int(b)
            ans = a + b if op == "+" else (a - b if op == "-" else a * b)
        else:
            m2 = re.search(r"Compute:\s*\((\d+)\s*\+\s*(\d+)\)\s*\*\s*(\d+)", last)
            if m2:
                a, b, c = map(int, m2.groups())
                ans = (a + b) * c
        # for other kinds, just 0 to keep it simple for dry-run
        is_nl = "problem solver" in last.lower()
        if is_nl:
            out = {"rationale": "Solve deterministically.", "answer": ans}
        else:
            out = {"rationale": f"```python\nprint({ans})\n```", "answer": ans}
        return json.dumps(out)


openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
openai_api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1/")
openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")


class OpenAIChatClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, seed: int = 0):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("pip install openai>=1.0 required") from e
        _api_key = api_key or openai_api_key
        _base_url = base_url or openai_api_base
        if not _api_key or _api_key == "EMPTY":
            raise RuntimeError("OPENAI_API_KEY is required for backend=openai/running")
        self.client = OpenAI(api_key=_api_key, base_url=_base_url)
        self.seed = seed
        print("Instantiated OPENAI!")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=top_p,
            max_completion_tokens=max_tokens,
            stop=stop,
            # seed=self.seed,
        )
        return str(resp.choices[0].message.content or "")


class OpenRouterChatClient(LLMClient):
    """Simple OpenRouter client using the OpenAI SDK."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, seed: int = 0):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("pip install openai>=1.0 required") from e
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY or pass --openrouter_api_key")
        self._base_url = base_url or openrouter_api_base
        self.client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        self.seed = seed
        print("Instantiated OpenRouter client!")

    @staticmethod
    def _extract_text(resp: Any) -> str:
        # Defensive guard so errors are clearer than a NoneType subscript failure.
        if resp is None or getattr(resp, "choices", None) in (None, []):
            return ""
            # raise RuntimeError(f"OpenRouter response missing choices: {resp}")
        choice0 = resp.choices[0]
        if getattr(choice0, "message", None) is None:
            raise RuntimeError(f"OpenRouter choice missing message: {resp}")
        return str(choice0.message.content or "")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=top_p,
            max_completion_tokens=max_tokens,
            stop=stop,
            # seed=self.seed,  # OpenRouter may ignore seed; kept for symmetry
        )
        return self._extract_text(resp)

    def chat_many(
        self,
        model: str,
        messages_list: List[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        request_timeout: float = 120.0,
    ) -> List[str]:
        # Create a fresh async client per batch to avoid event-loop reuse issues.
        from openai import AsyncOpenAI  # type: ignore

        async def _one(idx: int, msgs: List[Dict[str, str]]) -> tuple[int, str]:
            async with AsyncOpenAI(api_key=self._api_key, base_url=self._base_url) as async_client:
                try:
                    resp = await asyncio.wait_for(
                        async_client.chat.completions.create(
                            model=model,
                            messages=msgs,
                            top_p=top_p,
                            max_completion_tokens=max_tokens,
                            stop=stop,
                            timeout=request_timeout,
                        ),
                        timeout=request_timeout + 5,
                    )
                    return idx, self._extract_text(resp)
                except asyncio.TimeoutError:
                    return idx, ""
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"OpenRouter chat_many failed for idx={idx}: {exc}")
                    return idx, ""

        async def _run() -> List[str]:
            tasks = [asyncio.create_task(_one(i, m)) for i, m in enumerate(messages_list)]
            results: List[Optional[str]] = [None] * len(messages_list)
            try:
                async for task in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Chatting (openrouter)"):
                    idx, text = await task
                    results[idx] = text
            finally:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                for t in tasks:
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
            return [r if r is not None else "" for r in results]  # All slots should be filled

        return asyncio.run(_run())


class VLLMClient(LLMClient):
    """
    vLLM-powered local inference with the same .chat(...) signature you use
    everywhere else. Reuses a single engine; applies a chat template if the
    model provides one; otherwise falls back to the last user message content.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        download_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        seed: int = 0,
    ):
        # Lazy import vLLM only when this client is actually instantiated
        try:
            from vllm import LLM as VLLMEngine
            from vllm import SamplingParams
        except ImportError as e:
            raise RuntimeError(
                "vLLM is not installed. Install a CUDA-matching vLLM wheel "
                "(e.g. vllm-cu121) or build from source."
            ) from e

        self._SamplingParams = SamplingParams  # Store for use in chat methods
        # vLLM engine (persistent)
        self.seed = seed
        self.llm = VLLMEngine(
            model=model_name,
            dtype=dtype,  # "auto" | "float16"
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len) if max_model_len else None,
            trust_remote_code=bool(trust_remote_code),
            download_dir=download_dir,
            seed=seed,
            tokenizer_mode="auto",
            enable_prefix_caching=True,
        )
        # Use HF tokenizer to format chat prompts if available
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
        self.has_template = hasattr(self.tok, "apply_chat_template") and (self.tok.chat_template is not None)

    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.has_template:
            # Mirrors your HFLocalClient behavior
            return str(self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
        # Fallback: use the last user content (same as HFLocalClient fallback)
        return messages[-1]["content"]

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        prompt = self._to_prompt(messages)
        sp = self._SamplingParams(
            max_tokens=int(max_tokens),  # new tokens
            temperature=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 1.0,
            stop=stop or None,
            seed=self.seed,
        )
        # vLLM can batch; here we keep semantics identical (one request per call)
        outs = self.llm.generate([prompt], sp)
        # outs is a List[RequestOutput]; take first, first candidate
        return str(outs[0].outputs[0].text)

    def chat_many(
        self,
        model: str,
        messages_list: List[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        prompts = [self._to_prompt(msgs) for msgs in messages_list]
        sp = self._SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 1.0,
            stop=stop or None,
            seed=self.seed,
        )
        outs = self.llm.generate(prompts, sp)
        # preserve order, one candidate per request
        return [str(o.outputs[0].text) for o in outs]


class HFLocalClient(LLMClient):
    """Vanilla Hugging Face transformers inference (no vLLM)."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        _map = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = _map.get(dtype, None)
        cache_dir = "../models"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.has_template = hasattr(self.tokenizer, "apply_chat_template") and (self.tokenizer.chat_template is not None)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        if self.has_template:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = messages[-1]["content"]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = (temperature is not None) and (float(temperature) > 0.0)
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
        )
        out_ids = gen_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        if stop:
            idxs = [text.find(s) for s in stop if s in text]
            if idxs:
                cut = min(i for i in idxs if i >= 0)
                text = text[:cut]
        return str(text)


def llm(args: Any) -> Any:
    if args.backend == "vllm":
        client = VLLMClient(
            model_name=args.model,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.vllm_tensor_parallel,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            max_model_len=args.vllm_max_model_len,
            download_dir=args.vllm_download_dir,
            trust_remote_code=args.hf_trust_remote_code,
            seed=args.seed,
        )
        return client
    elif args.backend == "dummy":
        return DummyClient()
    elif args.backend == "running":
        return OpenAIChatClient(seed=args.seed)
    elif args.backend == "openai":
        return OpenAIChatClient(seed=args.seed)
    elif args.backend == "openrouter":
        api_key = getattr(args, "openrouter_api_key", None) or os.getenv("OPENROUTER_API_KEY")
        base_url = getattr(args, "openrouter_base_url", None) or openrouter_api_base
        return OpenRouterChatClient(api_key=api_key, base_url=base_url, seed=args.seed)


def run_batch(messages_list: List[List[Dict[str, str]]], args: Any, client: Any) -> List[str]:
    total = len(messages_list)
    if hasattr(client, "chat_many") and callable(getattr(client, "chat_many")) and args.batch_size > 1:
        outs = []
        with tqdm(total=total, desc="Chatting (overall)", unit="req") as overall:
            for start in range(0, total, args.batch_size):
                chunk = messages_list[start : start + args.batch_size]
                with tqdm(total=len(chunk), desc="Batch", unit="req", leave=False) as batchbar:
                    chunk_outs = client.chat_many(
                        args.model,
                        chunk,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        stop=None,
                        request_timeout=getattr(args, "request_timeout", 120),
                    )
                    outs.extend(chunk_outs)
                    batchbar.update(len(chunk))
                overall.update(len(chunk))
        return outs
    else:
        outs = []
        with tqdm(total=total, desc="Chatting (overall)", unit="req") as pbar:
            for m in messages_list:
                out = client.chat(args.model, m, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, stop=None)
                outs.append(out)
                pbar.update(1)
        return outs
