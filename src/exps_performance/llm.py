from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

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
except (AttributeError, RuntimeError):
    # AttributeError if method doesn't exist, RuntimeError if CUDA not available
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
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        raise NotImplementedError


@dataclass
class ChatResponse:
    text: str
    attempts: int = 1
    error: str = ""
    reasoning: str = ""
    reasoning_details_count: int = 0
    reasoning_tokens: int = 0
    structured_requested: bool = False
    reasoning_requested: bool = False
    finish_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    request_options: Dict[str, Any] = field(default_factory=dict)


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
        request_options: Optional[Dict[str, Any]] = None,
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
        request_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
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

    @staticmethod
    def _usage_to_dict(resp: Any) -> Dict[str, Any]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            dumped = usage.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(usage, dict):
            return usage
        return {}

    @staticmethod
    def _extract_reasoning_details(choice0: Any, message: Any) -> int:
        details = getattr(message, "reasoning_details", None)
        if not details:
            details = getattr(choice0, "reasoning_details", None)
        if isinstance(details, list):
            return len(details)
        return 0

    @staticmethod
    def _extract_reasoning_tokens(usage: Dict[str, Any]) -> int:
        candidates = [
            usage.get("reasoning_tokens"),
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        ]
        for candidate in candidates:
            if isinstance(candidate, int):
                return candidate
        return 0

    @staticmethod
    def _merge_extra_body(base: Dict[str, Any], incoming: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(base)
        if not incoming:
            return merged
        for key, value in incoming.items():
            if key == "plugins":
                merged[key] = list(value)
            elif key == "reasoning" and isinstance(value, dict) and isinstance(merged.get(key), dict):
                reasoning = dict(merged[key])
                reasoning.update(value)
                merged[key] = reasoning
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _extract_response(resp: Any, *, request_options: Optional[Dict[str, Any]], attempts: int, error: str = "") -> ChatResponse:
        if resp is None or getattr(resp, "choices", None) in (None, []):
            return ChatResponse(
                text="",
                attempts=attempts,
                error=error or "missing_choices",
                structured_requested=bool(request_options and request_options.get("response_format")),
                reasoning_requested=bool((request_options or {}).get("extra_body", {}).get("reasoning")),
                request_options=dict(request_options or {}),
            )

        choice0 = resp.choices[0]
        message = getattr(choice0, "message", None)
        usage = OpenRouterChatClient._usage_to_dict(resp)
        reasoning = str(getattr(message, "reasoning", "") or "")
        return ChatResponse(
            text=str(getattr(message, "content", "") or ""),
            attempts=attempts,
            error=error,
            reasoning=reasoning,
            reasoning_details_count=OpenRouterChatClient._extract_reasoning_details(choice0, message),
            reasoning_tokens=OpenRouterChatClient._extract_reasoning_tokens(usage),
            structured_requested=bool(request_options and request_options.get("response_format")),
            reasoning_requested=bool((request_options or {}).get("extra_body", {}).get("reasoning")),
            finish_reason=str(getattr(choice0, "finish_reason", "") or ""),
            usage=usage,
            request_options=dict(request_options or {}),
        )

    @staticmethod
    def _needs_retry(response: ChatResponse) -> bool:
        if response.text.strip():
            return False
        return response.structured_requested or response.reasoning_requested or bool(response.error)

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        # Short bounded backoff helps when OpenRouter returns transient malformed
        # responses under high concurrency.
        return float(min(8.0, 1.5 * (2 ** max(0, attempt - 1))))

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        req_opts = request_options or {}
        retries = max(1, int(req_opts.get("retry_attempts", 1)))
        base_extra_body = req_opts.get("extra_body", {})
        last_error = ""
        for attempt in range(1, retries + 1):
            extra_body = self._merge_extra_body(
                base_extra_body,
                {"plugins": [{"id": "response-healing"}]} if req_opts.get("enable_response_healing", False) else None,
            )
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "top_p": top_p,
                "max_completion_tokens": max_tokens,
                "stop": stop,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if req_opts.get("response_format") is not None:
                kwargs["response_format"] = req_opts["response_format"]
            if req_opts.get("timeout") is not None:
                kwargs["timeout"] = req_opts["timeout"]
            if extra_body:
                kwargs["extra_body"] = extra_body
            try:
                resp = self.client.chat.completions.create(**kwargs)
                extracted = self._extract_response(resp, request_options=req_opts, attempts=attempt)
                if not self._needs_retry(extracted) or attempt == retries:
                    return extracted
                last_error = extracted.error or "empty_response"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt == retries:
                    return ChatResponse(
                        text="",
                        attempts=attempt,
                        error=last_error,
                        structured_requested=bool(req_opts.get("response_format")),
                        reasoning_requested=bool(base_extra_body.get("reasoning")),
                        request_options=dict(req_opts),
                    )
            time.sleep(self._retry_delay_seconds(attempt))
        return ChatResponse(
            text="",
            attempts=retries,
            error=last_error or "empty_response",
            structured_requested=bool(req_opts.get("response_format")),
            reasoning_requested=bool(base_extra_body.get("reasoning")),
            request_options=dict(req_opts),
        )

    def chat_many(
        self,
        model: str,
        messages_list: List[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        request_timeout: float = 120.0,
        request_options_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        on_result: Optional[Callable[[int, ChatResponse], None]] = None,
    ) -> List[ChatResponse]:
        # Create a fresh async client per batch to avoid event-loop reuse issues.
        from openai import AsyncOpenAI  # type: ignore

        async def _one(
            async_client: Any,
            idx: int,
            msgs: List[Dict[str, str]],
            request_options: Optional[Dict[str, Any]],
        ) -> tuple[int, ChatResponse]:
            req_opts = request_options or {}
            retries = max(1, int(req_opts.get("retry_attempts", 1)))
            last_error = ""
            for attempt in range(1, retries + 1):
                extra_body = self._merge_extra_body(
                    req_opts.get("extra_body", {}),
                    {"plugins": [{"id": "response-healing"}]} if req_opts.get("enable_response_healing", False) else None,
                )
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": msgs,
                    "top_p": top_p,
                    "max_completion_tokens": max_tokens,
                    "stop": stop,
                    "timeout": req_opts.get("timeout", request_timeout),
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if req_opts.get("response_format") is not None:
                    kwargs["response_format"] = req_opts["response_format"]
                if extra_body:
                    kwargs["extra_body"] = extra_body
                try:
                    resp = await asyncio.wait_for(
                        async_client.chat.completions.create(**kwargs),
                        timeout=float(req_opts.get("timeout", request_timeout)) + 5,
                    )
                    extracted = self._extract_response(resp, request_options=req_opts, attempts=attempt)
                    if not self._needs_retry(extracted) or attempt == retries:
                        return idx, extracted
                    last_error = extracted.error or "empty_response"
                except asyncio.TimeoutError:
                    last_error = "timeout"
                    logger.warning(f"OpenRouter chat_many timed out for idx={idx} attempt={attempt}")
                except (ConnectionError, OSError, ValueError) as exc:
                    last_error = str(exc)
                    logger.warning(f"OpenRouter chat_many failed for idx={idx} attempt={attempt}: {exc}")
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    logger.warning(f"OpenRouter chat_many unexpected failure for idx={idx} attempt={attempt}: {exc}")
                await asyncio.sleep(self._retry_delay_seconds(attempt))
            return idx, ChatResponse(
                text="",
                attempts=retries,
                error=last_error or "empty_response",
                structured_requested=bool(req_opts.get("response_format")),
                reasoning_requested=bool((req_opts.get("extra_body") or {}).get("reasoning")),
                request_options=dict(req_opts),
            )

        async def _run() -> List[ChatResponse]:
            reqs = request_options_list or [None] * len(messages_list)
            results: List[Optional[ChatResponse]] = [None] * len(messages_list)
            concurrency_limit = len(messages_list)
            for req in reqs:
                if req and req.get("max_concurrency"):
                    concurrency_limit = max(1, int(req["max_concurrency"]))
                    break
            semaphore = asyncio.Semaphore(concurrency_limit)

            async def _bounded_one(idx: int, msgs: List[Dict[str, str]], req: Optional[Dict[str, Any]]) -> tuple[int, ChatResponse]:
                async with semaphore:
                    return await _one(async_client, idx, msgs, req)

            async with AsyncOpenAI(api_key=self._api_key, base_url=self._base_url) as async_client:
                tasks = [asyncio.create_task(_bounded_one(i, m, reqs[i])) for i, m in enumerate(messages_list)]
                try:
                    async for task in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Chatting (openrouter)"):
                        idx, response = await task
                        results[idx] = response
                        if on_result is not None:
                            on_result(idx, response)
                finally:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    for t in tasks:
                        with contextlib.suppress(asyncio.CancelledError):
                            await t
            return [r if r is not None else ChatResponse(text="", error="missing_result") for r in results]

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
            raise RuntimeError("vLLM is not installed. Install a CUDA-matching vLLM wheel (e.g. vllm-cu121) or build from source.") from e

        self._SamplingParams = SamplingParams  # Store for use in chat methods
        # vLLM engine (persistent)
        self.seed = seed
        self.llm = VLLMEngine(
            model=model_name,
            dtype=cast(Any, dtype),  # "auto" | "float16"
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
        request_options: Optional[Dict[str, Any]] = None,
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
        request_options_list: Optional[List[Optional[Dict[str, Any]]]] = None,
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
        # Use absolute path for cache directory to prevent path traversal issues
        cache_dir = str((Path(__file__).parent.parent / "models").resolve())
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
        request_options: Optional[Dict[str, Any]] = None,
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


def _normalize_batch_outputs(raw_outputs: List[Any]) -> List[ChatResponse]:
    normalized: List[ChatResponse] = []
    for item in raw_outputs:
        if isinstance(item, ChatResponse):
            normalized.append(item)
        else:
            normalized.append(ChatResponse(text=str(item or "")))
    return normalized


def run_batch(
    messages_list: List[List[Dict[str, str]]],
    args: Any,
    client: Any,
    request_options_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    return_responses: bool = False,
    on_response: Optional[Callable[[int, ChatResponse], None]] = None,
) -> List[Any]:
    total = len(messages_list)
    if hasattr(client, "chat_many") and callable(getattr(client, "chat_many")) and args.batch_size > 1:
        outs: List[ChatResponse] = []
        with tqdm(total=total, desc="Chatting (overall)", unit="req") as overall:
            for start in range(0, total, args.batch_size):
                chunk = messages_list[start : start + args.batch_size]
                chunk_request_options = None if request_options_list is None else request_options_list[start : start + args.batch_size]
                with tqdm(total=len(chunk), desc="Batch", unit="req", leave=False) as batchbar:
                    completed = 0

                    def _handle_response(local_idx: int, response: ChatResponse) -> None:
                        nonlocal completed
                        completed += 1
                        batchbar.update(1)
                        overall.update(1)
                        if on_response is not None:
                            on_response(start + local_idx, response)

                    try:
                        raw_chunk_outs = client.chat_many(
                            args.model,
                            chunk,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            stop=None,
                            request_timeout=getattr(args, "request_timeout", 120),
                            request_options_list=chunk_request_options,
                            on_result=_handle_response,
                        )
                    except TypeError as exc:
                        if "on_result" not in str(exc):
                            raise
                        raw_chunk_outs = client.chat_many(
                            args.model,
                            chunk,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            stop=None,
                            request_timeout=getattr(args, "request_timeout", 120),
                            request_options_list=chunk_request_options,
                        )
                    outs.extend(_normalize_batch_outputs(raw_chunk_outs))
                    if completed == 0:
                        batchbar.update(len(chunk))
                        overall.update(len(chunk))
        if return_responses:
            return list(outs)
        return [response.text for response in outs]
    else:
        sequential_outs: List[ChatResponse] = []
        with tqdm(total=total, desc="Chatting (overall)", unit="req") as pbar:
            for idx, m in enumerate(messages_list):
                request_options = None if request_options_list is None else request_options_list[idx]
                raw_out = client.chat(
                    args.model,
                    m,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=None,
                    request_options=request_options,
                )
                normalized = _normalize_batch_outputs([raw_out])
                sequential_outs.extend(normalized)
                if on_response is not None:
                    on_response(idx, normalized[0])
                pbar.update(1)
        if return_responses:
            return list(sequential_outs)
        return [response.text for response in sequential_outs]
