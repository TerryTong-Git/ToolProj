from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams
except Exception as _vllm_import_err:
    VLLMEngine = None
    SamplingParams = None

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


class OpenAIChatClient(LLMClient):
    def __init__(self, seed):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("pip install openai>=1.0 required") from e
        self.client = OpenAI()
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
        print("Currently Chatting OPENAI!")
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=top_p,
            max_completion_tokens=max_tokens,
            stop=stop,
            seed=self.seed,
        )
        return resp.choices[0].message.content


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
        assert dtype == "float16", "Wrong dtype"
        if VLLMEngine is None:
            raise RuntimeError("vLLM is not installed. Install a CUDA-matching vLLM wheel (e.g. vllm-cu121) or build from source.")
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
        )
        # Use HF tokenizer to format chat prompts if available
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.has_template = hasattr(self.tok, "apply_chat_template") and (self.tok.chat_template is not None)

    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.has_template:
            # Mirrors your HFLocalClient behavior
            return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
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
        sp = SamplingParams(
            max_tokens=int(max_tokens),  # new tokens
            temperature=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 1.0,
            stop=stop or None,
            seed=self.seed,
        )
        # vLLM can batch; here we keep semantics identical (one request per call)
        outs = self.llm.generate([prompt], sp)
        # outs is a List[RequestOutput]; take first, first candidate
        return outs[0].outputs[0].text

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
        sp = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 1.0,
            stop=stop or None,
            seed=self.seed,
        )
        outs = self.llm.generate(prompts, sp)
        # preserve order, one candidate per request
        return [o.outputs[0].text for o in outs]


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
        return text


def run_batch(messages_list, args, client):
    if hasattr(client, "chat_many") and callable(getattr(client, "chat_many")) and args.batch_size > 1:
        outs = []
        for start in tqdm(range(0, len(messages_list), args.batch_size), desc="Chatting"):
            chunk = messages_list[start : start + args.batch_size]
            outs.extend(
                client.chat_many(
                    args.model,
                    chunk,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=None,
                )
            )
        return outs
    else:
        return [client.chat(args.model, m, max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, stop=None) for m in tqdm(messages_list)]
