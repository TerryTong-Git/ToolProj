import json
import re
from pathlib import Path
from typing import Any

from src.reasoning_benchmark import runner as perf_main
from src.reasoning_benchmark.llm_clients import ChatResponse
from src.reasoning_benchmark.records import CheckpointManager


class FakeStructuredOpenRouterClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def chat_many(
        self,
        model: str,
        messages_list: list[list[dict[str, str]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        request_timeout: float = 120.0,
        request_options_list: list[dict[str, Any] | None] | None = None,
        on_result: Any = None,
    ) -> list[ChatResponse]:
        responses: list[ChatResponse] = []
        options = request_options_list or [None] * len(messages_list)
        for idx, messages in enumerate(messages_list):
            request_options = options[idx] or {}
            answer = self._answer_for_prompt(messages[-1]["content"])
            response = ChatResponse(
                text=json.dumps(self._payload_for_schema(request_options, answer)),
                attempts=1,
                reasoning="",
                reasoning_tokens=3,
                structured_requested=bool(request_options.get("response_format")),
                reasoning_requested=bool((request_options.get("extra_body") or {}).get("reasoning")),
                request_options=dict(request_options),
            )
            self.requests.append(
                {
                    "model": model,
                    "messages": messages,
                    "request_options": request_options,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop,
                    "request_timeout": request_timeout,
                }
            )
            responses.append(response)
            if on_result is not None:
                on_result(idx, response)
        return responses

    @staticmethod
    def _answer_for_prompt(prompt: str) -> int:
        match = re.search(r"Compute:\s*(\d+)\s*([+\-*])\s*(\d+)", prompt)
        assert match is not None, prompt
        left, op, right = match.groups()
        a = int(left)
        b = int(right)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        return a * b

    @staticmethod
    def _payload_for_schema(request_options: dict[str, Any], answer: int) -> dict[str, str]:
        schema_name = ((request_options.get("response_format") or {}).get("json_schema") or {}).get("name") or ""
        payload = {"Answer": str(answer), "simulation": f"The answer is {answer}."}
        if schema_name == "CodeReasoning":
            payload["code"] = f"def solution():\n    return {answer}"
        return payload


def test_frontier_structured_openrouter_e2e_with_fake_client(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake_client = FakeStructuredOpenRouterClient()
    monkeypatch.setattr(perf_main, "llm", lambda _args: fake_client)

    args = perf_main.Args(
        root=str(tmp_path),
        n=2,
        digits_list=[2],
        kinds=["add"],
        seed=1,
        backend="openrouter",
        model="fake/model",
        exp_id="structured-e2e",
        openrouter_structured_outputs=True,
        openrouter_structured_strict=True,
        openrouter_reasoning_enabled=True,
        openrouter_reasoning_effort="low",
        openrouter_reasoning_exclude=True,
        openrouter_retry_attempts=1,
        openrouter_max_concurrency=2,
        stage_batch_retry_attempts=1,
        max_tokens=256,
        batch_size=2,
        checkpoint_every=1,
        request_timeout=30.0,
        exec_code=True,
        controlled_sim=False,
        tb_disable=True,
    )

    perf_main.run(args)

    run_dir = Path(args.exp_dir)
    res_path = run_dir / "res.jsonl"
    stats_path = run_dir / "llm_stage_stats.jsonl"
    rows = [json.loads(line) for line in res_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = [json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 2
    assert all(row["sim_err_msg"] == "ok" and row["sim_parse_err"] is False for row in rows)
    assert all(row["code_err_msg"] == "ok,ok" and row["code_correct"] is True for row in rows)
    assert all(row["nl_err_msg"] == "ok" and row["nl_parse_err"] is False for row in rows)
    assert all(row["nl_correct"] and row["sim_correct"] for row in rows)

    assert any(stat["stage"] == "Arm2" and stat["structured_requested"] == 2 for stat in stats)
    assert any(stat["stage"] == "Arm1" and stat["reasoning_requested"] == 2 for stat in stats)
    assert all(request["request_options"].get("response_format") for request in fake_client.requests)
    assert all((request["request_options"].get("extra_body") or {}).get("reasoning") for request in fake_client.requests)

    checkpoint = CheckpointManager(str(res_path))
    assert len(checkpoint.all_records()) == 2
