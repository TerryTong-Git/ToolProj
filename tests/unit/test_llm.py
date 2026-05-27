from types import SimpleNamespace

from src.exps_performance import llm


def test_openrouter_env_fallback(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(
            self,
            api_key=None,
            base_url=None,
            seed=None,
            reasoning_enabled=None,
            reasoning_effort=None,
            reasoning_max_tokens=None,
            reasoning_exclude=None,
            verbosity=None,
        ):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["seed"] = seed
            captured["reasoning_enabled"] = reasoning_enabled
            captured["reasoning_effort"] = reasoning_effort
            captured["reasoning_max_tokens"] = reasoning_max_tokens
            captured["reasoning_exclude"] = reasoning_exclude
            captured["verbosity"] = verbosity

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setattr(llm, "OpenRouterChatClient", FakeClient)

    args = SimpleNamespace(
        backend="openrouter",
        seed=42,
        openrouter_reasoning_enabled=True,
        openrouter_reasoning_effort="xhigh",
        openrouter_reasoning_max_tokens=2048,
        openrouter_reasoning_exclude=False,
        openrouter_verbosity="max",
    )

    client = llm.llm(args)

    assert isinstance(client, FakeClient)
    assert captured["api_key"] == "env-key"
    assert captured["base_url"] == llm.openrouter_api_base
    assert captured["seed"] == 42
    assert captured["reasoning_enabled"] is True
    assert captured["reasoning_effort"] == "xhigh"
    assert captured["reasoning_max_tokens"] == 2048
    assert captured["reasoning_exclude"] is False
    assert captured["verbosity"] == "max"
