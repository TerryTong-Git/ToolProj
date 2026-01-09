from types import SimpleNamespace

from src.exps_performance import llm


def test_openrouter_env_fallback(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, api_key=None, base_url=None, seed=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["seed"] = seed

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setattr(llm, "OpenRouterChatClient", FakeClient)

    args = SimpleNamespace(backend="openrouter", seed=42)

    client = llm.llm(args)

    assert isinstance(client, FakeClient)
    assert captured["api_key"] == "env-key"
    assert captured["base_url"] == llm.openrouter_api_base
    assert captured["seed"] == 42
