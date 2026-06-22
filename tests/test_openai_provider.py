import json

from limen.providers import OpenAICompatibleProvider, ProviderSpec


def test_openai_compatible_provider_builds_chat_completion_payload() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout_s: float,
    ) -> dict:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["timeout_s"] = timeout_s
        return {"choices": [{"message": {"content": "hello"}}]}

    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://example.test/v1",
        transport=transport,
    )
    spec = ProviderSpec(agent_id=3, provider="openai_compatible", model="model-a", timeout_s=12)

    text = provider.complete(spec, [{"role": "user", "content": "Hi"}])

    assert text == "hello"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer sk-test",
        "Content-Type": "application/json",
    }
    assert json.loads(json.dumps(captured["body"])) == {
        "model": "model-a",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert captured["timeout_s"] == 12


def test_openai_compatible_provider_rejects_malformed_responses() -> None:
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://example.test/v1",
        transport=lambda *_args, **_kwargs: {"choices": []},
    )

    spec = ProviderSpec(agent_id=0, provider="openai_compatible", model="model-a")

    try:
        provider.complete(spec, [{"role": "user", "content": "Hi"}])
    except RuntimeError as exc:
        assert "malformed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
