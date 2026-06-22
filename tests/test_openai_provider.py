import json

import pytest

from limen.providers import OpenAICompatibleProvider, ProviderPool, ProviderSpec


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
        return {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }

    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://example.test/v1",
        transport=transport,
    )
    spec = ProviderSpec(agent_id=3, provider="openai_compatible", model="model-a", timeout_s=12)

    completion = provider.complete(spec, [{"role": "user", "content": "Hi"}])

    assert completion.text == "hello"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 7
    assert completion.usage.completion_tokens == 3
    assert completion.usage.total_tokens == 10
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


def test_provider_pool_estimates_cost_when_pricing_is_declared() -> None:
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://example.test/v1",
        transport=lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 2000, "total_tokens": 3000},
        },
    )

    pool = ProviderPool(
        specs=[
            ProviderSpec(
                agent_id=0,
                provider="openai_compatible",
                model="model-a",
                input_cost_per_million_tokens=1.0,
                output_cost_per_million_tokens=2.0,
            )
        ],
        adapters={"openai_compatible": provider},
    )

    response = pool.dispatch(0, [{"role": "user", "content": "Hi"}])

    assert response.usage is not None
    assert response.usage.total_tokens == 3000
    assert response.estimated_cost_usd == 0.005


def test_provider_spec_rejects_negative_pricing() -> None:
    with pytest.raises(ValueError, match="cost fields must be non-negative"):
        ProviderSpec(
            agent_id=0,
            provider="openai_compatible",
            model="model-a",
            input_cost_per_million_tokens=-1.0,
        )


def test_provider_pool_exposes_public_model_manifest() -> None:
    pool = ProviderPool(
        specs=[
            ProviderSpec(
                agent_id=0,
                provider="openai_compatible",
                model="model-a",
                name="fast",
                input_cost_per_million_tokens=1.0,
                output_cost_per_million_tokens=2.0,
            )
        ],
        adapters={},
    )

    assert pool.model_manifest() == [
        {
            "agent_id": 0,
            "name": "fast",
            "provider": "openai_compatible",
            "model": "model-a",
            "input_cost_per_million_tokens": 1.0,
            "output_cost_per_million_tokens": 2.0,
        }
    ]
