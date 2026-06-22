from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class ProviderSpec:
    agent_id: int
    provider: str
    model: str
    name: str | None = None
    base_url: str | None = None
    timeout_s: float | None = None
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None

    def __post_init__(self) -> None:
        costs = (self.input_cost_per_million_tokens, self.output_cost_per_million_tokens)
        if any(cost is not None and cost < 0 for cost in costs):
            raise ValueError("cost fields must be non-negative")


@dataclass(frozen=True)
class ProviderCompletion:
    text: str
    usage: TokenUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider: str
    model: str
    agent_id: int
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage | None = None
    estimated_cost_usd: float | None = None


@dataclass(frozen=True)
class ProviderCall:
    spec: ProviderSpec
    messages: list[dict[str, str]]
    role: str | None
    metadata: dict[str, Any]


class ProviderAdapter(Protocol):
    def complete(
        self,
        spec: ProviderSpec,
        messages: list[dict[str, str]],
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderCompletion | str: ...


class MockProvider:
    """Deterministic in-process provider for tests and dry runs."""

    def __init__(
        self,
        responses_by_role: dict[str, str | list[str]] | None = None,
        responses_by_agent: dict[int, str | list[str]] | None = None,
        usage_by_role: dict[str, dict[str, int] | TokenUsage] | None = None,
        usage_by_agent: dict[int, dict[str, int] | TokenUsage] | None = None,
        default_response: str = "MOCK",
    ) -> None:
        self.responses_by_role = responses_by_role or {}
        self.responses_by_agent = responses_by_agent or {}
        self.usage_by_role = usage_by_role or {}
        self.usage_by_agent = usage_by_agent or {}
        self.default_response = default_response
        self.calls: list[ProviderCall] = []
        self._cursors: dict[tuple[str, str], int] = {}

    def complete(
        self,
        spec: ProviderSpec,
        messages: list[dict[str, str]],
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderCompletion:
        call = ProviderCall(spec=spec, messages=list(messages), role=role, metadata=metadata or {})
        self.calls.append(call)

        if role is not None and role in self.responses_by_role:
            return ProviderCompletion(
                text=self._next(("role", role), self.responses_by_role[role]),
                usage=_coerce_usage(self.usage_by_role.get(role)),
            )
        if spec.agent_id in self.responses_by_agent:
            return ProviderCompletion(
                text=self._next(
                    ("agent", str(spec.agent_id)),
                    self.responses_by_agent[spec.agent_id],
                ),
                usage=_coerce_usage(self.usage_by_agent.get(spec.agent_id)),
            )
        return ProviderCompletion(text=self.default_response)

    def _next(self, key: tuple[str, str], value: str | list[str]) -> str:
        if isinstance(value, str):
            return value
        index = self._cursors.get(key, 0)
        self._cursors[key] = index + 1
        return value[min(index, len(value) - 1)]


class OpenAICompatibleProvider:
    """Minimal chat-completions adapter with injectable transport for tests.

    The adapter deliberately receives credentials through its constructor rather
    than reading process environment variables. Applications can decide how they
    materialize secrets and can keep that boundary out of reusable library code.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        transport: Any | None = None,
        default_timeout_s: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.transport = transport or self._urllib_transport
        self.default_timeout_s = default_timeout_s

    def complete(
        self,
        spec: ProviderSpec,
        messages: list[dict[str, str]],
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderCompletion:
        del role, metadata
        body: dict[str, object] = {"model": spec.model, "messages": messages}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.transport(
            f"{self.base_url}/chat/completions",
            headers,
            body,
            spec.timeout_s or self.default_timeout_s,
        )
        return ProviderCompletion(
            text=self._extract_text(response),
            usage=self._extract_usage(response),
        )

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"malformed OpenAI-compatible response: {response!r}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"malformed OpenAI-compatible response: {response!r}")
        return content

    @staticmethod
    def _extract_usage(response: dict[str, Any]) -> TokenUsage | None:
        usage = response.get("usage")
        if usage is None:
            return None
        if not isinstance(usage, dict):
            raise RuntimeError(f"malformed OpenAI-compatible response: {response!r}")
        return _coerce_usage(usage)

    @staticmethod
    def _urllib_transport(
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout_s: float,
    ) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise RuntimeError("OpenAI-compatible response must be a JSON object")
        return decoded


@dataclass
class ProviderPool:
    specs: list[ProviderSpec]
    adapters: dict[str, ProviderAdapter]

    def model_manifest(self) -> list[dict[str, object]]:
        return [
            {
                "agent_id": spec.agent_id,
                "name": spec.name,
                "provider": spec.provider,
                "model": spec.model,
                "input_cost_per_million_tokens": spec.input_cost_per_million_tokens,
                "output_cost_per_million_tokens": spec.output_cost_per_million_tokens,
            }
            for spec in self.specs
        ]

    def spec_for_agent(self, agent_id: int) -> ProviderSpec:
        for spec in self.specs:
            if spec.agent_id == agent_id:
                return spec
        raise KeyError(f"unknown agent_id={agent_id}")

    def dispatch(
        self,
        agent_id: int,
        messages: list[dict[str, str]],
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        spec = self.spec_for_agent(agent_id)
        try:
            adapter = self.adapters[spec.provider]
        except KeyError as exc:
            raise KeyError(f"unsupported provider={spec.provider}") from exc
        completion = _coerce_completion(
            adapter.complete(spec, messages, role=role, metadata=metadata)
        )
        estimated_cost_usd = _estimate_cost_usd(spec, completion.usage)
        return ProviderResponse(
            text=completion.text,
            provider=spec.provider,
            model=spec.model,
            agent_id=spec.agent_id,
            metadata={**(metadata or {}), **completion.metadata},
            usage=completion.usage,
            estimated_cost_usd=estimated_cost_usd,
        )


def _coerce_completion(value: ProviderCompletion | str) -> ProviderCompletion:
    if isinstance(value, ProviderCompletion):
        return value
    return ProviderCompletion(text=value)


def _coerce_usage(value: dict[str, int] | TokenUsage | None) -> TokenUsage | None:
    if value is None:
        return None
    if isinstance(value, TokenUsage):
        return value
    prompt_tokens = _require_non_negative_int(value, "prompt_tokens")
    completion_tokens = _require_non_negative_int(value, "completion_tokens")
    total_tokens = value.get("total_tokens", prompt_tokens + completion_tokens)
    if not isinstance(total_tokens, int) or total_tokens < 0:
        raise RuntimeError("token usage total_tokens must be a non-negative integer")
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _require_non_negative_int(value: dict[str, int], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or item < 0:
        raise RuntimeError(f"token usage {key} must be a non-negative integer")
    return item


def _estimate_cost_usd(spec: ProviderSpec, usage: TokenUsage | None) -> float | None:
    if usage is None:
        return None
    if spec.input_cost_per_million_tokens is None or spec.output_cost_per_million_tokens is None:
        return None
    input_cost = usage.prompt_tokens * spec.input_cost_per_million_tokens / 1_000_000
    output_cost = usage.completion_tokens * spec.output_cost_per_million_tokens / 1_000_000
    return input_cost + output_cost
