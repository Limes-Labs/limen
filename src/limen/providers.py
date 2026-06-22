from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderSpec:
    agent_id: int
    provider: str
    model: str
    name: str | None = None
    base_url: str | None = None
    timeout_s: float | None = None


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider: str
    model: str
    agent_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


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
    ) -> str: ...


class MockProvider:
    """Deterministic in-process provider for tests and dry runs."""

    def __init__(
        self,
        responses_by_role: dict[str, str | list[str]] | None = None,
        responses_by_agent: dict[int, str | list[str]] | None = None,
        default_response: str = "MOCK",
    ) -> None:
        self.responses_by_role = responses_by_role or {}
        self.responses_by_agent = responses_by_agent or {}
        self.default_response = default_response
        self.calls: list[ProviderCall] = []
        self._cursors: dict[tuple[str, str], int] = {}

    def complete(
        self,
        spec: ProviderSpec,
        messages: list[dict[str, str]],
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        call = ProviderCall(spec=spec, messages=list(messages), role=role, metadata=metadata or {})
        self.calls.append(call)

        if role is not None and role in self.responses_by_role:
            return self._next(("role", role), self.responses_by_role[role])
        if spec.agent_id in self.responses_by_agent:
            return self._next(("agent", str(spec.agent_id)), self.responses_by_agent[spec.agent_id])
        return self.default_response

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
    ) -> str:
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
        return self._extract_text(response)

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
        text = adapter.complete(spec, messages, role=role, metadata=metadata)
        return ProviderResponse(
            text=text,
            provider=spec.provider,
            model=spec.model,
            agent_id=spec.agent_id,
            metadata=metadata or {},
        )
