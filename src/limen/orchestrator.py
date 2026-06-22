from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Protocol

from limen.providers import ProviderPool
from limen.roles import RoleInjector, parse_thinker, parse_verifier
from limen.routers import RouteDecision
from limen.trace import JSONLTraceSink, TraceEvent, stable_hash


class Router(Protocol):
    def route(self, messages: list[dict[str, str]]) -> RouteDecision: ...


class OrchestrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrchestrationResult:
    status: str
    output: str | None
    turns: int
    run_id: str


class Orchestrator:
    def __init__(
        self,
        router: Router,
        provider_pool: ProviderPool,
        max_turns: int = 5,
        max_provider_calls: int | None = None,
        trace_sink: JSONLTraceSink | None = None,
        respect_thinker_suggestions: bool = True,
    ) -> None:
        if max_turns < 0:
            raise ValueError("max_turns must be non-negative")
        self.router = router
        self.provider_pool = provider_pool
        self.max_turns = max_turns
        self.max_provider_calls = max_provider_calls
        self.trace_sink = trace_sink
        self.respect_thinker_suggestions = respect_thinker_suggestions

    def run(self, messages: list[dict[str, str]]) -> OrchestrationResult:
        transcript = _normalize_messages(messages)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        latest_worker_response: str | None = None
        suggested_role: str | None = None
        self._emit(run_id, "run_started", {"max_turns": self.max_turns})

        for turn in range(self.max_turns):
            self._emit(
                run_id,
                "turn_started",
                {
                    "turn": turn,
                    "message_count": len(transcript),
                    "transcript_hash": stable_hash(transcript),
                },
            )
            decision = self.router.route(transcript)
            if suggested_role and self.respect_thinker_suggestions:
                decision = replace(
                    decision,
                    role=suggested_role,
                    diagnostics={**(decision.diagnostics or {}), "role_override": suggested_role},
                )
                suggested_role = None

            role = RoleInjector.role_name(decision.role)
            self._emit(
                run_id,
                "route_selected",
                {
                    "turn": turn,
                    "agent_id": decision.agent_id,
                    "route_id": decision.route_id,
                    "role": role,
                    "reason": decision.reason,
                    "margins": decision.margins,
                },
            )

            if role == "Verifier" and latest_worker_response is None:
                self._emit(run_id, "run_failed", {"turn": turn, "reason": "verifier_before_worker"})
                raise OrchestrationError("verifier_before_worker")

            if self.max_provider_calls is not None and turn >= self.max_provider_calls:
                self._emit(
                    run_id,
                    "run_completed",
                    {"turn": turn, "status": "budget_exceeded", "provider_calls": turn},
                )
                return OrchestrationResult(
                    status="budget_exceeded",
                    output=latest_worker_response,
                    turns=turn,
                    run_id=run_id,
                )

            agent_id = decision.agent_id if decision.agent_id is not None else 0
            injected = RoleInjector.inject(transcript, role)
            response = self.provider_pool.dispatch(
                agent_id,
                injected,
                role=role,
                metadata={"run_id": run_id, "turn": turn, "route_id": decision.route_id},
            )
            transcript.append({"role": "assistant", "content": response.text})
            self._emit(
                run_id,
                "provider_called",
                {
                    "turn": turn,
                    "provider": response.provider,
                    "provider_model": response.model,
                    "agent_id": response.agent_id,
                    "role": role,
                    "response_hash": stable_hash(response.text),
                },
            )

            accepted = False
            if role == "Worker":
                latest_worker_response = response.text
            elif role == "Thinker":
                thinker = parse_thinker(response.text)
                suggested_role = thinker.suggested_role
            elif role == "Verifier":
                verifier = parse_verifier(response.text)
                accepted = verifier.safe_status == "accepted"

            self._emit(
                run_id,
                "turn_completed",
                {
                    "turn": turn,
                    "role": role,
                    "accepted": accepted,
                    "response_hash": stable_hash(response.text),
                },
            )
            if accepted:
                self._emit(run_id, "run_completed", {"turn": turn, "status": "accepted"})
                return OrchestrationResult(
                    status="accepted",
                    output=response.text,
                    turns=turn + 1,
                    run_id=run_id,
                )

        self._emit(run_id, "run_completed", {"turn": self.max_turns, "status": "max_turns"})
        return OrchestrationResult(
            status="max_turns",
            output=latest_worker_response,
            turns=self.max_turns,
            run_id=run_id,
        )

    def _emit(self, run_id: str, event: str, fields: dict[str, object]) -> None:
        if self.trace_sink is None:
            return
        self.trace_sink.write(TraceEvent.new(event, run_id, fields))


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"invalid message: {message!r}")
        normalized.append({"role": role, "content": content})
    return normalized
