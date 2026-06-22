from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Protocol

from limen.providers import ProviderPool, TokenUsage
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
    usage: TokenUsage | None = None
    estimated_cost_usd: float | None = None


class Orchestrator:
    def __init__(
        self,
        router: Router,
        provider_pool: ProviderPool,
        max_turns: int = 5,
        max_provider_calls: int | None = None,
        max_estimated_cost_usd: float | None = None,
        trace_sink: JSONLTraceSink | None = None,
        respect_thinker_suggestions: bool = True,
    ) -> None:
        if max_turns < 0:
            raise ValueError("max_turns must be non-negative")
        if max_estimated_cost_usd is not None and max_estimated_cost_usd < 0:
            raise ValueError("max_estimated_cost_usd must be non-negative")
        self.router = router
        self.provider_pool = provider_pool
        self.max_turns = max_turns
        self.max_provider_calls = max_provider_calls
        self.max_estimated_cost_usd = max_estimated_cost_usd
        self.trace_sink = trace_sink
        self.respect_thinker_suggestions = respect_thinker_suggestions

    def run(self, messages: list[dict[str, str]]) -> OrchestrationResult:
        transcript = _normalize_messages(messages)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        latest_worker_response: str | None = None
        suggested_role: str | None = None
        total_usage: TokenUsage | None = None
        total_estimated_cost_usd = 0.0
        has_estimated_cost = False
        self._emit(
            run_id,
            "run_started",
            {
                "max_turns": self.max_turns,
                "model_manifest": self.provider_pool.model_manifest(),
                "max_estimated_cost_usd": self.max_estimated_cost_usd,
            },
        )

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
                    usage=total_usage,
                    estimated_cost_usd=(
                        total_estimated_cost_usd if has_estimated_cost else None
                    ),
                )

            agent_id = decision.agent_id if decision.agent_id is not None else 0
            injected = RoleInjector.inject(transcript, role)
            response = self.provider_pool.dispatch(
                agent_id,
                injected,
                role=role,
                metadata={"run_id": run_id, "turn": turn, "route_id": decision.route_id},
            )
            if response.usage is not None:
                total_usage = (
                    response.usage if total_usage is None else total_usage + response.usage
                )
            if response.estimated_cost_usd is not None:
                total_estimated_cost_usd += response.estimated_cost_usd
                has_estimated_cost = True
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
                    "usage": _usage_json(response.usage),
                    "estimated_cost_usd": response.estimated_cost_usd,
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
                self._emit(
                    run_id,
                    "run_completed",
                    {
                        "turn": turn,
                        "status": "accepted",
                        "usage": _usage_json(total_usage),
                        "estimated_cost_usd": (
                            total_estimated_cost_usd if has_estimated_cost else None
                        ),
                    },
                )
                return OrchestrationResult(
                    status="accepted",
                    output=response.text,
                    turns=turn + 1,
                    run_id=run_id,
                    usage=total_usage,
                    estimated_cost_usd=(
                        total_estimated_cost_usd if has_estimated_cost else None
                    ),
                )

            if (
                self.max_estimated_cost_usd is not None
                and has_estimated_cost
                and total_estimated_cost_usd >= self.max_estimated_cost_usd
            ):
                self._emit(
                    run_id,
                    "run_completed",
                    {
                        "turn": turn,
                        "status": "cost_budget_exceeded",
                        "usage": _usage_json(total_usage),
                        "estimated_cost_usd": total_estimated_cost_usd,
                        "max_estimated_cost_usd": self.max_estimated_cost_usd,
                    },
                )
                return OrchestrationResult(
                    status="cost_budget_exceeded",
                    output=latest_worker_response,
                    turns=turn + 1,
                    run_id=run_id,
                    usage=total_usage,
                    estimated_cost_usd=total_estimated_cost_usd,
                )

        self._emit(
            run_id,
            "run_completed",
            {
                "turn": self.max_turns,
                "status": "max_turns",
                "usage": _usage_json(total_usage),
                "estimated_cost_usd": total_estimated_cost_usd if has_estimated_cost else None,
            },
        )
        return OrchestrationResult(
            status="max_turns",
            output=latest_worker_response,
            turns=self.max_turns,
            run_id=run_id,
            usage=total_usage,
            estimated_cost_usd=total_estimated_cost_usd if has_estimated_cost else None,
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


def _usage_json(usage: TokenUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
