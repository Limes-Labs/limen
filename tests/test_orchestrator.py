from pathlib import Path

import pytest

from limen.orchestrator import OrchestrationError, Orchestrator
from limen.providers import MockProvider, ProviderPool, ProviderSpec
from limen.routers import ScriptedRouter
from limen.trace import JSONLTraceSink


def pool(provider: MockProvider) -> ProviderPool:
    return ProviderPool(
        specs=[ProviderSpec(agent_id=0, provider="mock", model="mock-worker")],
        adapters={"mock": provider},
    )


def test_orchestrator_runs_worker_then_verifier_until_acceptance(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    provider = MockProvider(responses_by_role={"Worker": ["draft answer"], "Verifier": ["ACCEPT"]})
    orchestrator = Orchestrator(
        router=ScriptedRouter([(0, "Worker"), (0, "Verifier")]),
        provider_pool=pool(provider),
        trace_sink=JSONLTraceSink(trace_path),
    )

    result = orchestrator.run([{"role": "user", "content": "Solve it."}])

    assert result.status == "accepted"
    assert result.output == "ACCEPT"
    assert provider.calls[0].messages[0]["role"] == "system"
    assert "concrete step" in provider.calls[0].messages[0]["content"]

    events = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    assert any('"event":"route_selected"' in line for line in events)
    assert any('"event":"run_completed"' in line for line in events)


def test_orchestrator_refuses_verifier_before_worker_response() -> None:
    orchestrator = Orchestrator(
        router=ScriptedRouter([(0, "Verifier")]),
        provider_pool=pool(MockProvider(responses_by_role={"Verifier": ["ACCEPT"]})),
    )

    with pytest.raises(OrchestrationError, match="verifier_before_worker"):
        orchestrator.run([{"role": "user", "content": "Solve it."}])


def test_orchestrator_honors_provider_call_budget() -> None:
    orchestrator = Orchestrator(
        router=ScriptedRouter([(0, "Worker"), (0, "Worker")]),
        provider_pool=pool(MockProvider(responses_by_role={"Worker": ["one", "two"]})),
        max_provider_calls=1,
    )

    result = orchestrator.run([{"role": "user", "content": "Solve it."}])

    assert result.status == "budget_exceeded"
    assert result.output == "one"
