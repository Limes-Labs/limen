import pytest

from limen.providers import MockProvider, ProviderPool, ProviderSpec
from limen.workflow import WorkflowExecutor, WorkflowPlan, WorkflowStep


def test_workflow_validates_topological_access_and_executes_with_isolated_context() -> None:
    plan = WorkflowPlan(
        steps=[
            WorkflowStep(id="plan", agent_id=0, prompt="Plan the solution."),
            WorkflowStep(id="solve", agent_id=1, prompt="Implement it.", access=["plan"]),
            WorkflowStep(id="verify", agent_id=0, prompt="Check it.", access=["solve"]),
        ]
    )
    provider = MockProvider(responses_by_agent={0: ["plan output", "verified"], 1: ["solution"]})
    pool = ProviderPool(
        specs=[
            ProviderSpec(agent_id=0, provider="mock", model="planner"),
            ProviderSpec(agent_id=1, provider="mock", model="worker"),
        ],
        adapters={"mock": provider},
    )

    result = WorkflowExecutor(pool).execute(plan, "Build a router.")

    assert result.outputs == {"plan": "plan output", "solve": "solution", "verify": "verified"}
    assert "plan output" in provider.calls[1].messages[-1]["content"]
    assert "plan output" not in provider.calls[2].messages[-1]["content"]
    assert "solution" in provider.calls[2].messages[-1]["content"]


def test_workflow_rejects_forward_references() -> None:
    plan = WorkflowPlan(
        steps=[
            WorkflowStep(id="solve", agent_id=0, prompt="Solve.", access=["verify"]),
            WorkflowStep(id="verify", agent_id=0, prompt="Verify."),
        ]
    )

    with pytest.raises(ValueError, match="forward or unknown"):
        plan.validate()
