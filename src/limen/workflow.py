from __future__ import annotations

from dataclasses import dataclass, field

from limen.providers import ProviderPool


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    agent_id: int
    prompt: str
    access: list[str] = field(default_factory=list)
    role: str = "Worker"


@dataclass(frozen=True)
class WorkflowPlan:
    steps: list[WorkflowStep]

    def validate(self) -> None:
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"duplicate workflow step id={step.id}")
            missing = [item for item in step.access if item not in seen]
            if missing:
                raise ValueError(
                    f"workflow step {step.id} has forward or unknown access references: {missing}"
                )
            seen.add(step.id)


@dataclass(frozen=True)
class WorkflowResult:
    outputs: dict[str, str]


class WorkflowExecutor:
    def __init__(self, provider_pool: ProviderPool) -> None:
        self.provider_pool = provider_pool

    def execute(self, plan: WorkflowPlan, user_request: str) -> WorkflowResult:
        plan.validate()
        outputs: dict[str, str] = {}

        for step in plan.steps:
            context = "\n".join(f"[{item}]\n{outputs[item]}" for item in step.access)
            prompt_parts = [f"User request:\n{user_request}", f"Subtask:\n{step.prompt}"]
            if context:
                prompt_parts.append(f"Context:\n{context}")
            messages = [{"role": "user", "content": "\n\n".join(prompt_parts)}]
            response = self.provider_pool.dispatch(
                step.agent_id,
                messages,
                role=step.role,
                metadata={"workflow_step_id": step.id},
            )
            outputs[step.id] = response.text

        return WorkflowResult(outputs=outputs)
