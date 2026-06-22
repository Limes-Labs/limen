from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from limen.routers import RouteDecision


@dataclass(frozen=True)
class RouteEvalCase:
    id: str
    prompt: str
    expected_route_id: str


@dataclass(frozen=True)
class RouteEvalResult:
    total: int
    correct: int
    misses: list[dict[str, str]]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


class TextRouter(Protocol):
    def route_text(self, text: str) -> RouteDecision: ...


def load_route_eval_cases(path: str | Path) -> list[RouteEvalCase]:
    cases: list[RouteEvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            try:
                cases.append(
                    RouteEvalCase(
                        id=str(item["id"]),
                        prompt=str(item["prompt"]),
                        expected_route_id=str(item["expected_route_id"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"missing field {exc!s} in {path}:{line_number}") from exc
    return cases


def evaluate_routes(router: TextRouter, cases: list[RouteEvalCase]) -> RouteEvalResult:
    correct = 0
    misses: list[dict[str, str]] = []
    for case in cases:
        decision = router.route_text(case.prompt)
        actual = str(decision.route_id)
        if actual == case.expected_route_id:
            correct += 1
        else:
            misses.append({"id": case.id, "expected": case.expected_route_id, "actual": actual})
    return RouteEvalResult(total=len(cases), correct=correct, misses=misses)
