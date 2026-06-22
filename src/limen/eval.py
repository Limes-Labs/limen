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
    confusion_matrix: dict[str, dict[str, int]]
    per_route: dict[str, dict[str, float | int]]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_jsonable(self) -> dict[str, object]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "misses": self.misses,
            "confusion_matrix": self.confusion_matrix,
            "per_route": self.per_route,
        }


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
    confusion_matrix: dict[str, dict[str, int]] = {}
    per_route_counts: dict[str, dict[str, int]] = {}
    for case in cases:
        decision = router.route_text(case.prompt)
        actual = str(decision.route_id)
        expected = case.expected_route_id
        confusion_matrix.setdefault(expected, {})
        confusion_matrix[expected][actual] = confusion_matrix[expected].get(actual, 0) + 1
        per_route_counts.setdefault(expected, {"total": 0, "correct": 0})
        per_route_counts[expected]["total"] += 1
        if actual == case.expected_route_id:
            correct += 1
            per_route_counts[expected]["correct"] += 1
        else:
            misses.append({"id": case.id, "expected": case.expected_route_id, "actual": actual})

    per_route: dict[str, dict[str, float | int]] = {}
    for route_id, counts in per_route_counts.items():
        route_total = counts["total"]
        route_correct = counts["correct"]
        per_route[route_id] = {
            "total": route_total,
            "correct": route_correct,
            "accuracy": route_correct / route_total if route_total else 0.0,
        }

    return RouteEvalResult(
        total=len(cases),
        correct=correct,
        misses=misses,
        confusion_matrix=confusion_matrix,
        per_route=per_route,
    )
