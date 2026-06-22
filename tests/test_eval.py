import json
from pathlib import Path

from limen.eval import RouteEvalCase, evaluate_routes, load_route_eval_cases
from limen.routers import RouteDecision


class PrefixRouter:
    def route_text(self, text: str) -> RouteDecision:
        return RouteDecision(route_id="code" if text.startswith("Fix") else "general")


def test_evaluate_routes_reports_accuracy_and_misses() -> None:
    result = evaluate_routes(
        PrefixRouter(),
        [
            RouteEvalCase(id="a", prompt="Fix pytest", expected_route_id="code"),
            RouteEvalCase(id="b", prompt="Explain routing", expected_route_id="general"),
            RouteEvalCase(id="c", prompt="Fix docs", expected_route_id="general"),
        ],
    )

    assert result.total == 3
    assert result.correct == 2
    assert result.accuracy == 2 / 3
    assert result.misses == [{"id": "c", "expected": "general", "actual": "code"}]


def test_load_route_eval_cases_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prompt": "Fix pytest", "expected_route_id": "code"}),
                json.dumps({"id": "b", "prompt": "Explain", "expected_route_id": "general"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_route_eval_cases(path)

    assert [case.id for case in cases] == ["a", "b"]
