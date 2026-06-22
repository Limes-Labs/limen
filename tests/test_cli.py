import json
import sys
from pathlib import Path

from limen.cli import main


def write_route(root: Path, name: str, body: str) -> None:
    (root / f"{name}.md").write_text(body.strip() + "\n", encoding="utf-8")


def test_cli_routes_prompt_and_preserves_legacy_shape(tmp_path: Path, capsys) -> None:
    write_route(
        tmp_path,
        "L0",
        """
        ---
        id: L0
        target: general
        ---
        ## Description
        General entry route.
        """,
    )
    write_route(
        tmp_path,
        "code",
        """
        ---
        id: code
        target: coding-worker
        priority: 100
        ---
        ## Description
        Coding work.
        ## Strong Signals
        pytest
        """,
    )

    assert main(["--library-dir", str(tmp_path), "Fix pytest"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["route_id"] == "code"
    assert output["target"] == "coding-worker"
    assert output["reason"] == "specialist_signal"


def test_cli_eval_reports_accuracy_and_misses(tmp_path: Path, capsys) -> None:
    write_route(
        tmp_path,
        "L0",
        """
        ---
        id: L0
        target: general
        ---
        ## Description
        General entry route.
        """,
    )
    write_route(
        tmp_path,
        "code",
        """
        ---
        id: code
        target: coding-worker
        priority: 100
        ---
        ## Description
        Coding work.
        ## Strong Signals
        pytest
        """,
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prompt": "Fix pytest", "expected_route_id": "code"}),
                json.dumps({"id": "b", "prompt": "Explain thresholds", "expected_route_id": "L0"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["eval", "--library-dir", str(tmp_path), "--cases", str(cases)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {"total": 2, "correct": 2, "accuracy": 1.0, "misses": []}


def test_cli_entrypoint_reads_sys_argv_for_eval(tmp_path: Path, capsys, monkeypatch) -> None:
    write_route(
        tmp_path,
        "L0",
        """
        ---
        id: L0
        target: general
        ---
        ## Description
        General entry route.
        """,
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps({"id": "a", "prompt": "Explain thresholds", "expected_route_id": "L0"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["limen", "eval", "--library-dir", str(tmp_path), "--cases", str(cases)],
    )

    assert main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["accuracy"] == 1.0
