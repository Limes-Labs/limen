from pathlib import Path

import pytest

from limen.routers import MetadataRouter
from limen.routes import RouteLibrary


def write_route(root: Path, name: str, body: str) -> None:
    (root / f"{name}.md").write_text(body.strip() + "\n", encoding="utf-8")


def test_metadata_router_prefers_clear_specialist_and_explains_signals(tmp_path: Path) -> None:
    write_route(
        tmp_path,
        "L0",
        """
        ---
        id: L0
        adapter_name: general
        priority: 0
        ---
        ## Description
        General chat and ambiguous work.
        """,
    )
    write_route(
        tmp_path,
        "L2",
        """
        ---
        id: L2
        adapter_name: coding
        priority: 10
        ---
        ## Description
        Coding, tests, patches, and repository work.
        ## Strong Signals
        pytest, failing test, apply_patch
        ## Positive Signals
        repository, code, verify
        ## Negative Signals
        explain, overview
        ## Examples
        Fix a failing pytest => L2
        """,
    )

    library = RouteLibrary.load(tmp_path)
    router = MetadataRouter(library)

    decision = router.route_text("Fix a failing pytest in this repository and verify it.")

    assert decision.route_id == "L2"
    assert decision.target == "coding"
    assert decision.reason == "specialist_signal"
    assert decision.diagnostics is not None
    assert decision.diagnostics["selected"]["strong_hits"] == ["pytest", "failing test"]


def test_metadata_router_falls_back_to_entry_for_general_or_ambiguous_requests(
    tmp_path: Path,
) -> None:
    write_route(
        tmp_path,
        "L0",
        """
        ---
        id: L0
        adapter_name: general
        ---
        ## Description
        General chat.
        """,
    )
    write_route(
        tmp_path,
        "L1",
        """
        ---
        id: L1
        adapter_name: research
        priority: 0
        ---
        ## Description
        Research tasks.
        ## Positive Signals
        compare
        """,
    )
    write_route(
        tmp_path,
        "L2",
        """
        ---
        id: L2
        adapter_name: coding
        priority: 0
        ---
        ## Description
        Coding tasks.
        ## Positive Signals
        compare
        """,
    )

    router = MetadataRouter(RouteLibrary.load(tmp_path))

    assert router.route_text("Explain model routing in one paragraph.").route_id == "L0"

    ambiguous = router.route_text("Compare the two approaches.")
    assert ambiguous.route_id == "L0"
    assert ambiguous.reason == "ambiguous_specialist_signal_default_entry"


def test_model_route_is_accepted_only_when_metadata_has_no_stronger_specialist(
    tmp_path: Path,
) -> None:
    write_route(
        tmp_path,
        "L0",
        """
        ---
        id: L0
        adapter_name: general
        ---
        ## Description
        General chat.
        """,
    )
    write_route(
        tmp_path,
        "L2",
        """
        ---
        id: L2
        adapter_name: coding
        priority: 10
        ---
        ## Description
        Coding tasks.
        ## Strong Signals
        pytest
        """,
    )

    router = MetadataRouter(RouteLibrary.load(tmp_path))

    assert router.apply_guardrail("model_id=L2", "Tell me a story.").reason == "model_route"

    guarded = router.apply_guardrail("model_id=L0", "Fix pytest.")
    assert guarded.route_id == "L2"
    assert guarded.raw_route == "L0"
    assert guarded.reason == "specialist_signal"


def test_route_library_rejects_duplicate_route_ids(tmp_path: Path) -> None:
    write_route(
        tmp_path,
        "L0",
        """
        ---
        id: L0
        target: general
        ---
        ## Description
        Entry route.
        """,
    )
    write_route(
        tmp_path,
        "code",
        """
        ---
        id: specialist
        target: coding-worker
        ---
        ## Description
        Coding route.
        """,
    )
    write_route(
        tmp_path,
        "research",
        """
        ---
        id: specialist
        target: research-worker
        ---
        ## Description
        Research route.
        """,
    )

    with pytest.raises(ValueError, match="duplicate route id 'specialist'"):
        RouteLibrary.load(tmp_path)
