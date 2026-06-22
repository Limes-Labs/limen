import numpy as np
import pytest

from limen.routers import LinearHeadRouter


def test_linear_head_routes_agent_and_role_from_hidden_vector() -> None:
    weights = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    router = LinearHeadRouter(weights, num_agents=3, role_names=("Worker", "Verifier"))

    decision = router.route_vector(np.array([0.2, 1.0, 0.4], dtype=np.float32))

    assert decision.agent_id == 1
    assert decision.role == "Verifier"
    assert decision.route_id == "agent:1"
    assert decision.margins["agent"] == pytest.approx(2.6)
    assert decision.margins["role"] == pytest.approx(1.6)


def test_linear_head_rejects_shape_that_cannot_be_split() -> None:
    with pytest.raises(ValueError, match="output rows"):
        LinearHeadRouter(np.zeros((2, 3), dtype=np.float32), num_agents=3)


def test_linear_head_can_route_workers_without_roles() -> None:
    router = LinearHeadRouter(
        np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32),
        num_agents=2,
        role_names=(),
    )

    decision = router.route_vector(np.array([0.5, 4.0], dtype=np.float32))

    assert decision.agent_id == 0
    assert decision.role is None
