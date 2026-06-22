from pathlib import Path

import numpy as np
import pytest

from limen.artifacts import LinearHeadArtifact
from limen.routers import LinearHeadRouter


def test_linear_head_artifact_round_trips_router_and_metadata(tmp_path: Path) -> None:
    weights = np.array(
        [
            [0.0, 2.0],
            [1.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    router = LinearHeadRouter(weights, num_agents=2, role_names=("Verifier",))
    artifact = LinearHeadArtifact.from_router(
        router,
        metadata={
            "source": "unit-test",
            "transcript_format": "raw_role_content_v1",
        },
    )

    path = tmp_path / "head.npz"
    artifact.save(path)
    loaded = LinearHeadArtifact.load(path)
    decision = loaded.to_router().route_vector(np.array([0.0, 1.0], dtype=np.float32))

    assert loaded.schema_version == 1
    assert loaded.num_agents == 2
    assert loaded.role_names == ("Verifier",)
    assert loaded.metadata == {
        "source": "unit-test",
        "transcript_format": "raw_role_content_v1",
    }
    np.testing.assert_array_equal(loaded.weights, weights)
    assert decision.agent_id == 0
    assert decision.role == "Verifier"


def test_linear_head_artifact_rejects_non_json_metadata(tmp_path: Path) -> None:
    router = LinearHeadRouter(np.ones((2, 2), dtype=np.float32), num_agents=2, role_names=())
    artifact = LinearHeadArtifact.from_router(router, metadata={"bad": {1, 2, 3}})

    with pytest.raises(TypeError, match="metadata must be JSON serializable"):
        artifact.save(tmp_path / "bad.npz")


def test_linear_head_artifact_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "unknown.npz"
    np.savez_compressed(
        path,
        weights=np.ones((2, 2), dtype=np.float32),
        manifest=np.array(
            '{"schema_version": 99, "num_agents": 2, "role_names": [], "metadata": {}}'
        ),
    )

    with pytest.raises(ValueError, match="unsupported linear head artifact schema"):
        LinearHeadArtifact.load(path)
