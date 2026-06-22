from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from limen.routers import LinearHeadRouter

LINEAR_HEAD_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LinearHeadArtifact:
    """Versioned artifact for a learned linear routing head."""

    weights: npt.NDArray[np.float32]
    num_agents: int
    role_names: tuple[str, ...] = ("Worker", "Thinker", "Verifier")
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = LINEAR_HEAD_ARTIFACT_SCHEMA_VERSION

    @classmethod
    def from_router(
        cls,
        router: LinearHeadRouter,
        metadata: dict[str, Any] | None = None,
    ) -> LinearHeadArtifact:
        return cls(
            weights=router.weights.copy(),
            num_agents=router.num_agents,
            role_names=router.role_names,
            metadata=dict(metadata or {}),
        )

    def to_router(self) -> LinearHeadRouter:
        return LinearHeadRouter(
            self.weights.copy(),
            num_agents=self.num_agents,
            role_names=self.role_names,
        )

    def save(self, path: Path | str) -> None:
        manifest = {
            "schema_version": self.schema_version,
            "num_agents": self.num_agents,
            "role_names": list(self.role_names),
            "metadata": self.metadata,
        }
        try:
            manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        except TypeError as exc:
            raise TypeError("metadata must be JSON serializable") from exc

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            weights=np.asarray(self.weights, dtype=np.float32),
            manifest=np.array(manifest_json),
        )

    @classmethod
    def load(cls, path: Path | str) -> LinearHeadArtifact:
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            weights = np.asarray(data["weights"], dtype=np.float32)
            manifest = _load_manifest(data["manifest"])

        schema_version = _require_int(manifest, "schema_version")
        if schema_version != LINEAR_HEAD_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported linear head artifact schema "
                f"{schema_version}; expected {LINEAR_HEAD_ARTIFACT_SCHEMA_VERSION}"
            )

        num_agents = _require_int(manifest, "num_agents")
        role_names = _require_string_tuple(manifest, "role_names")
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("linear head artifact metadata must be an object")

        artifact = cls(
            weights=weights,
            num_agents=num_agents,
            role_names=role_names,
            metadata=metadata,
            schema_version=schema_version,
        )
        artifact.to_router()
        return artifact


def _load_manifest(value: npt.NDArray[Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value.item()))
    except (AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("linear head artifact manifest must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("linear head artifact manifest must be an object")
    return parsed


def _require_int(manifest: dict[str, Any], key: str) -> int:
    value = manifest.get(key)
    if not isinstance(value, int):
        raise ValueError(f"linear head artifact {key} must be an integer")
    return value


def _require_string_tuple(manifest: dict[str, Any], key: str) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"linear head artifact {key} must be a list of strings")
    return tuple(value)
