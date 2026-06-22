from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {"api_key", "authorization", "password", "secret", "token"}


def stable_hash(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def redact(value: Any, secret_values: list[str] | None = None) -> Any:
    secrets = [item for item in (secret_values or []) if item]
    redacted = _redact_keys(value)
    return _redact_values(redacted, secrets)


@dataclass(frozen=True)
class TraceEvent:
    payload: dict[str, Any]

    @classmethod
    def new(cls, event: str, run_id: str, fields: dict[str, Any] | None = None) -> TraceEvent:
        payload = {
            "schema_version": 1,
            "event": event,
            "run_id": run_id,
            "timestamp_ms": int(time.time() * 1000),
        }
        payload.update(fields or {})
        return cls(redact(payload))

    def to_json(self) -> str:
        return json.dumps(self.payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class JSONLTraceSink:
    path: Path | str

    def write(self, event: TraceEvent) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(event.to_json() + "\n")


def _redact_keys(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in SENSITIVE_KEYS:
                out[key] = "<redacted>"
            else:
                out[key] = _redact_keys(item)
        return out
    if isinstance(value, list):
        return [_redact_keys(item) for item in value]
    if isinstance(value, str) and value.lower().startswith("bearer "):
        return "<redacted>"
    return value


def _redact_values(value: Any, secrets: list[str]) -> Any:
    if not secrets:
        return value
    if isinstance(value, dict):
        return {key: _redact_values(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_values(item, secrets) for item in value]
    if isinstance(value, str):
        out = value
        for secret in secrets:
            out = out.replace(secret, "<redacted>")
        return out
    return value
