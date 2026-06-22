import json

from limen.trace import TraceEvent, redact, stable_hash


def test_trace_event_redacts_sensitive_keys_and_hashes_content() -> None:
    event = TraceEvent.new(
        "provider_called",
        "run-1",
        {
            "api_key": "sk-secret",
            "authorization": "Bearer token",
            "prompt": "hello",
            "content_hash": stable_hash("hello"),
        },
    )

    encoded = event.to_json()
    decoded = json.loads(encoded)

    assert decoded["api_key"] == "<redacted>"
    assert decoded["authorization"] == "<redacted>"
    assert decoded["content_hash"] == stable_hash("hello")
    assert decoded["schema_version"] == 1


def test_redact_replaces_materialized_secret_values() -> None:
    payload = {"nested": ["prefix secret-token suffix"]}

    assert redact(payload, secret_values=["secret-token"]) == {
        "nested": ["prefix <redacted> suffix"]
    }
