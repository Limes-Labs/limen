from limen.roles import RoleInjector, parse_thinker, parse_verifier


def test_verifier_accepts_only_explicit_accept_token() -> None:
    assert parse_verifier("ACCEPT: complete").status == "accepted"
    assert parse_verifier("REVISE - missing edge case").status == "revised"

    unknown = parse_verifier("Looks good to me")
    assert unknown.status == "unknown"
    assert unknown.safe_status == "revised"


def test_thinker_suggestions_are_explicit_and_role_limited() -> None:
    parsed = parse_thinker(
        "<suggestion>Run the focused tests first.</suggestion>"
        "<suggested_role>solver</suggested_role>"
    )

    assert parsed.suggestion == "Run the focused tests first."
    assert parsed.suggested_role == "Worker"

    rejected = parse_thinker(
        "<suggestion>Do it.</suggestion><suggested_role>admin</suggested_role>"
    )
    assert rejected.suggested_role is None


def test_role_injector_uses_canonical_role_prompt() -> None:
    messages = [{"role": "user", "content": "Solve it."}]

    injected = RoleInjector.inject(messages, "verifier")

    assert injected[0]["role"] == "system"
    assert "ACCEPT or REVISE" in injected[0]["content"]
    assert injected[1:] == messages
