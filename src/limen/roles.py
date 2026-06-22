from __future__ import annotations

from dataclasses import dataclass

ROLE_PROMPTS = {
    "Thinker": (
        "Analyze the current state and provide high-level guidance, plans, "
        "decompositions, or critiques. Do not present unchecked final answers."
    ),
    "Worker": (
        "Execute the next concrete step of the plan. Write code, calculations, "
        "or concrete answer content that advances the solution."
    ),
    "Verifier": (
        "Check the current solution for correctness, completeness, and responsiveness. "
        "Start your response with exactly ACCEPT or REVISE. After REVISE, include a "
        "concise diagnosis."
    ),
}

ROLE_ALIASES = {
    "0": "Worker",
    "1": "Thinker",
    "2": "Verifier",
    "solver": "Worker",
    "worker": "Worker",
    "w": "Worker",
    "thinker": "Thinker",
    "t": "Thinker",
    "verifier": "Verifier",
    "v": "Verifier",
}


class RoleInjector:
    @staticmethod
    def role_name(role: str | int | None) -> str:
        if isinstance(role, int):
            return {0: "Worker", 1: "Thinker", 2: "Verifier"}.get(role, "Worker")
        if role is None:
            return "Worker"
        return ROLE_ALIASES.get(str(role).strip().lower(), str(role))

    @classmethod
    def inject(cls, messages: list[dict[str, str]], role: str | int | None) -> list[dict[str, str]]:
        role_name = cls.role_name(role)
        prompt = ROLE_PROMPTS.get(role_name, "You are a helpful assistant.")
        return [{"role": "system", "content": prompt}, *messages]


@dataclass(frozen=True)
class VerifierResult:
    status: str
    raw: str
    diagnosis: str | None = None
    token: str | None = None

    @property
    def safe_status(self) -> str:
        return "accepted" if self.status == "accepted" else "revised"


def parse_verifier(
    text: str,
    accept_token: str = "ACCEPT",
    revise_token: str = "REVISE",
) -> VerifierResult:
    raw = str(text).strip()
    normalized = raw.upper()
    accept = accept_token.strip().upper()
    revise = revise_token.strip().upper()

    if _token_prefix(normalized, accept):
        return VerifierResult("accepted", raw, _diagnosis_after_token(raw, accept), accept)
    if _token_prefix(normalized, revise):
        return VerifierResult("revised", raw, _diagnosis_after_token(raw, revise), revise)
    return VerifierResult("unknown", raw, raw or None, None)


@dataclass(frozen=True)
class ThinkerResult:
    raw: str
    suggestion: str | None = None
    suggested_role: str | None = None


def parse_thinker(text: str) -> ThinkerResult:
    raw = str(text)
    suggestion = _extract_tag(raw, "suggestion")
    role = _normalize_suggested_role(_extract_tag(raw, "suggested_role"))
    if suggestion and role:
        return ThinkerResult(raw=raw, suggestion=suggestion, suggested_role=role)
    return ThinkerResult(raw=raw)


def _token_prefix(text: str, token: str) -> bool:
    if not token:
        return False
    return (
        text == token
        or text.startswith(token + ":")
        or text.startswith(token + " ")
        or text.startswith(token + "-")
        or text.startswith(token + "\n")
        or text.startswith(token + "\r\n")
    )


def _diagnosis_after_token(raw: str, token: str) -> str | None:
    rest = raw[len(token) :].lstrip()
    while rest.startswith((":", "-")):
        rest = rest[1:].lstrip()
    return rest or None


def _extract_tag(text: str, tag: str) -> str | None:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag not in text or close_tag not in text:
        return None
    value = text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
    return value or None


def _normalize_suggested_role(role: str | None) -> str | None:
    if role is None:
        return None
    return {"solver": "Worker", "worker": "Worker", "verifier": "Verifier"}.get(
        role.strip().lower()
    )
