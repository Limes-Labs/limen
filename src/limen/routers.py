from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from limen.routes import RouteLibrary, RouteSpec
from limen.trace import stable_hash

RAW_TRANSCRIPT_FORMAT = "raw_role_content_v1"


@dataclass(frozen=True)
class RouteDecision:
    route_id: str
    target: str | None = None
    agent_id: int | None = None
    role: str | None = None
    reason: str = "selected"
    raw_route: str | None = None
    diagnostics: dict[str, Any] | None = None
    logits: list[float] | None = None
    margins: dict[str, float] | None = None


class MetadataRouter:
    """Deterministic route-library router with transparent guardrails."""

    def __init__(self, library: RouteLibrary) -> None:
        self.library = library

    def route(self, messages: list[dict[str, str]]) -> RouteDecision:
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_text = message.get("content", "")
                break
        return self.route_text(user_text)

    def route_text(self, user_text: str) -> RouteDecision:
        candidates: list[dict[str, Any]] = []
        signal_diagnostics: dict[str, Any] = {}

        for route_id, route in self.library.routes.items():
            if route_id == self.library.entry_route_id:
                continue
            negative_hits = self._signal_hits(user_text, route.negative_signals)
            strong_hits = self._signal_hits(user_text, route.strong_signals)
            positive_hits = self._signal_hits(user_text, route.positive_signals)
            signal_diagnostics[route_id] = {
                "negative_hits": negative_hits[:8],
                "strong_hits": strong_hits[:8],
                "positive_hits": positive_hits[:8],
            }
            if not strong_hits and not positive_hits:
                continue

            strong_score = self._weighted_signal_score(strong_hits)
            positive_score = self._weighted_signal_score(positive_hits)
            negative_score = self._weighted_signal_score(negative_hits)
            score = route.priority + strong_score * 100 + positive_score * 10 - negative_score * 250
            if score <= 0:
                continue
            candidates.append(
                {
                    "route_id": route_id,
                    "score": score,
                    "strong_count": len(strong_hits),
                    "strong_score": strong_score,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "strong_hits": strong_hits[:8],
                    "positive_hits": positive_hits[:8],
                    "negative_hits": negative_hits[:8],
                }
            )

        if not candidates:
            return self._entry_decision("default_entry", {"signals": signal_diagnostics})

        candidates.sort(key=lambda item: (item["score"], item["strong_count"]), reverse=True)
        top = candidates[0]

        if len(candidates) == 1 or top["score"] > candidates[1]["score"]:
            if self._is_general_entry_request(user_text) and top["strong_score"] <= 1:
                return self._entry_decision(
                    "general_entry_guard",
                    {"candidates": candidates, "signals": signal_diagnostics},
                )
            return self._decision(
                top["route_id"],
                "specialist_signal",
                {"selected": top, "candidates": candidates, "signals": signal_diagnostics},
            )

        return self._entry_decision(
            "ambiguous_specialist_signal_default_entry",
            {"candidates": candidates, "signals": signal_diagnostics},
        )

    def parse_router_output(self, text: str | None) -> str | None:
        if text is None:
            return None
        lower = text.lower()
        aliases = {self.library.entry_route_id.lower(): self.library.entry_route_id}
        for route_id, route in self.library.routes.items():
            aliases[route_id.lower()] = route_id
            if route.target:
                aliases[route.target.lower()] = route_id

        match = re.search(r"model_id\s*=\s*([A-Za-z0-9_.:/-]+)", lower)
        if match:
            return aliases.get(self._clean_alias(match.group(1)))

        stripped = self._clean_alias(lower)
        if stripped in aliases:
            return aliases[stripped]

        mentions = []
        for alias, route_id in aliases.items():
            pattern = rf"(?<![A-Za-z0-9_.:/-]){re.escape(alias)}(?![A-Za-z0-9_.:/-])"
            if re.search(pattern, lower):
                mentions.append(route_id)
        unique_mentions = list(dict.fromkeys(mentions))
        if len(unique_mentions) == 1:
            return unique_mentions[0]
        return None

    def apply_guardrail(self, raw_route_text: str | None, user_text: str) -> RouteDecision:
        raw_route = self.parse_router_output(raw_route_text)
        library_decision = self.route_text(user_text)

        if library_decision.route_id != self.library.entry_route_id:
            return RouteDecision(
                route_id=library_decision.route_id,
                target=library_decision.target,
                reason=library_decision.reason,
                raw_route=raw_route,
                diagnostics=library_decision.diagnostics,
            )

        if raw_route is not None and raw_route in self.library:
            return self._decision(raw_route, "model_route", raw_route=raw_route)

        return RouteDecision(
            route_id=library_decision.route_id,
            target=library_decision.target,
            reason=library_decision.reason,
            raw_route=raw_route,
            diagnostics=library_decision.diagnostics,
        )

    def _decision(
        self,
        route_id: str,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
        raw_route: str | None = None,
    ) -> RouteDecision:
        route = self.library.get(route_id)
        return RouteDecision(
            route_id=route.id,
            target=route.target,
            reason=reason,
            raw_route=raw_route,
            diagnostics=diagnostics,
        )

    def _entry_decision(
        self,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> RouteDecision:
        return self._decision(self.library.entry_route_id, reason, diagnostics)

    @staticmethod
    def _clean_alias(value: str) -> str:
        return value.strip().strip("`'\".,;:()[]{}<>").lower()

    @classmethod
    def _signal_hits(cls, user_text: str, signals: Iterable[str]) -> list[str]:
        lower = user_text.lower()
        return [signal for signal in signals if cls._signal_matches(lower, signal)]

    @staticmethod
    def _signal_matches(text: str, signal: str) -> bool:
        signal = signal.strip().lower()
        if not signal:
            return False
        if " " in signal:
            return signal in text or all(
                MetadataRouter._signal_part_present(text, part)
                for part in signal.split()
                if part.strip()
            )
        if re.fullmatch(r"[a-z0-9_+-]+", signal):
            pattern = rf"(?<![a-z0-9_+-]){re.escape(signal)}(?![a-z0-9_+-])"
            return re.search(pattern, text) is not None
        return signal in text

    @staticmethod
    def _signal_part_present(text: str, part: str) -> bool:
        part = part.strip().lower()
        if not part:
            return False
        return MetadataRouter._signal_matches(text, part) or part in text

    @classmethod
    def _weighted_signal_score(cls, hits: Iterable[str]) -> int:
        return sum(cls._signal_weight(hit) for hit in hits)

    @staticmethod
    def _signal_weight(signal: str) -> int:
        signal = signal.strip()
        if not signal:
            return 0
        if len(signal) >= 40:
            return 6
        if len(signal) >= 20:
            return 4
        if any(ch in signal for ch in ("/", "_", "-", ":", "`", "*")):
            return 4
        if " " in signal:
            return 3
        return 1

    @staticmethod
    def _is_general_entry_request(user_text: str) -> bool:
        lower = user_text.strip().lower()
        return lower.startswith(
            (
                "translate ",
                "rewrite ",
                "proofread ",
                "explain ",
                "summarize ",
                "compare ",
                "what is ",
                "why does ",
                "give me an overview",
            )
        )


class LinearHeadRouter:
    """Bias-free linear router over an externally supplied hidden vector."""

    def __init__(
        self,
        weights: npt.ArrayLike,
        num_agents: int,
        role_names: Sequence[str] = ("Worker", "Thinker", "Verifier"),
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float32)
        self.num_agents = num_agents
        self.role_names = tuple(role_names)
        if self.weights.ndim != 2:
            raise ValueError("weights must be a rank-2 array")
        expected_rows = self.num_agents + len(self.role_names)
        if self.weights.shape[0] != expected_rows:
            raise ValueError(
                f"output rows must equal num_agents + num_roles ({expected_rows}), "
                f"got {self.weights.shape[0]}"
            )
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")

    def route_vector(self, vector: npt.ArrayLike) -> RouteDecision:
        hidden = np.asarray(vector, dtype=np.float32)
        if hidden.ndim != 1 or hidden.shape[0] != self.weights.shape[1]:
            raise ValueError(f"vector shape must be ({self.weights.shape[1]},), got {hidden.shape}")

        logits = self.weights @ hidden
        agent_logits = logits[: self.num_agents]
        agent_id = int(np.argmax(agent_logits))
        role: str | None = None
        margins = {"agent": _top_margin(agent_logits)}

        if self.role_names:
            role_logits = logits[self.num_agents :]
            role_id = int(np.argmax(role_logits))
            role = self.role_names[role_id]
            margins["role"] = _top_margin(role_logits)

        return RouteDecision(
            route_id=f"agent:{agent_id}",
            target=f"agent:{agent_id}",
            agent_id=agent_id,
            role=role,
            reason="linear_head",
            logits=[float(item) for item in logits.tolist()],
            margins=margins,
        )


def format_raw_transcript(messages: Sequence[dict[str, str]]) -> str:
    """Render chat messages as the raw role/content transcript used by small routers."""

    lines: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"message {index} must include string role and content")
        lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


class TranscriptVectorRouter:
    """Route raw chat transcripts through an injected hidden-vector extractor."""

    def __init__(
        self,
        head: LinearHeadRouter,
        extractor: Callable[[str], npt.ArrayLike],
    ) -> None:
        self.head = head
        self.extractor = extractor

    def route(self, messages: list[dict[str, str]]) -> RouteDecision:
        transcript = format_raw_transcript(messages)
        vector = self.extractor(transcript)
        decision = self.head.route_vector(vector)
        diagnostics = dict(decision.diagnostics or {})
        diagnostics.update(
            {
                "message_count": len(messages),
                "transcript_format": RAW_TRANSCRIPT_FORMAT,
                "transcript_hash": stable_hash(transcript),
            }
        )
        return replace(
            decision,
            reason="transcript_vector_head",
            diagnostics=diagnostics,
        )


class ScriptedRouter:
    """Small deterministic router for tests, examples, and dry-run demos."""

    def __init__(self, decisions: Sequence[RouteDecision | tuple[int, str]]) -> None:
        if not decisions:
            raise ValueError("ScriptedRouter requires at least one decision")
        self._decisions = [self._coerce(item) for item in decisions]
        self._index = 0

    def route(self, _messages: list[dict[str, str]]) -> RouteDecision:
        decision = self._decisions[min(self._index, len(self._decisions) - 1)]
        self._index += 1
        return decision

    @staticmethod
    def _coerce(item: RouteDecision | tuple[int, str]) -> RouteDecision:
        if isinstance(item, RouteDecision):
            return item
        agent_id, role = item
        return RouteDecision(
            route_id=f"agent:{agent_id}",
            target=f"agent:{agent_id}",
            agent_id=agent_id,
            role=role,
            reason="scripted",
        )


def _top_margin(values: np.ndarray[Any, np.dtype[np.float32]]) -> float:
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return 0.0
    top = np.sort(values)[-2:]
    return float(top[-1] - top[-2])


def render_router_instruction(library: RouteLibrary) -> str:
    """Render a compact prompt for an L0 language-model router."""

    route_lines = "\n".join(
        f"- {route.id}: {route.description}" for route in _sorted_specs(library)
    )
    rules: list[str] = []
    examples: list[str] = []
    for route in _sorted_specs(library):
        rules.extend(f"{route.id}: {rule}" for rule in route.routing_rules)
        examples.extend(route.examples)
    rule_lines = "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))
    example_lines = "\n".join(examples)
    choices = "|".join(library.routes)
    return (
        "Router instruction:\n"
        "Choose exactly one model_id for the next phase. "
        "Classify by user goal, not keyword overlap. "
        "Use the entry route for ordinary or ambiguous requests.\n\n"
        f"Available model ids:\n{route_lines}\n\n"
        f"Routing rules:\n{rule_lines}\n\n"
        f"Examples:\n{example_lines}\n\n"
        f"Return exactly one line: model_id=<{choices}>\n\nmodel_id="
    )


def _sorted_specs(library: RouteLibrary) -> list[RouteSpec]:
    return sorted(
        library.routes.values(),
        key=lambda route: (route.id != library.entry_route_id, -route.priority, route.id),
    )
