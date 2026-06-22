from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RouteSpec:
    """Metadata for one route, model, provider slot, or LoRA adapter."""

    id: str
    target: str
    description: str = ""
    priority: int = 0
    level: str = ""
    routing_rules: tuple[str, ...] = ()
    strong_signals: tuple[str, ...] = ()
    positive_signals: tuple[str, ...] = ()
    negative_signals: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    source_path: str = ""
    library_path: str = ""

    @property
    def is_entry(self) -> bool:
        return self.id.upper() == "L0" or self.level.upper() == "L0"


@dataclass(frozen=True)
class RouteLibrary:
    """A validated collection of route specs loaded from markdown files."""

    routes: dict[str, RouteSpec] = field(default_factory=dict)
    entry_route_id: str = "L0"

    @classmethod
    def load(cls, root: str | Path, entry_route_id: str = "L0") -> RouteLibrary:
        path = Path(root)
        if not path.exists():
            raise FileNotFoundError(f"Missing route library directory: {path}")

        routes = {
            route.id: route
            for route in (parse_route_markdown(item) for item in sorted(path.glob("*.md")))
        }
        if not routes:
            raise FileNotFoundError(f"No .md route files found in: {path}")
        if entry_route_id not in routes:
            raise FileNotFoundError(
                f"Route library must include entry route {entry_route_id}.md: {path}"
            )
        return cls(routes=routes, entry_route_id=entry_route_id)

    @property
    def entry(self) -> RouteSpec:
        return self.routes[self.entry_route_id]

    def get(self, route_id: str) -> RouteSpec:
        return self.routes[route_id]

    def __contains__(self, route_id: object) -> bool:
        return route_id in self.routes


def parse_route_markdown(path: Path) -> RouteSpec:
    text = textwrap.dedent(path.read_text(encoding="utf-8")).strip() + "\n"
    meta, body = _split_front_matter(text)
    sections = _parse_sections(body)
    route_id = meta.get("id") or path.stem
    target = meta.get("target") or meta.get("adapter_name") or meta.get("model") or route_id
    datasets = _section_items(sections, "datasets")
    dataset = meta.get("dataset", "")
    if dataset and dataset not in datasets:
        datasets.insert(0, dataset)

    return RouteSpec(
        id=route_id,
        target=target,
        description=_section_text(sections, "description"),
        priority=int(meta.get("priority", "0") or 0),
        level=meta.get("level", ""),
        routing_rules=tuple(_bullets(sections, "routing_rules")),
        strong_signals=tuple(_comma_list(sections, "strong_signals")),
        positive_signals=tuple(_comma_list(sections, "positive_signals")),
        negative_signals=tuple(_comma_list(sections, "negative_signals")),
        examples=tuple(_section_items(sections, "examples")),
        datasets=tuple(datasets),
        source_path=meta.get("source_path", ""),
        library_path=str(path),
    )


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    try:
        _, raw_meta, body = text.split("---", 2)
    except ValueError:
        return {}, text

    meta: dict[str, str] = {}
    for raw_line in raw_meta.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def _parse_sections(body: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        header = line.strip()
        if header.startswith("## "):
            current = header[3:].strip().lower().replace(" ", "_")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line.strip())
    return sections


def _section_text(sections: dict[str, list[str]], name: str) -> str:
    return "\n".join(sections.get(name, [])).strip()


def _bullets(sections: dict[str, list[str]], name: str) -> list[str]:
    out: list[str] = []
    for line in sections.get(name, []):
        stripped = line.strip()
        if stripped.startswith("- "):
            out.append(stripped[2:].strip())
    return out


def _comma_list(sections: dict[str, list[str]], name: str) -> list[str]:
    raw = _section_text(sections, name)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _section_items(sections: dict[str, list[str]], name: str) -> list[str]:
    out: list[str] = []
    for line in sections.get(name, []):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        out.append(stripped)
    return out


def library_to_jsonable(library: RouteLibrary) -> dict[str, Any]:
    return {
        "entry_route_id": library.entry_route_id,
        "routes": {route_id: spec.__dict__ for route_id, spec in library.routes.items()},
    }
