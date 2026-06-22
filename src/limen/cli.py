from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from limen.eval import evaluate_routes, load_route_eval_cases
from limen.routers import MetadataRouter
from limen.routes import RouteLibrary


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "eval":
        return _eval_command(args[1:])
    return _route_command(args)


def _route_command(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Route a prompt through a Limen route library.")
    parser.add_argument("prompt", nargs="?", default="Fix a failing pytest in this repository.")
    parser.add_argument("--library-dir", type=Path, default=Path("examples/routes"))
    args = parser.parse_args(argv)

    library = RouteLibrary.load(args.library_dir)
    decision = MetadataRouter(library).route_text(args.prompt)
    print(
        json.dumps(
            {
                "route_id": decision.route_id,
                "target": decision.target,
                "reason": decision.reason,
                "diagnostics": decision.diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _eval_command(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Limen route fixtures.")
    parser.add_argument("--library-dir", type=Path, default=Path("examples/routes"))
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args(argv)

    library = RouteLibrary.load(args.library_dir)
    router = MetadataRouter(library)
    result = evaluate_routes(router, load_route_eval_cases(args.cases))
    print(
        json.dumps(
            {
                "total": result.total,
                "correct": result.correct,
                "accuracy": result.accuracy,
                "misses": result.misses,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0
