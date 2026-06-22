from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from limen.artifacts import LinearHeadArtifact
from limen.eval import evaluate_routes, load_route_eval_cases
from limen.routers import MetadataRouter
from limen.routes import RouteLibrary


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "eval":
        return _eval_command(args[1:])
    if args and args[0] == "artifact":
        return _artifact_command(args[1:])
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
            result.to_jsonable(),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _artifact_command(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Limen artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="Inspect a linear-head artifact.")
    inspect.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    if args.command == "inspect":
        artifact = LinearHeadArtifact.load(args.path)
        print(
            json.dumps(
                {
                    "type": "linear_head",
                    "schema_version": artifact.schema_version,
                    "weights_shape": list(artifact.weights.shape),
                    "num_agents": artifact.num_agents,
                    "role_names": list(artifact.role_names),
                    "metadata": artifact.metadata,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    parser.error(f"unknown artifact command: {args.command}")
    return 2
