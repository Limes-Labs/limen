from __future__ import annotations

import argparse
import json
from pathlib import Path

from limen.routers import MetadataRouter
from limen.routes import RouteLibrary


def main() -> int:
    parser = argparse.ArgumentParser(description="Route a prompt through a Limen route library.")
    parser.add_argument("prompt", nargs="?", default="Fix a failing pytest in this repository.")
    parser.add_argument("--library-dir", type=Path, default=Path("examples/routes"))
    args = parser.parse_args()

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
