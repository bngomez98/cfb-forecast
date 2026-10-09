#!/usr/bin/env python3
"""Inventory every nested file under the CFB workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path, nargs="?", default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = (args.output or workspace / "cfb_forecast" / "artifacts" / "workspace_inventory.json").resolve()
    files = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or any(part in EXCLUDED for part in path.parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({
            "path": path.relative_to(workspace).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "suffix": path.suffix.lower(),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"workspace": str(workspace), "files": files}, indent=2))
    print(f"Inventoried {len(files)} files")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
