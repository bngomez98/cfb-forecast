#!/usr/bin/env python3
"""Merge sibling workspace files into the nested cfb_forecast project.

Default behavior preserves existing target files and reports conflicts.

Examples:
    python scripts/merge_workspace.py
    python scripts/merge_workspace.py --dry-run
    python scripts/merge_workspace.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(r"C:\Users\brian\cfb-forecast")
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}


def iter_files(source: Path):
    for path in sorted(source.rglob("*")):
        if path.is_file() and not any(part in EXCLUDED_DIRS for part in path.parts):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT, help="Workspace root")
    parser.add_argument("target", nargs="?", type=Path, default=None, help="Merge target; default: root/cfb_forecast")
    parser.add_argument("--overwrite", "-f", action="store_true", help="Replace conflicting files")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without copying")
    parser.add_argument("--yes", "-y", action="store_true", help="Run without confirmation")
    parser.add_argument("--report", type=Path, default=None, help="JSON report path")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    target = (args.target or root / "cfb_forecast").expanduser().resolve()
    report_path = (args.report or target / "artifacts" / "merge_report.json").resolve()

    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")
    if not target.is_dir():
        raise SystemExit(f"Target directory not found: {target}")

    if args.overwrite and not args.yes and not args.dry_run:
        answer = input("Overwrite conflicting files? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled")
            return 0

    sources = [path for path in sorted(root.iterdir()) if path.is_dir() and path.resolve() != target]
    copied: list[str] = []
    conflicts: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for source in sources:
        for source_file in iter_files(source):
            relative = source_file.relative_to(source)
            destination = target / relative
            if destination.exists() and not args.overwrite:
                conflicts.append({"source": str(source_file), "target": str(destination)})
                continue
            if not args.dry_run:
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, destination)
                except OSError as exc:
                    errors.append({"source": str(source_file), "target": str(destination), "error": str(exc)})
                    continue
            copied.append(str(destination))

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "target": str(target),
        "sources": [str(path) for path in sources],
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "copied_count": len(copied),
        "conflict_count": len(conflicts),
        "error_count": len(errors),
        "copied": copied,
        "conflicts": conflicts,
        "errors": errors,
    }

    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Copied: {len(copied)}")
    print(f"Conflicts preserved: {len(conflicts)}")
    print(f"Errors: {len(errors)}")
    if not args.dry_run:
        print(f"Report: {report_path}")
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
