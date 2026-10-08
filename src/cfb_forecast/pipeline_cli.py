"""Installed wrapper for the human-friendly pipeline commands."""
from pathlib import Path
import runpy


def main():
    root = Path.cwd()
    script = root / "scripts" / "cfb.py"
    if not script.exists():
        raise FileNotFoundError(f"Run cfb-pipeline from the project root: {root}")
    runpy.run_path(str(script), run_name="__main__")
