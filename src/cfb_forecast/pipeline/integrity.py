"""Verify saved predictions and their archived pregame evidence."""
from pathlib import Path
from ..ingest.data import sha256


def verify_run_integrity(run_dir, manifest):
    root = Path(run_dir)
    predictions = root / "forecasts.csv"
    expected = manifest.get("prediction_sha256")
    if expected and (not predictions.is_file() or sha256(predictions) != expected):
        raise ValueError(f"Saved predictions changed: {predictions}")
    inputs = (root / "inputs").resolve()
    for name, expected in manifest.get("input_hashes", {}).items():
        path = (inputs / name).resolve()
        if not path.is_relative_to(inputs):
            raise ValueError(f"Saved input path escapes run directory: {name}")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Saved pregame inputs changed: {path}")
