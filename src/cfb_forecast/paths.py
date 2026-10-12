"""Shared runtime locations independent of the caller's working directory."""
import os
from pathlib import Path


def project_root():
    return Path(os.environ.get("CFB_PROJECT_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def data_root():
    return resolve_path(os.environ.get("CFB_DATA_DIR", "data"))


def artifacts_root():
    return resolve_path(os.environ.get("CFB_ARTIFACTS_DIR", "artifacts"))


def model_root():
    return resolve_path(os.environ.get("CFB_MODEL_DIR", artifacts_root() / "training" / "model"))


def cli_path(value):
    path = Path(value)
    if not path.is_absolute() and path.parts[0] in {"data", "artifacts"}:
        root = data_root() if path.parts[0] == "data" else artifacts_root()
        return root.joinpath(*path.parts[1:])
    return resolve_path(path)
