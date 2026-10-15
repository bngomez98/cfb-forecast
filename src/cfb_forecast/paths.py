"""Shared runtime locations independent of the caller's working directory."""
from __future__ import annotations

import os
from dataclasses import dataclass
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


def database_path():
    return resolve_path(os.environ.get("CFB_DATABASE", artifacts_root() / "cfb_forecast.sqlite"))


def market_path():
    return resolve_path(os.environ.get("CFB_MARKET_FILE", data_root() / "market_history.csv"))


@dataclass(frozen=True)
class ProjectPaths:
    project: Path
    data: Path
    artifacts: Path
    model: Path
    database: Path
    market: Path

    def ensure_runtime_dirs(self) -> "ProjectPaths":
        self.data.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "project": str(self.project),
            "data": str(self.data),
            "artifacts": str(self.artifacts),
            "model": str(self.model),
            "database": str(self.database),
            "market": str(self.market),
        }


def paths() -> ProjectPaths:
    return ProjectPaths(
        project=project_root(),
        data=data_root(),
        artifacts=artifacts_root(),
        model=model_root(),
        database=database_path(),
        market=market_path(),
    )


def cli_path(value):
    path = Path(value)
    if not path.is_absolute() and path.parts[0] in {"data", "artifacts"}:
        root = data_root() if path.parts[0] == "data" else artifacts_root()
        return root.joinpath(*path.parts[1:])
    return resolve_path(path)
