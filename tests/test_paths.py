from pathlib import Path

from cfb_forecast.paths import ProjectPaths


def test_project_paths_are_consistent(tmp_path: Path):
    paths = ProjectPaths(
        project=tmp_path,
        data=tmp_path / "data",
        artifacts=tmp_path / "artifacts",
        model=tmp_path / "artifacts" / "training" / "model",
        database=tmp_path / "artifacts" / "cfb_forecast.sqlite",
        market=tmp_path / "data" / "market_history.csv",
    ).ensure_runtime_dirs()
    assert paths.data.is_dir()
    assert paths.artifacts.is_dir()
    assert paths.database.parent.is_dir()
    assert Path(paths.as_dict()["model"]).as_posix().endswith("training/model")
