"""Runtime checks for mounted data and artifact paths."""
from pathlib import Path


def run_preflight(data_root="data", artifacts_root="artifacts", season=None):
    data = Path(data_root)
    artifacts = Path(artifacts_root)
    if not data.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data.resolve()}")
    if not artifacts.is_dir():
        artifacts.mkdir(parents=True, exist_ok=True)
    if season is not None:
        schedule = data / f"schedule_{int(season)}.csv"
        if not schedule.exists():
            raise FileNotFoundError(f"Missing required schedule: {schedule.resolve()}")
    print(f"data={data.resolve()}")
    print(f"artifacts={artifacts.resolve()}")
    if season is not None:
        print(f"schedule={data / f'schedule_{int(season)}.csv'}")
    return True
