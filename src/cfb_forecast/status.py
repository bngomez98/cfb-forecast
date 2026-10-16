"""One-screen view of what exists, what is missing and the next command to run.

Shared by ``cfb-forecast status`` and the dashboard's "Start here" page so both
interfaces always give the same answer.
"""
from __future__ import annotations

import json
from datetime import date

from .paths import artifacts_root, data_root, database_path, model_root


def current_season(today: date | None = None) -> int:
    """College seasons start in late summer and finish in January."""
    today = today or date.today()
    return today.year if today.month >= 6 else today.year - 1


def _read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def collect_status(season: int | None = None) -> dict:
    season = season or current_season()
    data, artifacts, model, database = data_root(), artifacts_root(), model_root(), database_path()
    manifest = _read(artifacts / "latest" / "run_manifest.json")
    data_files = sum(1 for p in data.rglob("*") if p.is_file()) if data.exists() else 0
    checks = [
        {"name": "Source data", "ok": data_files > 0, "detail": f"{data_files} files in {data}",
         "fix": f"cfb-forecast fetch --seasons {season}"},
        {"name": "Trained model", "ok": model.exists() and any(model.iterdir()), "detail": str(model),
         "fix": "cfb-forecast backtest"},
        {"name": "Latest forecast", "ok": bool(manifest), "fix": f"cfb-forecast update --season {season}",
         "detail": (f"{manifest.get('status', 'unknown')} · season {manifest.get('season', '?')} · "
                    f"{manifest.get('games', '?')} games · {manifest.get('as_of', '')}") if manifest else "none published yet"},
        {"name": "Database", "ok": database.exists(), "detail": str(database),
         "fix": f"cfb-forecast update --season {season}"},
    ]
    missing = next((c for c in checks if not c["ok"]), None)
    if missing is None:
        next_step = f"cfb-forecast update --season {season}   # refresh for the coming week"
    elif missing["name"] == "Source data" or missing["name"] == "Trained model":
        # update fetches and trains as needed, so it is the single answer for a new setup
        next_step = f"cfb-forecast update --season {season}   # fetches data, trains, forecasts, publishes"
    else:
        next_step = missing["fix"]
    return {"season": season, "checks": checks, "ready": missing is None, "next_step": next_step,
            "dashboard": "cfb-forecast dashboard", "api": "cfb-forecast serve"}


def format_status(report: dict) -> str:
    lines = [f"cfb-forecast status · season {report['season']}", ""]
    for c in report["checks"]:
        lines.append(f"  [{'ok' if c['ok'] else '--'}] {c['name']:<16} {c['detail']}")
    lines += ["", f"Next:       {report['next_step']}",
              f"Dashboard:  {report['dashboard']}", f"API:        {report['api']}",
              "All commands: cfb-forecast --help"]
    return "\n".join(lines)
