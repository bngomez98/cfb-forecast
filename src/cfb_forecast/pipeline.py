"""End-to-end pipeline orchestration.

One entry point replaces the earlier shell and Docker wrapper scripts. Every
stage runs in-process, so the same command works on a laptop, inside the
container image and in GitHub Actions:

    cfb-forecast pipeline weekly --season 2026
    cfb-pipeline weekly --season 2026            # installed alias

Stages are either required (a failure stops the run with a nonzero exit code)
or optional (a failure is recorded in the run receipt and the run continues).
Every run writes ``<artifacts>/pipeline/latest.json`` plus a timestamped copy,
and a log file in ``CFB_LOG_DIR`` (default ``<artifacts>/logs``).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import __version__
from .logs import add_file, close_file, configure, get

log = get("pipeline")

ACTIONS = ("weekly", "results", "pregame", "backtest", "data", "sources")


@dataclass
class Context:
    data: Path
    artifacts: Path
    model: Path
    season: int
    draws: int = 50000
    lead_hours: float = 48.0
    week: int | None = None
    refresh_data: bool = False
    sources: str = "all"
    log_dir: Path | None = None
    outputs: dict = field(default_factory=dict)

    @classmethod
    def from_args(cls, args):
        artifacts = Path(args.artifacts)
        return cls(
            data=Path(args.data),
            artifacts=artifacts,
            model=Path(args.model) if args.model else artifacts / "backtest" / "model",
            season=int(args.season),
            draws=int(args.draws),
            lead_hours=float(args.lead_hours),
            week=args.week,
            refresh_data=bool(args.refresh_data),
            sources=args.sources,
            log_dir=Path(args.log_dir) if args.log_dir else artifacts / "logs",
        )


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[Context], dict | None]
    required: bool = True
    description: str = ""


def _season_files(ctx):
    return [ctx.data / f"schedule_{ctx.season}.csv", ctx.data / f"adv_team_{ctx.season}.parquet"]


def stage_data(ctx):
    """Download season inputs when missing, or refresh them when requested."""
    from .data import fetch

    files = _season_files(ctx)
    missing = [p for p in files if not p.exists()]
    if not missing and not ctx.refresh_data:
        return {"action": "cached", "files": [str(p) for p in files]}
    try:
        fetch(ctx.data, [ctx.season], refresh=ctx.refresh_data)
    except Exception as exc:
        if missing:
            raise
        # Self-healing: a transient source outage falls back to the cached snapshot.
        return {"action": "refresh-failed-using-cache", "error": f"{type(exc).__name__}: {exc}"}
    return {"action": "refreshed" if ctx.refresh_data else "downloaded", "files": [str(p) for p in files]}


def stage_preflight(ctx):
    from .preflight import run_preflight

    run_preflight(ctx.data, ctx.artifacts, ctx.season)
    model_file = ctx.model / "model.json"
    if not model_file.exists():
        raise FileNotFoundError(f"Production model not found: {model_file}. Run `cfb-forecast backtest` first.")
    return {"model": str(model_file)}


def stage_results(ctx):
    from .results import refresh_results

    out = ctx.artifacts / "results"
    results, manifest = refresh_results(ctx.data, out, ctx.season, ctx.week)
    ctx.outputs["results"] = str(out)
    return {"rows": int(len(results)), "final": int(results.is_final.sum()) if len(results) else 0, "output": str(out)}


def _sources_stage(tag):
    def stage(ctx):
        from .sources import collect

        if ctx.sources.strip().lower() in ("", "none", "off", "0"):
            return {"action": "disabled"}
        names = None if ctx.sources.strip().lower() == "all" else [n.strip() for n in ctx.sources.split(",") if n.strip()]
        report = collect(ctx.data, ctx.season, ctx.artifacts, names=names, tag=tag, week=ctx.week)
        statuses = {k: v["status"] for k, v in report["sources"].items()}
        ok = [k for k, v in statuses.items() if v == "ok"]
        if not ok:
            raise RuntimeError(f"no external source produced records: {statuses}")
        ctx.outputs["sources"] = str(ctx.data / "sources")
        return {"ok": ok, "statuses": statuses, **{k: v for k, v in report.items() if k not in ("sources",)}}
    return stage


def _market_path(ctx):
    path = ctx.data / "market_history.csv"
    return path if path.exists() and path.stat().st_size else None


def stage_market(ctx):
    """Point-in-time SportsDataverse spread/total snapshot into the market history."""
    from .market import capture_sportsdataverse_snapshot

    store = ctx.data / "market_history.csv"
    rows = capture_sportsdataverse_snapshot(ctx.season, store)
    return {"store": str(store), "rows": int(len(rows))}


def stage_forecast(ctx):
    from .weekly import run_weekly_forecast

    result, manifest = run_weekly_forecast(ctx.data, ctx.model, ctx.artifacts, ctx.season, ctx.week,
                                           market_path=_market_path(ctx), draws=ctx.draws)
    ctx.outputs["forecast"] = manifest["output"]
    ctx.outputs["latest_weekly"] = str(ctx.artifacts / "latest-weekly")
    return manifest


def stage_backtest(ctx):
    from .season_backtest import run_season_backtest

    _, summary = run_season_backtest(ctx.data, ctx.model, ctx.artifacts / "backtest", ctx.season, ctx.draws, ctx.lead_hours)
    ctx.outputs["season_backtest"] = str(ctx.artifacts / "backtest" / f"season-{ctx.season}")
    return summary


def stage_verify(ctx):
    expected = {
        "forecast": ctx.artifacts / "latest-weekly" / "forecasts.csv",
        "manifest": ctx.artifacts / "latest-weekly" / "weekly_manifest.json",
    }
    missing = {k: str(v) for k, v in expected.items() if not v.exists()}
    if missing:
        raise FileNotFoundError(f"Pipeline finished without expected outputs: {missing}")
    return {k: str(v) for k, v in expected.items()}


def stage_any_results(ctx):
    got = [k for k in ("results", "sources") if k in ctx.outputs]
    if not got:
        raise RuntimeError("Neither ESPN nor any external result source returned data")
    return {"from": got}


DATA = Stage("data", stage_data, True, "Download or refresh season schedule and advanced-team inputs")
PREFLIGHT = Stage("preflight", stage_preflight, True, "Validate data, artifact and model paths")
RESULTS = Stage("results", stage_results, False, "Final scores of completed games from the ESPN scoreboard")
SOURCES = Stage("sources", _sources_stage(None), False, "Collect external results, picks, lines and stats")
SOURCE_RESULTS = Stage("sources", _sources_stage("results"), False, "Collect external result sources")
SOURCE_PREGAME = Stage("sources", _sources_stage(("picks", "lines")), False, "Collect external picks and lines before kickoff")
MARKET = Stage("market", stage_market, False, "Snapshot the SportsDataverse spread/total into market history")

FORECAST = Stage("forecast", stage_forecast, True, "Forecast the next upcoming FBS week")
BACKTEST = Stage("backtest", stage_backtest, False, "Point-in-time backtest: re-forecast every completed week from its pre-kickoff state, compare with final scores")
VERIFY = Stage("verify", stage_verify, True, "Check that the forecast outputs exist")

PLANS = {
    "weekly": (DATA, PREFLIGHT, MARKET, RESULTS, SOURCES, FORECAST, BACKTEST, VERIFY),
    "pregame": (DATA, MARKET, SOURCE_PREGAME),
    "results": (DATA, RESULTS, SOURCE_RESULTS, Stage("check", stage_any_results, True, "Require results from at least one provider")),
    "sources": (DATA, Stage("sources", _sources_stage(None), True, SOURCES.description)),
    "backtest": (DATA, PREFLIGHT, Stage("backtest", stage_backtest, True, BACKTEST.description)),
    "data": (DATA,),
}


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def run(action, ctx):
    configure()
    log_file = add_file(ctx.log_dir, action) if ctx.log_dir else None
    started = _now()
    receipt = {
        "action": action,
        "version": __version__,
        "python": platform.python_version(),
        "started_at": started.isoformat(),
        "season": ctx.season,
        "week": ctx.week,
        "draws": ctx.draws,
        "lead_hours": ctx.lead_hours,
        "refresh_data": ctx.refresh_data,
        "data": str(ctx.data),
        "artifacts": str(ctx.artifacts),
        "model": str(ctx.model),
        "sources": ctx.sources,
        "log_file": str(log_file) if log_file else None,
        "stages": [],
    }
    status = "success"
    for stage in PLANS[action]:
        log.info("==> [%s] %s", stage.name, stage.description)
        t0 = time.perf_counter()
        entry = {"stage": stage.name, "required": stage.required}
        try:
            entry["detail"] = stage.run(ctx)
            entry["status"] = "ok"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if stage.required:
                log.error("[%s] %s\n%s", stage.name, entry["error"], traceback.format_exc().rstrip())
            else:
                log.warning("[%s] optional stage failed: %s", stage.name, entry["error"])
            if stage.required:
                status = "failed"
            elif status == "success":
                status = "degraded"
        entry["seconds"] = round(time.perf_counter() - t0, 2)
        receipt["stages"].append(entry)
        log.info("<== [%s] %s in %ss", stage.name, entry["status"], entry["seconds"])
        if status == "failed":
            break
    receipt["status"] = status
    receipt["finished_at"] = _now().isoformat()
    receipt["outputs"] = ctx.outputs
    path = write_receipt(ctx.artifacts, receipt, started)
    log.info("Pipeline %s: %s. Receipt: %s%s", action, status, path, f" Log: {log_file}" if log_file else "")
    close_file()
    return receipt


def write_receipt(artifacts, receipt, started):
    folder = Path(artifacts) / "pipeline"
    folder.mkdir(parents=True, exist_ok=True)
    body = json.dumps(receipt, indent=2, default=str)
    path = folder / f"{receipt['action']}_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(body)
    (folder / "latest.json").write_text(body)
    return path


def add_arguments(p):
    env = os.environ.get
    p.add_argument("action", choices=ACTIONS, help=(
        "weekly: full run; results: final scores of completed games; pregame: capture external predictions and lines; "
        "backtest: point-in-time season backtest; data: fetch inputs; sources: all external sites"))
    p.add_argument("--season", type=int, default=int(env("CFB_FORECAST_SEASON", "2026")))
    p.add_argument("--week", type=int, default=None, help="Forecast or refresh one specific week")
    p.add_argument("--draws", type=int, default=int(env("CFB_FORECAST_DRAWS", "50000")))
    p.add_argument("--lead-hours", type=float, default=float(env("CFB_BACKTEST_LEAD_HOURS", "48")))
    p.add_argument("--data", default=env("CFB_DATA_DIR", "data"))
    p.add_argument("--artifacts", default=env("CFB_ARTIFACTS_DIR", "artifacts"))
    p.add_argument("--model", default=env("CFB_MODEL_DIR"), help="Defaults to <artifacts>/backtest/model")
    p.add_argument("--sources", default=env("CFB_SOURCES", "all"),
                   help="External sources to collect: 'all', 'none' or a comma-separated list (see sources-list)")
    p.add_argument("--log-dir", default=env("CFB_LOG_DIR"), help="Defaults to <artifacts>/logs")
    p.add_argument("--refresh-data", action="store_true", default=env("CFB_REFRESH_DATA", "0") == "1",
                   help="Re-download the season inputs even when cached copies exist")
    return p


def run_from_args(args):
    receipt = run(args.action, Context.from_args(args))
    return 1 if receipt["status"] == "failed" else 0


def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser(prog="cfb-pipeline", description="Run the CFB forecast pipeline"))
    raise SystemExit(run_from_args(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
