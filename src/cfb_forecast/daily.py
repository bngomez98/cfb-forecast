"""Daily continuous-learning cycle: the application's autonomous heartbeat.

The system is designed to run *as an application*, not as a notebook of tests.
Each invocation of :func:`run_daily_cycle` performs one complete learning pass
over the real calendar:

1. **Ingest** — refresh final scores from ESPN's public scoreboard API for any
   day that has elapsed since the last completed cycle (``results.refresh``),
   so yesterday's games become today's training signal.
2. **Settle and learn** — replay every frozen pre-kickoff prediction whose
   result is now available, score each expert against the observed winner,
   update the Beta posteriors, and persist the new blend weights
   (``learning.settle_learning``). This is the reinforcement step: experts
   that keep being right keep gaining weight; experts that miss lose weight,
   automatically and without any manual re-tuning.
3. **Forecast forward** — run the leakage-free weekly forecast for the next
   upcoming slate (``weekly.run_weekly_forecast``) using the freshly learned
   state.
4. **Freeze** — write immutable pre-kickoff prediction records for every game
   in that slate, capturing all expert probabilities and the conditional
   provenance (weather/venue/injury/player blocks actually attached), so the
   next settlement can grade exactly what was believed before kickoff.

The cycle is idempotent and crash-safe: it stores a marker file recording the
last completed day, skips ingestion when nothing new has elapsed, and the
settlement ledger refuses to double-count a game. A scheduler (cron inside
Docker, GitHub Actions, or ``--loop`` mode) can call it once a day forever;
calling it twice on the same day is harmless.

Weather, injuries, rosters, and player production are captured opportunistically
when the operator supplies the corresponding CSV paths; when they are absent the
cycle still runs, with explicit missingness indicators recorded in provenance.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from .data import timestamp
from .learning import BLEND_STATE_FILE, PREDICTIONS_FILE, freeze_upcoming, settle_learning, train_heads
from .weekly import run_weekly_forecast

log = logging.getLogger(__name__)

#: Name of the completion-marker file inside the learning directory.
CYCLE_MARKER_FILE = "daily_cycle.json"

#: Default lookback window (days) for refreshing results after an outage.
DEFAULT_RESULTS_LOOKBACK_DAYS = 7

#: Minimum new settled games that trigger a conditional-heads refit when a
#: previous fit exists. Below this threshold the extra compute is not worth
#: the posterior churn; above it the heads relearn from the enlarged sample.
HEADS_REFIT_MIN_NEW_GAMES = 15


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("daily.cycle: unreadable marker at %s; treating as first run", path)
        return None


def _load_capture(path_str):
    """Read an optional capture CSV (weather/injuries/player ppg)."""
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        log.info("daily.cycle: optional capture %s not present; skipping block", path)
        return None
    return pd.read_csv(path)


def run_daily_cycle(
    data_root="data",
    model_root="artifacts/backtest/model",
    out_root="artifacts",
    season=None,
    *,
    as_of=None,
    market_path=None,
    weather_path=None,
    injuries_path=None,
    player_ppg_path=None,
    draws=50000,
    decay=1.0,
    force=False,
    results_refresh=True,
    results_lookback_days=DEFAULT_RESULTS_LOOKBACK_DAYS,
):
    """Execute one full ingest -> settle -> forecast -> freeze learning pass.

    Parameters mirror the CLI: ``season`` defaults to the latest season found
    in the schedule store; ``as_of`` lets operators replay history; ``force``
    reruns even if today's cycle already completed. Returns a report dict that
    is also persisted to ``<learning>/daily_cycle.json``.
    """
    data_root = Path(data_root)
    out_root = Path(out_root)
    learning_dir = out_root / "learning"
    marker_path = learning_dir / CYCLE_MARKER_FILE
    at = timestamp(as_of or datetime.now(UTC).isoformat())
    day_key = at.strftime("%Y-%m-%d")

    marker = _read_json(marker_path)
    if marker and marker.get("day") == day_key and not force:
        log.info("daily.cycle: day %s already completed; nothing to do", day_key)
        return {"skipped": True, "day": day_key, "previous": marker}

    schedules, _, _ = _load_schedules(data_root)
    if season is None:
        season = int(schedules.season.max())
    season = int(season)

    report: dict = {
        "day": day_key,
        "started_at": at.isoformat(),
        "season": season,
        "steps": {},
    }

    # 1. Ingest: pull any results that landed since the previous cycle.
    if results_refresh:
        report["steps"]["results"] = _refresh_results(
            data_root, out_root, season, marker, at, lookback_days=results_lookback_days
        )

    # Re-load so settlement sees the newly ingested finals.
    schedules, _, _ = _load_schedules(data_root)

    # 2. Settle & learn: grade frozen predictions, update expert posteriors.
    summary = settle_learning(
        data_root,
        ledger_dir=learning_dir,
        blend_out=learning_dir / BLEND_STATE_FILE,
        as_of=at.isoformat(),
        decay=decay,
    )
    report["steps"]["learning"] = {
        "settled": summary["settled"],
        "pending": summary["pending"],
        "weights": summary["weights"],
    }
    log.info(
        "daily.cycle: settled %d games (%d pending); weights %s",
        summary["settled"],
        summary["pending"],
        summary["weights"],
    )

    # 2b. Retrain the conditional score heads on the enlarged settled sample.
    # The heads are the production experts (ridge/logit/gbm over the rating-
    # free conditioning matrix). We refit when none exists yet, or when the
    # day's settlement added a meaningful number of new graded games.
    heads_dir = out_root / "model-conditional"
    capture_kwargs = {
        "venues": _load_capture(out_root / "venues.csv"),
        "weather": _load_capture(weather_path),
        "injuries": _load_capture(injuries_path),
        "player_ppg": _load_capture(player_ppg_path),
    }
    trained_before = 0
    manifest_path = heads_dir / "manifest.json"
    if manifest_path.exists():
        try:
            trained_before = int(json.loads(manifest_path.read_text()).get("trained_at_games", 0))
        except (json.JSONDecodeError, ValueError, OSError):
            trained_before = 0
    total_settled = int(schedules.result_usable.sum()) if "result_usable" in schedules.columns else 0
    should_refit = (
        not (heads_dir / "heads.pkl").exists()
        or (total_settled - trained_before) >= HEADS_REFIT_MIN_NEW_GAMES
    )
    if should_refit:
        try:
            report["steps"]["heads"] = train_heads(
                data_root, heads_dir, as_of=at.isoformat(), **capture_kwargs
            )
            log.info(
                "daily.cycle: conditional heads refit on %s games",
                report["steps"]["heads"]["trained_at_games"],
            )
        except ValueError as error:
            # Too few settled games to fit yet: freeze falls back to matchup/
            # market/base experts plus the legacy model column. Never abort.
            report["steps"]["heads"] = {"status": "skipped", "reason": str(error)}
            log.info("daily.cycle: heads refit skipped (%s)", error)
    else:
        report["steps"]["heads"] = {"status": "carried_over", "trained_at_games": trained_before}

    # 3. Forecast forward with the freshly learned state.
    result, manifest = run_weekly_forecast(
        data_root=data_root,
        model_root=model_root,
        out_root=out_root,
        season=season,
        as_of=at.isoformat(),
        market_path=market_path,
        draws=draws,
    )
    report["steps"]["forecast"] = manifest

    # 4. Freeze pre-kickoff records so tomorrow can grade today's beliefs.
    enriched = freeze_upcoming(
        data_root,
        result,
        as_of=at.isoformat(),
        market_path=market_path,
        venues=capture_kwargs["venues"],
        weather=capture_kwargs["weather"],
        injuries=capture_kwargs["injuries"],
        player_ppg=capture_kwargs["player_ppg"],
        blend_path=learning_dir / BLEND_STATE_FILE,
        ledger_dir=learning_dir,
        heads_dir=heads_dir,
        decay=decay,
    )
    report["steps"]["freeze"] = {
        "fixtures_frozen": int(len(enriched)),
        "ledger": str(learning_dir / PREDICTIONS_FILE),
        "conditional_blocks": sorted(
            {
                column.split("_")[0]
                for column in (enriched.columns if "blend_p_home" in enriched.columns else [])
                if column.startswith(("weather_", "home_injury", "away_injury", "venue_"))
            }
        ),
    }

    report["completed_at"] = datetime.now(UTC).isoformat()
    _write_marker(marker_path, report)
    return report


def _load_schedules(data_root: Path):
    from .data import load_data

    return load_data(data_root)


def _refresh_results(data_root, out_root, season, marker, at, *, lookback_days):
    """Fetch scoreboard finals covering the gap since the last cycle.

    One call per cycle: ESPN's scoreboard endpoint accepts a date range, so we
    pull ``[last completed day - lookback, today]`` in a single request rather
    than hammering the API once per day. Failures are logged and reported but
    never abort the learning loop; settlement simply has nothing new to grade.
    """
    from .results import refresh_results

    if marker and marker.get("completed_at"):
        last = pd.Timestamp(marker["completed_at"]).tz_convert("UTC")
        start = min(last.normalize(), (at - timedelta(days=lookback_days)).normalize())
    else:
        start = (at - timedelta(days=lookback_days)).normalize()
    end = at.normalize() + timedelta(days=1)
    try:
        _, manifest = refresh_results(
            data_root=data_root,
            output_root=Path(out_root) / "results",
            season=int(season),
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return {
            "status": "ok",
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "games": manifest["games"],
            "final_games": manifest["final_games"],
        }
    except Exception as error:  # network flakiness must never kill the loop
        log.warning("daily.cycle: results refresh failed (%s: %s)", type(error).__name__, error)
        return {"status": f"error:{type(error).__name__}", "start": str(start.date()), "end": str(end.date())}


def _write_marker(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def run_loop(interval_minutes: float, **kwargs):
    """Run :func:`run_daily_cycle` forever, sleeping between passes.

    Intended for container foreground mode; exceptions in one pass are logged
    and retried on the next tick rather than crashing the worker.
    """
    while True:
        try:
            report = run_daily_cycle(**kwargs)
            log.info("daily.loop: pass complete (%s)", report.get("day"))
        except Exception:
            log.exception("daily.loop: pass failed; will retry next tick")
        time.sleep(max(1.0, interval_minutes * 60.0))
