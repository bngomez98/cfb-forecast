"""Ongoing weekly cycle: R source analysis -> point-in-time forecast -> Sunday validation.

Each scheduled FPS week of the season gets one forecast issued at the start of
the week (first kickoff minus ``lead_hours``) and is validated on the Sunday
closing that week's slate. No artifacts, no manifests, no simulations beyond
the model's own score expectation: every result lands in the SQLite schema
(:mod:`cfb_forecast.schema`) and an email goes to the recipient each time a
forecast completes and each time a validation completes.

Pipeline per week (:func:`run_week`):

1. ``Rscript r/analyze_sources.R <data> <as_of> <season>`` — sources are
   analyzed in R *before* any Python forecasting; failure aborts the week.
2. :func:`cfb_forecast.forecast.forecast` scores the week's slate from a state
   replayed only from results available by the issue time.
3. Forecast rows are logged to the schema and emailed.
4. If the week's games have final results usable by the validation boundary,
   they are graded (Brier, hit rate, margin/total MAE), logged, and emailed.
   Weeks whose Sunday has not arrived yet stay open with ``validated_at NULL``
   and are graded by :func:`validate_pending` on the next autonomous pass.

:func:`run_cycle` walks all weeks whose start has arrived, then validates any
still-open past weeks; it is safe to run repeatedly (cron / scheduler).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data import load_data, timestamp
from .evaluate import metrics
from .notify import send, write_queue_file
from .schema import RunLog
from .score import score_games

DEFAULT_LEAD_HOURS = 48.0
DEFAULT_RECIPIENT = "bngomez@ieee.org"


def fps_weeks(schedules, season):
    """Chronological ``(week, season_type, first_kickoff, last_kickoff)`` boundaries."""
    games = schedules[(schedules.season == int(season)) & schedules.fbs_game]
    if games.empty:
        raise ValueError(f"No FBS games scheduled for season {int(season)}")
    out = (
        games.groupby(["week", "season_type"], sort=False)
        .agg(first_kickoff=("start_date", "min"), last_kickoff=("start_date", "max"),
             games=("game_id", "size"))
        .reset_index().sort_values("first_kickoff").reset_index(drop=True)
    )
    out["week"] = out.week.astype(int)
    return out


def forecast_issue_time(first_kickoff, lead_hours=DEFAULT_LEAD_HOURS):
    return timestamp(first_kickoff) - pd.Timedelta(hours=float(lead_hours))


def validation_sunday(last_kickoff):
    """Monday-00Z boundary closing the Sunday that ends the week's slate."""
    return (timestamp(last_kickoff) + pd.Timedelta(days=1)).normalize()


def analyze_sources_r(data_root, as_of, season, repo_root=None):
    """Run the R source-analysis stage; returns its JSON report or raises."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    script = root / "r" / "analyze_sources.R"
    if not script.exists():
        raise FileNotFoundError(str(script))
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript not available: sources must be analyzed in R first")
    proc = subprocess.run(
        ["Rscript", str(script), str(data_root), str(as_of), str(int(season))],
        capture_output=True, text=True, timeout=600,
    )
    report = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            report = json.loads(line)
            break
    if proc.returncode != 0 or not report:
        raise RuntimeError(f"R source analysis failed: {proc.stderr.strip()[:400]}")
    return report


def _grade(predictions, season_games, validated_at):
    actual = season_games[["game_id", "result_usable", "result_available_at",
                           "home_points", "away_points"]]
    joined = predictions.merge(actual, on="game_id", how="left")
    ready = joined.result_usable.fillna(False) & (joined.result_available_at <= timestamp(validated_at))
    graded = joined[ready].copy()
    if graded.empty:
        return graded
    graded["actual_home_win"] = (graded.home_points > graded.away_points).astype(int)
    graded["correct"] = ((graded.home_win_probability >= 0.5).astype(int) == graded.actual_home_win).astype(int)
    graded["brier"] = (graded.home_win_probability - graded.actual_home_win) ** 2
    graded["margin_error"] = graded.projected_home_margin - (graded.home_points - graded.away_points)
    graded["total_error"] = graded.projected_total - (graded.home_points + graded.away_points)
    graded["validated_at"] = str(validated_at)
    return graded


def _email(recipient, log, run_id, kind, subject, body):
    transport = detail = None
    delivered = False
    try:
        transport = send(recipient, subject, body)
        delivered = True
    except Exception as exc:  # noqa: BLE001 - queue instead of losing the notice
        detail = f"{type(exc).__name__}: {exc}"
        transport = "queued"
        written = write_queue_file(recipient, subject, body)
        detail = f"{detail}; queued to {written}" if detail else written
    log.log_notification(kind, recipient, subject, delivered, transport, detail)
    return delivered, transport


def _forecast_email(preds):
    lines = [f"{r.away_team} @ {r.home_team}  {r.kickoff}"
             f"  proj {r.expected_away_score:.1f}-{r.expected_home_score:.1f}"
             f"  home win {r.home_win_probability:.0%}"
             for r in preds.itertuples(index=False)]
    return "\n".join(lines)


def _validation_email(graded):
    graded = graded.copy()
    # rows restored from the ledger store finals as home_points/away_points
    if "home_points" not in graded.columns and "actual_home_points" in graded.columns:
        graded["home_points"] = graded["actual_home_points"]
        graded["away_points"] = graded["actual_away_points"]
    m = metrics(graded.actual_home_win, graded.home_win_probability,
                graded.projected_home_margin, graded.home_points - graded.away_points,
                graded.projected_total, graded.home_points + graded.away_points)
    m["margin_mae"] = float(graded.margin_error.abs().mean())
    m["total_mae"] = float(graded.total_error.abs().mean())
    head = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())
    lines = [f"{r.away_team} @ {r.home_team}: proj margin {r.projected_home_margin:+.1f}"
             f" -> final {r.away_points}-{r.home_points} (err {r.margin_error:+.1f})"
             for r in graded.itertuples(index=False)]
    return head + "\n\n" + "\n".join(lines)


def run_week(data_root, model_root, db_path, season, week_row, now=None,
             lead_hours=DEFAULT_LEAD_HOURS, recipient=DEFAULT_RECIPIENT, log=None):
    """One week end to end: R analysis, forecast, logging, email, validation if due."""
    log = log or RunLog(db_path)
    at = forecast_issue_time(week_row.first_kickoff, lead_hours)
    boundary = validation_sunday(week_row.last_kickoff)
    horizon = timestamp(now) if now is not None else datetime.now(UTC)
    validated_at = boundary if boundary <= horizon else None
    run_id = log.start_run(season, week_row.week, week_row.season_type, at, validated_at)
    try:
        report = analyze_sources_r(data_root, at.isoformat(), season)
        log.log_source_check(run_id, True, report)
        preds = score_games(
            data_root=data_root, model_root=model_root,
            as_of=at.isoformat(), through=week_row.last_kickoff.isoformat(),
            season=int(season), target_game_ids=week_row.game_ids,
        )
        if "week" not in preds.columns:
            preds.insert(0, "week", int(week_row.week))
        n = log.log_forecasts(run_id, preds)
        _email(recipient, log, run_id, "forecast",
               f"[CFB] {int(season)} week {int(week_row.week)} forecast complete ({n} games)",
               f"Issued at cutoff {at.isoformat()} after R source audit "
               f"({report['games_total']} schedule games checked, "
               f"{report['training_games_pit']} point-in-time training games).\n\n"
               + _forecast_email(preds))
        graded = pd.DataFrame()
        if validated_at is not None:
            graded = _grade(preds, week_row.season_games_all, validated_at)
            if len(graded):
                g = log.log_validations(run_id, graded)
                _email(recipient, log, run_id, "validation",
                       f"[CFB] {int(season)} week {int(week_row.week)} validation complete ({g} games)",
                       _validation_email(graded))
            else:
                log.conn.execute("UPDATE runs SET validated_at=NULL WHERE run_id=?", (run_id,))
                log.conn.commit()
        log.finish_run(run_id, "completed")
    except Exception as exc:
        log.finish_run(run_id, "failed", str(exc))
        raise
    return run_id, preds, graded


def validate_pending(data_root, model_root, db_path, season, recipient=DEFAULT_RECIPIENT,
                     now=None, log=None):
    """Grade weeks left open by earlier passes once their Sunday has arrived."""
    log = log or RunLog(db_path)
    horizon = timestamp(now) if now is not None else datetime.now(UTC)
    schedules, _, _ = load_data(data_root)
    boundaries = fps_weeks(schedules, season)
    season_games = schedules[(schedules.season == int(season)) & schedules.fbs_game]
    done = []
    for run_id, _s, week, season_type, as_of in log.pending_validation_weeks(season):
        row = boundaries[(boundaries.week == week) & (boundaries.season_type == season_type)]
        if row.empty or validation_sunday(row.iloc[0].last_kickoff) > horizon:
            continue
        preds = log.week_forecasts(run_id)
        if preds.empty:
            continue
        graded = _grade(preds, season_games, validation_sunday(row.iloc[0].last_kickoff))
        if graded.empty:
            continue
        g = log.log_validations(run_id, graded)
        # email uses the forecast row's projections; carry them onto the graded frame
        graded["as_of"] = graded["issued_at"]
        graded["expected_home_score"] = graded["projected_home_margin"] / 2 + graded["projected_total"] / 2
        graded["expected_away_score"] = graded["projected_total"] - graded["expected_home_score"]
        log.conn.execute("UPDATE runs SET validated_at=? WHERE run_id=?",
                         (validation_sunday(row.iloc[0].last_kickoff).isoformat(), run_id))
        log.conn.commit()
        _email(recipient, log, run_id, "validation",
               f"[CFB] {int(season)} week {int(week)} validation complete ({g} games)",
               _validation_email(graded.rename(columns={
                   "home_points": "actual_home_points",
                   "away_points": "actual_away_points"})))
        done.append((run_id, week, g))
    return done


def run_cycle(data_root, model_root, db_path, season, now=None,
              lead_hours=DEFAULT_LEAD_HOURS, recipient=DEFAULT_RECIPIENT):
    """Autonomous pass: re-issue every started week's forecast, validate due Sundays.

    Safe to run repeatedly; each pass rebuilds the season's rows from real
    sources so the schema always reflects the latest schedule/results data.
    """
    log = RunLog(db_path)
    schedules, _, _ = load_data(data_root)
    boundaries = fps_weeks(schedules, season)
    season_games = schedules[(schedules.season == int(season)) & schedules.fbs_game]
    horizon = timestamp(now) if now is not None else datetime.now(UTC)
    log.conn.execute("DELETE FROM runs WHERE season=?", (int(season),))
    log.conn.commit()
    results = []
    for row in boundaries.itertuples(index=False):
        if row.first_kickoff > horizon:
            break
        week_games = season_games[season_games.week == int(row.week)]
        item = row._asdict() | {"game_ids": week_games.game_id.astype(int).tolist(),
                                "season_games_all": season_games}
        results.append(run_week(data_root, model_root, db_path, season,
                                pd.Series(item), now=now, lead_hours=lead_hours,
                                recipient=recipient, log=log))
    pending = validate_pending(data_root, model_root, db_path, season,
                               recipient=recipient, now=now, log=log)
    log.close()
    return results, pending
