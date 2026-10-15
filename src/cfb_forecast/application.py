"""The forecasting application: configuration, scheduler daemon, service commands.

This module is the entry point of the *application*, not a pipeline script.
It owns:

* ``config.toml`` loading — paths, season, cadence and recipient are declared
  once in a config file; every command runs from it with no arguments
  (``cfb-forecast up``, ``cfb-forecast status``, ``cfb-forecast run-once``).
* :func:`daemon` — a long-running process that wakes on an interval, executes
  the autonomous weekly pass (R source analysis -> joint possession forecast
  at each FPS week start -> Sunday validation -> SQLite schema log -> email),
  refreshes live results before each pass, and logs everything through the
  standard logger so ``journalctl``/container logs show real activity.
* :func:`status` / :func:`show` — read the live state straight out of the
  ledger database: current week, open/pending validations, latest metrics,
  per-game forecasts.

There is no statistical "application state" here: all football state lives in
the sources and the ledger; the process only executes the system.
"""
from __future__ import annotations

import tomllib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration resolved from ``config.toml`` plus defaults."""

    data_root: str = "data"
    model_root: str = "artifacts/backtest/model"
    db_path: str = "cfb_ledger.db"
    season: int = 2026
    lead_hours: float = 48.0
    recipient: str = "bngomez@ieee.org"
    interval_seconds: int = 900
    market_path: str | None = None
    root: Path = field(default_factory=lambda: Path.cwd())

    @property
    def resolved(self) -> "AppConfig":
        """Return a copy with relative paths anchored at the config directory."""
        base = self.root

        def _p(value: str) -> str:
            path = Path(value)
            return str(path if path.is_absolute() else base / path)

        return AppConfig(
            data_root=_p(self.data_root),
            model_root=_p(self.model_root),
            db_path=_p(self.db_path),
            season=self.season,
            lead_hours=self.lead_hours,
            recipient=self.recipient,
            interval_seconds=self.interval_seconds,
            market_path=_p(self.market_path) if self.market_path else None,
            root=base,
        )


def load_config(path=None) -> AppConfig:
    """Read ``config.toml`` (or ``CFB_FORECAST_CONFIG``) into an AppConfig."""
    candidate = (
        Path(path)
        if path
        else Path(__import__("os").environ.get("CFB_FORECAST_CONFIG", DEFAULT_CONFIG_PATH))
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"No config at {candidate}. Copy config.example.toml to config.toml."
        )
    raw = tomllib.loads(candidate.read_text())
    app = raw.get("app", {})
    season = int(app.get("season", datetime.now(UTC).year))
    return AppConfig(
        data_root=str(app.get("data_root", "data")),
        model_root=str(app.get("model_root", "artifacts/backtest/model")),
        db_path=str(app.get("db_path", "cfb_ledger.db")),
        season=season,
        lead_hours=float(app.get("lead_hours", 48.0)),
        recipient=str(app.get("recipient", "bngomez@ieee.org")),
        interval_seconds=int(app.get("interval_seconds", 900)),
        market_path=app.get("market_path"),
        root=candidate.resolve().parent,
    ).resolved


# --------------------------------------------------------------------------- daemon


def run_pass(cfg: AppConfig, log) -> tuple[int, list]:
    """One autonomous pass: refresh live results, then the weekly cycle."""
    from .results import refresh_results
    from .weekly import run_cycle

    try:
        refresh_results(data_root=cfg.data_root, season=cfg.season)
    except Exception as exc:  # noqa: BLE001 - offline is fine, ledger keeps last data
        log.warning("live results refresh skipped: %s", exc)
    results, pending = run_cycle(
        cfg.data_root, cfg.model_root, cfg.db_path, cfg.season,
        lead_hours=cfg.lead_hours, recipient=cfg.recipient,
    )
    return len(results), pending


def daemon(cfg: AppConfig, once: bool = False) -> None:
    """Long-running application loop. Wakes every ``interval_seconds``."""
    import logging

    log = logging.getLogger("cfb_forecast.application")
    pid_file = cfg.root / "cfb_forecast.pid"
    pid_file.write_text(str(__import__("os").getpid()))
    log.info("cfb-forecast up: season=%s db=%s interval=%ss recipient=%s",
             cfg.season, cfg.db_path, cfg.interval_seconds, cfg.recipient)
    try:
        while True:
            started = datetime.now(UTC)
            try:
                weeks, pending = run_pass(cfg, log)
                log.info("pass complete: %d week(s) issued, %d validation(s) closed%s",
                         weeks, len(pending),
                         f" at {started.isoformat(timespec='seconds')}" if pending else "")
            except Exception:  # noqa: BLE001 - a failed pass must not kill the app
                log.exception("pass failed; retrying next interval")
            if once:
                break
            time.sleep(max(1, cfg.interval_seconds))
    finally:
        pid_file.unlink(missing_ok=True)


def stop(cfg: AppConfig) -> bool:
    """Terminate a running daemon via its pid file."""
    import os
    import signal

    pid_file = cfg.root / "cfb_forecast.pid"
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return False
    pid_file.unlink(missing_ok=True)
    return True


# --------------------------------------------------------------------------- status


def status(cfg: AppConfig) -> dict:
    """Live application state read directly from the ledger database."""
    from .schema import RunLog

    if not Path(cfg.db_path).exists():
        return {"database": str(cfg.db_path), "initialized": False}
    log = RunLog(cfg.db_path)
    try:
        from .schema import pd_read

        runs = pd_read(log.conn, """
            SELECT r.run_id, r.week, r.season_type, r.as_of, r.validated_at, r.status,
                   (SELECT COUNT(*) FROM forecasts f WHERE f.run_id = r.run_id) AS games,
                   (SELECT AVG(v.brier) FROM validations v WHERE v.run_id = r.run_id) AS avg_brier,
                   (SELECT AVG(v.correct) FROM validations v WHERE v.run_id = r.run_id) AS hit_rate
            FROM runs r WHERE r.season = ? ORDER BY r.as_of DESC LIMIT 12
        """, (cfg.season,))
        pending = log.pending_validation_weeks(cfg.season)
        counts = {}
        for table in ("runs", "source_checks", "forecasts", "validations", "notifications"):
            counts[table] = log.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {
            "database": str(cfg.db_path),
            "initialized": True,
            "now_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "season": cfg.season,
            "running": (cfg.root / "cfb_forecast.pid").exists(),
            "counts": counts,
            "recent_runs": runs.to_dict("records"),
            "pending_validations": [list(p) for p in pending],
        }
    finally:
        log.close()


def show(cfg: AppConfig, week: int | None = None, team: str | None = None):
    """Latest issued forecasts per game from the ledger, optionally filtered.

    A game's newest run wins, so re-issued weeks supersede older rows; games
    already final are still shown but flagged with their validation result.
    """
    from .schema import RunLog, pd_read

    log = RunLog(cfg.db_path)
    try:
        frame = pd_read(log.conn, """
            SELECT f.run_id, f.game_id, f.week, f.home_team, f.away_team, f.kickoff, r.as_of,
                   f.expected_home_score, f.expected_away_score, f.home_win_probability,
                   v.home_points, v.away_points, v.brier, v.correct
            FROM forecasts f
            JOIN runs r ON r.run_id = f.run_id
            JOIN (SELECT f2.game_id, MAX(f2.run_id) AS run_id FROM forecasts f2
                  JOIN runs r2 ON r2.run_id = f2.run_id
                  WHERE r2.season = ? GROUP BY f2.game_id) m USING (game_id, run_id)
            LEFT JOIN validations v ON v.game_id = f.game_id AND v.run_id = f.run_id
            ORDER BY f.kickoff
        """, (cfg.season,))
    finally:
        log.close()
    if week is not None:
        frame = frame[frame.week == int(week)]
    if team:
        key = team.lower()
        frame = frame[frame.home_team.str.lower().str.contains(key)
                      | frame.away_team.str.lower().str.contains(key)]
    return frame
