"""Built-in automation: run each pipeline job when the game calendar says it is due.

    cfb-forecast scheduler                 # run forever (the `scheduler` compose service)
    cfb-forecast scheduler --once          # run whatever is due now and exit (cron / Task Scheduler)
    cfb-forecast scheduler --status        # show each job's last run and whether it is due
    cfb-forecast scheduler --run forecast  # force one job now

Jobs (all times UTC, overridable with environment variables):

=============  ======================================================================  ==========
job            when it runs                                                            action
=============  ======================================================================  ==========
forecast       every CFB_FORECAST_HOURS (24) in season, and at CFB_FORECAST_HOUR (9):  forecast
               the ongoing pipeline — refresh data, completed results and player
               ratings, then forecast every remaining upcoming FBS game
backtest       CFB_BACKTEST_DAY (tue) at CFB_BACKTEST_HOUR (11), in season             backtest
=============  ======================================================================  ==========

There is no in-game polling and no chronological ledger: each forecast run simply
recomputes from current and historical data. Outside the season (no FBS game within
21 days either side) nothing is due, so the calendar quietly resumes next season.
State, a heartbeat and a single-instance lock live in ``<artifacts>/scheduler/``.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .logs import add_file, configure, get

log = get("scheduler")

DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def env_int(name, default):
    return int(os.environ.get(name, default))


def default_season(now=None):
    value = os.environ.get("CFB_FORECAST_SEASON", "auto")
    if value != "auto":
        return int(value)
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1  # January bowls belong to the previous season


class Calendar:
    """FBS kickoff times for the season, reloaded when the schedule file changes."""

    def __init__(self, data_root, season):
        self.path = Path(data_root) / f"schedule_{season}.csv"
        self._mtime = None
        self.kickoffs = pd.Series(dtype="datetime64[ns, UTC]")

    def refresh(self):
        if not self.path.exists():
            self.kickoffs = pd.Series(dtype="datetime64[ns, UTC]")
            return
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            s = pd.read_csv(self.path, usecols=lambda c: c in {"start_date", "home_division", "away_division"}, low_memory=False)
            if {"home_division", "away_division"} <= set(s.columns):
                s = s[s.home_division.eq("fbs") & s.away_division.eq("fbs")]
            self.kickoffs = pd.to_datetime(s.start_date, utc=True).sort_values().reset_index(drop=True)
            self._mtime = mtime

    def any_between(self, start, end):
        k = self.kickoffs
        return bool(((k >= start) & (k <= end)).any())

    def in_season(self, now):
        return self.any_between(now - timedelta(days=21), now + timedelta(days=21))

    def first_kickoff_today_within(self, now, hours):
        upcoming = self.kickoffs[(self.kickoffs >= now) & (self.kickoffs <= now + timedelta(hours=hours))]
        return not upcoming.empty


@dataclass(frozen=True)
class Job:
    name: str
    action: str
    due: Callable[[datetime, dict, Calendar], bool]
    description: str


def _since(state, name, now):
    last = state.get(name, {}).get("last_started")
    return now - datetime.fromisoformat(last) if last else timedelta.max


def _daily_slot(now, hour):
    slot = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return slot if now >= slot else slot - timedelta(days=1)


def _ran_after(state, name, moment):
    last = state.get(name, {}).get("last_started")
    return bool(last) and datetime.fromisoformat(last) >= moment


def build_jobs():
    forecast_every = timedelta(hours=env_int("CFB_FORECAST_HOURS", 24))
    forecast_hour = env_int("CFB_FORECAST_HOUR", 9)
    backtest_day = DAYS[os.environ.get("CFB_BACKTEST_DAY", "tue")[:3].lower()]
    backtest_hour = env_int("CFB_BACKTEST_HOUR", 11)

    def forecast(now, st, cal):
        # Once per CFB_FORECAST_HOURS during the season; also right at the daily slot so a
        # restarted scheduler still produces the morning run. This is the ongoing pipeline.
        if not cal.in_season(now):
            return False
        return _since(st, "forecast", now) >= forecast_every - timedelta(minutes=1) or not _ran_after(st, "forecast", _daily_slot(now, forecast_hour))

    def backtest(now, st, cal):
        slot = _daily_slot(now, backtest_hour)
        while slot.weekday() != backtest_day:
            slot -= timedelta(days=1)
        return cal.in_season(now) and not _ran_after(st, "backtest", slot)

    return [
        Job("forecast", "forecast", forecast, "Daily full-slate forecast: refresh data/results/players, forecast every upcoming game"),
        Job("backtest", "backtest", backtest, "Weekly point-in-time season backtest"),
    ]


class Scheduler:
    def __init__(self, data="data", artifacts="artifacts", season=None, jobs=None, poll_seconds=60):
        self.data, self.artifacts = Path(data), Path(artifacts)
        self.season = season or default_season()
        self.jobs = [j for j in build_jobs() if not jobs or j.name in jobs]
        self.poll = poll_seconds
        self.home = self.artifacts / "scheduler"
        self.home.mkdir(parents=True, exist_ok=True)
        self.state_path = self.home / "state.json"
        self.calendar = Calendar(self.data, self.season)
        self._stop = False
        self.current = None

    # state -----------------------------------------------------------------
    def load_state(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def save_state(self, state):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)

    def heartbeat(self, now):
        (self.home / "heartbeat").write_text(now.isoformat())
        (self.home / "current_job").write_text(self.current or "")

    def _beat_forever(self):
        """Keep the heartbeat fresh while a long job runs, so the container stays healthy."""
        while not self._stop:
            try:
                self.heartbeat(datetime.now(timezone.utc))
            except OSError:
                log.warning("could not write heartbeat")
            time.sleep(30)

    # running ---------------------------------------------------------------
    def _context(self):
        from .pipeline import Context

        return Context(
            data=self.data, artifacts=self.artifacts, model=Path(os.environ.get("CFB_MODEL_DIR") or self.artifacts / "backtest" / "model"),
            season=self.season, draws=env_int("CFB_FORECAST_DRAWS", 50000),
            lead_hours=float(os.environ.get("CFB_BACKTEST_LEAD_HOURS", "48")),
            refresh_data=True, sources=os.environ.get("CFB_SOURCES", "all"),
            log_dir=Path(os.environ.get("CFB_LOG_DIR") or self.artifacts / "logs"),
        )

    def run_job(self, job, state):
        from .pipeline import run

        started = datetime.now(timezone.utc).replace(microsecond=0)
        entry = state.setdefault(job.name, {"runs": 0})
        entry.update(last_started=started.isoformat(), runs=entry.get("runs", 0) + 1)
        self.save_state(state)
        log.info("job %s starting (%s)", job.name, job.description)
        self.current = job.name
        try:
            ctx = self._context()
            # Every scheduled run refreshes current data; the daily cadence makes this cheap.
            ctx.refresh_data = True
            receipt = run(job.action, ctx)
            entry.update(last_status=receipt["status"], last_error=None)
        except Exception as exc:  # the scheduler itself must survive any job failure
            log.exception("job %s crashed", job.name)
            entry.update(last_status="crashed", last_error=f"{type(exc).__name__}: {exc}")
        finally:
            self.current = None
        entry["last_finished"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.save_state(state)
        return entry

    def due(self, now, state):
        self.calendar.refresh()
        return [j for j in self.jobs if j.due(now, state, self.calendar)]

    def tick(self):
        now = datetime.now(timezone.utc)
        state = self.load_state()
        ran = []
        for job in self.due(now, state):
            self.run_job(job, state)
            ran.append(job.name)
        self.heartbeat(datetime.now(timezone.utc))
        return ran

    def status(self):
        now = datetime.now(timezone.utc)
        state = self.load_state()
        self.calendar.refresh()
        due = {j.name for j in self.due(now, state)}
        rows = []
        for j in self.jobs:
            s = state.get(j.name, {})
            rows.append({"job": j.name, "action": j.action, "due_now": j.name in due, "last_started": s.get("last_started"),
                         "last_status": s.get("last_status"), "runs": s.get("runs", 0)})
        return {"season": self.season, "in_season": self.calendar.in_season(now), "jobs": rows}

    def _lock(self):
        import fcntl

        handle = open(self.home / "lock", "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit(f"Another scheduler is already running for {self.artifacts}") from exc
        handle.write(str(os.getpid()))
        handle.flush()
        return handle

    def forever(self):
        lock = self._lock()
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stop", True))
        log.info("scheduler started: season %s, jobs %s, poll %ss", self.season, [j.name for j in self.jobs], self.poll)
        threading.Thread(target=self._beat_forever, name="heartbeat", daemon=True).start()
        try:
            while not self._stop:
                try:
                    self.tick()
                except Exception:
                    log.exception("scheduler tick failed; retrying next poll")
                for _ in range(self.poll):
                    if self._stop:
                        break
                    time.sleep(1)
        finally:
            lock.close()
            log.info("scheduler stopped")


def add_arguments(p):
    p.add_argument("--data", default=os.environ.get("CFB_DATA_DIR", "data"))
    p.add_argument("--artifacts", default=os.environ.get("CFB_ARTIFACTS_DIR", "artifacts"))
    p.add_argument("--season", type=int, default=None, help="Default: CFB_FORECAST_SEASON, or the current season")
    p.add_argument("--jobs", nargs="+", default=None, help="Only these jobs")
    p.add_argument("--poll", type=int, default=env_int("CFB_SCHEDULER_POLL", 60), help="Seconds between checks")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run due jobs once and exit")
    mode.add_argument("--status", action="store_true", help="Print job status and exit")
    mode.add_argument("--run", default=None, help="Force one job now and exit")
    return p


def run_from_args(args):
    configure()
    sched = Scheduler(args.data, args.artifacts, args.season, args.jobs, args.poll)
    if args.status:
        print(json.dumps(sched.status(), indent=2))
        return 0
    add_file(Path(args.artifacts) / "logs", "scheduler", slot="cfb-scheduler-file")
    if args.run:
        job = next((j for j in build_jobs() if j.name == args.run), None)
        if job is None:
            raise SystemExit(f"Unknown job {args.run}; choose from {[j.name for j in build_jobs()]}")
        entry = sched.run_job(job, sched.load_state())
        return 0 if entry.get("last_status") in ("success", "degraded") else 1
    if args.once:
        log.info("ran: %s", sched.tick() or "nothing due")
        return 0
    sched.forever()
    return 0


def healthcheck(artifacts, max_age_seconds=600):
    """Exit 0 when the scheduler heartbeat is fresh (used by the container HEALTHCHECK)."""
    path = Path(artifacts) / "scheduler" / "heartbeat"
    try:
        beat = datetime.fromisoformat(path.read_text().strip())
    except (OSError, ValueError):
        return 1
    return 0 if (datetime.now(timezone.utc) - beat).total_seconds() <= max_age_seconds else 1
