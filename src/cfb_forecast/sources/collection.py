"""Collect, interpret, reconcile and publish external source data.

Outputs (under ``<data>/sources/``):

* ``results.csv`` / ``picks.csv`` / ``lines.csv``: every accepted record, one row per
  source, game and retrieval, with the evidence text it came from
* ``team_stats/<source>.csv``: team stat tables
* ``results_crosscheck.csv``: per game, the latest score from each source, whether
  sources agree with each other and with the schedule
* ``collections/<timestamp>.json`` and ``collections/latest.json``: per-source status

Agreed final scores are also written to ``<artifacts>/results/sources/results.csv``
(the dashboard reads it with the ESPN results), and pregame lines are appended
to ``data/market_history.csv`` with ``observed_at`` = retrieval time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..logs import get
from .extract import parse_file
from .fetch import FetchError, fetch, latest_snapshot, store
from .interpret import GameBook, interpret, team_stats
from .registry import SOURCES, get_source, select

log = get("sources")

KEYS = ["source", "game_id", "retrieved_at"]


def load_book(data_root, season):
    from ..data import normalize_schedule

    path = Path(data_root) / f"schedule_{int(season)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run `cfb-forecast fetch --seasons {season}` first")
    return GameBook(normalize_schedule(pd.read_csv(path, low_memory=False)))


def _append(path, frame, keys=KEYS):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame["retrieved_at"] = pd.to_datetime(frame["retrieved_at"], utc=True, format="mixed").map(lambda t: t.isoformat())
    frame = frame.drop_duplicates([k for k in keys if k in frame], keep="last")
    frame.to_csv(path, index=False)
    return frame


def process_snapshot(snap, source, book, data_root, season, allow_past_picks=False):
    """Interpret one stored snapshot and append its records to the source stores."""
    doc = parse_file(snap.path, snap.content_type)
    retrieved = pd.Timestamp(snap.retrieved_at).tz_convert("UTC")
    meta = {"source": source.name, "url": snap.url, "retrieved_at": retrieved.isoformat(), "raw_sha256": snap.sha256}
    summary = {"lines_extracted": len(doc.lines), "results": 0, "picks": 0, "lines": 0, "team_stats": 0}
    root = Path(data_root) / "sources"
    if source.kind == "team_stats":
        rows = team_stats(doc, book)
        if rows:
            frame = pd.DataFrame(rows).assign(**meta)
            _append(root / "team_stats" / f"{source.name}.csv", frame, ["source", "team", "retrieved_at"])
        summary["team_stats"] = len(rows)
        return summary
    ctx = {"season": int(season), "retrieved_at": retrieved, "final_only": source.final_only, "allow_past_picks": allow_past_picks}
    found = interpret(doc, book, source.kind, ctx)
    for kind, rows in found.items():
        if rows:
            _append(root / f"{kind}.csv", pd.DataFrame(rows).assign(**meta))
        summary[kind] = len(rows)
    return summary


def collect(data_root, season, artifacts_root="artifacts", names=None, tag=None, week=None, offline=False, market_store=None):
    """Fetch (or reload, with ``offline``) each selected source and publish the reconciled outputs."""
    unknown = [n for n in (names or []) if n not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown sources {unknown}; choose from {sorted(SOURCES)}")
    book = load_book(data_root, season)
    started = datetime.now(timezone.utc).replace(microsecond=0)
    report = {"started_at": started.isoformat(), "season": int(season), "offline": offline, "sources": {}}
    for source in select(names, tag):
        entry = {"kind": source.kind}
        try:
            if offline:
                snap = latest_snapshot(source.name, data_root)
                if snap is None:
                    raise FetchError("no-snapshot", "no stored snapshot to re-interpret")
            else:
                if source.needs_week and week is None:
                    week = current_week(book)
                snap = fetch(source.name, source.resolve(season, week), data_root, source.min_interval, respect_robots=source.respect_robots)
            entry.update(snapshot=snap.path, url=snap.url, sha256=snap.sha256)
            entry.update(process_snapshot(snap, source, book, data_root, season))
            useful = sum(entry[k] for k in ("results", "picks", "lines", "team_stats"))
            entry["status"] = "ok" if useful else "no-records"
            if not useful:
                log.warning("%s: page retrieved but no %s records matched scheduled games (%d text lines). "
                            "The page may render its data with JavaScript; save it from a browser and use sources-import.",
                            source.name, source.kind, entry["lines_extracted"])
        except FetchError as exc:
            entry.update(status="failed", reason=exc.reason, error=str(exc))
            log.warning("%s: %s", source.name, exc)
        except Exception as exc:  # one broken source never stops the others
            entry.update(status="failed", reason="error", error=f"{type(exc).__name__}: {exc}")
            log.exception("%s: unexpected error", source.name)
        else:
            log.info("%s: %s (%s)", source.name, entry["status"], ", ".join(f"{k}={entry[k]}" for k in ("results", "picks", "lines", "team_stats") if entry.get(k)))
        report["sources"][source.name] = entry
    report.update(publish(data_root, artifacts_root, season, market_store))
    report["finished_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    _write_report(data_root, report, started)
    return report


def import_file(path, source_name, data_root, season, artifacts_root="artifacts", observed_at=None, market_store=None):
    """Interpret a saved page, export or PDF as if it had been fetched from ``source_name``.

    ``observed_at`` is when the content was published or captured; it defaults to now.
    Picks are only accepted for games that had not kicked off by that time.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    source = get_source(source_name)
    when = pd.Timestamp(observed_at).tz_convert("UTC").to_pydatetime() if observed_at else None
    ctype = {".pdf": "application/pdf", ".json": "application/json", ".csv": "text/csv", ".txt": "text/plain"}.get(path.suffix.lower(), "text/html")
    snap = store(source.name, f"file://{path.resolve()}", path.read_bytes(), ctype, data_root, retrieved_at=when)
    book = load_book(data_root, season)
    summary = process_snapshot(snap, source, book, data_root, season)
    summary.update(publish(data_root, artifacts_root, season, market_store))
    summary.update(source=source.name, snapshot=snap.path, sha256=snap.sha256, observed_at=snap.retrieved_at)
    log.info("imported %s as %s: %s", path, source.name, {k: summary[k] for k in ("results", "picks", "lines", "team_stats")})
    return summary


def current_week(book):
    now = pd.Timestamp.now(tz="UTC")
    upcoming = sorted((g for games in book.pairs.values() for g in games if g.kickoff >= now - pd.Timedelta(days=2)), key=lambda g: g.kickoff)
    return upcoming[0].week if upcoming else max(g.week for games in book.pairs.values() for g in games)


def _final_observations(data_root, artifacts_root, season):
    """Every final-score sighting: external result sources plus ESPN scoreboard snapshots."""
    frames = []
    path = Path(data_root) / "sources" / "results.csv"
    if path.exists():
        res = pd.read_csv(path)
        res = res[(res.season == int(season)) & res.is_final.astype(str).str.lower().eq("true")]
        frames.append(res[["source", "game_id", "home_points", "away_points", "retrieved_at"]])
    for snap in sorted(Path(artifacts_root).glob(f"results/{int(season)}-*/snapshots/*.csv")):
        espn = pd.read_csv(snap)
        espn = espn[espn.is_final.astype(str).str.lower().eq("true")]
        espn = espn[pd.to_numeric(espn.game_id, errors="coerce").notna()]
        if len(espn):
            frames.append(espn.assign(source="espn")[["source", "game_id", "home_points", "away_points", "retrieved_at"]])
    if not frames:
        return pd.DataFrame()
    obs = pd.concat(frames, ignore_index=True).dropna(subset=["home_points", "away_points"])
    obs["game_id"] = pd.to_numeric(obs.game_id).astype("int64")
    obs["home_points"] = obs.home_points.astype(int)
    obs["away_points"] = obs.away_points.astype(int)
    obs["retrieved_at"] = pd.to_datetime(obs.retrieved_at, utc=True, format="mixed")
    return obs


def crosscheck(data_root, season, artifacts_root="artifacts"):
    """Per game: the latest final from each provider (ESPN included) and agreement flags.

    ``available_at`` is the earliest moment at least two providers had reported the
    agreed score (each provider's first sighting, second-earliest across providers).
    """
    from ..data import normalize_schedule

    obs = _final_observations(data_root, artifacts_root, season)
    if obs.empty:
        return pd.DataFrame()
    sched = normalize_schedule(pd.read_csv(Path(data_root) / f"schedule_{int(season)}.csv", low_memory=False)).set_index("game_id")
    latest = obs.sort_values("retrieved_at").drop_duplicates(["source", "game_id"], keep="last")
    rows = []
    for gid, g in latest.groupby("game_id"):
        if gid not in sched.index:
            continue
        s = sched.loc[gid]
        scores = sorted({(h, a) for h, a in zip(g.home_points, g.away_points)})
        agree = len(scores) == 1
        sched_score = (int(s.home_points), int(s.away_points)) if bool(s.result_usable) else None
        available_at = None
        if agree:
            hits = obs[(obs.game_id == gid) & (obs.home_points == scores[0][0]) & (obs.away_points == scores[0][1])]
            firsts = sorted(hits.groupby("source").retrieved_at.min())
            available_at = firsts[1].isoformat() if len(firsts) >= 2 else None
        rows.append({
            "game_id": int(gid), "season": int(season), "week": int(s.week), "kickoff": s.start_date.isoformat(),
            "home_team": s.home_team, "away_team": s.away_team,
            "sources": ";".join(sorted(g.source)), "n_sources": int(g.source.nunique()),
            "home_points": scores[0][0] if agree else None, "away_points": scores[0][1] if agree else None,
            "sources_agree": agree,
            "source_scores": ";".join(f"{src}={h}-{a}" for src, h, a in zip(g.source, g.home_points, g.away_points)),
            "schedule_score": f"{sched_score[0]}-{sched_score[1]}" if sched_score else "",
            "schedule_agrees": (sched_score == scores[0]) if sched_score and agree else None,
            "available_at": available_at,
            "latest_retrieved_at": g.retrieved_at.max().isoformat(),
        })
    return pd.DataFrame(rows).sort_values(["kickoff", "game_id"]) if rows else pd.DataFrame()


def write_overlay(data_root, cons):
    """Results the model may use before the schedule snapshot catches up (>= 2 agreeing providers)."""
    from ..data import RESULTS_OVERLAY

    path = Path(data_root) / RESULTS_OVERLAY
    usable = cons[cons.sources_agree & cons.available_at.notna() & cons.schedule_score.eq("")] if not cons.empty else cons
    new = pd.DataFrame({
        "game_id": usable.game_id, "season": usable.season, "home_points": usable.home_points,
        "away_points": usable.away_points, "completed_at": usable.available_at, "providers": usable.sources,
    }) if len(usable) else pd.DataFrame(columns=["game_id", "season", "home_points", "away_points", "completed_at", "providers"])
    if path.exists():
        old = pd.read_csv(path)
        # Keep first-written availability times; never move a result earlier or change a recorded score silently.
        keep = old[~old.game_id.isin(new.game_id)]
        both = new.merge(old[["game_id", "home_points", "away_points", "completed_at"]], on="game_id", how="left", suffixes=("", "_old"))
        changed = both.home_points_old.notna() & ((both.home_points != both.home_points_old) | (both.away_points != both.away_points_old))
        if changed.any():
            log.warning("overlay: agreed provider score changed for games %s; using the new agreed score", both.game_id[changed].tolist())
        both["completed_at"] = both.completed_at_old.where(both.completed_at_old.notna() & ~changed, both.completed_at)
        new = pd.concat([keep, both[new.columns]], ignore_index=True)
    new.sort_values("game_id").to_csv(path, index=False)
    return int(len(new))


def publish(data_root, artifacts_root, season, market_store=None):
    """Rebuild the results crosscheck, export agreed finals, update the results overlay and append pregame lines."""
    out = {}
    root = Path(data_root) / "sources"
    root.mkdir(parents=True, exist_ok=True)
    cons = crosscheck(data_root, season, artifacts_root)
    if not cons.empty:
        cons.to_csv(root / "results_crosscheck.csv", index=False)
        agreed = cons[cons.sources_agree & cons.schedule_agrees.ne(False)]
        dest = Path(artifacts_root) / "results" / "sources"
        dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "provider_game_id": agreed.game_id, "game_id": agreed.game_id, "season": agreed.season, "week": agreed.week,
            "status": "STATUS_FINAL", "status_detail": "Final", "is_final": True, "kickoff": agreed.kickoff,
            "home_team": agreed.home_team, "away_team": agreed.away_team,
            "home_points": agreed.home_points, "away_points": agreed.away_points,
            "retrieved_at": agreed.latest_retrieved_at, "provider": "sources:" + agreed.sources,
        }).to_csv(dest / "results.csv", index=False)
        out["crosscheck_games"] = int(len(cons))
        out["crosscheck_conflicts"] = int((~cons.sources_agree).sum() + cons.schedule_agrees.eq(False).sum())
        out["overlay_games"] = write_overlay(data_root, cons)
    lines_path = root / "lines.csv"
    if lines_path.exists():
        out["market_rows_appended"] = _lines_to_market(pd.read_csv(lines_path), market_store or Path(data_root) / "market_history.csv")
    return out


def _lines_to_market(lines, store_path):
    from ..market import append_market_history, read_market_history

    usable = lines.dropna(subset=["home_spread", "total_line", "home_ml_decimal", "away_ml_decimal"], how="all").copy()
    if usable.empty:
        return 0
    frame = pd.DataFrame({
        "game_id": usable.game_id.astype("int64"),
        "observed_at": pd.to_datetime(usable.retrieved_at, utc=True, format="mixed"),
        "provider": usable.source,
        "home_spread": usable.home_spread, "total_line": usable.total_line,
        "home_ml_decimal": usable.home_ml_decimal, "away_ml_decimal": usable.away_ml_decimal,
        "source": usable.url,
    }).drop_duplicates(["game_id", "provider", "observed_at"], keep="last")
    before = len(read_market_history(store_path)) if Path(store_path).exists() else 0
    after = len(append_market_history(store_path, frame))
    return int(after - before)


def _write_report(data_root, report, started):
    folder = Path(data_root) / "sources" / "collections"
    folder.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, indent=2, default=str)
    (folder / f"{started:%Y%m%dT%H%M%SZ}.json").write_text(body)
    (folder / "latest.json").write_text(body)


def list_sources():
    return pd.DataFrame([{"name": s.name, "kind": s.kind, "url": s.url, "final_only": s.final_only, "description": s.description}
                         for s in SOURCES.values()])

