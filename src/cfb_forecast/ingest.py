"""Source acquisition: preserve every retrieval, then load normalized football records.

Acquisition performs no forecasting calculation. Each retrieval is stored with its
source identifier, retrieval time, schema version and content hash; every
normalized row carries the retrieval it came from and the time it became known.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    "sportsdataverse": ("SportsDataverse / cfbfastR", "https://github.com/sportsdataverse/sportsdataverse-data", "schedules, play-by-play, drives, team and player statistics, EPA"),
    "cfbfastr_models": ("cfbfastR model registry", "https://github.com/sportsdataverse/cfbfastR-cfb-data/blob/main/models/REGISTRY.md", "published expected-points, win-probability and completion-probability models"),
    "cfbd": ("CollegeFootballData API", "https://api.collegefootballdata.com", "games, plays, drives, statistics, players, rosters, recruiting, transfers, usage, returning production, ratings, weather, venues"),
    "footballdb": ("FootballDB", "https://www.footballdb.com/college-football/stats/index.html", "independent team and player statistical tables"),
    "cfbtrack": ("CFBTrack", "https://cfbtrack.com/", "independent team/player statistics and contextual records"),
    "fanduel": ("FanDuel", "https://www.fanduel.com/research/ncaaf", "timestamped spreads, totals, moneylines"),
    "cfbtxt": ("CFB TXT", "https://cfbtxt.com/schedule/", "schedule and result reconciliation, weekly validation"),
    "mrtimo": ("mrtimo game data", "https://mrtimo.github.io/cfb-games/", "secondary game metadata and cross-checking"),
}
SCHEMA_VERSIONS = {"sportsdataverse": "sdv-2026", "cfbd": "cfbd-v1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_sources(conn) -> None:
    conn.executemany(
        """INSERT INTO sources(source_id,name,url,role,schema_version) VALUES(?,?,?,?,?)
           ON CONFLICT(source_id) DO UPDATE SET name=excluded.name,url=excluded.url,role=excluded.role""",
        [(sid, name, url, role, SCHEMA_VERSIONS.get(sid)) for sid, (name, url, role) in SOURCES.items()])
    conn.commit()


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def record_retrieval(conn, source_id, url, body=None, *, season=None, status="ok", retrieved_at=None,
                     name=None, schema_version=None, store_root=None, suffix=".bin", path=None,
                     receipt=None) -> int:
    """Store one retrieval (and its bytes when given) and return its id."""
    register_sources(conn)
    retrieved_at = retrieved_at or utc_now()
    sha, size = None, None
    if body is not None:
        sha, size = _digest(body), len(body)
    elif path is not None and Path(path).exists():
        data = Path(path).read_bytes()
        sha, size = _digest(data), len(data)
    version = schema_version or SCHEMA_VERSIONS.get(source_id)
    cursor = conn.execute(
        """INSERT INTO source_retrievals(source_id,season,name,url,status,retrieved_at,schema_version,sha256,bytes,receipt_json)
           VALUES(?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(source_id,season,retrieved_at) DO UPDATE SET status=excluded.status,sha256=excluded.sha256""",
        (source_id, season, name, url, status, retrieved_at, version, sha, size, json.dumps(receipt or {}, default=str)))
    retrieval_id = conn.execute("SELECT id FROM source_retrievals WHERE source_id IS ? AND season IS ? AND retrieved_at=?",
                                (source_id, season, retrieved_at)).fetchone()[0]
    stored = path
    if body is not None and store_root is not None and sha:
        target = Path(store_root) / "sources" / source_id / f"{sha}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(body)
        stored = target
    if stored is not None and sha:
        conn.execute("""INSERT OR IGNORE INTO source_files(retrieval_id,path,format,sha256,bytes,stored_at) VALUES(?,?,?,?,?,?)""",
                     (retrieval_id, str(stored), Path(str(stored)).suffix.lstrip("."), sha, size, retrieved_at))
    conn.commit()
    return retrieval_id


def _none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return int(value)
    return value


def _rows(frame, columns):
    return [tuple(_none(v) for v in row) for row in frame[columns].itertuples(index=False, name=None)]


# ------------------------------------------------------------------ SportsDataverse

OUTCOME_CLASSES = {
    "TD": "touchdown", "FG": "field_goal", "MISSED FG": "missed_field_goal", "PUNT": "punt", "DOWNS": "turnover_on_downs",
    "INT": "turnover", "FUMBLE": "turnover", "END OF HALF": "end_of_period", "END OF GAME": "end_of_period",
    "END OF QUARTER": "end_of_period",
}


def outcome_class(result) -> str:
    """Possession outcome class; defensive/special-teams scores and safeties are rare events."""
    text = str(result or "").upper().strip()
    if text in OUTCOME_CLASSES:
        return OUTCOME_CLASSES[text]
    if text in ("NOT PROVIDED", "UNKNOWN", "", "NONE", "NAN"):
        return "unknown"
    return "rare_score"


def _clock_seconds(value):
    try:
        m, s = str(value).split(":")
        return int(m) * 60 + int(s)
    except ValueError:
        return None


PBP_COLUMNS = [
    "game_id", "id", "drive.id", "pos_team_id", "def_pos_team_id", "period", "clock.displayValue", "down", "distance",
    "start.yardLine", "start.yardsToEndzone", "pos_score_diff_start", "type.text", "statYardage", "first_down_created",
    "touchdown", "completion", "pass_attempt", "int", "fumble_vec", "fumble_lost", "sack", "TFL", "penalty_flag",
    "yds_penalty", "fg_attempt", "field_goal_result", "punt", "kickoff_play", "yds_punt_return", "yds_kickoff_return",
    "yds_int_return", "scrimmage_play", "pass", "rush", "qb_hurry", "early_down", "late_down", "rz_play", "goal_to_go",
    "stuffed_run", "short_rush_attempt", "short_rush_success", "scoring_opp", "pos_score_pts", "passer_player_id",
    "rusher_player_id", "receiver_player_id", "passer_player_name", "rusher_player_name", "receiver_player_name",
    "EPA", "EP_start", "EP_end", "cp", "cpoe", "EPA_success", "EPA_explosive", "drive.result", "drive.start.yardLine",
    "drive.end.yardLine", "drive.yards", "drive.offensivePlays", "drive.timeElapsed.displayValue",
    "drive.start.period.number", "season", "week", "air_yards", "yards_after_catch", "penalty_team_id", "yds_punted",
    "yds_kickoff", "kickoff_tb", "punt_tb", "yds_fg",
]


def _flag(series):
    return series.astype("boolean").fillna(False).astype(int)


def ingest_play_by_play(conn, path, season, retrieval_id, available_at=None) -> dict:
    """Load plays, drives, derived play observations and players from a cfbfastR/SportsDataverse PBP file."""
    import pyarrow.parquet as pq
    present = set(pq.ParquetFile(path).schema_arrow.names)
    d = pd.read_parquet(path, columns=[c for c in PBP_COLUMNS if c in present])
    for column in PBP_COLUMNS:
        if column not in d:
            d[column] = np.nan
    known = {int(r[0]): r[1] for r in conn.execute("SELECT game_id, result_available_at FROM games WHERE season=?", (int(season),))}
    d = d[d.game_id.astype("int64").isin(known)].copy()
    if d.empty:
        return {"plays": 0, "drives": 0, "players": 0}
    d["game_id"] = d.game_id.astype("int64")
    avail = d.game_id.map(known)
    d["play_id"] = d["id"].astype(str)
    d["drive_key"] = d["game_id"].astype(str) + "-" + d["drive.id"].astype(str)
    ret = d[["yds_punt_return", "yds_kickoff_return", "yds_int_return"]].apply(pd.to_numeric, errors="coerce")
    d["return_yards"] = ret.bfill(axis=1).iloc[:, 0]
    is_pass, is_rush = _flag(d["pass"]), _flag(d["rush"])
    attempt = _flag(d["pass_attempt"])
    d["incompletion"] = ((attempt == 1) & (_flag(d["completion"]) == 0) & (_flag(d["int"]) == 0)).astype(int)
    d["score_diff"] = pd.to_numeric(d["pos_score_diff_start"], errors="coerce")
    play_columns = ["play_id", "game_id", "drive_key", "pos_team_id", "def_pos_team_id", "period", "clock.displayValue", "down",
                    "distance", "start.yardLine", "start.yardsToEndzone", "score_diff", "type.text", "statYardage"]
    frame = pd.DataFrame({
        "play_id": d.play_id, "game_id": d.game_id, "drive_id": d.drive_key, "offense": d.pos_team_id, "defense": d.def_pos_team_id,
        "period": d.period, "clock": d["clock.displayValue"], "down": d.down, "distance": d.distance,
        "yard_line": d["start.yardLine"], "yards_to_goal": d["start.yardsToEndzone"], "score_difference": d.score_diff,
        "play_type": d["type.text"], "yards_gained": d.statYardage, "first_down": _flag(d.first_down_created),
        "touchdown": _flag(d.touchdown), "completion": _flag(d.completion), "incompletion": d.incompletion,
        "interception": _flag(d["int"]), "fumble": _flag(d.fumble_vec), "fumble_lost": _flag(d.fumble_lost),
        "sack": _flag(d.sack), "tfl": _flag(d.TFL), "penalty": _flag(d.penalty_flag), "penalty_yards": d.yds_penalty,
        "fg_attempt": _flag(d.fg_attempt), "fg_result": d.field_goal_result, "punt": _flag(d.punt),
        "kickoff": _flag(d.kickoff_play), "return_yards": d.return_yards, "scrimmage": _flag(d.scrimmage_play),
        "is_pass": is_pass, "is_rush": is_rush, "qb_hurry": _flag(d.qb_hurry), "early_down": _flag(d.early_down),
        "late_down": _flag(d.late_down), "red_zone": _flag(d.rz_play), "goal_to_go": _flag(d.goal_to_go),
        "stuffed_run": _flag(d.stuffed_run), "sy_attempt": _flag(d.short_rush_attempt), "sy_success": _flag(d.short_rush_success),
        "scoring_opportunity": _flag(d.scoring_opp), "points_scored": pd.to_numeric(d.pos_score_pts, errors="coerce"),
        "passer": d.passer_player_id.astype("string"), "rusher": d.rusher_player_id.astype("string"),
        "receiver": d.receiver_player_id.astype("string"), "air_yards": pd.to_numeric(d.air_yards, errors="coerce"),
        "yac": pd.to_numeric(d.yards_after_catch, errors="coerce"), "penalty_team": pd.to_numeric(d.penalty_team_id, errors="coerce"),
        "punt_yards": pd.to_numeric(d.yds_punted, errors="coerce"), "kickoff_yards": pd.to_numeric(d.yds_kickoff, errors="coerce"),
        "touchback": (_flag(d.kickoff_tb) | _flag(d.punt_tb)), "fg_distance": pd.to_numeric(d.yds_fg, errors="coerce"),
        "source": "sportsdataverse", "retrieval": retrieval_id, "available": avail,
    })
    conn.executemany("""INSERT OR REPLACE INTO plays(play_id,game_id,drive_id,offense_team_id,defense_team_id,period,clock,down,distance,
        yard_line,yards_to_goal,score_difference,play_type,yards_gained,first_down,touchdown,completion,incompletion,interception,
        fumble,fumble_lost,sack,tackle_for_loss,penalty,penalty_yards,field_goal_attempt,field_goal_result,punt,kickoff,return_yards,
        scrimmage_play,is_pass,is_rush,qb_hurry,early_down,late_down,red_zone,goal_to_go,stuffed_run,short_yardage_attempt,
        short_yardage_success,scoring_opportunity,points_scored,passer_id,rusher_id,receiver_id,air_yards,yards_after_catch,penalty_team_id,punt_yards,kickoff_yards,touchback,fg_distance,source_id,retrieval_id,available_at)
        VALUES(""" + ",".join("?" * 56) + ")", _rows(frame, list(frame.columns)))

    # Derived observations from the published cfbfastR models (EP/EPA/CP) plus deterministic indicators.
    version = f"sdv-pbp-{int(season)}"
    now = utc_now()
    yards = pd.to_numeric(d.statYardage, errors="coerce")
    epa = pd.to_numeric(d.EPA, errors="coerce")
    derived = pd.DataFrame({
        "play_id": d.play_id, "model": "cfbfastR_ep", "version": version, "ep_before": pd.to_numeric(d.EP_start, errors="coerce"),
        "ep_after": pd.to_numeric(d.EP_end, errors="coerce"), "epa": epa,
        "cp": pd.to_numeric(d.cp, errors="coerce"), "cpoe": pd.to_numeric(d.cpoe, errors="coerce") / 100.0,
        "success": _flag(d.EPA_success), "explosive": _flag(d.EPA_explosive),
        "negative": ((yards < 0) & (_flag(d.scrimmage_play) == 1)).astype(int),
        "definition": "EP/EPA/CP from cfbfastR registered models; success=EPA_success; explosive=EPA_explosive; negative=scrimmage yards<0",
        "at": now,
    })
    conn.executemany("""INSERT OR REPLACE INTO derived_play_observations(play_id,derivation_model,derivation_model_version,
        expected_points_before,expected_points_after,epa,expected_completion_probability,completion_over_expected,
        success_indicator,explosive_play_indicator,negative_play_indicator,input_definition,calculation_timestamp)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", _rows(derived, list(derived.columns)))

    # Drives: one row per ESPN drive, points from the offense's scoring plays inside the drive.
    signed = pd.to_numeric(d.pos_score_pts, errors="coerce")
    pts = d.assign(points=signed.clip(lower=0), against=(-signed).clip(lower=0))
    grouped = pts.groupby("drive_key")
    drives = pd.DataFrame({
        "game_id": grouped.game_id.first(), "team_id": grouped.pos_team_id.first(), "start": grouped["drive.start.yardLine"].first(),
        "end": grouped["drive.end.yardLine"].first(), "plays": grouped["drive.offensivePlays"].first(),
        "yards": grouped["drive.yards"].first(), "duration": grouped["drive.timeElapsed.displayValue"].first().map(_clock_seconds),
        "first_downs": grouped.first_down_created.apply(lambda s: int(s.fillna(False).astype(bool).sum())),
        "result": grouped["drive.result"].first(), "points": grouped.points.sum(), "against": grouped.against.sum(),
        "period": grouped["drive.start.period.number"].first(),
    })
    drives["outcome"] = drives.result.map(outcome_class)
    drives["source"], drives["retrieval"] = "sportsdataverse", retrieval_id
    drives["available"] = drives.game_id.map(known)
    drives = drives.reset_index().rename(columns={"drive_key": "drive_id"})
    conn.executemany("""INSERT OR REPLACE INTO drives(drive_id,game_id,team_id,start_field_position,end_field_position,plays,yards,
        duration_seconds,first_downs,result,points,opponent_points,start_period,outcome_class,source_id,retrieval_id,available_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        _rows(drives, ["drive_id", "game_id", "team_id", "start", "end", "plays", "yards", "duration", "first_downs", "result",
                       "points", "against", "period", "outcome", "source", "retrieval", "available"]))

    # Players observed in plays: id, name, role-inferred position group, team membership span.
    roles = []
    for column, name_col, role in (("passer_player_id", "passer_player_name", "QB"), ("rusher_player_id", "rusher_player_name", "RB"),
                                   ("receiver_player_id", "receiver_player_name", "REC")):
        part = d[[column, name_col, "pos_team_id", "game_id"]].dropna(subset=[column]).rename(columns={column: "pid", name_col: "name"})
        part["role"] = role
        roles.append(part)
    seen = pd.concat(roles, ignore_index=True)
    seen["pid"] = seen.pid.astype(str)
    counts = seen.groupby(["pid", "role"]).size().unstack(fill_value=0)
    position = counts.idxmax(axis=1)
    names = seen.dropna(subset=["name"]).groupby("pid").name.first()
    conn.executemany("INSERT INTO players(player_id,name,position) VALUES(?,?,?) ON CONFLICT(player_id) DO UPDATE SET name=excluded.name,position=COALESCE(players.position,excluded.position)",
                     [(pid, names.get(pid), pos) for pid, pos in position.items()])
    span = seen.assign(start=seen.game_id.map(known)).groupby(["pid", "pos_team_id"]).agg(start=("start", "min"), end=("start", "max")).reset_index()
    conn.execute("DELETE FROM player_team_membership WHERE source_id='sportsdataverse' AND start_date >= ? AND start_date <= ?",
                 (f"{int(season)}-01-01", f"{int(season)}-12-31"))
    conn.executemany("INSERT INTO player_team_membership(player_id,team_id,start_date,end_date,status,source_id,available_at) VALUES(?,?,?,?,?,?,?)",
                     [(r.pid, int(r.pos_team_id), r.start, r.end, "observed_in_plays", "sportsdataverse", r.end) for r in span.itertuples()])
    conn.commit()
    return {"plays": len(frame), "drives": len(drives), "players": len(position)}


def ingest_sportsdataverse(conn, data_root, season, artifacts=None, include_plays=True) -> dict:
    """Load one season from the downloaded SportsDataverse files (schedule, team rates, play-by-play)."""
    from .data import load_data, sha256
    from .warehouse import load_games, load_team_statistics
    data_root = Path(data_root)
    manifest_path = data_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()).get("files", {}) if manifest_path.exists() else {}
    schedules, adv, _ = load_data(data_root)
    summary = {"season": int(season)}
    retrievals = {}
    for name in (f"schedule_{season}.csv", f"adv_team_{season}.parquet", f"pbp_{season}.parquet"):
        path = data_root / name
        if path.exists():
            meta = manifest.get(name, {})
            retrievals[name] = record_retrieval(
                conn, "sportsdataverse", meta.get("url", str(path)), None, season=int(season), name=name, path=path,
                retrieved_at=meta.get("downloaded_at") or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                receipt={"sha256_manifest": meta.get("sha256"), "sha256_file": sha256(path)})
    summary["games"] = load_games(conn, schedules, season)
    summary["team_game_statistics"] = load_team_statistics(conn, schedules, adv, season, retrievals.get(f"adv_team_{season}.parquet"))
    pbp = data_root / f"pbp_{season}.parquet"
    if include_plays and pbp.exists():
        summary.update(ingest_play_by_play(conn, pbp, season, retrievals[f"pbp_{season}.parquet"]))
    conn.commit()
    return summary


# ------------------------------------------------------------------ CollegeFootballData

CFBD = "https://api.collegefootballdata.com"


class CFBDClient:
    """Thin CFBD client; ``fetch`` is injectable so tests and offline replays never touch the network."""

    def __init__(self, api_key=None, fetch=None):
        self.api_key = api_key or os.environ.get("CFBD_API_KEY")
        self._fetch = fetch or self._http

    def _http(self, url, headers):
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    def get(self, path, **params):
        if not self.api_key and self._fetch == self._http:
            raise RuntimeError("CFBD_API_KEY is not set; collection skipped rather than inventing data")
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{CFBD}{path}" + (f"?{query}" if query else "")
        body = self._fetch(url, {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json", "User-Agent": "cfb-forecast/0.5"})
        return url, body, json.loads(body)


def _g(row, *keys, default=None):
    for key in keys:
        if isinstance(row, dict) and row.get(key) is not None:
            return row[key]
    return default


def _venue_rows(payload):
    return [(str(_g(v, "id")), _g(v, "name"), _g(v, "city"), _g(v, "state"), _g(v, "latitude") or (_g(v, "location") or {}).get("x"),
             _g(v, "longitude") or (_g(v, "location") or {}).get("y"), _g(v, "elevation"),
             "dome" if _g(v, "dome") else ("grass" if _g(v, "grass") else "turf"), int(bool(_g(v, "dome"))), _g(v, "timezone")) for v in payload]


def _team_rows(payload):
    rows = []
    for t in payload:
        loc = t.get("location") or {}
        rows.append((int(t["id"]), _g(t, "school", "team"), _g(t, "abbreviation"), _g(t, "conference"), _g(t, "classification"),
                     loc.get("latitude"), loc.get("longitude"), loc.get("timezone")))
    return rows


def _clock(c):
    if isinstance(c, dict):
        return f"{int(c.get('minutes') or 0):02d}:{int(c.get('seconds') or 0):02d}"
    return c


def ingest_cfbd(conn, season, artifacts=None, client=None, weeks=None) -> dict:
    """Collect every CFBD endpoint named in the design; each call is preserved with its hash and loaded."""
    client = client or CFBDClient()
    summary: dict[str, int] = {}
    now = utc_now()

    def pull(label, path, **params):
        url, body, payload = client.get(path, **params)
        rid = record_retrieval(conn, "cfbd", url, body, season=int(season), name=label, store_root=artifacts, suffix=".json",
                               retrieved_at=utc_now())
        return rid, payload

    rid, payload = pull("venues", "/venues")
    conn.executemany("""INSERT INTO venues(venue_id,name,city,state,latitude,longitude,elevation,surface,indoor,timezone)
        VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(venue_id) DO UPDATE SET name=excluded.name,latitude=excluded.latitude,
        longitude=excluded.longitude,elevation=excluded.elevation,surface=excluded.surface,indoor=excluded.indoor""", _venue_rows(payload))
    summary["venues"] = len(payload)

    rid, payload = pull("teams", "/teams/fbs", year=season)
    conn.executemany("""INSERT INTO teams(team_id,name,abbreviation,conference,division,latitude,longitude,timezone)
        VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(team_id) DO UPDATE SET name=excluded.name,conference=excluded.conference,
        latitude=excluded.latitude,longitude=excluded.longitude,timezone=excluded.timezone""", _team_rows(payload))
    summary["teams"] = len(payload)

    rid, payload = pull("games", "/games", year=season)
    game_rows = []
    for g in payload:
        game_rows.append((int(g["id"]), _g(g, "venueId", "venue_id"), _g(g, "homeConference", "home_conference"), int(_g(g, "homeId", "home_id")),
                          int(_g(g, "awayId", "away_id"))))
    # CFBD enriches games that exist from the schedule source; it never overwrites scores or kickoff.
    conn.executemany("UPDATE games SET venue_id=COALESCE(?,venue_id), conference=COALESCE(?,conference) WHERE game_id=? AND home_id=? AND away_id=?",
                     [(None if v is None else str(v), c, gid, h, a) for gid, v, c, h, a in game_rows])
    summary["games"] = len(payload)

    for week in (weeks or [None]):
        wk = {"week": week} if week is not None else {}
        rid, payload = pull("games_weather", "/games/weather", year=season, **wk)
        conn.executemany("""INSERT INTO weather(game_id,temperature,wind_speed,precipitation,humidity,conditions,observed_at,source_id,retrieval_id,available_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)""", [(int(w["id"]), w.get("temperature"), w.get("windSpeed"), w.get("precipitation"), w.get("humidity"),
                                              w.get("weatherCondition"), w.get("startTime"), "cfbd", rid, now) for w in payload])
        summary["weather"] = summary.get("weather", 0) + len(payload)

        rid, payload = pull("games_teams", "/games/teams", year=season, **wk)
        rows = []
        for game in payload:
            for team in game.get("teams", []):
                stats = {s["category"]: s["stat"] for s in team.get("stats", []) if "category" in s}
                rows.append((int(game["id"]), int(team["teamId"]), "cfbd", int(season), _num(stats.get("firstDowns")), _num(stats.get("turnovers")),
                             _num(stats.get("sacks")), _num(stats.get("tacklesForLoss")), _num(stats.get("totalPenaltiesYards", "0-0").split("-")[0]) if isinstance(stats.get("totalPenaltiesYards"), str) else None,
                             _num(stats.get("netPassingYards")), _num(stats.get("rushingYards")), rid, now))
        conn.executemany("""INSERT OR REPLACE INTO team_game_statistics(game_id,team_id,source_id,season,first_downs,turnovers,sacks_created,
            tackles_for_loss,penalties,passing_yards,rushing_yards,retrieval_id,available_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        summary["team_game_statistics"] = summary.get("team_game_statistics", 0) + len(rows)

        rid, payload = pull("games_players", "/games/players", year=season, **wk)
        stat_rows = []
        for game in payload:
            for team in game.get("teams", []):
                for cat in team.get("categories", []):
                    for typ in cat.get("types", []):
                        for athlete in typ.get("athletes", []):
                            stat_rows.append((int(game["id"]), str(athlete["id"]), _g(team, "teamId"), cat.get("name"), typ.get("name"), athlete.get("stat")))
        summary["player_game_statistics"] = summary.get("player_game_statistics", 0) + _load_player_game_stats(conn, stat_rows, rid, now)

        rid, payload = pull("drives", "/drives", year=season, **wk)
        conn.executemany("""INSERT OR REPLACE INTO drives(drive_id,game_id,team_id,start_field_position,end_field_position,plays,yards,duration_seconds,
            result,outcome_class,points,source_id,retrieval_id,available_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(f"cfbd-{d['id']}", int(d["gameId"]), _g(d, "offenseId"), _g(d, "startYardline"), _g(d, "endYardline"), d.get("plays"), d.get("yards"),
              _elapsed(d.get("elapsed")), d.get("driveResult"), outcome_class(d.get("driveResult")), None, "cfbd", rid, now) for d in payload])
        summary["drives"] = summary.get("drives", 0) + len(payload)

        rid, payload = pull("plays", "/plays", year=season, **wk)
        conn.executemany("""INSERT OR REPLACE INTO plays(play_id,game_id,drive_id,offense_team_id,defense_team_id,period,clock,down,distance,yard_line,
            yards_to_goal,play_type,yards_gained,source_id,retrieval_id,available_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(f"cfbd-{p['id']}", int(p["gameId"]), f"cfbd-{p['driveId']}" if p.get("driveId") is not None else None, _g(p, "offenseId"), _g(p, "defenseId"),
              p.get("period"), _clock(p.get("clock")), p.get("down"), p.get("distance"), p.get("yardline"), p.get("yardsToGoal"),
              p.get("playType"), p.get("yardsGained"), "cfbd", rid, now) for p in payload])
        summary["plays"] = summary.get("plays", 0) + len(payload)

    rid, payload = pull("player_usage", "/player/usage", year=season)
    conn.executemany("""INSERT INTO player_usage(player_id,team_id,season,source_id,usage_overall,usage_pass,usage_rush,usage_first_down,
        usage_second_down,usage_third_down,usage_standard_downs,usage_passing_downs,retrieval_id,available_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(str(u["id"]), _g(u, "teamId"), int(season), "cfbd", *[(u.get("usage") or {}).get(k) for k in
          ("overall", "pass", "rush", "firstDown", "secondDown", "thirdDown", "standardDowns", "passingDowns")], rid, now) for u in payload])
    summary["player_usage"] = len(payload)

    rid, payload = pull("returning_production", "/player/returning", year=season)
    team_ids = {r[1]: r[0] for r in conn.execute("SELECT team_id, name FROM teams WHERE name IS NOT NULL")}
    conn.executemany("""INSERT OR REPLACE INTO returning_production(season,team_id,total_ppa,total_passing_ppa,total_receiving_ppa,
        total_rushing_ppa,percent_ppa,usage,source_id,available_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        [(int(season), team_ids.get(r["team"]), r.get("totalPPA"), r.get("totalPassingPPA"), r.get("totalReceivingPPA"), r.get("totalRushingPPA"),
          r.get("percentPPA"), r.get("usage"), "cfbd", now) for r in payload if team_ids.get(r.get("team")) is not None])
    summary["returning_production"] = len(payload)

    rid, payload = pull("recruiting", "/recruiting/players", year=season)
    conn.executemany("INSERT INTO recruiting(player_id,season,team_id,position,rating,stars,national_rank,source_id,available_at) VALUES(?,?,?,?,?,?,?,?,?)",
                     [(str(_g(r, "athleteId", "id")), int(season), team_ids.get(r.get("committedTo")), r.get("position"), r.get("rating"),
                       r.get("stars"), r.get("ranking"), "cfbd", now) for r in payload])
    summary["recruiting"] = len(payload)

    rid, payload = pull("transfer_portal", "/player/portal", year=season)
    conn.executemany("INSERT INTO transfers(player_id,season,previous_team_id,new_team_id,transfer_date,rating,eligibility,source_id,available_at) VALUES(?,?,?,?,?,?,?,?,?)",
                     [(f"{r.get('firstName', '')} {r.get('lastName', '')}".strip(), int(season), team_ids.get(r.get("origin")), team_ids.get(r.get("destination")),
                       r.get("transferDate"), r.get("rating"), r.get("eligibility"), "cfbd", r.get("transferDate") or now) for r in payload])
    summary["transfers"] = len(payload)

    ratings = 0
    for kind in ("sp", "fpi", "elo", "srs"):
        rid, payload = pull(f"ratings_{kind}", f"/ratings/{kind}", year=season)
        rows = [(int(season), None, team_ids.get(r.get("team")), kind, r.get("rating") if r.get("rating") is not None else r.get("elo"), _g(r, "ranking"),
                 "cfbd", rid, now) for r in payload]
        conn.executemany("""INSERT OR IGNORE INTO external_ratings(season,week,team_id,rating_source,rating,rank,source_id,retrieval_id,available_at)
            VALUES(?,?,?,?,?,?,?,?,?)""", [r for r in rows if r[2] is not None])
        ratings += len(rows)
    summary["external_ratings"] = ratings

    roster_rows = 0
    for team_name in team_ids:
        rid, payload = pull("roster", "/roster", year=season, team=team_name)
        conn.executemany("""INSERT OR IGNORE INTO rosters(season,team_id,player_id,position,class_year,jersey,height,weight,source_id,retrieval_id,available_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""", [(int(season), team_ids[team_name], str(r["id"]), r.get("position"), r.get("year"), r.get("jersey"),
                                                  r.get("height"), r.get("weight"), "cfbd", rid, now) for r in payload])
        roster_rows += len(payload)
    summary["rosters"] = roster_rows
    conn.commit()
    return summary


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _elapsed(value):
    if isinstance(value, dict):
        return int(value.get("minutes") or 0) * 60 + int(value.get("seconds") or 0)
    return None


PLAYER_STAT_MAP = {
    ("passing", "ATT"): "pass_attempts", ("passing", "C/ATT"): "completions", ("passing", "YDS"): "passing_yards", ("passing", "INT"): "interceptions",
    ("rushing", "CAR"): "carries", ("rushing", "YDS"): "rushing_yards", ("receiving", "TAR"): "targets", ("receiving", "REC"): "receptions",
    ("receiving", "YDS"): "receiving_yards", ("defensive", "SACKS"): "sacks", ("defensive", "TOT"): "tackles", ("interceptions", "INT"): "interceptions_made",
    ("kicking", "FG"): "field_goal_makes", ("punting", "NO"): "punt_attempts", ("punting", "YDS"): "punt_yards",
}


def _load_player_game_stats(conn, rows, retrieval_id, now) -> int:
    table: dict[tuple, dict] = {}
    for game_id, pid, team_id, category, kind, value in rows:
        column = PLAYER_STAT_MAP.get((str(category).lower(), str(kind).upper()))
        if column is None:
            continue
        text = str(value)
        number = _num(text.split("/")[0]) if column == "completions" else _num(text)
        table.setdefault((game_id, pid, team_id), {})[column] = number
    for (game_id, pid, team_id), stats in table.items():
        columns = list(stats)
        conn.execute(f"INSERT OR REPLACE INTO player_game_statistics(game_id,player_id,source_id,team_id,retrieval_id,available_at,{','.join(columns)}) "
                     f"VALUES(?,?,?,?,?,?,{','.join('?' * len(columns))})", (game_id, pid, "cfbd", team_id, retrieval_id, now, *stats.values()))
    return len(table)


# ------------------------------------------------------------------ market quotes

def ingest_market_quotes(conn, quotes, source_id="fanduel", retrieval_id=None) -> int:
    """Store timestamped quotes exactly as observed; a quote needs a game, book, market, price/line and retrieval time."""
    rows = []
    for q in quotes:
        if not all(q.get(k) is not None for k in ("game_id", "sportsbook", "market_type", "retrieved_at")):
            raise ValueError(f"Incomplete market quote (game_id, sportsbook, market_type, retrieved_at required): {q}")
        rows.append((int(q["game_id"]), q["sportsbook"], q["market_type"], q.get("side"), _none(q.get("line")), _none(q.get("price")),
                     pd.Timestamp(q["retrieved_at"]).tz_convert("UTC").isoformat(), source_id, retrieval_id))
    conn.executemany("INSERT INTO market_quotes(game_id,sportsbook,market_type,side,line,price,retrieved_at,source_id,retrieval_id) VALUES(?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)
