"""Derived football observations and the assembled data for the joint model.

``derive_observations`` turns plays and drives into per-game team observations
(rates, means and the counts that give them their precision) and stores them in
``derived_team_observations``. ``load_model_data`` reads the database as it stood at
a cutoff and assembles every array the joint model conditions on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import model_spec as sp

MODEL_NAME = "cfb_forecast_streams"
MODEL_VERSION = "1"
DEPTH_BINS = {"short": (-np.inf, 7), "intermediate": (7, 15), "deep": (15, np.inf)}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------- team streams

def _group(frame, mask, x=None, val=None, cov=None):
    """Aggregate one stream per (game, offense): count n, successes x, mean value."""
    f = frame.loc[mask]
    g = f.groupby(["game_id", "offense_team_id"])
    out = pd.DataFrame({"n": g.size()})
    if x is not None:
        out["x"] = g[x].sum() if isinstance(x, str) else x(f).groupby([f.game_id, f.offense_team_id]).sum()
    if val is not None:
        out["val"] = g[val].mean()
    if cov is not None:
        out["cov"] = g[cov].mean()
    return out.reset_index()


def team_streams(plays: pd.DataFrame, drives: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """All team-level observation streams as rows (game_id, offense_team_id, stream, n, x, val, cov)."""
    p = plays
    scrim = p.scrimmage_play == 1
    pass_np = scrim & (p.is_pass == 1) & (p.sack == 0)
    dropback = scrim & (p.is_pass == 1)
    rush = scrim & (p.is_rush == 1)
    out = []

    def add(name, frame, kind):
        if frame is None or frame.empty:
            return
        f = frame.copy()
        f["stream"] = name
        if kind == "B":
            f["val"] = f.x / f.n
        else:
            f["x"] = np.nan
        if "cov" not in f:
            f["cov"] = 0.0
        out.append(f[["game_id", "offense_team_id", "stream", "n", "x", "val", "cov"]])

    add("pass_epa", _group(p, pass_np, val="epa"), "N")
    add("rush_epa", _group(p, rush, val="epa"), "N")
    add("success_rate", _group(p, scrim, x="success_indicator"), "B")
    add("pass_explosive", _group(p, pass_np, x="explosive_play_indicator"), "B")
    add("rush_explosive", _group(p, rush, x="explosive_play_indicator"), "B")
    add("early_down_epa", _group(p, scrim & (p.early_down == 1), val="epa"), "N")
    add("late_down_epa", _group(p, scrim & (p.late_down == 1), val="epa"), "N")
    add("short_yardage_success", _group(p, scrim & (p.short_yardage_attempt == 1), x="short_yardage_success"), "B")
    add("red_zone_epa", _group(p, scrim & (p.red_zone == 1), val="epa"), "N")
    add("goal_to_go_epa", _group(p, scrim & (p.goal_to_go == 1), val="epa"), "N")
    add("first_down_rate", _group(p, scrim, x="first_down"), "B")

    db = _group(p, dropback, x="sack")
    db["x"] = db.n - db.x
    add("sack_avoid", db, "B")
    pressured = p.assign(_pr=((p.sack == 1) | (p.qb_hurry == 1)).astype(int))
    hv = _group(pressured, dropback, x="_pr")
    hv["x"] = hv.n - hv.x
    add("hurry_avoid", hv, "B")
    sv = _group(p, rush, x="stuffed_run")
    sv["x"] = sv.n - sv.x
    add("stuff_avoid", sv, "B")
    tv = _group(p, rush, x="tackle_for_loss")
    tv["x"] = tv.n - tv.x
    add("tfl_avoid", tv, "B")
    add("pass_int", _group(p, pass_np, x="interception"), "B")
    fv = _group(p, rush, x="fumble_lost")
    fv["x"] = fv.n - fv.x
    add("rush_fumble_safe", fv, "B")

    own_pen = p.assign(_own=((p.penalty == 1) & (p.penalty_team_id == p.offense_team_id)).astype(int),
                       _opp=((p.penalty == 1) & (p.penalty_team_id == p.defense_team_id)).astype(int))
    add("offense_penalty_rate", _group(own_pen, scrim, x="_own"), "B")
    add("defense_penalty_rate", _group(own_pen, scrim, x="_opp"), "B")

    tempo = _group(p, scrim, val="epa")
    tempo["val"] = tempo.n / 80.0
    tempo["n"] = 1
    add("tempo_plays", tempo, "N")
    fourth = p.assign(_conv=((p.first_down == 1) | (p.touchdown == 1)).astype(int))
    add("fourth_down_conversion", _group(fourth, scrim & (p.down == 4) & (p.punt == 0) & (p.field_goal_attempt == 0), x="_conv"), "B")

    if not drives.empty:
        d = drives
        g = d.groupby(["game_id", "team_id"])
        created = pd.DataFrame({"n": g.size(), "x": g.opp.sum()}).reset_index().rename(columns={"team_id": "offense_team_id"})
        add("scoring_opp_created", created, "B")
        c = d[d.opp == 1].groupby(["game_id", "team_id"])
        conv = pd.DataFrame({"n": c.size(), "x": c.scored.sum()}).reset_index().rename(columns={"team_id": "offense_team_id"})
        add("scoring_opp_converted", conv, "B")
        fp = d[d.start_ytg <= 50].groupby(["game_id", "team_id"])
        fpp = pd.DataFrame({"n": fp.size(), "val": fp.points.mean()}).reset_index().rename(columns={"team_id": "offense_team_id"})
        add("field_position_points", fpp, "N")
        od = g.agg(n=("start_ytg", "size"), val=("start_ytg", "mean")).reset_index().rename(columns={"team_id": "offense_team_id"})
        od["val"] = od.val / 100.0
        add("opponent_drive_start", od, "N")

    for name, (lo, hi) in (("pass_epa_short", DEPTH_BINS["short"]), ("pass_epa_intermediate", DEPTH_BINS["intermediate"]),
                           ("pass_epa_deep", DEPTH_BINS["deep"])):
        add(name, _group(p, pass_np & p.air_yards.notna() & (p.air_yards >= lo) & (p.air_yards < hi), val="epa"), "N")
    add("completion_over_expected", _group(p, pass_np & p.completion_over_expected.notna(), val="completion_over_expected"), "N")
    add("yards_after_catch", _group(p, pass_np & (p.completion == 1) & p.yards_after_catch.notna(), val="yards_after_catch"), "N")
    is_rb = p.receiver_id.map(positions).eq("RB")
    add("rb_target_epa", _group(p, pass_np & is_rb, val="epa"), "N")

    fg = _group(p.assign(_made=p.field_goal_result.astype(str).str.lower().str.contains("good|made").astype(int)),
                (p.field_goal_attempt == 1) & p.fg_distance.notna(), x="_made", cov="fg_distance")
    fg["cov"] = (fg["cov"] - 38.0) / 10.0
    add("fg_make", fg, "B")
    kick = p.kickoff == 1
    tb = _group(p, kick, x="touchback")
    add("kickoff_touchback", tb, "B")
    ky = _group(p, kick & p.kickoff_yards.notna(), val="kickoff_yards")
    ky["val"] = ky.val / 70.0
    add("kickoff_yards", ky, "N")
    pu = _group(p, (p.punt == 1) & p.punt_yards.notna(), val="punt_yards")
    pu["val"] = pu.val / 50.0
    add("punt_yards", pu, "N")
    kr = _group(p, kick & (p.touchback == 0) & p.return_yards.notna(), val="return_yards")
    kr["val"] = kr.val / 30.0
    add("kick_return_yards", kr, "N")
    pr = _group(p, (p.punt == 1) & (p.touchback == 0) & p.return_yards.notna(), val="return_yards")
    pr["val"] = pr.val / 10.0
    add("punt_return_yards", pr, "N")
    return pd.concat(out, ignore_index=True)


def drive_frame(plays: pd.DataFrame, drives: pd.DataFrame) -> pd.DataFrame:
    """Drives with scoring-opportunity flag and the starting yards to goal from their first scrimmage play."""
    s = plays[plays.scrimmage_play == 1].copy()
    s["pid"] = pd.to_numeric(s.play_id, errors="coerce")
    s = s.sort_values(["drive_id", "pid"])
    first = s.groupby("drive_id").yards_to_goal.first().rename("start_ytg")
    opp = s.groupby("drive_id").scoring_opportunity.max().rename("opp")
    d = drives.set_index("drive_id").join(first).join(opp).reset_index()
    d["opp"] = d.opp.fillna(0).astype(int)
    d["start_ytg"] = d.start_ytg.fillna(75.0)
    d["scored"] = (d.points > 0).astype(int)
    return d


def derive_observations(conn, seasons, database_version=MODEL_VERSION) -> dict:
    """Compute and store derived_team_observations for FBS-vs-FBS games of the given seasons."""
    games = pd.read_sql_query(
        f"SELECT game_id FROM games WHERE fbs_game=1 AND result_usable=1 AND season IN ({','.join('?' * len(seasons))})", conn, params=list(seasons))
    if games.empty:
        return {"games": 0, "observations": 0}
    plays = _load_plays(conn, seasons)
    drives = drive_frame(plays, _load_drives(conn, seasons))
    positions = pd.read_sql_query("SELECT player_id, position FROM players", conn).set_index("player_id").position
    streams = team_streams(plays, drives, positions)
    now = _utc_now()
    definition = "per-game team stream from plays/drives; value=mean or rate, sample_size=n"
    rows = [(int(r.game_id), int(r.offense_team_id), MODEL_NAME, database_version, r.stream,
             None if pd.isna(r.val) else float(r.val), int(r.n), None if pd.isna(r.cov) else float(r.cov), definition, now)
            for r in streams.itertuples(index=False)]
    ids = tuple(int(g) for g in games.game_id)
    conn.execute(f"DELETE FROM derived_team_observations WHERE derivation_model=? AND derivation_model_version=? AND game_id IN ({','.join('?' * len(ids))})",
                 (MODEL_NAME, database_version, *ids))
    conn.executemany("""INSERT OR REPLACE INTO derived_team_observations(game_id,team_id,derivation_model,derivation_model_version,metric,
        value,sample_size,covariate,input_definition,calculation_timestamp) VALUES(?,?,?,?,?,?,?,?,?,?)""", rows)
    conn.commit()
    return {"games": len(games), "observations": len(rows)}


def _load_plays(conn, seasons):
    q = f"""SELECT p.*, d.epa, d.success_indicator, d.explosive_play_indicator, d.expected_completion_probability,
                   d.completion_over_expected
            FROM plays p
            JOIN games g ON g.game_id = p.game_id
            LEFT JOIN derived_play_observations d ON d.play_id = p.play_id AND d.derivation_model='cfbfastR_ep'
            WHERE g.fbs_game=1 AND g.result_usable=1 AND g.season IN ({','.join('?' * len(seasons))})"""
    plays = pd.read_sql_query(q, conn, params=list(seasons))
    for c in plays.columns:
        if plays[c].dtype == object and c not in ("play_id", "drive_id", "clock", "play_type", "field_goal_result", "passer_id",
                                                  "rusher_id", "receiver_id", "available_at", "source_id"):
            plays[c] = pd.to_numeric(plays[c], errors="coerce")
    return plays


def _load_drives(conn, seasons):
    q = f"""SELECT dr.* FROM drives dr JOIN games g ON g.game_id=dr.game_id
            WHERE g.fbs_game=1 AND g.result_usable=1 AND g.season IN ({','.join('?' * len(seasons))})
              AND dr.source_id='sportsdataverse'"""
    return pd.read_sql_query(q, conn, params=list(seasons))


# ---------------------------------------------------------------------- schedule conditions

def derive_schedule_conditions(conn) -> int:
    """Rest, short week, bye and (when coordinates exist) travel and time-zone change for every game."""
    g = pd.read_sql_query("SELECT game_id, season, start_date, home_id, away_id, neutral_site FROM games WHERE fbs_game=1", conn)
    if g.empty:
        return 0
    g["start"] = pd.to_datetime(g.start_date, utc=True, format="ISO8601")
    long = pd.concat([g[["game_id", "season", "start", "home_id"]].rename(columns={"home_id": "team"}).assign(home=1),
                      g[["game_id", "season", "start", "away_id"]].rename(columns={"away_id": "team"}).assign(home=0)])
    long = long.sort_values(["team", "start"])
    long["prev"] = long.groupby(["team", "season"]).start.shift(1)
    long["rest"] = (long.start - long.prev).dt.total_seconds() / 86400.0
    long["short"] = (long.rest < 6.5).astype(float).where(long.rest.notna())
    long["bye"] = (long.rest > 10).astype(float).where(long.rest.notna())
    teams = pd.read_sql_query("SELECT team_id, latitude, longitude, timezone FROM teams", conn).set_index("team_id")
    venues = pd.read_sql_query("SELECT g.game_id, v.latitude, v.longitude FROM games g JOIN venues v ON v.venue_id=g.venue_id", conn).set_index("game_id")
    rows = []
    sides = {1: long[long.home == 1].set_index("game_id"), 0: long[long.home == 0].set_index("game_id")}
    for r in g.itertuples(index=False):
        h, a = sides[1].loc[r.game_id], sides[0].loc[r.game_id]
        travel = {1: None, 0: None}
        if r.game_id in venues.index and not pd.isna(venues.loc[r.game_id].latitude):
            v = venues.loc[r.game_id]
            for flag, team in ((1, r.home_id), (0, r.away_id)):
                if team in teams.index and not pd.isna(teams.loc[team].latitude):
                    t = teams.loc[team]
                    travel[flag] = _haversine(t.latitude, t.longitude, v.latitude, v.longitude)
        nan = lambda x: None if pd.isna(x) else float(x)
        rows.append((int(r.game_id), nan(h.rest), nan(a.rest), nan(h.short), nan(a.short), nan(h.bye), nan(a.bye),
                     travel[1], travel[0], None, None, _utc_now()))
    conn.executemany("""INSERT OR REPLACE INTO schedule_conditions(game_id,home_days_rest,away_days_rest,home_short_week,away_short_week,
        home_bye_week,away_bye_week,home_travel_distance,away_travel_distance,home_timezone_change,away_timezone_change,computed_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    conn.commit()
    return len(rows)


def _haversine(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


# ---------------------------------------------------------------------- joint-model data

@dataclass
class ModelData:
    seasons: list
    team_ids: list
    cutoff: str | None
    # game directions (offense i against defense j)
    gd_game: np.ndarray
    gd_i: np.ndarray
    gd_j: np.ndarray
    gd_per: np.ndarray
    gd_y: np.ndarray
    gd_h: np.ndarray
    gd_X: np.ndarray
    cov_names: list
    cov_stats: dict
    # observed stream rows
    obs_gd: np.ndarray
    obs_s: np.ndarray
    obs_n: np.ndarray
    obs_x: np.ndarray
    obs_val: np.ndarray
    obs_cov: np.ndarray
    obs_src: np.ndarray
    n_src: int
    stream_scale: np.ndarray          # empirical per-play scale of each stream (prior scale for noise)
    stream_init: np.ndarray           # empirical stream mean (initial offset)
    # roster-change covariates
    R: np.ndarray                     # (T, Y, r)
    R_names: list
    # players
    nodes: dict = field(default_factory=dict)       # group -> dict(player_id, season, parent, cov, team, y)
    entries: dict = field(default_factory=dict)     # group -> dict(row, node, comp, w)
    participation: dict = field(default_factory=dict)   # group -> dict(node, gd, z)
    usage: dict = field(default_factory=dict)       # group -> dict(seg, node, count, n_seg)
    # possessions
    drive_gd: np.ndarray = None
    drive_class: np.ndarray = None
    drive_fp: np.ndarray = None
    td_idx: np.ndarray = None
    rare_off_idx: np.ndarray = None
    rare_opp_idx: np.ndarray = None
    game_gd1: np.ndarray = None
    game_gd2: np.ndarray = None
    game_drives: np.ndarray = None
    # external ratings
    rating_team: np.ndarray = None
    rating_per: np.ndarray = None
    rating_src: np.ndarray = None
    rating_val: np.ndarray = None
    # bookkeeping for forecasting
    gd_index: dict = field(default_factory=dict)    # (game_id, offense_team_id) -> gd
    team_index: dict = field(default_factory=dict)  # team_id -> idx
    last_period: np.ndarray = None                  # per team: last observed period

    @property
    def T(self):
        return len(self.team_ids)

    @property
    def Y(self):
        return len(self.seasons)

    @property
    def n_gd(self):
        return len(self.gd_game)


def _standardize(frame: pd.DataFrame, stats: dict | None = None):
    out, new = frame.copy(), {}
    for c in out.columns:
        if stats and c in stats:
            mean, sd = stats[c]
        else:
            mean, sd = float(out[c].mean()), float(out[c].std() or 1.0)
        new[c] = (mean, sd)
        out[c] = ((out[c] - mean) / (sd if sd > 0 else 1.0)).fillna(0.0)
    return out, new


def load_model_data(conn, seasons, cutoff=None, min_participation=3) -> ModelData:
    """Read the database as it stood at ``cutoff`` (result_available_at < cutoff) into model arrays."""
    seasons = sorted(int(s) for s in seasons)
    where_cut = "AND g.result_available_at < ?" if cutoff else ""
    args = [*seasons] + ([pd.Timestamp(cutoff).tz_convert("UTC").isoformat()] if cutoff else [])
    games = pd.read_sql_query(
        f"""SELECT g.game_id, g.season, g.week, g.start_date, g.home_id, g.away_id, g.neutral_site FROM games g
            WHERE g.fbs_game=1 AND g.result_usable=1 AND g.season IN ({','.join('?' * len(seasons))}) {where_cut}""", conn, params=args)
    if games.empty:
        raise ValueError("No completed FBS games available at the cutoff")
    games["y"] = games.season.map({s: n for n, s in enumerate(seasons)})
    games["per"] = games.y * sp.WEEKS + games.week.clip(1, sp.WEEKS).astype(int) - 1
    team_ids = sorted(set(games.home_id) | set(games.away_id))
    tix = {t: n for n, t in enumerate(team_ids)}

    # game directions: home offense and away offense
    home = pd.DataFrame({"game_id": games.game_id, "off": games.home_id, "def": games.away_id, "h": np.where(games.neutral_site == 1, 0.0, 1.0)})
    away = pd.DataFrame({"game_id": games.game_id, "off": games.away_id, "def": games.home_id, "h": np.where(games.neutral_site == 1, 0.0, -1.0)})
    gd = pd.concat([home, away], ignore_index=True).merge(games[["game_id", "y", "per", "season", "start_date"]], on="game_id")
    gd = gd.sort_values(["per", "game_id", "h"], ascending=[True, True, False]).reset_index(drop=True)
    gd["gd"] = np.arange(len(gd))
    gd_index = {(int(r.game_id), int(r.off)): int(r.gd) for r in gd.itertuples(index=False)}

    # schedule/weather covariates for each direction
    sc = pd.read_sql_query("SELECT * FROM schedule_conditions", conn).set_index("game_id")
    cov = pd.DataFrame(index=gd.index)
    first = gd.h >= 0
    for name, home_col, away_col in (("rest_diff", "home_days_rest", "away_days_rest"), ("short_week_diff", "home_short_week", "away_short_week"),
                                      ("bye_diff", "home_bye_week", "away_bye_week"), ("travel_diff", "home_travel_distance", "away_travel_distance")):
        if home_col in sc:
            a = pd.to_numeric(gd.game_id.map(sc[home_col]), errors="coerce").to_numpy(float)
            b = pd.to_numeric(gd.game_id.map(sc[away_col]), errors="coerce").to_numpy(float)
            own = np.where(first, a, b)
            opp = np.where(first, b, a)
            cov[name] = own - opp
    w = pd.read_sql_query("SELECT game_id, temperature, wind_speed, precipitation FROM weather WHERE source_id IS NOT NULL", conn)
    if len(w):
        w = w.groupby("game_id").mean(numeric_only=True)
        for c in ("temperature", "wind_speed", "precipitation"):
            cov[c] = gd.game_id.map(w[c])
    keep = [c for c in cov.columns if cov[c].notna().mean() >= 0.5]
    cov, cov_stats = _standardize(cov[keep]) if keep else (pd.DataFrame(index=gd.index), {})

    # observed stream rows
    obs = pd.read_sql_query(
        f"""SELECT game_id, team_id, metric, value, sample_size, covariate FROM derived_team_observations
            WHERE derivation_model=? AND derivation_model_version=? AND game_id IN ({','.join('?' * len(games))})""",
        conn, params=[MODEL_NAME, MODEL_VERSION, *[int(g) for g in games.game_id]])
    sidx = {s.name: n for n, s in enumerate(sp.STREAMS)}
    obs["s"] = obs.metric.map(sidx)
    obs = obs.dropna(subset=["s", "value"]).copy()
    obs["gd"] = [gd_index.get((int(g), int(t)), -1) for g, t in zip(obs.game_id, obs.team_id)]
    obs = obs[obs.gd >= 0]
    obs["s"] = obs.s.astype(int)
    kinds = np.array([s.kind for s in sp.STREAMS])
    obs["kind"] = kinds[obs.s.values]
    obs["n"] = obs.sample_size.astype(int)
    obs["x"] = np.where(obs.kind == "B", np.rint(obs.value * obs.n), 0).astype(int)
    S = len(sp.STREAMS)
    scale, init = np.ones(S), np.zeros(S)
    for s in range(S):
        o = obs[obs.s == s]
        if o.empty:
            continue
        if kinds[s] == "B":
            p = float(np.clip(o.x.sum() / max(o.n.sum(), 1), 1e-3, 1 - 1e-3))
            init[s] = np.log(p / (1 - p))
            scale[s] = 1.0
        else:
            init[s] = float(np.average(o.value, weights=o.n))
            scale[s] = max(float(np.sqrt(np.average(o.n * (o.value - init[s]) ** 2))), 1e-2)

    data = ModelData(
        seasons=seasons, team_ids=team_ids, cutoff=cutoff,
        gd_game=gd.game_id.to_numpy(), gd_i=gd.off.map(tix).to_numpy(), gd_j=gd["def"].map(tix).to_numpy(),
        gd_per=gd.per.to_numpy(), gd_y=gd.y.to_numpy(), gd_h=gd.h.to_numpy(float), gd_X=cov.to_numpy(float).reshape(len(gd), -1),
        cov_names=list(cov.columns), cov_stats=cov_stats,
        obs_gd=obs.gd.to_numpy(), obs_s=obs.s.to_numpy(), obs_n=obs.n.to_numpy(), obs_x=obs.x.to_numpy(),
        obs_val=obs.value.to_numpy(float), obs_cov=obs.covariate.fillna(0).to_numpy(float), obs_src=np.zeros(len(obs), int), n_src=1,
        stream_scale=scale, stream_init=init, R=np.zeros((len(team_ids), len(seasons), len(sp.TEAM_COVARIATES))), R_names=list(sp.TEAM_COVARIATES),
        gd_index=gd_index, team_index=tix)
    last = np.full(len(team_ids), -1)
    for i, per in zip(data.gd_i, data.gd_per):
        last[i] = max(last[i], per)
    data.last_period = last
    _attach_players(conn, data, seasons, gd, min_participation)
    _attach_possessions(conn, data, seasons, gd)
    _attach_ratings(conn, data, cutoff)
    return data


def _attach_players(conn, data: ModelData, seasons, gd, min_participation):
    plays = pd.read_sql_query(
        f"""SELECT p.game_id, p.offense_team_id, p.scrimmage_play, p.is_pass, p.is_rush, p.sack, p.passer_id, p.rusher_id, p.receiver_id
            FROM plays p WHERE p.game_id IN ({','.join('?' * len(set(data.gd_game)))}) AND p.scrimmage_play=1""",
        conn, params=[int(g) for g in sorted(set(data.gd_game))])
    pos = pd.read_sql_query("SELECT player_id, position FROM players", conn).set_index("player_id").position
    gmap = {"QB": "QB", "RB": "RB", "REC": "REC"}
    plays["gd"] = [data.gd_index.get((int(g), int(t)), -1) for g, t in zip(plays.game_id, plays.offense_team_id)]
    plays = plays[plays.gd >= 0]
    y_of_gd = pd.Series(data.gd_y, index=np.arange(data.n_gd))
    plays["y"] = plays.gd.map(y_of_gd)
    team_of_gd = pd.Series(data.gd_i, index=np.arange(data.n_gd))

    def role_counts(frame, col, group, role):
        f = frame[frame[col].notna()].copy()
        f["player_id"] = f[col].astype(str)
        f["group"] = f.player_id.map(pos)
        f = f[f.group == group]
        c = f.groupby(["gd", "player_id"]).size().rename("c").reset_index()
        c["role"] = role
        c["group"] = group
        return c

    scrim = plays
    pass_np = scrim[(scrim.is_pass == 1) & (scrim.sack == 0)]
    dropbacks = scrim[scrim.is_pass == 1]
    rush = scrim[scrim.is_rush == 1]
    den = {"dropback": dropbacks.groupby("gd").size(), "target": pass_np.groupby("gd").size(), "carry": rush.groupby("gd").size()}
    counts = [
        role_counts(dropbacks, "passer_id", "QB", "dropback"),
        role_counts(pass_np, "receiver_id", "REC", "target"),
        role_counts(rush, "rusher_id", "QB", "carry_qb"),
        role_counts(rush, "rusher_id", "RB", "carry_rb"),
        role_counts(pass_np, "receiver_id", "RB", "rb_target"),
    ]
    rc = pd.concat(counts, ignore_index=True)
    rb_den = rc[rc.role == "rb_target"].groupby("gd").c.sum()
    def denom(row_role, g):
        if row_role in ("carry_qb", "carry_rb"):
            return den["carry"].reindex(g).to_numpy()
        if row_role == "rb_target":
            return rb_den.reindex(g).to_numpy()
        return den[row_role].reindex(g).to_numpy()
    rc["w"] = 0.0
    for role in rc.role.unique():
        m = rc.role == role
        rc.loc[m, "w"] = rc.loc[m, "c"].to_numpy() / np.maximum(denom(role, rc.loc[m, "gd"].to_numpy()), 1)
    rc["y"] = rc.gd.map(y_of_gd)

    S = len(sp.STREAMS)
    for group, comps in sp.PLAYER_GROUPS.items():
        usage_role = {"QB": "dropback", "RB": "carry_rb", "REC": "target"}[group]
        mine = rc[(rc.group == group)]
        # nodes: one per (player, season) seen in any role of this group
        seen = mine.groupby(["player_id", "y"]).agg(total=("c", "sum"), games=("gd", "nunique")).reset_index()
        # The last (target) season also carries a node for every prior-season player not yet seen in it, so
        # forecasts for that season draw their states from the same model (parent link, no observations).
        ylast = len(seasons) - 1
        if ylast > 0:
            prev = seen[seen.y == ylast - 1]
            add = prev[~prev.player_id.isin(set(seen[seen.y == ylast].player_id))].copy()
            add["y"], add["total"], add["games"] = ylast, 0, 0
            seen = pd.concat([seen, add], ignore_index=True)
        seen = seen.sort_values(["y", "player_id"]).reset_index(drop=True)
        seen["node"] = np.arange(len(seen))
        key = {(r.player_id, int(r.y)): int(r.node) for r in seen.itertuples(index=False)}
        team_seen = mine.assign(team=mine.gd.map(team_of_gd)).groupby(["player_id", "y"]).team.agg(lambda s: s.mode().iat[0])

        class _Teams:
            def get(self, k, default=None):
                return team_seen.get(k, team_seen.get((k[0], k[1] - 1), default))
        team_modal = _Teams()
        parent = np.array([key.get((r.player_id, int(r.y) - 1), -1) for r in seen.itertuples(index=False)])
        team_tot = rc[(rc.group == group) & (rc.role == usage_role)].assign(team=lambda f: f.gd.map(team_of_gd)).groupby(["team", "y"]).c.sum()
        prior_share = np.zeros(len(seen))
        for n_, r in enumerate(seen.itertuples(index=False)):
            if parent[n_] >= 0:
                pr = seen.iloc[parent[n_]]
                tm = team_modal.get((pr.player_id, int(pr.y)))
                tot = team_tot.get((tm, int(pr.y)), np.nan)
                prior_share[n_] = pr.total / tot if tot and not np.isnan(tot) else 0.0
        data.nodes[group] = dict(player_id=seen.player_id.to_numpy(), y=seen.y.to_numpy(), parent=parent, cov=prior_share.reshape(-1, 1),
                                 team=np.array([int(team_modal.get((r.player_id, int(r.y)), -1)) for r in seen.itertuples(index=False)]),
                                 n_obs=seen.games.to_numpy())
        # additive-effect entries on design rows
        rows, nodes, comps, ws = [], [], [], []
        mine = mine.assign(node=[key[(p, int(y))] for p, y in zip(mine.player_id, mine.y)])
        for s_i, spec in enumerate(sp.STREAMS):
            for pg, comp, role in spec.players:
                if pg != group:
                    continue
                sel = mine[mine.role == role]
                rows.append(sel.gd.to_numpy() * S + s_i)
                nodes.append(sel.node.to_numpy())
                comps.append(np.full(len(sel), comps_index(group, comp)))
                ws.append(sel.w.to_numpy(float))
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
        data.entries[group] = dict(row=cat(rows, np.int64), node=cat(nodes, np.int64), comp=cat(comps, np.int64), w=cat(ws, float))
        # usage (multinomial over participants) and participation (Bernoulli over team games)
        use = mine[mine.role == usage_role].groupby(["gd", "node"]).c.sum().reset_index()
        data.usage[group] = dict(seg=use.gd.to_numpy(), node=use.node.to_numpy(), count=use.c.to_numpy(float), n_seg=data.n_gd)
        appear = {(int(g), int(n)) for g, n in zip(use.gd, use.node)}
        by_team_year: dict = {}
        for g in range(data.n_gd):
            by_team_year.setdefault((int(data.gd_i[g]), int(data.gd_y[g])), []).append(g)
        part_node, part_gd, part_z = [], [], []
        active = seen[seen.games >= min_participation]
        for r in active.itertuples(index=False):
            tm = data.nodes[group]["team"][r.node]
            for g in by_team_year.get((tm, int(r.y)), []):
                part_node.append(r.node)
                part_gd.append(g)
                part_z.append(1.0 if (g, r.node) in appear else 0.0)
        data.participation[group] = dict(node=np.array(part_node, np.int64), gd=np.array(part_gd, np.int64), z=np.array(part_z, float))

    # roster-change covariates: returning offensive usage and quarterback change
    opp = rc[rc.role.isin(["dropback", "carry_rb", "target"])].assign(team=lambda f: f.gd.map(team_of_gd))
    R = data.R
    names = {n: i for i, n in enumerate(sp.TEAM_COVARIATES)}
    totals = opp.groupby(["team", "y"]).c.sum()
    tp = opp.groupby(["team", "y", "player_id"]).c.sum().reset_index()
    for (team, y), tot in totals.items():
        if y == 0 or team < 0:
            continue
        prev = tp[(tp.team == team) & (tp.y == y - 1)]
        now = set(tp[(tp.team == team) & (tp.y == y)].player_id)
        if prev.c.sum() > 0:
            R[team, y, names["returning_usage_offense"]] = prev[prev.player_id.isin(now)].c.sum() / prev.c.sum()
    qb = opp[opp.role == "dropback"].groupby(["team", "y", "player_id"]).c.sum().reset_index()
    top = qb.sort_values("c", ascending=False).drop_duplicates(["team", "y"]).set_index(["team", "y"]).player_id
    for (team, y), pid in top.items():
        if y > 0 and (team, y - 1) in top.index:
            R[team, y, names["qb_change"]] = float(top[(team, y - 1)] != pid)
    rp = pd.read_sql_query("SELECT season, team_id, total_ppa FROM returning_production", conn)
    for r in rp.itertuples(index=False):
        if r.season in seasons and r.team_id in data.team_index and not pd.isna(r.total_ppa):
            R[data.team_index[r.team_id], seasons.index(r.season), names["returning_production_ppa"]] = r.total_ppa
    for k in range(R.shape[2]):
        col = R[:, 1:, k]
        sd = col.std()
        if sd > 0:
            R[:, 1:, k] = (col - col.mean()) / sd


def comps_index(group, comp):
    return sp.PLAYER_GROUPS[group].index(comp)


def _attach_possessions(conn, data: ModelData, seasons, gd):
    d = pd.read_sql_query(
        f"""SELECT dr.drive_id, dr.game_id, dr.team_id, dr.outcome_class, dr.points, dr.opponent_points FROM drives dr
            WHERE dr.source_id='sportsdataverse' AND dr.game_id IN ({','.join('?' * len(set(data.gd_game)))})""",
        conn, params=[int(g) for g in sorted(set(data.gd_game))])
    plays = pd.read_sql_query(
        f"""SELECT drive_id, play_id, yards_to_goal FROM plays WHERE scrimmage_play=1 AND game_id IN ({','.join('?' * len(set(data.gd_game)))})""",
        conn, params=[int(g) for g in sorted(set(data.gd_game))])
    plays["pid"] = pd.to_numeric(plays.play_id, errors="coerce")
    first = plays.sort_values(["drive_id", "pid"]).groupby("drive_id").yards_to_goal.first()
    d["ytg"] = d.drive_id.map(first).astype(float).fillna(75.0)
    d["gd"] = [data.gd_index.get((int(g), int(t)), -1) for g, t in zip(d.game_id, d.team_id)]
    d = d[(d.gd >= 0)]
    games_total = d.groupby("game_id").size()
    d = d[d.outcome_class.isin(sp.OUTCOMES)]
    cls = {c: n for n, c in enumerate(sp.OUTCOMES)}
    data.drive_gd = d.gd.to_numpy()
    data.drive_class = d.outcome_class.map(cls).to_numpy()
    data.drive_fp = ((d.ytg.to_numpy() - 75.0) / 25.0).astype(float)
    near = lambda v, bins: int(np.argmin(np.abs(np.asarray(bins) - v)))
    data.td_idx = np.array([near(p, sp.TD_POINTS) if c == cls["touchdown"] else -1 for p, c in zip(d.points, data.drive_class)])
    data.rare_off_idx = np.array([near(p, sp.RARE_POINTS) if c == cls["rare_score"] else -1 for p, c in zip(d.points, data.drive_class)])
    data.rare_opp_idx = np.array([near(p, sp.RARE_POINTS) if c == cls["rare_score"] else -1 for p, c in zip(d.opponent_points.fillna(0), data.drive_class)])
    by_game = {}
    for g, k in data.gd_index.items():
        by_game.setdefault(g[0], []).append(k)
    pairs = [(sorted(v), g) for g, v in by_game.items() if len(v) == 2 and g in games_total.index]
    data.game_gd1 = np.array([p[0][0] for p in pairs])
    data.game_gd2 = np.array([p[0][1] for p in pairs])
    data.game_drives = np.array([int(games_total[p[1]]) for p in pairs])


def _attach_ratings(conn, data: ModelData, cutoff):
    q = "SELECT season, week, team_id, rating_source, rating, available_at FROM external_ratings WHERE rating IS NOT NULL"
    r = pd.read_sql_query(q, conn)
    keep = r.season.isin(data.seasons) & r.team_id.isin(data.team_index)
    if cutoff:
        keep &= pd.to_datetime(r.available_at, utc=True, errors="coerce") < pd.Timestamp(cutoff).tz_convert("UTC")
    r = r[keep]
    srcs = {s: n for n, s in enumerate(sp.RATING_SOURCES)}
    r = r[r.rating_source.isin(srcs)]
    wk = r.week.fillna(sp.WEEKS).clip(1, sp.WEEKS).astype(int)
    data.rating_team = r.team_id.map(data.team_index).to_numpy(int)
    data.rating_per = (r.season.map({s: n for n, s in enumerate(data.seasons)}) * sp.WEEKS + wk - 1).to_numpy(int)
    data.rating_src = r.rating_source.map(srcs).to_numpy(int)
    data.rating_val = r.rating.to_numpy(float)
