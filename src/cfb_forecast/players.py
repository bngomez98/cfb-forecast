"""Hierarchical dynamic player ratings from play-by-play.

This is the player layer of the forecasting model. It estimates, from the
yardage of individual rush and pass plays, a latent offensive rating for every
ball-carrier, pooled toward their team and the league (a Bayesian hierarchy),
adjusted for the defense faced, and discounted by how long ago each play
happened (the dynamic, continuously-updated part).

Model, per role r in {pass, rush}, for a play with ball-carrier j on team t
against defense d, on standardized yards y:

    y = alpha_j + beta_d + epsilon,    epsilon ~ N(0, sigma_r^2)
    alpha_j ~ N(theta_t, tau_player^2)       players pooled toward their team
    theta_t ~ N(0,       tau_team^2)          teams pooled toward the league
    beta_d  ~ N(0,       tau_def^2)           defenses pooled toward the league

The posterior means are obtained by conjugate Gaussian shrinkage, iterated a
few times between the player/team offense block and the defense block
(coordinate ascent). Each play carries weight ``decay ** weeks_before_cutoff``,
so the estimate is an exponential-forgetting state filter on a slowly varying
latent rather than a flat average: this is the "dynamic continuous" state.

Everything is point-in-time: a rating as of a cutoff uses only plays from games
whose result was available strictly before the cutoff (kickoff + 8 hours). A team's pre-kickoff offensive
rating is the usage-weighted average of its expected players' alphas, where
expected usage comes from the share of recent plays each player took.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RESULT_LAG, require_columns, timestamp
from .logs import get

log = get("players")

ROLES = ("pass", "rush")
# Yardage clips keep a single explosive play from dominating a player's standardized mean.
CLIP = {"pass": (-15.0, 75.0), "rush": (-10.0, 60.0)}
RAW_COLUMNS = ["game_id", "pos_team_id", "def_pos_team_id", "rush", "pass", "sack", "statYardage",
               "passer_player_id", "rusher_player_id"]
NAME_COLUMNS = ["passer_player_name", "rusher_player_name"]  # optional; used only to label the published tables


@dataclass(frozen=True)
class PlayerConfig:
    tau_player: float = 1.0     # prior SD of a player around their team (standardized units)
    tau_team: float = 0.6       # prior SD of a team around the league
    tau_def: float = 0.5        # prior SD of a defense around the league
    half_life_weeks: float = 8.0  # recency: a play this many weeks old counts half
    iterations: int = 6
    min_player_plays: float = 8.0   # effective plays below which a player stays near its team mean

    def decay(self):
        return 0.5 ** (1.0 / self.half_life_weeks)

    def to_dict(self):
        return asdict(self)


def _flag(series):
    return series.astype(str).str.lower().isin(["true", "1"])


def player_plays(plays):
    """Offensive rush/pass plays reduced to one row per play with its ball-carrier.

    A pass (including a sack, charged to the passer) is credited to ``passer_player_id``;
    a run to ``rusher_player_id``. Plays without a usable carrier or yardage are dropped.
    """
    require_columns(plays, RAW_COLUMNS, "play-by-play")
    p = plays.reindex(columns=RAW_COLUMNS + NAME_COLUMNS).copy()
    for c in ("rush", "pass", "sack"):
        p[c] = _flag(p[c])
    p["statYardage"] = pd.to_numeric(p.statYardage, errors="coerce")
    for c in ("game_id", "pos_team_id", "def_pos_team_id", "passer_player_id", "rusher_player_id"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    is_pass = p["pass"] | p.sack
    rows = []
    for role, mask, carrier, name_col in (("pass", is_pass, "passer_player_id", "passer_player_name"),
                                          ("rush", p.rush & ~is_pass, "rusher_player_id", "rusher_player_name")):
        part = p[mask & p[carrier].notna() & p.pos_team_id.notna() & p.def_pos_team_id.notna() & p.statYardage.notna()].copy()
        if part.empty:
            continue
        lo, hi = CLIP[role]
        rows.append(pd.DataFrame({
            "game_id": part.game_id.astype("int64"), "role": role,
            "player_id": part[carrier].astype("int64"),
            "player_name": part[name_col].fillna("").astype(str),
            "team_id": part.pos_team_id.astype("int64"), "def_id": part.def_pos_team_id.astype("int64"),
            "yards": part.statYardage.clip(lo, hi).astype(float),
        }))
    if not rows:
        return pd.DataFrame(columns=["game_id", "role", "player_id", "player_name", "team_id", "def_id", "yards"])
    return pd.concat(rows, ignore_index=True)


def player_game_aggregates(plays, schedule):
    """Per (game, player, role): plays, total and mean yards, team, opponent, kickoff.

    This is the compact, reusable artifact; the per-play file never has to be kept.
    """
    pp = player_plays(plays)
    if pp.empty:
        return pp
    sched = schedule.loc[:, ["game_id", "season", "week", "start_date"]].copy()
    sched["game_id"] = pd.to_numeric(sched.game_id, errors="coerce").astype("int64")
    sched["start_date"] = pd.to_datetime(sched.start_date, utc=True)
    agg = pp.groupby(["game_id", "role", "player_id", "team_id", "def_id"], as_index=False).agg(
        player_name=("player_name", "first"), plays=("yards", "size"), yards=("yards", "sum"), yards_per_play=("yards", "mean"))
    return agg.merge(sched, on="game_id", how="inner").sort_values(["start_date", "game_id"]).reset_index(drop=True)


def _shrink(sum_w, sum_wy, prior_mean, prior_pseudocount):
    """Posterior mean of a Gaussian mean: (Σw·y + κ·prior) / (Σw + κ)."""
    return (sum_wy + prior_pseudocount * prior_mean) / (sum_w + prior_pseudocount)


def fit_player_ratings(aggregates, cutoff, config=None, roles=ROLES):
    """Fit hierarchical, opponent-adjusted, recency-weighted player/team/defense ratings.

    Only games whose kickoff plus the result lag is strictly before ``cutoff`` contribute. Returns a dict
    with per-role player, team-offense and defense rating tables (standardized units;
    positive offense = more yards than league average, positive defense = fewer allowed).
    """
    config = config or PlayerConfig()
    cutoff = timestamp(cutoff)
    out = {"cutoff": cutoff.isoformat(), "config": config.to_dict(), "players": {}, "teams": {}, "defense": {}, "league": {}}
    if aggregates.empty:
        return out
    # Same availability rule as the team states: a game counts only once kickoff + lag is before the cutoff.
    a = aggregates[pd.to_datetime(aggregates.start_date, utc=True) + RESULT_LAG < cutoff].copy()
    if a.empty:
        return out
    weeks_old = (cutoff - pd.to_datetime(a.start_date, utc=True)).dt.total_seconds() / 604800.0
    a["w"] = config.decay() ** weeks_old.clip(lower=0) * a.plays  # weight by recency and volume
    decay, players, teams, defense, league = config.decay(), {}, {}, {}, {}
    for role in roles:
        r = a[a.role == role]
        if r.empty:
            continue
        mu = np.average(r.yards_per_play, weights=r.w)
        sd = np.sqrt(np.average((r.yards_per_play - mu) ** 2, weights=r.w)) or 1.0
        league[role] = {"mean_ypp": float(mu), "sd_ypp": float(sd), "games": int(r.game_id.nunique())}
        # standardized per-game residual y and its weight for every (player, team, def) row
        y = (r.yards_per_play - mu) / sd
        kappa_player = config.min_player_plays / sd ** 2 if sd else config.min_player_plays
        kappa_team = (1.0 / config.tau_team ** 2)
        kappa_def = (1.0 / config.tau_def ** 2)
        kappa_pl = (1.0 / config.tau_player ** 2)
        beta = pd.Series(0.0, index=sorted(r.def_id.unique()))
        theta = pd.Series(0.0, index=sorted(r.team_id.unique()))
        alpha = pd.Series(0.0, index=sorted(r.player_id.unique()))
        df = pd.DataFrame({"player": r.player_id.to_numpy(), "team": r.team_id.to_numpy(),
                           "defn": r.def_id.to_numpy(), "y": y.to_numpy(), "w": r.w.to_numpy()})
        w = df.w
        team_key, player_key, def_key = [df.team], [df.team, df.player], [df.defn]
        sw_team = w.groupby(team_key).sum()
        sw_def = w.groupby(def_key).sum()
        sw_player = w.groupby(player_key).sum()
        for _ in range(config.iterations):
            adj = df.y - df.defn.map(beta).to_numpy()
            swy_team = (w * adj).groupby(team_key).sum()
            swy_player = (w * adj).groupby(player_key).sum()
            theta = _shrink(sw_team, swy_team, 0.0, kappa_team)
            prior = sw_player.index.get_level_values(0).map(theta)
            alpha = pd.Series(_shrink(sw_player.to_numpy(), swy_player.to_numpy(), prior.to_numpy(), kappa_pl),
                              index=sw_player.index.get_level_values(1))
            if alpha.index.has_duplicates:  # a player who changed teams midseason: average the stints
                alpha = alpha.groupby(level=0).mean()
            adj_d = df.y - df.player.map(alpha).to_numpy()
            swy_def = (w * adj_d).groupby(def_key).sum()
            beta = _shrink(sw_def, swy_def, 0.0, kappa_def)
        players[role] = {int(k): float(v) for k, v in alpha.items()}
        teams[role] = {int(k): float(v) for k, v in theta.items()}
        defense[role] = {int(k): float(-v) for k, v in beta.items()}  # + defense = fewer yards allowed
    out.update(players=players, teams=teams, defense=defense, league=league)
    _ = decay, kappa_player
    return out


def recent_usage(aggregates, cutoff, lookback_games=4):
    """Each team's expected player usage share (by plays) over its most recent games before cutoff."""
    cutoff = timestamp(cutoff)
    a = aggregates[pd.to_datetime(aggregates.start_date, utc=True) + RESULT_LAG < cutoff]
    usage = {}
    for (team, role), g in a.groupby(["team_id", "role"]):
        recent_games = g.sort_values("start_date").game_id.drop_duplicates().tail(lookback_games)
        gg = g[g.game_id.isin(recent_games)]
        shares = gg.groupby("player_id").plays.sum()
        total = shares.sum()
        if total > 0:
            usage[(int(team), role)] = (shares / total).to_dict()
    return usage


def team_offense_ratings(ratings, usage):
    """Usage-weighted team offensive rating per role from fitted player alphas.

    Players with no fitted alpha (new/returning) fall back to their team's theta,
    so a team that loses a high-alpha player to absent recent usage is rated lower.
    """
    columns = ["team_id", "pass_offense", "rush_offense", "player_offense"]
    rows = []
    teams = {t for t, _ in usage}
    if not teams:
        return pd.DataFrame(columns=columns)
    for team in sorted(teams):
        row = {"team_id": team}
        for role in ROLES:
            share = usage.get((team, role), {})
            alpha = ratings.get("players", {}).get(role, {})
            theta = ratings.get("teams", {}).get(role, {}).get(team, 0.0)
            if share:
                row[f"{role}_offense"] = float(sum(w * alpha.get(int(p), theta) for p, w in share.items()))
            else:
                row[f"{role}_offense"] = float(theta)
        row["player_offense"] = 0.6 * row.get("pass_offense", 0.0) + 0.4 * row.get("rush_offense", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_player_layer(aggregates, cutoff, config=None, lookback_games=4):
    """Fit ratings as of ``cutoff`` and roll them up to per-team offensive ratings."""
    ratings = fit_player_ratings(aggregates, cutoff, config)
    usage = recent_usage(aggregates, cutoff, lookback_games)
    team = team_offense_ratings(ratings, usage)
    return ratings, team


def team_defense_rating(ratings, team_id):
    dp = ratings.get("defense", {}).get("pass", {})
    dr = ratings.get("defense", {}).get("rush", {})
    return 0.6 * dp.get(int(team_id), 0.0) + 0.4 * dr.get(int(team_id), 0.0)


def player_features_by_week(aggregates, games, lead_hours=48.0, config=None, lookback_games=4):
    """Per-game home/away player offense and defense ratings, fit point-in-time by week.

    The player layer is re-fit once per (season, week) at that week's first kickoff
    minus ``lead_hours``, using only earlier games, and every game in the week reads
    those ratings. Returns a frame indexed by game_id with four columns plus the net
    offense-minus-defense signal, matching the 26-column team-state feature layout so
    the score head can consume them as extra inputs.
    """
    config = config or PlayerConfig()
    g = games.copy()
    g["start_date"] = pd.to_datetime(g.start_date, utc=True)
    rows = []
    for (season, week), wk in g.groupby(["season", "week"]):
        cutoff = wk.start_date.min() - pd.Timedelta(hours=lead_hours)
        ratings = fit_player_ratings(aggregates, cutoff, config)
        usage = recent_usage(aggregates, cutoff, lookback_games)
        team = team_offense_ratings(ratings, usage).set_index("team_id")
        offense = team.player_offense if "player_offense" in team else pd.Series(dtype=float)
        for r in wk.itertuples(index=False):
            ho = float(offense.get(int(r.home_id), 0.0))
            ao = float(offense.get(int(r.away_id), 0.0))
            hd = team_defense_rating(ratings, r.home_id)
            ad = team_defense_rating(ratings, r.away_id)
            rows.append({"game_id": int(r.game_id),
                         "home_player_offense": ho, "away_player_offense": ao,
                         "home_player_defense": hd, "away_player_defense": ad,
                         "player_net_signal": (ho - ad) - (ao - hd)})
    cols = ["game_id", "home_player_offense", "away_player_offense", "home_player_defense", "away_player_defense", "player_net_signal"]
    return pd.DataFrame(rows, columns=cols).set_index("game_id")


def player_feature_matrix(game_ids, features_by_week):
    """Two-sided (n, 2, 2) player feature block aligned to a list of game_ids.

    Column 0 is the team's own player offense, column 1 the opponent's player defense,
    for home (side 0) and away (side 1); missing games default to league-average zero.
    """
    f = features_by_week.reindex([int(g) for g in game_ids]).fillna(0.0)
    block = np.zeros((len(f), 2, 2))
    block[:, 0, 0] = f.home_player_offense.to_numpy()
    block[:, 0, 1] = f.away_player_defense.to_numpy()
    block[:, 1, 0] = f.away_player_offense.to_numpy()
    block[:, 1, 1] = f.home_player_defense.to_numpy()
    return block


def _available(aggregates, cutoff):
    cutoff = timestamp(cutoff)
    return aggregates[pd.to_datetime(aggregates.start_date, utc=True) + RESULT_LAG < cutoff]


def player_table(aggregates, ratings, cutoff, team_names=None, min_plays=25, lookback_games=4):
    """Named player ratings as of ``cutoff``: one row per (player, role) with enough plays.

    ``rating`` is the pooled, defense-adjusted player effect in standardized yards-per-play
    units (positive = better than the league). ``usage_share`` is the player's share of
    their team's recent plays in that role and ``recent_plays`` the plays behind it, so a
    high rating with no recent usage is visible. ``team_id`` is the player's latest team.
    """
    columns = ["role", "player_id", "player_name", "team_id", "team", "rating", "usage_share", "plays", "recent_plays", "games"]
    a = _available(aggregates, cutoff)
    if a.empty or not ratings.get("players"):
        return pd.DataFrame(columns=columns)
    a = a.sort_values("start_date")
    if "player_name" not in a:
        a = a.assign(player_name="")
    last = a.groupby(["role", "player_id"]).tail(1).set_index(["role", "player_id"])
    totals = a.groupby(["role", "player_id"]).agg(plays=("plays", "sum"), games=("game_id", "nunique"))
    usage = recent_usage(aggregates, cutoff, lookback_games)
    rows = []
    for role, alphas in ratings["players"].items():
        for pid, rating in alphas.items():
            key = (role, int(pid))
            if key not in totals.index:
                continue
            team = int(last.loc[key, "team_id"])
            share = usage.get((team, role), {}).get(int(pid), 0.0)
            rows.append({"role": role, "player_id": int(pid), "player_name": str(last.loc[key, "player_name"]),
                         "team_id": team, "team": (team_names or {}).get(team, str(team)), "rating": float(rating),
                         "usage_share": float(share), "plays": int(totals.loc[key, "plays"]),
                         "recent_plays": int(round(share * a[(a.team_id == team) & (a.role == role)].plays.sum())) if share else 0,
                         "games": int(totals.loc[key, "games"])})
    out = pd.DataFrame(rows, columns=columns)
    out = out[out.plays >= min_plays]
    return out.sort_values(["role", "rating"], ascending=[True, False]).reset_index(drop=True)


def team_table(ratings, usage, team_names=None):
    """Team offense (usage-weighted players) and defense ratings by role, with a net rating."""
    team = team_offense_ratings(ratings, usage)
    columns = ["team_id", "team", "pass_offense", "rush_offense", "player_offense", "player_defense", "net_rating"]
    if team.empty:
        return pd.DataFrame(columns=columns)
    team["player_defense"] = [team_defense_rating(ratings, t) for t in team.team_id]
    team["net_rating"] = team.player_offense + team.player_defense
    team["team"] = team.team_id.map(lambda t: (team_names or {}).get(int(t), str(t)))
    return team[columns].sort_values("net_rating", ascending=False).reset_index(drop=True)


def write_player_report(data_root, out_dir, as_of, team_names=None, config=None):
    """Fit the player layer as of ``as_of`` and write ``player_ratings.csv`` and ``team_ratings.csv``."""
    agg = load_aggregates(data_root)
    if agg.empty:
        return {"players": 0, "teams": 0}
    config = config or PlayerConfig()
    ratings = fit_player_ratings(agg, as_of, config)
    players = player_table(agg, ratings, as_of, team_names)
    teams = team_table(ratings, recent_usage(agg, as_of), team_names)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    players.to_csv(out_dir / "player_ratings.csv", index=False)
    teams.to_csv(out_dir / "team_ratings.csv", index=False)
    return {"players": int(len(players)), "teams": int(len(teams))}


PBP_RELEASE ="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_pbp/play_by_play_{season}.parquet"


def build_aggregates(data_root, seasons, pbp_dir=None, refresh=False):
    """Aggregate raw play-by-play into compact per-player-game files, one per season.

    Reads ``play_by_play_<season>.parquet`` from ``pbp_dir`` (or downloads it to a temp
    file), writes ``player_games_<season>.parquet`` next to the schedules, and deletes
    the large raw file unless it was supplied locally. The raw PBP is never committed.
    """
    import tempfile
    import urllib.request

    import pyarrow.parquet as pq

    from .data import normalize_schedule

    data_root = Path(data_root)
    written = []
    for season in seasons:
        schedule_path = data_root / f"schedule_{int(season)}.csv"
        if not schedule_path.exists():
            raise FileNotFoundError(f"Need {schedule_path} first (run fetch --seasons {season})")
        out = data_root / f"player_games_{int(season)}.parquet"
        if out.exists() and not refresh:
            log.info("player aggregates cached: %s", out)
            written.append(out)
            continue
        local = Path(pbp_dir) / f"play_by_play_{int(season)}.parquet" if pbp_dir else None
        tmp = None
        if local and local.exists():
            source = local
        else:
            tmp = Path(tempfile.mkstemp(suffix=".parquet")[1])
            url = PBP_RELEASE.format(season=int(season))
            log.info("downloading play-by-play %s", season)
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed SportsDataverse release URL
            source = tmp
        try:
            wanted = RAW_COLUMNS + NAME_COLUMNS
            available = set(pq.ParquetFile(source).schema.names)
            plays = pd.read_parquet(source, columns=[c for c in wanted if c in available])
            schedule = normalize_schedule(pd.read_csv(schedule_path, low_memory=False))
            agg = player_game_aggregates(plays, schedule)
            agg.to_parquet(out, index=False)
            log.info("wrote %s (%d player-games)", out, len(agg))
            written.append(out)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    return written


def load_aggregates(data_root, seasons=None):
    """Read cached per-player-game aggregates (written by the ``player-aggregates`` command)."""
    data_root = Path(data_root)
    paths = sorted(data_root.glob("player_games_*.parquet"))
    if seasons is not None:
        seasons = {int(s) for s in seasons}
        paths = [p for p in paths if any(str(s) in p.stem for s in seasons)]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
