"""Past-game observations taken from the recorded source files."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .config import METRICS
from .forecast_model import HISTORY_COLUMNS

PLAY_RATE_NAMES = (
    "epa",
    "success_rate",
    "explosive_rate",
    "turnover_rate",
    "qb_yards_per_play",
    "qb_epa",
    "qb_success",
    "rb_yards_per_play",
    "rb_epa",
    "wr_yards_per_play",
    "wr_epa",
    "dl_epa",
    "db_epa",
)
SOURCE_PLAY_COLUMNS = (
    "game_id",
    "pos_team_id",
    "def_pos_team_id",
    "statYardage",
    "pass",
    "rush",
    "sack",
    "kneel_down",
    "EPA",
    "EPA_success",
    "is_turnover",
    "passer_player_name",
    "rusher_player_name",
    "receiver_player_name",
    "sack_player_name",
    "sack_player_name2",
    "interception_player_name",
    "def_EPA",
)

def observations(games, summaries):
    """No provider EPA, win probability, final-season ratings or betting inputs."""
    Y = np.full((len(games), 2, len(METRICS)), np.nan)
    Y[:,0,0] = games.home_points.to_numpy()
    Y[:,1,0] = games.away_points.to_numpy()
    if len(summaries):
        z = summaries.set_index(["game_id", "pos_team_id"])
        for side, team in enumerate(("home_id", "away_id")):
            ix = pd.MultiIndex.from_arrays([games.game_id, games[team]])
            Y[:,side,1:] = z.reindex(ix)[list(METRICS[1:])].to_numpy(dtype=float)
    Y[:,:,1] = np.clip(Y[:,:,1],0,15)
    Y[:,:,2] = np.clip(Y[:,:,2],0,20)
    Y[:,:,3] = np.clip(Y[:,:,3],-2,15)
    Y[:,:,4] = np.clip(Y[:,:,4],0,1)
    return Y

def _flag(frame, column):
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)

def _mean_table(frame, team_column, value_column, mask=None):
    """Mean of one recorded column for each game and team. Missing stays missing."""
    if frame is None or len(frame) == 0 or "game_id" not in frame.columns:
        return {}
    if team_column not in frame.columns or value_column not in frame.columns:
        return {}
    part = frame if mask is None else frame.loc[mask]
    if part.empty:
        return {}
    values = pd.to_numeric(part[value_column], errors="coerce")
    teams = pd.to_numeric(part[team_column], errors="coerce")
    games = pd.to_numeric(part["game_id"], errors="coerce")
    keep = values.notna() & teams.notna() & games.notna()
    if not keep.any():
        return {}
    grouped = (
        pd.DataFrame({
            "game_id": games[keep].astype("int64"),
            "team_id": teams[keep].astype("int64"),
            "value": values[keep].astype(float),
        })
        .groupby(["game_id", "team_id"], as_index=False)["value"]
        .mean()
    )
    return {
        (int(row.game_id), int(row.team_id)): float(row.value)
        for row in grouped.itertuples(index=False)
    }

def _explosive_table(frame):
    if "statYardage" not in frame.columns:
        return {}
    yards = pd.to_numeric(frame["statYardage"], errors="coerce")
    keep = yards.notna()
    if not keep.any():
        return {}
    is_pass = _flag(frame, "pass") | _flag(frame, "sack")
    recorded = frame.loc[keep].copy()
    recorded["_explosive"] = np.where(is_pass.loc[keep], yards.loc[keep].ge(20), yards.loc[keep].ge(10)).astype(float)
    return _mean_table(recorded, "pos_team_id", "_explosive")

def play_rate_observations(games, plays):
    """Per-play rates from recorded source plays, including kneel-downs.

    Yards, EPA, and success are means on the plays that recorded them. A kneel
    stays on the rusher. A team or position with no recorded play stays missing.
    """
    if plays is None or len(plays) == 0 or games is None or len(games) == 0:
        return None
    if "game_id" not in plays.columns or "pos_team_id" not in plays.columns:
        return None
    if "game_id" not in games.columns or "home_id" not in games.columns or "away_id" not in games.columns:
        return None
    kneel = _flag(plays, "kneel_down")
    rush = _flag(plays, "rush")
    passed = _flag(plays, "pass") if "pass" in plays.columns else pd.Series(True, index=plays.index)
    passer = plays["passer_player_name"].notna() if "passer_player_name" in plays.columns else pd.Series(False, index=plays.index)
    rusher = plays["rusher_player_name"].notna() if "rusher_player_name" in plays.columns else pd.Series(False, index=plays.index)
    receiver = plays["receiver_player_name"].notna() if "receiver_player_name" in plays.columns else pd.Series(False, index=plays.index)
    sack = _flag(plays, "sack")
    if "sack_player_name" in plays.columns:
        sack = sack | plays["sack_player_name"].notna()
    if "sack_player_name2" in plays.columns:
        sack = sack | plays["sack_player_name2"].notna()
    interception = plays["interception_player_name"].notna() if "interception_player_name" in plays.columns else pd.Series(False, index=plays.index)
    lookups = {
        "epa": _mean_table(plays, "pos_team_id", "EPA"),
        "success_rate": _mean_table(plays, "pos_team_id", "EPA_success"),
        "explosive_rate": _explosive_table(plays),
        "turnover_rate": _mean_table(plays, "pos_team_id", "is_turnover"),
        "qb_yards_per_play": _mean_table(plays, "pos_team_id", "statYardage", passer & passed),
        "qb_epa": _mean_table(plays, "pos_team_id", "EPA", passer & passed),
        "qb_success": _mean_table(plays, "pos_team_id", "EPA_success", passer & passed),
        "rb_yards_per_play": _mean_table(plays, "pos_team_id", "statYardage", rusher & (rush | kneel)),
        "rb_epa": _mean_table(plays, "pos_team_id", "EPA", rusher & (rush | kneel)),
        "wr_yards_per_play": _mean_table(plays, "pos_team_id", "statYardage", receiver),
        "wr_epa": _mean_table(plays, "pos_team_id", "EPA", receiver),
        "dl_epa": _mean_table(plays, "def_pos_team_id", "def_EPA", sack),
        "db_epa": _mean_table(plays, "def_pos_team_id", "def_EPA", interception),
    }
    rates = np.full((len(games), 2, len(PLAY_RATE_NAMES)), np.nan)
    game_ids = pd.to_numeric(games.game_id, errors="coerce").to_numpy()
    home_ids = pd.to_numeric(games.home_id, errors="coerce").to_numpy()
    away_ids = pd.to_numeric(games.away_id, errors="coerce").to_numpy()
    for metric, table in lookups.items():
        if not table:
            continue
        column = PLAY_RATE_NAMES.index(metric)
        for row, game_id, home_id, away_id in zip(range(len(games)), game_ids, home_ids, away_ids):
            if not np.isfinite(game_id):
                continue
            game_key = int(game_id)
            if np.isfinite(home_id):
                value = table.get((game_key, int(home_id)))
                if value is not None:
                    rates[row, 0, column] = value
            if np.isfinite(away_id):
                value = table.get((game_key, int(away_id)))
                if value is not None:
                    rates[row, 1, column] = value
    if not np.isfinite(rates).any():
        return None
    return rates

def load_source_plays(data_root):
    """Read stored play-by-play files. Kneel rows are kept. A bad file is skipped."""
    root = Path(data_root)
    if not root.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(root.glob("pbp_*.parquet")):
        try:
            columns = _existing_columns(path, SOURCE_PLAY_COLUMNS)
            if "game_id" not in columns or "pos_team_id" not in columns:
                continue
            frames.append(pd.read_parquet(path, columns=columns))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def extend_standardization(center, scale, observations_y, warmup_mask):
    """Keep the saved location for the original metrics and scale any added rates."""
    center = np.asarray(center, float)
    scale = np.asarray(scale, float)
    width = int(observations_y.shape[-1])
    if len(center) == width and len(scale) == width:
        return center, scale
    if len(center) > width or len(scale) != len(center):
        raise ValueError("Saved state does not match the observation width")
    extra_center = []
    extra_scale = []
    warmup = np.asarray(warmup_mask, bool)
    for index in range(len(center), width):
        sample = observations_y[warmup, :, index]
        sample = sample[np.isfinite(sample)]
        if len(sample) < 2 or float(np.std(sample)) < 1e-8:
            extra_center.append(0.0)
            extra_scale.append(1.0)
        else:
            extra_center.append(float(np.mean(sample)))
            extra_scale.append(float(np.std(sample)))
    return np.r_[center, extra_center], np.r_[scale, extra_scale]

def feature_width(n_metrics):
    return 4 * int(n_metrics) + 6

def normalization(Y, mask):
    center = np.nanmean(Y[mask], axis=(0,1))
    scale = np.nanstd(Y[mask], axis=(0,1))
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale < 1e-8).any():
        raise ValueError("Warmup data must contain variable observations for every metric")
    return center, scale

def _year_from_name(path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def _fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _existing_columns(path, wanted):
    import pyarrow.parquet as pq
    names = set(pq.read_schema(path).names)
    return [column for column in wanted if column in names]


def _facts_from_database(database):
    if not database:
        return pd.DataFrame(), pd.DataFrame()
    try:
        from contextlib import closing
        from .database import connect
        with closing(connect(database)) as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            teams = pd.read_sql_query("SELECT * FROM team_game_facts", conn) if "team_game_facts" in names else pd.DataFrame()
            players = pd.read_sql_query("SELECT * FROM player_game_facts", conn) if "player_game_facts" in names else pd.DataFrame()
            return teams, players
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _combine_facts(base, stored, keys):
    if stored is None or len(stored) == 0:
        return base
    if base is None or len(base) == 0:
        return stored
    combined = pd.concat([base, stored], ignore_index=True, sort=False)
    return combined.drop_duplicates(keys, keep="last")


def _facts_from_files(data_root):
    """Clean historical parquet into team and player game rows. This is ingest, not the forecast."""
    root = Path(data_root)
    paths = sorted(
        path for path in list(root.glob("pbp_*.parquet")) + list(root.glob("adv_team_*.parquet"))
        if (_year_from_name(path) or 0) >= 2000
    )
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    cache = root / "_history"
    signature = _fingerprint(paths)
    team_cache = cache / "team_game_facts.parquet"
    player_cache = cache / "player_game_facts.parquet"
    stamp = cache / "fingerprint.txt"
    if team_cache.exists() and player_cache.exists() and stamp.exists() and stamp.read_text().strip() == signature:
        return pd.read_parquet(team_cache), pd.read_parquet(player_cache)
    team_frames = []
    player_frames = []
    for path in paths:
        year = _year_from_name(path)
        if "adv_team" in path.name:
            columns = _existing_columns(path, ["game_id", "pos_team_id", "season", "week", "yards_per_play", "yards_per_pass", "yards_per_rush", "first_downs_created_rate"])
            if "game_id" not in columns or "pos_team_id" not in columns:
                continue
            frame = pd.read_parquet(path, columns=columns)
            frame = frame.rename(columns={"pos_team_id": "team_id"})
            if "season" not in frame:
                frame["season"] = year
            team_frames.append(frame)
            continue
        columns = _existing_columns(
            path,
            [
                "game_id", "game_play_number", "pos_team_id", "season", "week", "statYardage",
                "pass", "rush", "sack", "kneel_down", "EPA", "EPA_success", "is_turnover", "scrimmage_play",
                "passer_player_name", "rusher_player_name", "receiver_player_name",
            ],
        )
        if "game_id" not in columns or "pos_team_id" not in columns or "statYardage" not in columns:
            continue
        frame = pd.read_parquet(path, columns=columns)
        if "game_play_number" in frame:
            frame = frame.drop_duplicates(["game_id", "game_play_number"])
        if "scrimmage_play" in frame:
            recorded = frame.scrimmage_play.fillna(False).astype(bool) | frame.statYardage.notna()
            frame = frame.loc[recorded].copy()
        frame = frame[frame.statYardage.notna() | frame.get("EPA", pd.Series(index=frame.index)).notna()].copy()
        if "season" not in frame:
            frame["season"] = year
        if "pass" in frame:
            frame["pass"] = frame["pass"].fillna(False).astype(bool)
        if "rush" in frame:
            frame["rush"] = frame["rush"].fillna(False).astype(bool)
        if "pass" in frame and "rush" in frame:
            frame["explosive_play"] = frame.statYardage.ge(np.where(frame["pass"], 20, 10))
        else:
            frame["explosive_play"] = frame.statYardage.ge(20)
        grouped = frame.groupby(["game_id", "pos_team_id"], as_index=False)
        team = grouped.agg(
            plays=("statYardage", "size"),
            yards_per_play=("statYardage", "mean"),
            explosive_rate=("explosive_play", "mean"),
        )
        if "EPA" in frame:
            team = team.merge(grouped.agg(epa=("EPA", "mean")), on=["game_id", "pos_team_id"], how="left")
        if "EPA_success" in frame:
            team = team.merge(grouped.agg(success_rate=("EPA_success", "mean")), on=["game_id", "pos_team_id"], how="left")
        if "is_turnover" in frame:
            team = team.merge(grouped.agg(turnover_rate=("is_turnover", "mean")), on=["game_id", "pos_team_id"], how="left")
        if "pass" in frame:
            passed = frame.loc[frame["pass"]].groupby(["game_id", "pos_team_id"], as_index=False).agg(pass_yards=("statYardage", "sum"))
            team = team.merge(passed, on=["game_id", "pos_team_id"], how="left")
            team["pass_yards"] = team["pass_yards"].fillna(0.0)
        if "rush" in frame:
            rushed = frame.loc[frame["rush"]].groupby(["game_id", "pos_team_id"], as_index=False).agg(rush_yards=("statYardage", "sum"))
            team = team.merge(rushed, on=["game_id", "pos_team_id"], how="left")
            team["rush_yards"] = team["rush_yards"].fillna(0.0)
        if "receiver_player_name" in frame:
            caught = frame.loc[frame.receiver_player_name.notna()].groupby(["game_id", "pos_team_id"], as_index=False).agg(receiving_yards=("statYardage", "sum"))
            team = team.merge(caught, on=["game_id", "pos_team_id"], how="left")
        season_week = frame.groupby(["game_id", "pos_team_id"], as_index=False).agg(season=("season", "first"))
        if "week" in frame:
            season_week = season_week.merge(frame.groupby(["game_id", "pos_team_id"], as_index=False).agg(week=("week", "first")), on=["game_id", "pos_team_id"], how="left")
        team = team.merge(season_week, on=["game_id", "pos_team_id"], how="left")
        team = team.rename(columns={"pos_team_id": "team_id"})
        team_frames.append(team)
        for role, name_column, flag in (
            ("passing", "passer_player_name", "pass"),
            ("rushing", "rusher_player_name", "rush"),
            ("receiving", "receiver_player_name", None),
        ):
            if name_column not in frame:
                continue
            role_rows = frame[frame[name_column].notna()].copy()
            if flag and flag in role_rows:
                keep = role_rows[flag].fillna(False).astype(bool)
                if role == "rushing" and "kneel_down" in role_rows.columns:
                    keep = keep | role_rows["kneel_down"].fillna(False).astype(bool)
                role_rows = role_rows.loc[keep]
            if role_rows.empty:
                continue
            player = role_rows.groupby(["game_id", "pos_team_id", name_column], as_index=False).agg(
                plays=("statYardage", "size"),
                yards=("statYardage", "sum"),
                **({"epa": ("EPA", "mean")} if "EPA" in role_rows else {}),
            )
            player = player.rename(columns={"pos_team_id": "team_id", name_column: "player_name"})
            player["position_group"] = role
            player["season"] = year
            player_frames.append(player)
    if not team_frames:
        empty = pd.DataFrame()
        return empty, empty
    teams = pd.concat(team_frames, ignore_index=True)
    teams["game_id"] = pd.to_numeric(teams.game_id, errors="coerce")
    teams["team_id"] = pd.to_numeric(teams.team_id, errors="coerce")
    teams = teams.dropna(subset=["game_id", "team_id"])
    teams["game_id"] = teams.game_id.astype("int64")
    teams["team_id"] = teams.team_id.astype("int64")
    value_columns = [column for column in teams.columns if column not in {"game_id", "team_id", "season", "week"}]
    teams = teams.groupby(["game_id", "team_id"], as_index=False).agg(
        {column: "mean" if column not in {"plays", "pass_yards", "rush_yards", "receiving_yards"} else "sum" for column in value_columns}
        | {"season": "first", **({"week": "first"} if "week" in teams.columns else {})}
    )
    players = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
    cache.mkdir(parents=True, exist_ok=True)
    teams.to_parquet(team_cache, index=False)
    players.to_parquet(player_cache, index=False)
    stamp.write_text(signature)
    return teams, players


def _write_facts(database, teams, players):
    from contextlib import closing
    from .database import connect, record_history_facts
    with closing(connect(database)) as conn:
        record_history_facts(conn, teams, players)
        conn.commit()


def load_history_facts(data_root, database=None):
    """Cleaned team and player rows for a forecast.

    A populated database is the calculation source. Files are read only when
    that database has no team facts, and those rows are written back first.
    """
    stored_teams, stored_players = _facts_from_database(database)
    if stored_teams is not None and len(stored_teams):
        return stored_teams, stored_players
    teams, players = _facts_from_files(data_root)
    if database and teams is not None and len(teams):
        _write_facts(database, teams, players)
        return _facts_from_database(database)
    return teams, players


def sync_cleaned_facts(data_root, database, schedules=None, summaries=None):
    """Parse source files, store the cleaned rows, and return the database copy."""
    teams, players = _facts_from_files(data_root)
    scored = _scoring_facts(schedules) if schedules is not None and len(schedules) else pd.DataFrame()
    if len(scored):
        if teams is None or len(teams) == 0:
            teams = scored
        else:
            teams = teams.merge(scored, on=["game_id", "team_id"], how="outer", suffixes=("", "_schedule"))
            for column in ("points_for", "points_against", "season", "week"):
                scheduled = f"{column}_schedule"
                if scheduled not in teams:
                    continue
                teams[column] = teams[scheduled].combine_first(teams[column]) if column in teams else teams[scheduled]
                teams = teams.drop(columns=[scheduled])
    if summaries is not None and len(summaries) and {"game_id", "pos_team_id"}.issubset(summaries.columns):
        metrics = [column for column in ("yards_per_play", "yards_per_pass", "yards_per_rush", "first_downs_created_rate") if column in summaries.columns]
        adv = summaries.rename(columns={"pos_team_id": "team_id"})[["game_id", "team_id", *metrics]].copy()
        adv["game_id"] = pd.to_numeric(adv.game_id, errors="coerce")
        adv["team_id"] = pd.to_numeric(adv.team_id, errors="coerce")
        adv = adv.dropna(subset=["game_id", "team_id"]).drop_duplicates(["game_id", "team_id"], keep="last")
        if teams is None or len(teams) == 0:
            teams = adv
        else:
            teams = teams.merge(adv, on=["game_id", "team_id"], how="outer", suffixes=("", "_adv"))
            for column in metrics:
                adv_column = f"{column}_adv"
                if adv_column not in teams:
                    continue
                teams[column] = teams[adv_column].combine_first(teams[column]) if column in teams else teams[adv_column]
                teams = teams.drop(columns=[adv_column])
    if not database:
        return teams, players
    if (teams is not None and len(teams)) or (players is not None and len(players)):
        _write_facts(database, teams if teams is not None else pd.DataFrame(), players)
    return _facts_from_database(database)


def summaries_from_database(database):
    """Advanced team metrics in the shape the dynamic state update reads."""
    from .config import METRICS
    teams, _players = _facts_from_database(database)
    columns = ["game_id", "pos_team_id", *METRICS[1:]]
    if teams is None or len(teams) == 0:
        return pd.DataFrame(columns=columns)
    frame = teams.rename(columns={"team_id": "pos_team_id"})
    for column in METRICS[1:]:
        if column not in frame:
            frame[column] = np.nan
    frame = frame.dropna(subset=["game_id", "pos_team_id"])
    return frame[columns]


def points_from_database(games, teams):
    """Replace schedule points with cleaned database points when a row exists."""
    if games is None or len(games) == 0 or teams is None or len(teams) == 0 or "points_for" not in teams.columns:
        return games
    view = games.copy()
    facts = teams.dropna(subset=["game_id", "team_id"]).copy()
    facts["game_id"] = pd.to_numeric(facts.game_id, errors="coerce")
    facts["team_id"] = pd.to_numeric(facts.team_id, errors="coerce")
    facts = facts.dropna(subset=["game_id", "team_id"]).drop_duplicates(["game_id", "team_id"], keep="last")
    indexed = facts.set_index(["game_id", "team_id"])["points_for"]
    for team_column, points_column in (("home_id", "home_points"), ("away_id", "away_points")):
        keys = list(zip(view.game_id.astype("int64"), view[team_column].astype("int64")))
        stored = indexed.reindex(keys).to_numpy(dtype=float)
        current = pd.to_numeric(view[points_column], errors="coerce").to_numpy(dtype=float)
        view[points_column] = np.where(np.isfinite(stored), stored, current)
    return view


def _scoring_facts(completed):
    rows = []
    usable = completed.dropna(subset=["home_points", "away_points"])
    for game in usable.itertuples(index=False):
        if not np.isfinite(game.home_points) or not np.isfinite(game.away_points):
            continue
        base = {"game_id": int(game.game_id), "season": int(game.season), "week": int(game.week)}
        rows.append({**base, "team_id": int(game.home_id), "points_for": float(game.home_points), "points_against": float(game.away_points)})
        rows.append({**base, "team_id": int(game.away_id), "points_for": float(game.away_points), "points_against": float(game.home_points)})
    return pd.DataFrame(rows)


def pregame_history(query, data_root, completed, database=None):
    """Rates each team had recorded before the game, from parquet facts and final scores.

    The current game does not update its own row. A team with no earlier game stays missing.
    """
    if query is None or len(query) == 0 or completed is None or len(completed) == 0:
        return None
    facts, _players = load_history_facts(data_root, database=database)
    scored = _scoring_facts(completed)
    if facts is None or len(facts) == 0:
        facts = scored
    elif len(scored):
        facts = facts.merge(scored[["game_id", "team_id", "points_for", "points_against"]], on=["game_id", "team_id"], how="outer", suffixes=("", "_score"))
        for column in ("points_for", "points_against"):
            scored_column = f"{column}_score"
            if scored_column in facts:
                facts[column] = facts[scored_column].combine_first(facts[column]) if column in facts else facts[scored_column]
                facts = facts.drop(columns=[scored_column])
    if facts is None or len(facts) == 0:
        return None
    season = int(pd.to_numeric(query.season, errors="coerce").max())
    past_ids = set(pd.to_numeric(completed.game_id, errors="coerce").dropna().astype("int64"))
    if "season" in facts:
        keep = facts.game_id.isin(past_ids) | (pd.to_numeric(facts.season, errors="coerce") < season)
        facts = facts.loc[keep].copy()
    else:
        facts = facts.loc[facts.game_id.isin(past_ids)].copy()
    if facts.empty:
        return None
    order = completed[["game_id", "start_date"]].drop_duplicates("game_id")
    facts = facts.merge(order, on="game_id", how="left")
    facts["_season"] = pd.to_numeric(facts.season, errors="coerce") if "season" in facts else season
    facts["_week"] = pd.to_numeric(facts.week, errors="coerce") if "week" in facts else 0
    facts = facts.sort_values(["_season", "_week", "start_date", "game_id"], kind="mergesort")
    metrics = [
        "points_for", "points_against", "yards_per_play", "success_rate", "explosive_rate",
        "turnover_rate", "pass_yards", "rush_yards", "receiving_yards", "epa",
    ]
    state = {}
    emitted = {}
    query_ids = {int(game_id): index for index, game_id in enumerate(pd.to_numeric(query.game_id, errors="coerce").astype("int64"))}
    home_ids = pd.to_numeric(query.home_id, errors="coerce").astype("int64").to_numpy()
    away_ids = pd.to_numeric(query.away_id, errors="coerce").astype("int64").to_numpy()
    home_for = {int(game_id): int(home) for game_id, home in zip(query_ids, home_ids[list(query_ids.values())])}
    away_for = {int(game_id): int(away) for game_id, away in zip(query_ids, away_ids[list(query_ids.values())])}

    def snapshot(team_id):
        record = state.get(int(team_id))
        if not record:
            return {}
        rates = {}
        for metric in metrics:
            season_n = record["n"].get(metric, 0)
            prior_n = record["prior_n"].get(metric, 0)
            total = prior_n + season_n
            if total == 0:
                continue
            total_sum = 0.0
            if prior_n and metric in record["prior"]:
                total_sum += record["prior"][metric] * prior_n
            if season_n and metric in record["sum"]:
                total_sum += record["sum"][metric]
            rates[metric] = total_sum / total
        return rates

    def remember(team_id, row):
        team_id = int(team_id)
        try:
            season = int(row["_season"])
        except (TypeError, ValueError, KeyError):
            season = None
        record = state.setdefault(team_id, {"season": season, "prior": {}, "prior_n": {}, "sum": {}, "n": {}})
        if season is not None and record["season"] is not None and season != record["season"]:
            record["prior"] = {
                metric: record["sum"][metric] / record["n"][metric]
                for metric in record["sum"]
                if record["n"].get(metric)
            }
            record["prior_n"] = {metric: count for metric, count in record["n"].items() if count}
            record["sum"] = {}
            record["n"] = {}
            record["season"] = season
        elif record["season"] is None:
            record["season"] = season
        for metric in metrics:
            if metric not in row.index:
                continue
            value = row[metric]
            if pd.isna(value):
                continue
            record["sum"][metric] = record["sum"].get(metric, 0.0) + float(value)
            record["n"][metric] = record["n"].get(metric, 0) + 1

    for game_id, group in facts.groupby("game_id", sort=False):
        game_id = int(game_id)
        if game_id in query_ids:
            home_team = home_for[game_id]
            away_team = away_for[game_id]
            emitted[game_id] = {
                home_team: snapshot(home_team),
                away_team: snapshot(away_team),
            }
        for row in group.itertuples(index=False):
            remember(row.team_id, pd.Series(row._asdict()))
    matrix = np.full((len(query), len(HISTORY_COLUMNS)), np.nan)
    pairs = (
        ("points_for", 0, 1),
        ("points_against", 2, 3),
        ("yards_per_play", 4, 5),
        ("success_rate", 6, 7),
        ("explosive_rate", 8, 9),
        ("turnover_rate", 10, 11),
        ("pass_yards", 12, 13),
        ("rush_yards", 14, 15),
        ("receiving_yards", 16, 17),
        ("epa", 18, 19),
    )
    final_rates = {team_id: snapshot(team_id) for team_id in state}
    for game_id, slot in query_ids.items():
        home = int(home_ids[slot])
        away = int(away_ids[slot])
        if game_id in emitted:
            home_state = emitted[game_id].get(home, {})
            away_state = emitted[game_id].get(away, {})
        else:
            home_state = final_rates.get(home, {})
            away_state = final_rates.get(away, {})
        for metric, home_column, away_column in pairs:
            if metric in home_state:
                matrix[slot, home_column] = home_state[metric]
            if metric in away_state:
                matrix[slot, away_column] = away_state[metric]
    if not np.isfinite(matrix).any():
        return None
    return matrix


def feature_columns(variant, n_metrics=None):
    # Layout: offense, defense, offense-defense, products, score-state variance,
    # play-rate variance, signed venue, early-season, offense pass-rush
    # difference, defense pass-rush difference. The five original metrics keep
    # the score columns at 0, 5, 10, 20, 22, and 23.
    n = len(METRICS) if n_metrics is None else int(n_metrics)
    width = feature_width(n)
    if variant == "score":
        columns = [0, n, 2 * n, 4 * n, 4 * n + 2, 4 * n + 3]
        for index in range(len(METRICS), n):
            columns.extend((index, n + index, 2 * n + index))
        return columns
    if variant == "pbp":
        return list(range(width))
    raise ValueError(f"Unknown variant {variant}")
