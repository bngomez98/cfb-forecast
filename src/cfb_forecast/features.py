"""Past-game observations and deterministic optional raw-play cleaning."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .config import METRICS
from .data import require_columns
from .forecast_model import HISTORY_COLUMNS

def clean_plays(plays):
    cols = ["game_id", "game_play_number", "pos_team_id", "def_pos_team_id", "down", "distance", "statYardage", "rush", "pass", "sack", "kneel_down", "penalty_no_play", "period", "start.pos_score_diff"]
    require_columns(plays, cols, "raw plays")
    p = plays.drop_duplicates(["game_id", "game_play_number"]).copy()
    flags = ["rush", "pass", "sack", "kneel_down", "penalty_no_play"]
    for c in flags:
        if not p[c].dropna().isin([True, False, 0, 1]).all():
            raise ValueError(f"Invalid raw-play flag {c}")
        p[c] = p[c].fillna(False).astype(bool)
    # Keep overtime and plays at any score. Kneels and penalty no-plays have no scrimmage result.
    p = p[(p.rush | p["pass"] | p.sack) & ~p.kneel_down & ~p.penalty_no_play & p.down.between(1, 4) & p.distance.between(1, 99) & p.period.ge(1) & p.statYardage.notna()].copy()
    p["pass_play"] = p["pass"] | p.sack
    threshold = p.distance * np.select([p.down.eq(1), p.down.eq(2)], [0.5, 0.7], default=1.)
    p["success"] = p.statYardage.ge(threshold).astype(float)
    p["explosive"] = p.statYardage.ge(np.where(p.pass_play,20,10)).astype(float)
    p["clipped_yards"] = p.statYardage.clip(-20,80)
    return p

def aggregate_plays(plays):
    p = clean_plays(plays)
    return p.groupby(["game_id", "pos_team_id", "def_pos_team_id"], as_index=False).agg(plays=("game_play_number", "size"), success_rate=("success", "mean"), explosive_rate=("explosive", "mean"), yards_per_play=("clipped_yards", "mean"))

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


def load_history_facts(data_root, database=None):
    """Team and player game facts from parquet seasons on disk.

    Seasons 2000 through 2025 are the training history. A later season file is
    read too, and the forecast caller keeps only games already completed.
    Missing seasons are skipped. No stat is filled with a league average here.
    """
    root = Path(data_root)
    paths = sorted(
        path for path in list(root.glob("pbp_*.parquet")) + list(root.glob("adv_team_*.parquet"))
        if (_year_from_name(path) or 0) >= 2000
    )
    db_teams, db_players = _facts_from_database(database)
    if not paths:
        return db_teams, db_players
    cache = root / "_history"
    signature = _fingerprint(paths)
    team_cache = cache / "team_game_facts.parquet"
    player_cache = cache / "player_game_facts.parquet"
    stamp = cache / "fingerprint.txt"
    if team_cache.exists() and player_cache.exists() and stamp.exists() and stamp.read_text().strip() == signature:
        teams = _combine_facts(pd.read_parquet(team_cache), db_teams, ["game_id", "team_id"])
        players = _combine_facts(pd.read_parquet(player_cache), db_players, ["game_id", "team_id", "player_name", "position_group"])
        return teams, players
    team_frames = []
    player_frames = []
    for path in paths:
        year = _year_from_name(path)
        if "adv_team" in path.name:
            columns = _existing_columns(path, ["game_id", "pos_team_id", "season", "week", "yards_per_play", "yards_per_pass", "yards_per_rush"])
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
                "pass", "rush", "EPA", "EPA_success", "is_turnover", "scrimmage_play",
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
                role_rows = role_rows[role_rows[flag]]
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
    teams = _combine_facts(teams, db_teams, ["game_id", "team_id"])
    players = _combine_facts(players, db_players, ["game_id", "team_id", "player_name", "position_group"])
    return teams, players


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


def _schedule_kickoffs(data_root):
    """Kickoff times for every scheduled game on disk, including non-FBS opponents."""
    frames = []
    for path in sorted(Path(data_root).glob("schedule_*.csv")):
        try:
            frames.append(pd.read_csv(path, usecols=["game_id", "start_date"]))
        except ValueError:
            continue
    if not frames:
        return pd.DataFrame(columns=["game_id", "start_date"])
    kickoffs = pd.concat(frames, ignore_index=True)
    kickoffs["start_date"] = pd.to_datetime(kickoffs.start_date, utc=True, format="ISO8601", errors="coerce")
    return kickoffs.dropna(subset=["start_date"])


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
    # Kickoff time is the only safe order: postseason games restart week numbering
    # at 1, so a season/week sort would replay bowl games before October games.
    order = pd.concat(
        [completed[["game_id", "start_date"]], _schedule_kickoffs(data_root)],
        ignore_index=True,
    ).drop_duplicates("game_id")
    order["game_id"] = pd.to_numeric(order.game_id, errors="coerce")
    order["start_date"] = pd.to_datetime(order.start_date, utc=True)
    facts = facts.drop(columns=["start_date"], errors="ignore").merge(order.dropna(subset=["game_id"]).astype({"game_id": "int64"}), on="game_id", how="left")
    # A fact without a known kickoff cannot be placed before or after a game.
    facts = facts.loc[facts.start_date.notna()].copy()
    # remember() starts a new season's running rates from this column. It has no
    # leading underscore because itertuples() renames such fields.
    facts["rate_season"] = pd.to_numeric(facts.season, errors="coerce") if "season" in facts else season
    facts = facts.sort_values(["start_date", "game_id"], kind="mergesort")
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
    query_seasons = pd.to_numeric(query.season, errors="coerce").to_numpy()
    season_for = {
        game_id: (int(query_seasons[slot]) if np.isfinite(query_seasons[slot]) else None)
        for game_id, slot in query_ids.items()
    }

    def snapshot(team_id, target_season=None):
        record = state.get(int(team_id))
        if not record:
            return {}
        # A team's first game of a new season starts from the previous season's
        # mean alone; older seasons no longer count.
        new_season = (
            target_season is not None
            and record["season"] is not None
            and record["season"] < target_season
        )
        rates = {}
        for metric in metrics:
            season_n = record["n"].get(metric, 0)
            prior_n = 0 if new_season else record["prior_n"].get(metric, 0)
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
            season = int(row["rate_season"])
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
                home_team: snapshot(home_team, season_for[game_id]),
                away_team: snapshot(away_team, season_for[game_id]),
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
    for game_id, slot in query_ids.items():
        home = int(home_ids[slot])
        away = int(away_ids[slot])
        if game_id in emitted:
            home_state = emitted[game_id].get(home, {})
            away_state = emitted[game_id].get(away, {})
        else:
            home_state = snapshot(home, season_for[game_id])
            away_state = snapshot(away, season_for[game_id])
        for metric, home_column, away_column in pairs:
            if metric in home_state:
                matrix[slot, home_column] = home_state[metric]
            if metric in away_state:
                matrix[slot, away_column] = away_state[metric]
    if not np.isfinite(matrix).any():
        return None
    return matrix


def feature_columns(variant):
    # Layout: offense(5), defense(5), offense-defense(5), products(5),
    # score-state variance, PBP variance, signed venue, early-season,
    # offense pass-rush difference, defense pass-rush difference (indices 0-25).
    # The player layer appends two columns: own player-offense rating (26) and
    # opponent player-defense rating (27), both standardized, point-in-time.
    if variant == "score":
        return [0,5,10,20,22,23]
    if variant == "pbp":
        return list(range(26))
    if variant == "pbp_player":
        return list(range(28))
    raise ValueError(f"Unknown variant {variant}")


PLAYER_VARIANTS = ("pbp_player",)


def attach_player_features(F, game_ids, features_by_week):
    """Concatenate the (n, 2, 2) player block onto the (n, 2, 26) state features."""
    from .players import player_feature_matrix
    import numpy as np
    block = player_feature_matrix(game_ids, features_by_week)
    return np.concatenate([F, block], axis=2)
