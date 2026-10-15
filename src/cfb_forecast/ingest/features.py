"""Past-game observations and deterministic optional raw-play cleaning."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import METRICS
from .data import require_columns
from ..modeling.forecast_model import HISTORY_COLUMNS, PLAYER_METRICS, BASE_METRICS, TEMPORAL_TRANSFORMS

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
    digest = hashlib.sha256(b"history-facts-v2")
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
        from ..storage.database import connect
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
    # Current on-disk measurements take precedence; SQLite fills absent fields.
    fresh = base.drop_duplicates(keys, keep="last").set_index(keys)
    saved = stored.drop_duplicates(keys, keep="last").set_index(keys)
    return fresh.combine_first(saved).reset_index()


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
        {column: "last" for column in value_columns}
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


def player_game_rates(players):
    """Observed role efficiency and usage; no assumed starters or replacements."""
    if players is None or players.empty:
        return pd.DataFrame(columns=["game_id", "team_id", *PLAYER_METRICS])
    required = ["game_id", "team_id", "position_group", "plays", "yards"]
    require_columns(players, required, "player game facts")
    frame = players.copy()
    for column in ("plays", "yards", "epa"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
        frame.loc[~np.isfinite(frame[column]), column] = np.nan
    frame = frame[frame.plays.gt(0)].copy()
    keys = ["game_id", "team_id", "position_group"]
    frame["plays_squared"] = frame.plays ** 2
    frame["epa_weighted"] = frame.epa * frame.plays
    frame["epa_plays"] = frame.plays.where(frame.epa.notna())
    frame["yards_plays"] = frame.plays.where(frame.yards.notna())
    grouped = frame.groupby(keys)
    totals = grouped[["plays", "plays_squared", "yards", "yards_plays", "epa_weighted", "epa_plays"]].sum(min_count=1)
    totals["yards_per_play"] = totals.yards / totals.yards_plays
    totals["epa"] = totals.epa_weighted / totals.epa_plays
    totals["usage_concentration"] = totals.plays_squared / totals.plays ** 2
    rates = totals[["yards_per_play", "epa", "usage_concentration"]].unstack("position_group")
    rates.columns = [f"player_{role}_{metric}" for metric, role in rates.columns]
    return rates.reindex(columns=PLAYER_METRICS).reset_index()


def pregame_history(query, data_root, completed, database=None):
    """Average observed games in the prior season and eligible current season.

    Replay finals by result availability, with each query preceding results at
    exactly its cutoff. Only the immediately preceding season supplies a prior.
    Missing metrics remain NaN; player role rates enter the same score model.
    """
    if query is None or len(query) == 0 or completed is None or len(completed) == 0:
        return None
    facts, players = load_history_facts(data_root, database=database)
    scores = _scoring_facts(completed)
    facts = facts.copy() if facts is not None and len(facts) else pd.DataFrame(columns=["game_id", "team_id"])
    # Schedule finals are authoritative for scoring, parquet for efficiency.
    facts = facts.drop(columns=["points_for", "points_against"], errors="ignore").merge(
        scores[["game_id", "team_id", "points_for", "points_against"]], on=["game_id", "team_id"], how="outer")
    player_rates = player_game_rates(players)
    if len(player_rates):
        facts = facts.merge(player_rates, on=["game_id", "team_id"], how="outer")
    schedules = [completed.copy()]
    for path in sorted(Path(data_root).glob("schedule_*.csv")):
        from .data import normalize_schedule
        schedules.append(normalize_schedule(pd.read_csv(path)))
    games = pd.concat(schedules, ignore_index=True, sort=False).drop_duplicates("game_id", keep="first")
    games["start_date"] = pd.to_datetime(games.start_date, utc=True)
    if "result_available_at" not in games:
        games["result_available_at"] = games.start_date + pd.Timedelta(hours=8)
    else:
        games["result_available_at"] = pd.to_datetime(games.result_available_at, utc=True).fillna(games.start_date + pd.Timedelta(hours=8))
    supplied = set(completed.game_id)
    usable = games.game_id.isin(supplied)
    if "result_usable" in games:
        usable |= games.result_usable.fillna(False).astype(bool)
    games = games[usable & games.result_available_at.gt(games.start_date)].copy()
    # Include verified finals against non-FBS opponents in each team's rates.
    all_scores = _scoring_facts(games)
    facts = facts.drop(columns=["points_for", "points_against"], errors="ignore").merge(
        all_scores[["game_id", "team_id", "points_for", "points_against"]],
        on=["game_id", "team_id"], how="outer")
    # Bind each fact to the verified scheduled participants and season.
    facts = facts.drop(columns=["season", "week", "start_date", "result_available_at"], errors="ignore").merge(
        games[["game_id", "season", "home_id", "away_id", "result_available_at"]], on="game_id", how="inner", validate="many_to_one")
    facts = facts[facts.team_id.eq(facts.home_id) | facts.team_id.eq(facts.away_id)].copy()
    if facts.empty:
        return None
    metrics = list(BASE_METRICS)
    state = {}
    sequences = {}
    column_index = {name: index for index, name in enumerate(HISTORY_COLUMNS)}
    result = np.full((len(query), len(HISTORY_COLUMNS)), np.nan)
    events = []
    for index, game in enumerate(query.itertuples(index=False)):
        cutoff = getattr(game, "cutoff_utc", getattr(game, "as_of", game.start_date))
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is None or pd.isna(cutoff) or cutoff >= game.start_date:
            # A default historical query uses kickoff itself, before the final.
            if not hasattr(game, "cutoff_utc") and not hasattr(game, "as_of"):
                cutoff = game.start_date
            else:
                raise ValueError("History cutoff must be timezone-aware and before kickoff")
        events.append((cutoff.value, 0, index))
    groups = list(facts.groupby("game_id", sort=False))
    for index, (_, rows) in enumerate(groups):
        events.append((rows.result_available_at.iloc[0].value, 1, index))
    queries = list(query.itertuples(index=False))
    for _, kind, index in sorted(events):
        if kind == 1:
            for row in groups[index][1].to_dict("records"):
                key = (int(row["team_id"]), int(row["season"]))
                sums, counts = state.setdefault(key, ({}, {}))
                sequence = sequences.setdefault(key, {metric: [] for metric in metrics})
                for metric in metrics:
                    value = pd.to_numeric(row.get(metric), errors="coerce")
                    if pd.notna(value) and np.isfinite(value):
                        sums[metric] = sums.get(metric, 0.) + float(value)
                        counts[metric] = counts.get(metric, 0) + 1
                        sequence[metric].append(float(value))
                    else:
                        # Keep the game's missing slot: lag 1 must not skip an unknown game.
                        sequence[metric].append(np.nan)
            continue
        game = queries[index]
        for side_index, team in enumerate((game.home_id, game.away_id)):
            for metric_index, metric in enumerate(metrics):
                total, count = 0., 0
                for season in (int(game.season) - 1, int(game.season)):
                    sums, counts = state.get((int(team), season), ({}, {}))
                    total += sums.get(metric, 0.)
                    count += counts.get(metric, 0)
                if count:
                    result[index, 2 * metric_index + side_index] = total / count
                observations = []
                for season in (int(game.season) - 1, int(game.season)):
                    observations.extend(sequences.get((int(team), season), {}).get(metric, []))
                values = np.asarray(observations, dtype=float)
                features = {}
                for lag in (1, 2):
                    features[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
                for window in (3, 5):
                    part = values[-window:]
                    complete = len(part) == window and np.isfinite(part).all()
                    features[f"rolling_{window}_mean"] = float(part.mean()) if complete else np.nan
                    features[f"rolling_{window}_std"] = float(part.std(ddof=0)) if complete else np.nan
                finite = values[np.isfinite(values)]
                features["expanding_mean"] = float(finite.mean()) if len(finite) else np.nan
                side = "home" if side_index == 0 else "away"
                for transform in TEMPORAL_TRANSFORMS:
                    result[index, column_index[f"{side}_{metric}_{transform}"]] = features[transform]
    return result if np.isfinite(result).any() else None


def feature_columns(variant):
    # Layout: offense(5), defense(5), offense-defense(5), products(5),
    # score-state variance, PBP variance, signed venue, early-season,
    # offense pass-rush difference, defense pass-rush difference.
    if variant == "score":
        return [0,5,10,20,22,23]
    if variant == "pbp":
        return list(range(26))
    raise ValueError(f"Unknown variant {variant}")
