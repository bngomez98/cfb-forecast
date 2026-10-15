"""Secondary public data sources and independent weekly validation.

These sources are deliberately separated from the production forecast calculation.
They are timestamped and archived before inference so they can be used for
cross-checking, later feature training, and source-quality evaluation without
silently changing the deployed model.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request

import pandas as pd

from .decision import american_to_decimal
from .paths import data_root, project_root
from .sources import TableParser, utc_now

CFBD = "https://api.collegefootballdata.com"


def cfbd_api_key():
    """Key used by the forecast. The tracked project file is the default."""
    env = os.environ.get("CFBD_API_KEY")
    if env:
        return env
    path = project_root() / "pyproject.toml"
    if not path.exists():
        return None
    import tomllib
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    key = data.get("tool", {}).get("cfb-forecast", {}).get("cfbd_api_key")
    text = str(key).strip() if key else ""
    return text or None


_LINE_NUMBERS = ("spread", "spreadOpen", "overUnder", "overUnderOpen", "homeMoneyline", "awayMoneyline")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _line_record(line):
    record = {"provider": line.get("provider")}
    for field in _LINE_NUMBERS:
        record[field] = _number(line.get(field))
    record["formattedSpread"] = line.get("formattedSpread")
    return record


def quotes_from_cfbd_lines(payload):
    """Keep every provider line. The consensus line supplies the forecast prices."""
    by_id = {}
    by_name = {}
    if not isinstance(payload, list):
        return by_id, by_name
    for game in payload:
        stored = [_line_record(line) for line in (game.get("lines") or []) if line.get("provider")]
        if not stored:
            continue
        chosen = next((line for line in stored if str(line.get("provider", "")).lower() == "consensus"), stored[0])
        quote = {"lines": stored, "line_provider": chosen.get("provider"), "formatted_spread": chosen.get("formattedSpread")}
        if chosen.get("spread") is not None:
            quote["home_spread"] = chosen["spread"]
        if chosen.get("spreadOpen") is not None:
            quote["spread_open"] = chosen["spreadOpen"]
        if chosen.get("overUnder") is not None:
            quote["total_line"] = chosen["overUnder"]
        if chosen.get("overUnderOpen") is not None:
            quote["total_open"] = chosen["overUnderOpen"]
        if chosen.get("homeMoneyline") is not None:
            quote["book_home_moneyline"] = chosen["homeMoneyline"]
        if chosen.get("awayMoneyline") is not None:
            quote["book_away_moneyline"] = chosen["awayMoneyline"]
        for side, field in (("home_ml_decimal", "homeMoneyline"), ("away_ml_decimal", "awayMoneyline")):
            price = chosen.get(field)
            if price is None:
                continue
            try:
                quote[side] = american_to_decimal(price)
            except (TypeError, ValueError):
                continue
        if game.get("id") is not None:
            by_id[int(game["id"])] = quote
        home = _normalize_team(str(game.get("homeTeam") or ""))
        away = _normalize_team(str(game.get("awayTeam") or ""))
        if home and away:
            by_name[(home, away)] = quote
    return by_id, by_name


def _cfbd_cached(path, params, destination, api_key):
    if destination.exists() and destination.stat().st_size > 0:
        try:
            return json.loads(destination.read_text(encoding="utf-8")), {"status": "cached", "url": f"{CFBD}{path}"}
        except (OSError, json.JSONDecodeError):
            pass
    return _cfbd_get(path, params, destination, api_key)


def _week_pairs(games):
    if games is None or len(games) == 0 or "season" not in getattr(games, "columns", []):
        return []
    return sorted({
        (int(season), int(week))
        for season, week in zip(games.season, games.week)
        if pd.notna(season) and pd.notna(week)
    })


def cfbd_lines_for_games(games, api_key):
    by_id = {}
    by_name = {}
    root = data_root() / "raw"
    for season, week in _week_pairs(games):
        payload, _receipt = _cfbd_cached(
            "/lines",
            {"year": season, "week": week, "seasonType": "both"},
            root / f"cfbd_lines_{season}_week_{week}_both.json",
            api_key,
        )
        ids, names = quotes_from_cfbd_lines(payload)
        by_id.update(ids)
        by_name.update(names)
    return by_id, by_name


def _index_named(payload, id_field):
    by_id = {}
    by_name = {}
    if not isinstance(payload, list):
        return by_id, by_name
    for row in payload:
        if row.get(id_field) is not None:
            by_id[int(row[id_field])] = row
        home = _normalize_team(str(row.get("homeTeam") or ""))
        away = _normalize_team(str(row.get("awayTeam") or ""))
        if home and away:
            by_name[(home, away)] = row
    return by_id, by_name


def cfbd_context_for_games(games, api_key):
    """Weather and pregame win probability for the same weeks as the lines."""
    weather_id, weather_name = {}, {}
    probability_id, probability_name = {}, {}
    root = data_root() / "raw"
    for season, week in _week_pairs(games):
        weather, _receipt = _cfbd_cached(
            "/games/weather",
            {"year": season, "week": week, "seasonType": "both", "classification": "fbs"},
            root / f"cfbd_weather_{season}_week_{week}_both.json",
            api_key,
        )
        ids, names = _index_named(weather, "id")
        weather_id.update(ids)
        weather_name.update(names)
        probabilities, _receipt = _cfbd_cached(
            "/metrics/wp/pregame",
            {"year": season, "week": week, "seasonType": "both"},
            root / f"cfbd_wp_pregame_{season}_week_{week}_both.json",
            api_key,
        )
        ids, names = _index_named(probabilities, "gameId")
        probability_id.update(ids)
        probability_name.update(names)
    return weather_id, weather_name, probability_id, probability_name


def _lookup(by_id, by_name, game):
    found = by_id.get(int(game.game_id))
    if found is not None:
        return found
    home = _normalize_team(str(getattr(game, "home_team", "") or ""))
    away = _normalize_team(str(getattr(game, "away_team", "") or ""))
    return by_name.get((home, away))


def _fill(row, column, value):
    if row.get(column) is None and value is not None:
        row[column] = value


def merge_cfbd_lines(games, market_rows, season):
    """Fill missing CollegeFootballData lines, weather, and pregame probability."""
    rows = [dict(row) for row in market_rows]
    key = cfbd_api_key()
    if not key or games is None or len(games) == 0:
        return rows
    seasons = {int(season), int(season) - 1}
    selected = games.loc[games.season.astype(int).isin(seasons)]
    by_id, by_name = cfbd_lines_for_games(selected, key)
    weather_id, weather_name, probability_id, probability_name = cfbd_context_for_games(selected, key)
    filled = []
    for game, row in zip(games.itertuples(index=False), rows):
        if int(game.season) not in seasons:
            filled.append(row)
            continue
        quote = _lookup(by_id, by_name, game)
        if quote:
            for column in (
                "home_spread", "total_line", "home_ml_decimal", "away_ml_decimal",
                "spread_open", "total_open", "formatted_spread", "line_provider",
                "book_home_moneyline", "book_away_moneyline",
            ):
                _fill(row, column, quote.get(column))
            row["cfbd_lines"] = quote.get("lines")
            row["line_source"] = row.get("line_source") or "collegefootballdata"
        weather = _lookup(weather_id, weather_name, game)
        if weather:
            _fill(row, "weather_temperature_f", _number(weather.get("temperature")))
            row["cfbd_weather"] = {
                "game_indoors": weather.get("gameIndoors"),
                "venue_id": weather.get("venueId"),
                "venue": weather.get("venue"),
                "temperature": _number(weather.get("temperature")),
                "dew_point": _number(weather.get("dewPoint")),
                "humidity": _number(weather.get("humidity")),
                "precipitation": _number(weather.get("precipitation")),
                "snowfall": _number(weather.get("snowfall")),
                "wind_direction": _number(weather.get("windDirection")),
                "wind_speed": _number(weather.get("windSpeed")),
                "pressure": _number(weather.get("pressure")),
                "condition_code": _number(weather.get("weatherConditionCode")),
                "condition": weather.get("weatherCondition"),
            }
        probability = _lookup(probability_id, probability_name, game)
        if probability:
            _fill(row, "home_spread", _number(probability.get("spread")))
            row["cfbd_pregame_home_win_probability"] = _number(probability.get("homeWinProbability"))
            row["cfbd_pregame_spread"] = _number(probability.get("spread"))
        filled.append(row)
    return filled

PUBLIC_SOURCES = (
    {
        "name": "cfbfastR",
        "url": "https://cfbfastr.sportsdataverse.org/",
        "role": "data_documentation",
    },
    {
        "name": "cfbfastR_registry",
        "url": "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-data/main/models/REGISTRY.md",
        "role": "data_schema_registry",
    },
    {
        "name": "radsportsanalytics",
        "url": "https://radsportsanalytics.com/blog/introduction-to-cfb-analytics/",
        "role": "methodology_reference",
    },
    {
        "name": "cfbtrack",
        "url": "https://cfbtrack.com/",
        "role": "team_player_context",
    },
    {
        "name": "cfbtracker",
        "url": "https://cfbtracker.com/",
        "role": "team_player_context",
    },
    {
        "name": "cfbresults",
        "url": "https://cfbresults.com/",
        "role": "results_validation",
    },
    {
        "name": "college_football_results",
        "url": "https://college-football-results.com/",
        "role": "results_validation",
    },
    {
        "name": "cfbstats",
        "url": "https://cfbstats.com/",
        "role": "team_player_context",
    },
    {
        "name": "collegefootballdb",
        "url": "https://collegefootballdb.com/",
        "role": "team_player_context",
    },
    {
        "name": "collegefootballdata",
        "url": "https://collegefootballdata.com/",
        "role": "ratings_and_schedule_validation",
    },
    {
        "name": "cfbtxt_schedule",
        "url": "https://cfbtxt.com/schedule/",
        "role": "weekly_schedule_validation",
    },
    {
        "name": "fanduel_research",
        "url": "https://www.fanduel.com/research/ncaaf",
        "role": "market_reference",
    },
    {
        "name": "statistical_prediction_reference",
        "url": "https://en.wikipedia.org/wiki/Statistical_association_football_predictions",
        "role": "methodology_reference",
    },
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalize_team(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"\b(university|college|state university|the)\b", " ", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def _fetch(url: str, destination: Path, headers: dict | None = None, expect_json=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    requested_at = utc_now()
    request_headers = {
        "User-Agent": "cfb-forecast/0.4 (+source-audit)",
        "Accept": "application/json" if expect_json else "text/html,*/*;q=0.8",
        **(headers or {}),
    }
    receipt = {"url": url, "requested_at": requested_at}
    try:
        req = urllib.request.Request(url, headers=request_headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            receipt.update(
                status="ok",
                retrieved_at=utc_now(),
                resolved_url=response.url,
                http_status=response.status,
                bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
            )
        destination.write_bytes(body)
        payload = json.loads(body) if expect_json else body.decode("utf-8", errors="replace")
    except Exception as exc:
        payload = None
        receipt.update(
            status="unavailable",
            retrieved_at=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
        )
    destination.with_suffix(destination.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )
    return payload, receipt


def _table_rows(html: str | None) -> list[list[str]]:
    if not html:
        return []
    parser = TableParser()
    parser.feed(html)
    return [row for table in parser.tables for row in table]


def validate_schedule_rows(games: pd.DataFrame, rows: Iterable[Iterable[str]], provider: str):
    """Cross-check matchup presence without making the external site authoritative."""
    searchable = [(_normalize_team(" ".join(map(str, row))), list(row)) for row in rows]
    results = []
    for game in games.itertuples(index=False):
        home = _normalize_team(str(game.home_team))
        away = _normalize_team(str(game.away_team))
        matches = [row for text, row in searchable if home and away and home in text and away in text]
        results.append(
            {
                "game_id": int(game.game_id),
                "week": int(game.week),
                "home_team": game.home_team,
                "away_team": game.away_team,
                "provider": provider,
                "status": "matched" if matches else "not_found",
                "matched_row": matches[0] if matches else None,
            }
        )
    return results


def _cfbd_get(path: str, params: dict, destination: Path, api_key: str):
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{CFBD}{path}" + (f"?{query}" if query else "")
    return _fetch(
        url,
        destination,
        headers={"Authorization": f"Bearer {api_key}"},
        expect_json=True,
    )


def _cfbd_week_sources(games: pd.DataFrame, season: int, root: Path, api_key: str):
    records = []
    validation = []
    weeks = sorted({int(x) for x in games.week.dropna().tolist()})
    fbs = {"seasonType": "both", "classification": "fbs"}
    week_routes = (
        ("/games", {**fbs}, "cfbd_games_week_{week}", "schedule_results"),
        ("/games/teams", {**fbs}, "cfbd_game_team_stats_week_{week}", "team_box_score"),
        ("/games/players", {**fbs}, "cfbd_game_player_stats_week_{week}", "player_box_score"),
        ("/games/media", {**fbs}, "cfbd_media_week_{week}", "broadcast"),
        ("/games/weather", {**fbs}, "cfbd_weather_week_{week}_both", "weather"),
        ("/lines", {"seasonType": "both"}, "cfbd_lines_week_{week}_both", "market_lines"),
        ("/drives", {**fbs}, "cfbd_drives_week_{week}", "drives"),
        ("/plays", {**fbs}, "cfbd_plays_week_{week}", "play_by_play"),
        ("/plays/stats", {"seasonType": "both"}, "cfbd_play_stats_week_{week}", "play_player_stats"),
        ("/stats/game/advanced", {"seasonType": "both"}, "cfbd_advanced_game_week_{week}", "advanced_game"),
        ("/stats/game/havoc", {"seasonType": "both"}, "cfbd_havoc_week_{week}", "havoc"),
        ("/ppa/games", {**fbs}, "cfbd_ppa_games_week_{week}", "team_game_ppa"),
        ("/ppa/players/games", {"seasonType": "both"}, "cfbd_ppa_players_week_{week}", "player_game_ppa"),
        ("/metrics/wp/pregame", {"seasonType": "both"}, "cfbd_wp_pregame_week_{week}_both", "pregame_win_probability"),
        ("/passing/plays", {**fbs}, "cfbd_passing_plays_week_{week}", "passing_plays"),
        ("/passing/players/games", {**fbs}, "cfbd_passing_players_week_{week}", "passing_player_games"),
        ("/passing/teams/games", {**fbs}, "cfbd_passing_teams_week_{week}", "passing_team_games"),
        ("/rushing/plays", {**fbs}, "cfbd_rushing_plays_week_{week}", "rushing_plays"),
        ("/rushing/players/games", {**fbs}, "cfbd_rushing_players_week_{week}", "rushing_player_games"),
        ("/rushing/teams/games", {**fbs}, "cfbd_rushing_teams_week_{week}", "rushing_team_games"),
        ("/stats/player/success/game", {"seasonType": "both"}, "cfbd_player_success_week_{week}", "player_game_success"),
        ("/rankings", {"seasonType": "both"}, "cfbd_rankings_week_{week}", "rankings"),
        ("/ratings/elo", {"seasonType": "both"}, "cfbd_elo_week_{week}", "elo_benchmark"),
    )
    for week in weeks:
        for path, params, name, role in week_routes:
            filename = name.format(week=week)
            payload, receipt = _cfbd_cached(path, {"year": season, "week": week, **params}, root / f"{filename}.json", api_key)
            record = {"name": filename, "role": role, **receipt}
            if isinstance(payload, list):
                record["rows"] = len(payload)
            records.append(record)
            if path == "/games" and isinstance(payload, list):
                rows = [[g.get("awayTeam", ""), g.get("homeTeam", ""), g.get("startDate", ""), g.get("completed")] for g in payload]
                validation.extend(validate_schedule_rows(games[games.week.astype(int) == week], rows, "collegefootballdata"))

    season_routes = (
        ("/calendar", {"year": season}, "cfbd_calendar", "calendar"),
        ("/conferences", {"year": season, "classification": "fbs"}, "cfbd_conferences", "conferences"),
        ("/conferences/affiliations", {"year": season, "classification": "fbs"}, "cfbd_affiliations", "conference_affiliations"),
        ("/conferences/changes", {"year": season}, "cfbd_conference_changes", "conference_changes"),
        ("/coaches", {"year": season}, "cfbd_coaches", "coaches"),
        ("/coaches/seasons", {"year": season}, "cfbd_coach_seasons", "coach_seasons"),
        ("/coaches/tenures", {"year": season}, "cfbd_coach_tenures", "coach_tenures"),
        ("/draft/picks", {"year": season}, "cfbd_draft_picks", "draft_picks"),
        ("/draft/positions", {}, "cfbd_draft_positions", "draft_positions"),
        ("/draft/teams", {}, "cfbd_draft_teams", "draft_teams"),
        ("/player/portal", {"year": season}, "cfbd_transfer_portal", "transfer_portal"),
        ("/player/returning", {"year": season}, "cfbd_returning_production", "returning_production"),
        ("/player/usage", {"year": season}, "cfbd_player_usage", "player_usage"),
        ("/playoffs/cfp", {"year": season}, "cfbd_cfp", "cfp"),
        ("/playoffs/cfp/games", {"year": season}, "cfbd_cfp_games", "cfp_games"),
        ("/playoffs/cfp/participants", {"year": season}, "cfbd_cfp_participants", "cfp_participants"),
        ("/plays/stats/types", {}, "cfbd_play_stat_types", "play_stat_types"),
        ("/plays/types", {}, "cfbd_play_types", "play_types"),
        ("/ppa/players/season", {"year": season}, "cfbd_ppa_players_season", "player_season_ppa"),
        ("/ppa/teams", {"year": season, "classification": "fbs"}, "cfbd_ppa_teams", "team_season_ppa"),
        ("/rankings", {"year": season, "seasonType": "both"}, "cfbd_rankings", "rankings"),
        ("/ratings/core", {"year": season}, "cfbd_core_ratings", "benchmark_only_not_forecast_rating"),
        ("/ratings/elo", {"year": season}, "cfbd_elo", "benchmark_only_not_forecast_rating"),
        ("/ratings/fpi", {"year": season}, "cfbd_fpi", "benchmark_only_not_forecast_rating"),
        ("/ratings/sp", {"year": season}, "cfbd_sp", "benchmark_only_not_forecast_rating"),
        ("/ratings/sp/conferences", {"year": season}, "cfbd_sp_conferences", "benchmark_only_not_forecast_rating"),
        ("/ratings/srs", {"year": season}, "cfbd_srs", "benchmark_only_not_forecast_rating"),
        ("/ratings/srs/expanded", {"year": season, "classification": "fbs"}, "cfbd_srs_expanded", "benchmark_only_not_forecast_rating"),
        ("/records", {"year": season}, "cfbd_records", "team_records"),
        ("/recruiting/players", {"year": season}, "cfbd_recruits", "recruits"),
        ("/recruiting/teams", {"year": season}, "cfbd_recruiting_teams", "recruiting_teams"),
        ("/recruiting/groups", {"startYear": season, "endYear": season}, "cfbd_recruiting_groups", "recruiting_groups"),
        ("/roster", {"year": season, "classification": "fbs"}, "cfbd_rosters", "rosters"),
        ("/passing/players/season", {"year": season, "classification": "fbs"}, "cfbd_passing_players_season", "passing_player_season"),
        ("/passing/teams/season", {"year": season, "classification": "fbs"}, "cfbd_passing_teams_season", "passing_team_season"),
        ("/rushing/players/season", {"year": season, "classification": "fbs"}, "cfbd_rushing_players_season", "rushing_player_season"),
        ("/rushing/teams/season", {"year": season, "classification": "fbs"}, "cfbd_rushing_teams_season", "rushing_team_season"),
        ("/scoreboard", {"classification": "fbs"}, "cfbd_scoreboard", "scoreboard"),
        ("/stats/categories", {}, "cfbd_stat_categories", "stat_categories"),
        ("/stats/season", {"year": season, "classification": "fbs"}, "cfbd_team_season_stats", "team_season_stats"),
        ("/stats/season/advanced", {"year": season, "classification": "fbs"}, "cfbd_advanced_season", "advanced_season"),
        ("/stats/player/season", {"year": season, "seasonType": "both"}, "cfbd_player_season_stats", "player_season_stats"),
        ("/stats/player/success", {"year": season, "seasonType": "both"}, "cfbd_player_success", "player_season_success"),
        ("/talent", {"year": season}, "cfbd_talent", "team_talent"),
        ("/teams", {"year": season}, "cfbd_teams", "teams"),
        ("/teams/ats", {"year": season}, "cfbd_ats", "against_the_spread"),
        ("/teams/fbs", {"year": season}, "cfbd_fbs_teams", "fbs_teams"),
        ("/venues", {}, "cfbd_venues", "venues"),
        ("/wepa/team/season", {"year": season}, "cfbd_wepa_teams", "benchmark_only_not_forecast_rating"),
        ("/wepa/players/passing", {"year": season}, "cfbd_wepa_passing", "benchmark_only_not_forecast_rating"),
        ("/wepa/players/rushing", {"year": season}, "cfbd_wepa_rushing", "benchmark_only_not_forecast_rating"),
        ("/wepa/players/kicking", {"year": season}, "cfbd_wepa_kicking", "benchmark_only_not_forecast_rating"),
        ("/info", {}, "cfbd_account", "account"),
    )
    for path, params, name, role in season_routes:
        payload, receipt = _cfbd_cached(path, params, root / f"{name}.json", api_key)
        record = {"name": name, "role": role, **receipt}
        if isinstance(payload, list):
            record["rows"] = len(payload)
        records.append(record)

    for game_id in sorted({int(game_id) for game_id in games.game_id.dropna().tolist()}):
        for path, params, name, role in (
            ("/game/box/advanced", {"id": game_id}, f"cfbd_advanced_box_{game_id}", "advanced_box_score"),
            (f"/games/{game_id}/preview", {}, f"cfbd_preview_{game_id}", "game_preview"),
            ("/metrics/wp", {"gameId": game_id}, f"cfbd_win_probability_{game_id}", "in_game_win_probability"),
        ):
            payload, receipt = _cfbd_cached(path, params, root / f"{name}.json", api_key)
            records.append({"name": name, "role": role, **receipt})
    return records, validation


def collect_extended_sources(games: pd.DataFrame, season: int, root) -> dict:
    """Archive independent sources and create a weekly schedule cross-check report.

    This function does not alter forecast values. Data can enter training only
    through an explicitly versioned feature/model change with out-of-sample
    validation.
    """
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    def public_job(spec):
        suffix = ".md" if spec["name"] == "cfbfastR_registry" else ".html"
        payload, receipt = _fetch(spec["url"], raw / f"{spec['name']}{suffix}")
        return spec, payload, receipt

    public = list(ThreadPoolExecutor(max_workers=6).map(public_job, PUBLIC_SOURCES))
    sources = [{**spec, **receipt} for spec, _, receipt in public]

    # FootballDB and CFBTrack have model-facing parsers in sources.py. Archive
    # their broader pages here as independent source receipts as well.
    extra_pages = [
        (
            "footballdb_stats_index",
            "https://www.footballdb.com/college-football/stats/index.html",
            "player_team_scoring_index",
        ),
        (
            "footballdb_team_offense",
            f"https://www.footballdb.com/college-football/stats/teamstat.html?group=O&cat=T&yr={season}&lg=FBS",
            "team_offense",
        ),
        (
            "footballdb_team_offense_passing",
            f"https://www.footballdb.com/college-football/stats/teamstat.html?group=O&cat=P&yr={season}&lg=FBS",
            "team_passing",
        ),
        (
            "footballdb_team_offense_rushing",
            f"https://www.footballdb.com/college-football/stats/teamstat.html?group=O&cat=R&yr={season}&lg=FBS",
            "team_rushing",
        ),
        (
            "footballdb_team_scoring",
            f"https://www.footballdb.com/college-football/stats/teamstat.html?group=O&cat=S&yr={season}&lg=FBS",
            "team_scoring",
        ),
        (
            "footballdb_team_defense",
            f"https://www.footballdb.com/college-football/stats/teamstat.html?group=D&cat=T&yr={season}&lg=FBS",
            "team_defense",
        ),
        (
            "footballdb_team_defense_scoring",
            f"https://www.footballdb.com/college-football/stats/teamstat.html?group=D&cat=S&yr={season}&lg=FBS",
            "team_defense_scoring",
        ),
        (
            "footballdb_player_passing",
            f"https://www.footballdb.com/college-football/stats/stats.html?lg=FBS&yr={season}&mode=P",
            "player_passing",
        ),
        (
            "footballdb_player_rushing",
            f"https://www.footballdb.com/college-football/stats/stats.html?lg=FBS&yr={season}&mode=R",
            "player_rushing",
        ),
        (
            "footballdb_player_receiving",
            f"https://www.footballdb.com/college-football/stats/stats.html?lg=FBS&yr={season}&mode=C",
            "player_receiving",
        ),
        (
            "footballdb_player_scoring",
            f"https://www.footballdb.com/college-football/stats/stats.html?lg=FBS&yr={season}&mode=S",
            "player_scoring",
        ),
        (
            "footballdb_player_defense",
            f"https://www.footballdb.com/college-football/stats/stats.html?lg=FBS&yr={season}&mode=D",
            "player_defense",
        ),
    ]
    for name, url, role in extra_pages:
        _, receipt = _fetch(url, raw / f"{name}.html")
        sources.append({"name": name, "url": url, "role": role, **receipt})

    validation = []
    cfbtxt = next((payload for spec, payload, _ in public if spec["name"] == "cfbtxt_schedule"), None)
    if cfbtxt:
        validation.extend(validate_schedule_rows(games, _table_rows(cfbtxt), "cfbtxt"))

    # mrTimo is week-specific, so capture each selected week separately.
    for week in sorted({int(x) for x in games.week.dropna().tolist()}):
        url = f"https://mrtimo.github.io/cfb-games/games-{season}.html?week=Week+{week}&sort=thrill"
        html, receipt = _fetch(url, raw / f"mrtimo_week_{week}.html")
        sources.append({"name": f"mrtimo_week_{week}", "url": url, "role": "schedule_crosscheck", **receipt})
        if html:
            validation.extend(validate_schedule_rows(games[games.week.astype(int) == week], _table_rows(html), "mrtimo"))

    api_key = cfbd_api_key()
    if api_key:
        cfbd_sources, cfbd_validation = _cfbd_week_sources(games, season, raw, api_key)
        sources.extend(cfbd_sources)
        validation.extend(cfbd_validation)
        cfbd_status = "configured"
    else:
        cfbd_status = "not_configured"
        sources.append(
            {
                "name": "collegefootballdata",
                "url": "https://api.collegefootballdata.com",
                "role": "schedule_weather_player_stats_and_benchmark_source",
                "status": "not_configured",
                "reason": "CollegeFootballData key missing from pyproject.toml and CFBD_API_KEY.",
            }
        )

    validation_frame = pd.DataFrame(validation)
    if len(validation_frame):
        validation_frame.to_csv(root / "schedule_validation.csv", index=False)
    else:
        pd.DataFrame(columns=["game_id", "provider", "status"]).to_csv(
            root / "schedule_validation.csv", index=False
        )

    report = {
        "schema_version": 1,
        "season": int(season),
        "collected_at": utc_now(),
        "cfbd_api": cfbd_status,
        "sources": sources,
        "schedule_validation": validation,
        "policy": {
            "cfbtxt": "independent weekly schedule validation",
            "cfbfastR": "primary schema/data ecosystem; production historical files remain SportsDataverse release assets",
            "footballdb_and_cfbtrack": "supplemental team/player statistics with identity checks",
            "cfbd": "lines, weather, pregame probability, plays, player and team stats, PPA, and venues are collected from the current API; Elo, SRS, SP, FPI, CORE, and WEPA stay benchmarks",
            "fanduel_research": "market/reference material only; not a substitute for timestamped sportsbook quotes",
            "methodology_pages": "documentation/reference only; never treated as observations",
            "core_ratings": "benchmark only; never a production team-rating input",
        },
    }
    (root / "external_sources.json").write_text(json.dumps(report, indent=2))
    return report
