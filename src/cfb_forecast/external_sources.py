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

from .sources import TableParser, utc_now

CFBD = "https://api.collegefootballdata.com"

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
    for week in weeks:
        payload, receipt = _cfbd_get(
            "/games",
            {"year": season, "week": week, "seasonType": "regular", "classification": "fbs"},
            root / f"cfbd_games_week_{week}.json",
            api_key,
        )
        records.append({"name": f"cfbd_games_week_{week}", "role": "schedule_results_crosscheck", **receipt})
        if isinstance(payload, list):
            rows = [[g.get("awayTeam", ""), g.get("homeTeam", ""), g.get("startDate", ""), g.get("completed")] for g in payload]
            subset = games[games.week.astype(int) == week]
            validation.extend(validate_schedule_rows(subset, rows, "collegefootballdata"))

        weather, weather_receipt = _cfbd_get(
            "/games/weather",
            {"year": season, "week": week, "seasonType": "regular", "classification": "fbs"},
            root / f"cfbd_weather_week_{week}.json",
            api_key,
        )
        records.append({"name": f"cfbd_weather_week_{week}", "role": "weather_crosscheck", **weather_receipt})

    ratings, rating_receipt = _cfbd_get(
        "/ratings/core",
        {"year": season},
        root / "cfbd_core_ratings.json",
        api_key,
    )
    records.append({"name": "cfbd_core_ratings", "role": "benchmark_only_not_forecast_rating", **rating_receipt})
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
            "footballdb_team_stats",
            f"https://www.footballdb.com/college-football/stats/teamstat.html?cat=S&yr={season}&lg=FBS",
            "team_statistics",
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

    api_key = os.environ.get("CFBD_API_KEY")
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
                "reason": "Set CFBD_API_KEY in the runtime environment; never commit the key.",
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
            "cfbd": "supplemental schedule, weather, player/team data and benchmark collection when CFBD_API_KEY is configured",
            "fanduel_research": "market/reference material only; not a substitute for timestamped sportsbook quotes",
            "methodology_pages": "documentation/reference only; never treated as observations",
            "core_ratings": "benchmark only; never a production team-rating input",
        },
    }
    (root / "external_sources.json").write_text(json.dumps(report, indent=2))
    return report
