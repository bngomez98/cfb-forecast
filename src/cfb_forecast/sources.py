"""Timestamped public-source collection used before forecast calculations."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import json
import re
import urllib.parse
import urllib.request

import pandas as pd

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

SOURCE_REGISTRY = {
    "cfbfastR": "https://cfbfastr.sportsdataverse.org/",
    "cfbfastR_models": "https://github.com/sportsdataverse/cfbfastR-cfb-data/blob/main/models/REGISTRY.md",
    "sportsdataverse": "https://github.com/sportsdataverse/sportsdataverse-data",
    "radsportsanalytics": "https://radsportsanalytics.com/blog/introduction-to-cfb-analytics/",
    "cfbtrack": "https://cfbtrack.com/",
    "cfbtracker": "https://cfbtracker.com/",
    "cfbresults": "https://cfbresults.com/",
    "college_football_results": "https://college-football-results.com/",
    "collegefootballdb": "https://collegefootballdb.com/",
    "cfbstats": "https://cfbstats.com/",
    "collegefootballdata": "https://collegefootballdata.com/",
    "mrtimo_schedule": "https://mrtimo.github.io/cfb-games/games-{season}.html",
    "cfbd_key": "https://collegefootballdata.com/key",
    "cfbd_ratings": "https://api.collegefootballdata.com/ratings/core",
    "cfbtxt_validation": "https://cfbtxt.com/schedule/",
    "fanduel_comparison": "https://www.fanduel.com/research/ncaaf",
    "football_db": "https://www.footballdb.com/college-football/stats/teamstat.html?yr={season}&lg=FBS",
    "espn_scoreboard": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
}


def source_registry(season):
    """Return the named source configuration used by collection and weekly review."""
    return {name: url.format(season=int(season)) for name, url in SOURCE_REGISTRY.items()}


def ingest_catalog(season, destination):
    """Fetch each catalog source and keep the response or the failure receipt."""
    destination = Path(destination)
    receipts = []
    jobs = source_registry(season)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(
                collect,
                url,
                destination / f"{name}.bin",
                "api." in url or url.endswith(".json"),
            ): name
            for name, url in jobs.items()
        }
        for future, name in futures.items():
            _, receipt = future.result()
            receipt["name"] = name
            receipts.append(receipt)
    return receipts


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def collect(url, destination, expect_json=True):
    """Preserve actual responses and explicit failures; never synthesize a payload."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"url": url, "requested_at": utc_now()}
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "cfb-forecast/0.4",
                "Accept": "application/json" if expect_json else "text/html",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            receipt.update(
                status="ok",
                retrieved_at=utc_now(),
                resolved_url=response.url,
                bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
            )
        payload = (
            json.loads(body) if expect_json else body.decode("utf-8", errors="replace")
        )
        destination.write_bytes(body)
    except Exception as exc:
        receipt.update(
            status="unavailable",
            retrieved_at=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        payload = None
    destination.with_suffix(destination.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )
    return payload, receipt


class TableParser(HTMLParser):
    """Extract server-rendered tables without executing third-party scripts."""

    def __init__(self):
        super().__init__()
        self.tables, self.table, self.row, self.cell = [], None, None, None
        self.links = []
        self.table_links, self.table_fragments = [], []
        self._table_links, self._table_fragments = [], []
        self._row_links, self._row_fragments = [], []
        self._cell_links, self._anchor = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attr = dict(attrs)
            link = attr.get("href")
            if link:
                self.links.append(link)
                if self.cell is not None:
                    self._anchor = {
                        "href": link,
                        "title": attr.get("title"),
                        "text": [],
                    }
        if tag == "table":
            self.table = []
            self._table_links, self._table_fragments = [], []
        elif tag == "tr" and self.table is not None:
            self.row = []
            self._row_links, self._row_fragments = [], []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
            self._cell_links = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        if self._anchor is not None:
            self._anchor["text"].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._anchor is not None:
            self._anchor["text"] = " ".join("".join(self._anchor["text"]).split())
            self._cell_links.append(self._anchor)
            self._anchor = None
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self._row_links.append(self._cell_links)
            self._row_fragments.append([x.strip() for x in self.cell if x.strip()])
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self._table_links.append(self._row_links)
            self._table_fragments.append(self._row_fragments)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table_links.append(self._table_links)
            self.table_fragments.append(self._table_fragments)
            self.table = None


def provider_tables(season, root):
    """Inspect the user's named providers and preserve accessible stat tables."""
    jobs = [
        (
            "football_database",
            f"https://www.footballdb.com/college-football/stats/stats.html?yr={season}",
            False,
        ),
        (
            "cfbtrack_team_stats",
            "https://cfbtrack.com/stats/team-season-leaders",
            False,
        ),
        (
            "cfbtrack_player_stats",
            "https://cfbtrack.com/stats/player-season-leaders",
            False,
        ),
    ]
    records = []
    for name, url, _ in jobs:
        payload, receipt = collect(url, Path(root) / f"{name}.html", False)
        record = {"provider": name, **receipt}
        if payload:
            parser = TableParser()
            parser.feed(payload)
            record["tables"] = parser.tables
            record["table_count"] = len(parser.tables)
            record["table_links"] = parser.table_links
            record["table_fragments"] = parser.table_fragments
            record["season_verified"] = bool(
                re.search(rf"<title>[^<]*\b{season}\b", payload)
                or re.search(rf'<option[^>]*value="{season}"[^>]*selected', payload)
            )
        records.append(record)
    return records


def named_provider_features(providers, team_id, athletes):
    """Join verified-season stats by provider team ID or an exact unique player name."""
    result, evidence = {}, []
    names = {a.get("displayName") for a in athletes if a.get("displayName")}
    for source in providers:
        if not source.get("season_verified"):
            continue
        for table, links, fragments in zip(
            source.get("tables", []),
            source.get("table_links", []),
            source.get("table_fragments", []),
        ):
            for row, row_links, chunks in zip(table[1:], links[1:], fragments[1:]):
                flat_links = [link for cell in row_links for link in cell]
                team_links = [
                    int(m.group(1))
                    for link in flat_links
                    if (m := re.match(r"/teams/(\d+)(?:-|/|$)", link["href"]))
                ]
                if (
                    source["provider"] == "cfbtrack_team_stats"
                    and team_id in team_links
                ):
                    if len(row) > 4 and table[0][4].startswith("Total Yards / Game"):
                        value = pd.to_numeric(row[4].replace(",", ""), errors="coerce")
                        if pd.notna(value):
                            result["cfbtrack.total_yards_per_game"] = float(value)
                            evidence.append(
                                {
                                    "provider": source["provider"],
                                    "team_id": team_id,
                                    "row": row,
                                    "url": source["url"],
                                }
                            )
                elif (
                    source["provider"] == "cfbtrack_player_stats"
                    and team_id in team_links
                ):
                    if (
                        len(chunks) > 4
                        and chunks[4]
                        and table[0][4].startswith("Passing YDS")
                    ):
                        value = pd.to_numeric(
                            chunks[4][0].replace(",", ""), errors="coerce"
                        )
                        if pd.notna(value):
                            result["cfbtrack.leading_passing_yards"] = float(value)
                            evidence.append(
                                {
                                    "provider": source["provider"],
                                    "team_id": team_id,
                                    "row": row,
                                    "url": source["url"],
                                }
                            )
                elif source["provider"] == "football_database":
                    player = next(
                        (
                            link["text"]
                            for link in flat_links
                            if link["href"].startswith("/players/")
                            and link["text"] in names
                        ),
                        None,
                    )
                    if player and len(row) == len(table[0]):
                        attempts = pd.to_numeric(
                            row[1].replace(",", ""), errors="coerce"
                        )
                        if pd.isna(attempts) or attempts < result.get(
                            "football_database.passing.Att", -1
                        ):
                            continue
                        for heading, value in zip(table[0][1:], row[1:]):
                            number = pd.to_numeric(
                                value.replace(",", ""), errors="coerce"
                            )
                            if pd.notna(number):
                                result[f"football_database.passing.{heading}"] = float(
                                    number
                                )
                        evidence.append(
                            {
                                "provider": source["provider"],
                                "player": player,
                                "row": row,
                                "url": source["url"],
                            }
                        )
    return result, evidence


def _statistics(payload, opponent=False):
    results = (payload or {}).get("results", {})
    stats = results.get("opponent" if opponent else "stats", {})
    categories = stats if isinstance(stats, list) else stats.get("categories", [])
    return {
        f"{category['name']}.{stat['name']}": stat.get("value")
        for category in categories
        for stat in category.get("stats", [])
    }


def _roster(payload):
    return [
        athlete
        for group in (payload or {}).get("athletes", [])
        for athlete in group.get("items", [])
    ]


def current_injuries(reports, as_of, max_age_days=14):
    """Ignore stale, future-dated, or undated reports rather than treating them as live."""
    at = pd.Timestamp(as_of)
    current, rejected = [], []
    for report in reports or []:
        when = pd.to_datetime(report.get("date"), utc=True, errors="coerce")
        if (
            pd.isna(when)
            or when > at
            or (at - when).total_seconds() > max_age_days * 86400
        ):
            rejected.append(report)
        else:
            current.append(report)
    return current, rejected


def collect_game_inputs(games, season, root, workers=6):
    """Gather weather, injuries, lineups, player and team stats before inference.

    An empty injury response is unknown coverage, never evidence of a healthy team.
    A roster is never relabeled as a confirmed starting lineup.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    team_ids = sorted(set(games.home_id.astype(int)) | set(games.away_id.astype(int)))
    jobs = []
    for gid in games.game_id.astype(int):
        jobs.append((f"game_{gid}", f"{ESPN}/summary?event={gid}"))
    for tid in team_ids:
        for kind in ("roster", "statistics", "depthcharts", "injuries"):
            jobs.append(
                (f"team_{tid}_{kind}", f"{ESPN}/teams/{tid}/{kind}?season={season}")
            )

    def get(job):
        name, url = job
        return name, collect(url, root / "raw" / f"{name}.json")

    fetched = dict(ThreadPoolExecutor(max_workers=workers).map(get, jobs))
    league_injuries, league_receipt = collect(
        f"{ESPN}/injuries", root / "raw" / "league_injuries.json"
    )
    injury_teams = {
        int(t["id"]): t.get("injuries", [])
        for t in (league_injuries or {}).get("injuries", [])
        if str(t.get("id", "")).isdigit()
    }
    providers = provider_tables(season, root / "raw")
    rows, documents = [], []
    for game in games.itertuples(index=False):
        summary, receipt = fetched[f"game_{int(game.game_id)}"]
        summary = summary or {}
        header = summary.get("header", {})
        season_value = header.get("season", {}).get("year")
        if season_value is not None and int(season_value) != int(season):
            raise ValueError(f"Wrong season in ESPN game {game.game_id}")
        weather = summary.get("gameInfo", {}).get("weather") or {}
        record = {
            "game_id": int(game.game_id),
            "season": int(season),
            "kickoff": game.start_date.isoformat(),
            "weather": weather,
            "venue": summary.get("gameInfo", {}).get("venue"),
            "source_url": receipt["url"],
            "retrieved_at": receipt["retrieved_at"],
            "teams": {},
        }
        features = {
            "game_id": int(game.game_id),
            "season": int(season),
            "weather_temperature_f": weather.get("temperature"),
            "weather_gust_mph": weather.get("gust"),
            "weather_precipitation_probability": weather.get("precipitation"),
            "weather_status": "available" if weather else "unavailable",
        }
        for side in ("home", "away"):
            tid = int(getattr(game, f"{side}_id"))
            stats, stats_receipt = fetched[f"team_{tid}_statistics"]
            roster, roster_receipt = fetched[f"team_{tid}_roster"]
            depth, depth_receipt = fetched[f"team_{tid}_depthcharts"]
            injuries, injury_receipt = fetched[f"team_{tid}_injuries"]
            for payload in (stats, roster, depth):
                if (
                    payload
                    and payload.get("season", {}).get("year") is not None
                    and int(payload["season"]["year"]) != season
                ):
                    raise ValueError(f"Wrong-season team input: {tid}")
            athletes = _roster(roster)
            reports, stale_reports = current_injuries(
                injury_teams.get(tid, []), utc_now()
            )
            confirmed_depth = (depth or {}).get("depthchart") or (depth or {}).get(
                "depthcharts"
            )
            leaders = next(
                (
                    team.get("leaders", [])
                    for team in summary.get("leaders", [])
                    if int(team.get("team", {}).get("id", -1)) == tid
                ),
                [],
            )
            # Summary leaders sometimes use a nested structure; retain the original too.
            lineup_status = "available" if confirmed_depth else "unavailable"
            injury_status = (
                "reported"
                if reports
                else "stale_reports"
                if stale_reports
                else "unavailable"
            )
            team = {
                "team_id": tid,
                "statistics": _statistics(stats),
                "opponent_statistics": _statistics(stats, True),
                "roster": athletes,
                "depthchart": confirmed_depth,
                "injuries": reports,
                "stale_injury_reports": stale_reports,
                "reported_injuries": reports,
                "leaders": leaders,
                "lineup_status": lineup_status,
                "injury_status": injury_status,
                "sources": [
                    stats_receipt,
                    roster_receipt,
                    depth_receipt,
                    injury_receipt,
                    league_receipt,
                ],
            }
            record["teams"][side] = team
            features[f"{side}_team_stats_status"] = (
                "available" if team["statistics"] else "unavailable"
            )
            features[f"{side}_player_stats_status"] = (
                "available" if leaders else "unavailable"
            )
            features[f"{side}_lineup_status"] = lineup_status
            features[f"{side}_injury_status"] = injury_status
            features[f"{side}_reported_injuries"] = len(reports) if reports else None
            for group in leaders:
                for player in group.get("leaders", []):
                    features[f"{side}_player.{group.get('name')}.value"] = player.get(
                        "value"
                    )
            for name, value in team["statistics"].items():
                features[f"{side}_{name}"] = value
            external, external_evidence = named_provider_features(
                providers, tid, athletes
            )
            team["named_provider_statistics"] = external
            team["named_provider_evidence"] = external_evidence
            for name, value in external.items():
                features[f"{side}_{name}"] = value
        record["summary_leaders"] = summary.get("leaders", [])
        record["boxscore"] = summary.get("boxscore", {})
        record["sources_complete"] = all(
            features.get(f"{s}_{kind}_status") in ("available", "reported")
            for s in ("home", "away")
            for kind in ("team_stats", "player_stats", "lineup", "injury")
        ) and bool(weather)
        features["inputs_complete"] = record["sources_complete"]
        rows.append(features)
        documents.append(record)
    (root / "game_inputs.json").write_text(json.dumps(documents, indent=2))
    pd.DataFrame(rows).to_csv(root / "features.csv", index=False)
    (root / "provider_status.json").write_text(json.dumps(providers, indent=2))
    (root / "source_registry.json").write_text(json.dumps(source_registry(season), indent=2))
    return pd.DataFrame(rows), documents
