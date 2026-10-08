"""Retrieve live college-football scores from ESPN's public scoreboard API."""
import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"


def _request_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "cfb-forecast/0.3"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _event_row(event, retrieved_at):
    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    home = next((x for x in competitors if x.get("homeAway") == "home"), {})
    away = next((x for x in competitors if x.get("homeAway") == "away"), {})
    status = competition.get("status") or event.get("status") or {}
    status_type = status.get("type") or {}
    event_id = event.get("id")
    if not event_id or not home or not away:
        return None
    return {
        "game_id": int(event_id) if str(event_id).isdigit() else str(event_id),
        "season": int((event.get("season") or {}).get("year", 0)),
        "week": int(((event.get("week") or {}).get("number")) or 0),
        "status": status_type.get("name") or status_type.get("state") or "unknown",
        "status_detail": status_type.get("shortDetail") or status.get("type", {}).get("shortDetail"),
        "is_final": bool(status_type.get("completed", False)),
        "period": status.get("period"),
        "clock": status.get("displayClock"),
        "kickoff": event.get("date"),
        "home_team": (home.get("team") or {}).get("displayName"),
        "away_team": (away.get("team") or {}).get("displayName"),
        "home_id": (home.get("team") or {}).get("id"),
        "away_id": (away.get("team") or {}).get("id"),
        "home_points": float(home.get("score")) if home.get("score") not in (None, "") else None,
        "away_points": float(away.get("score")) if away.get("score") not in (None, "") else None,
        "provider_updated_at": event.get("date"),
        "retrieved_at": retrieved_at,
        "provider": "espn-public-scoreboard",
    }


def fetch_scoreboard(season, dates):
    rows = []
    for date in dates:
        query = urllib.parse.urlencode({"dates": pd.Timestamp(date).strftime("%Y%m%d"), "limit": 500})
        payload = _request_json(f"{ESPN_SCOREBOARD}?{query}")
        retrieved_at = datetime.now(UTC).isoformat()
        for event in payload.get("events", []):
            row = _event_row(event, retrieved_at)
            if row is not None:
                row["season"] = int(season)
                rows.append(row)
    return pd.DataFrame(rows)


def refresh_results(data_root="data", output_root="artifacts/results", season=None, week=None, start=None, end=None):
    """Poll ESPN for a date range and update normalized results for a season/week."""
    data_root = Path(data_root)
    output_root = Path(output_root)
    if season is None:
        raise ValueError("season is required")
    schedule_paths = sorted(data_root.glob(f"schedule_{int(season)}.csv"))
    if not schedule_paths:
        raise FileNotFoundError(f"No schedule_{season}.csv found in {data_root}")
    schedule = pd.read_csv(schedule_paths[0], low_memory=False)
    schedule["start_date"] = pd.to_datetime(schedule["start_date"], utc=True)
    if week is not None:
        schedule = schedule[pd.to_numeric(schedule["week"], errors="coerce") == int(week)]
    if start:
        start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
    else:
        start_ts = schedule.start_date.min().normalize()
    if end:
        end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")
    else:
        end_ts = schedule.start_date.max().normalize()
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("No scheduled games match the selected season/week")
    dates = pd.date_range(start_ts, end_ts, freq="D", tz="UTC")
    current = fetch_scoreboard(season, dates)
    if current.empty:
        raise RuntimeError("ESPN returned no games for the requested date range")

    label = f"{int(season)}-week-{int(week)}" if week is not None else f"{int(season)}-{start_ts:%Y%m%d}-{end_ts:%Y%m%d}"
    destination = output_root / label
    snapshot_dir = destination / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(UTC)
    current.to_csv(snapshot_dir / f"{retrieved_at:%Y%m%dT%H%M%SZ}.csv", index=False)

    results_path = destination / "results.csv"
    if results_path.exists():
        previous = pd.read_csv(results_path)
        current = pd.concat([previous, current], ignore_index=True)
    current["retrieved_at"] = pd.to_datetime(current["retrieved_at"], utc=True, format="mixed")
    current = current.sort_values("retrieved_at").drop_duplicates("game_id", keep="last")
    current.to_csv(results_path, index=False)
    manifest = {
        "provider": "ESPN public scoreboard API",
        "endpoint": ESPN_SCOREBOARD,
        "season": int(season),
        "week": int(week) if week is not None else None,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "retrieved_at": retrieved_at.isoformat(),
        "games": int(len(current)),
        "final_games": int(current.is_final.sum()),
    }
    (destination / "results_manifest.json").write_text(json.dumps(manifest, indent=2))
    return current, manifest
