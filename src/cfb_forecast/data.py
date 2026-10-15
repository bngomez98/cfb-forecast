"""Free source ingestion, schema guards, snapshots and outcome availability."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import urllib.error
import urllib.request
import numpy as np
import pandas as pd

RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SCHEDULE_FIELDS = ["game_id", "season", "week", "start_date", "completed", "neutral_site", "home_id", "away_id", "home_team", "away_team", "home_points", "away_points"]
ADV_FIELDS = ["game_id", "pos_team_id", "yards_per_play", "yards_per_pass", "yards_per_rush", "first_downs_created_rate"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def timestamp(value):
    t = pd.Timestamp(value)
    if t.tz is None:
        raise ValueError("Use an explicit UTC offset, e.g. 2026-10-02T12:00:00Z")
    return t.tz_convert("UTC")

def require_columns(df, cols, label):
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}; source schema changed")

def booleans(series, name):
    mapping = {"true": True, "false": False, "1": True, "0": False}
    s = series.astype(str).str.lower().map(mapping)
    if s.isna().any():
        raise ValueError(f"{name}: missing or invalid boolean values")
    return s.astype(bool)

def normalize_schedule(df):
    require_columns(df, SCHEDULE_FIELDS, "schedule")
    df = df.copy()
    for c in ("completed", "neutral_site"):
        df[c] = booleans(df[c], c)
    for c in ("game_id", "home_id", "away_id", "season", "week"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    df["start_date"] = pd.to_datetime(df.start_date, utc=True, format="ISO8601", errors="raise")
    if df.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs in schedule; refusing ambiguous outcomes")
    if (df.home_id == df.away_id).any():
        raise ValueError("A game cannot list the same team on both sides")
    for c in ("home_points", "away_points"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df.completed & (df.home_points.isna() | df.away_points.isna() | (df.home_points < 0) | (df.away_points < 0))
    # Some lower-division/canceled source rows are marked completed with NA
    # scores. Quarantine their outcomes explicitly rather than guessing scores.
    df["invalid_completed_outcome"] = bad
    df["result_usable"] = df.completed & ~bad & df.home_points.ne(df.away_points)
    if "home_division" in df and "away_division" in df:
        df["fbs_game"] = df.home_division.eq("fbs") & df.away_division.eq("fbs")
    elif "fbs_game" in df:
        df["fbs_game"] = booleans(df.fbs_game, "fbs_game")
    else:
        raise ValueError("Schedule must identify FBS games via division columns or fbs_game")
    # No maximum-play-score substitution. Schedule outcomes are authoritative.
    # Unknown actual final whistle is conservatively proxied at kickoff + 8h.
    df["result_available_at"] = df.start_date + RESULT_LAG
    if "completed_at" in df:
        actual = pd.to_datetime(df.completed_at, utc=True, format="ISO8601", errors="coerce")
        df["result_available_at"] = actual.fillna(df.result_available_at)
    if (df.result_available_at <= df.start_date).any():
        raise ValueError("Result availability must be later than kickoff")
    return df.sort_values(["start_date", "game_id"]).reset_index(drop=True)

def read_frame(path):
    return pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)

RESULTS_OVERLAY = "results_overlay.csv"
# A final score is treated as available this long after kickoff when the real completion time is unknown.
RESULT_LAG = pd.Timedelta(hours=8)


def apply_results_overlay(raw, overlay):
    """Fill final scores the schedule snapshot does not have yet from cross-checked live results.

    Only games the schedule lists as unfinished (or without scores) are filled, and
    ``completed_at`` is set to when the agreeing sources were observed, so the
    result cannot enter any forecast cut off before that moment. Scores the schedule
    already has are never overwritten.
    """
    if overlay.empty:
        return raw
    raw = raw.copy()
    ov = overlay.drop_duplicates("game_id", keep="last").set_index("game_id")
    ids = pd.to_numeric(raw.game_id, errors="coerce")
    done = raw.completed.astype(str).str.lower().isin(["true", "1"])
    scored = pd.to_numeric(raw.home_points, errors="coerce").notna() & pd.to_numeric(raw.away_points, errors="coerce").notna()
    fill = ids.isin(ov.index) & ~(done & scored)
    if "completed_at" not in raw:
        raw["completed_at"] = pd.NA
    for col in ("completed", "home_points", "away_points", "completed_at"):
        raw[col] = raw[col].astype(object)
    sel = ids[fill]
    raw.loc[fill, "home_points"] = ov.loc[sel, "home_points"].to_numpy()
    raw.loc[fill, "away_points"] = ov.loc[sel, "away_points"].to_numpy()
    raw.loc[fill, "completed_at"] = ov.loc[sel, "completed_at"].to_numpy()
    raw.loc[fill, "completed"] = True
    raw["completed"] = raw.completed.astype(str)
    return raw


def load_data(root):
    root = Path(root)
    paths = sorted(root.glob("schedule_*.csv")) + sorted(root.glob("schedule_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No schedule files in {root}; run the fetch command")
    raw = pd.concat([read_frame(p) for p in paths], ignore_index=True)
    overlay_path = root / RESULTS_OVERLAY
    if overlay_path.exists():
        raw = apply_results_overlay(raw, pd.read_csv(overlay_path))
        paths = paths + [overlay_path]
    schedules = normalize_schedule(raw)
    paths_adv = sorted(root.glob("adv_team_*.parquet")) + sorted(root.glob("adv_team_*.csv"))
    adv = pd.concat([read_frame(p) for p in paths_adv], ignore_index=True) if paths_adv else pd.DataFrame(columns=ADV_FIELDS)
    require_columns(adv, ADV_FIELDS, "play-by-play summaries")
    for column in ("game_id", "pos_team_id"):
        adv[column] = pd.to_numeric(adv[column], errors="raise").astype("int64")
    for column in ADV_FIELDS[2:]:
        adv[column] = pd.to_numeric(adv[column], errors="raise")
        if np.isinf(adv[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Nonfinite advanced statistic: {column}")
    known = adv.merge(schedules[["game_id", "home_id", "away_id"]], on="game_id", how="inner", validate="many_to_one")
    if (~(known.pos_team_id.eq(known.home_id) | known.pos_team_id.eq(known.away_id))).any():
        raise ValueError("Advanced statistics team does not belong to the scheduled game")
    if adv.duplicated(["game_id", "pos_team_id"]).any():
        raise ValueError("Duplicate (game_id, pos_team_id) PBP summary rows")
    return schedules, adv, paths + paths_adv

def fetch(root, seasons, refresh=False, raw_pbp=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    for year in seasons:
        jobs = [(f"schedule_{year}.csv", f"{RELEASE}/cfb_schedules/cfb_schedules_{year}.csv.gz"),
                (f"adv_team_{year}.parquet", f"{RELEASE}/espn_cfb_adv_team/adv_team_{year}.parquet")]
        if raw_pbp:
            jobs.append((f"pbp_{year}.parquet", f"{RELEASE}/espn_cfb_pbp/play_by_play_{year}.parquet"))
        for name, url in jobs:
            target = root / name
            if target.exists() and not refresh:
                print(f"cached {name}", flush=True); continue
            tmp = target.with_suffix(".download")
            saved = False
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "cfb-forecast/0.1"})
                    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as f:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            f.write(block)
                    if name.startswith("schedule_"):
                        frame = pd.read_csv(tmp, compression="gzip")
                        normalize_schedule(frame); frame.to_csv(tmp, index=False)
                    else:
                        import pyarrow.parquet as pq
                        pq.ParquetFile(tmp)  # fail on HTML / incomplete downloads
                    saved = True
                    break
                except urllib.error.HTTPError as exc:
                    tmp.unlink(missing_ok=True)
                    if exc.code == 404:
                        manifest["files"][name] = {"url": url, "status": "unavailable", "http_status": 404}
                        manifest_path.write_text(json.dumps(manifest, indent=2))
                        print(f"unavailable {name}", flush=True)
                        break
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
                except Exception:
                    tmp.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            if not saved:
                continue
            tmp.replace(target)
            manifest["files"][name] = {"url": url, "sha256": sha256(target), "downloaded_at": datetime.now(timezone.utc).isoformat(), "bytes": target.stat().st_size}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"downloaded {name}", flush=True)
    return manifest

def audit(schedules, summaries):
    rows = []
    known = set(summaries.game_id)
    for year, g in schedules.groupby("season"):
        f = g[g.fbs_game & g.result_usable]
        rows.append({"season": int(year), "schedule_games": len(g), "completed_fbs_games": len(f), "invalid_completed_outcomes":int(g.invalid_completed_outcome.sum()), "invalid_fbs_outcomes":int((g.invalid_completed_outcome & g.fbs_game).sum()), "summary_games": int(f.game_id.isin(known).sum()), "missing_summary_games": int((~f.game_id.isin(known)).sum())})
    return pd.DataFrame(rows)
