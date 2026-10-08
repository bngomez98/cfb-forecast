"""Timestamped market history and cutoff-safe market diagnostics."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import require_columns, timestamp

NUMERIC = ["home_spread", "total_line", "home_decimal", "away_decimal", "home_ml_decimal", "away_ml_decimal"]
KEYS = ["game_id", "provider", "observed_at"]


def normalize_market_history(frame, default_provider="unspecified", source=None):
    require_columns(frame, ["game_id", "observed_at"], "market snapshots")
    f = frame.copy()
    f["game_id"] = pd.to_numeric(f.game_id, errors="raise").astype("int64")
    f["observed_at"] = pd.to_datetime(f.observed_at, utc=True, errors="raise")
    if "provider" not in f:
        f["provider"] = default_provider
    f["provider"] = f.provider.fillna(default_provider).astype(str).str.strip()
    if (f.provider == "").any():
        raise ValueError("Market provider cannot be blank")
    if "source" not in f:
        f["source"] = source
    elif source is not None:
        f["source"] = f["source"].fillna(source)
    for col in NUMERIC:
        if col in f:
            f[col] = pd.to_numeric(f[col], errors="coerce")
            bad = f[col].notna() & ~np.isfinite(f[col])
            if bad.any():
                raise ValueError(f"Invalid market value {col}")
    for col in ["home_decimal", "away_decimal", "home_ml_decimal", "away_ml_decimal"]:
        if col in f and (f[col].dropna() <= 1).any():
            raise ValueError(f"Decimal odds in {col} must exceed 1")
    if "total_line" in f and (f.total_line.dropna() < 0).any():
        raise ValueError("total_line cannot be negative")
    if f.duplicated(KEYS).any():
        raise ValueError("Market snapshots require unique game_id/provider/observed_at")
    if not any(col in f and f[col].notna().any() for col in NUMERIC):
        raise ValueError("Market history contains no usable prices or lines")
    return f.sort_values(KEYS).reset_index(drop=True)


def read_market_history(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if raw.empty:
        return raw
    return normalize_market_history(raw)


def write_market_history(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    f = normalize_market_history(frame)
    if path.suffix.lower() in {".parquet", ".pq"}:
        f.to_parquet(path, index=False)
    else:
        f.to_csv(path, index=False)
    return path


def append_market_history(store_path, incoming, default_provider="unspecified", source=None):
    """Append snapshots without silently overwriting conflicting historical records."""
    new = normalize_market_history(incoming, default_provider=default_provider, source=source)
    old = read_market_history(store_path)
    if old.empty:
        combined = new
    else:
        overlap = old.merge(new, on=KEYS, how="inner", suffixes=("_old", "_new"))
        for col in NUMERIC:
            a, b = f"{col}_old", f"{col}_new"
            if a in overlap and b in overlap:
                same = (overlap[a].isna() & overlap[b].isna()) | np.isclose(overlap[a], overlap[b], equal_nan=True)
                if not bool(np.all(same)):
                    raise ValueError(f"Conflicting market history for duplicate key in {col}")
        combined = pd.concat([old, new], ignore_index=True).drop_duplicates(KEYS, keep="first")
    write_market_history(combined, store_path)
    return combined


def import_market_file(input_path, store_path, provider="unspecified", source=None):
    p = Path(input_path)
    frame = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
    return append_market_history(store_path, frame, default_provider=provider, source=source or str(p))


def _eligible(frame, game_id, as_of, kickoff, provider=None):
    f = normalize_market_history(frame)
    rows = f[(f.game_id == int(game_id)) & (f.observed_at <= timestamp(as_of)) & (f.observed_at < timestamp(kickoff))]
    if provider is not None:
        rows = rows[rows.provider == provider]
    return rows


def market_at_cutoff(frame, game_id, as_of, kickoff, provider=None):
    """Return a named provider's latest quote, or a median consensus of latest provider quotes."""
    rows = _eligible(frame, game_id, as_of, kickoff, provider)
    if rows.empty:
        return {}
    if provider is not None or rows.provider.nunique() == 1:
        r = rows.sort_values("observed_at").iloc[-1]
        result = {"market_observed_at": r.observed_at.isoformat(), "market_provider": str(r.provider), "market_provider_count": 1}
        for col in NUMERIC:
            if col in r and pd.notna(r[col]):
                result[col] = float(r[col])
    else:
        latest = rows.sort_values("observed_at").groupby("provider", as_index=False).tail(1)
        result = {
            "market_observed_at": latest.observed_at.max().isoformat(),
            "market_provider": "median_consensus_latest_by_provider",
            "market_provider_count": int(latest.provider.nunique()),
            "market_providers": ",".join(sorted(latest.provider.unique())),
        }
        for col in NUMERIC:
            if col in latest and latest[col].notna().any():
                result[col] = float(latest[col].median())
    if "home_ml_decimal" in result and "away_ml_decimal" in result:
        h, a = 1 / result["home_ml_decimal"], 1 / result["away_ml_decimal"]
        result["market_home_win_probability_no_vig"] = h / (h + a)
    return result


def market_history_summary(frame, game_id, as_of, kickoff, provider=None):
    rows = _eligible(frame, game_id, as_of, kickoff, provider)
    if rows.empty:
        return {"market_snapshot_count": 0}
    # summarize consensus by timestamp to make opening/current movement inspectable
    series = []
    for observed, g in rows.groupby("observed_at"):
        rec = {"observed_at": observed}
        for col in ["home_spread", "total_line"]:
            if col in g and g[col].notna().any():
                rec[col] = float(g[col].median())
        series.append(rec)
    s = pd.DataFrame(series).sort_values("observed_at")
    out = {
        "market_snapshot_count": int(len(rows)),
        "market_distinct_timestamps": int(rows.observed_at.nunique()),
        "market_first_observed_at": rows.observed_at.min().isoformat(),
        "market_last_observed_at": rows.observed_at.max().isoformat(),
    }
    for col, prefix in [("home_spread", "spread"), ("total_line", "total")]:
        if col in s and s[col].notna().any():
            ss = s.dropna(subset=[col])
            out[f"market_open_{prefix}"] = float(ss.iloc[0][col])
            out[f"market_current_{prefix}"] = float(ss.iloc[-1][col])
            out[f"market_{prefix}_move"] = float(ss.iloc[-1][col] - ss.iloc[0][col])
    return out


def empty_market_history(path):
    cols = KEYS + NUMERIC + ["source"]
    f = pd.DataFrame({c: pd.Series(dtype="float64" if c in NUMERIC else "object") for c in cols})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        f.to_parquet(path, index=False)
    else:
        f.to_csv(path, index=False)
    meta = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "append-only point-in-time bookmaker snapshots for prospective collection",
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return path


def _download_csv(url):
    import io
    import urllib.request
    req=urllib.request.Request(url,headers={"User-Agent":"cfb-forecast/0.2"})
    with urllib.request.urlopen(req,timeout=60) as response:
        return pd.read_csv(io.BytesIO(response.read()))


def capture_sportsdataverse_snapshot(season, store_path, observed_at=None):
    """Capture the public SportsDataverse resolved ESPN spread/total as observed now.

    Repeating this command over time creates a prospective line-history series.
    It intentionally records retrieval time, not an invented bookmaker quote time.
    """
    at=timestamp(observed_at or datetime.now(UTC))
    url=f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_betting/betting_{int(season)}.csv"
    raw=_download_csv(url)
    require_columns(raw,["game_id","home_team_spread","over_under","game_spread_available"],"SportsDataverse betting")
    usable=raw[raw.game_spread_available.astype(str).str.lower().isin(["true","1"])].copy()
    snap=pd.DataFrame({
        "game_id":pd.to_numeric(usable.game_id,errors="raise").astype("int64"),
        "provider":"sportsdataverse_espn_resolved",
        "observed_at":at,
        "home_spread":pd.to_numeric(usable.home_team_spread,errors="coerce"),
        "total_line":pd.to_numeric(usable.over_under,errors="coerce"),
        "source":url,
    }).dropna(subset=["home_spread","total_line"],how="all")
    return append_market_history(store_path,snap,default_provider="sportsdataverse_espn_resolved",source=url)


def backfill_sportsdataverse_closing(seasons, out_path):
    """Download public resolved historical lines as a benchmark-only table.

    Historical rows are not assigned observed_at and must never be used as
    point-in-time forecast inputs. They are closing/resolved market benchmarks.
    """
    rows=[]; retrieved=datetime.now(UTC).isoformat()
    for season in seasons:
        url=f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_betting/betting_{int(season)}.csv"
        f=_download_csv(url).copy(); f["retrieved_at"]=retrieved; f["source_url"]=url; rows.append(f)
    out=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    path=Path(out_path); path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path,index=False)
    return out
