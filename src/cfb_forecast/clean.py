"""Data cleaning and organization layer.

Raw source files (sportsdataverse schedules, PBP summaries, market history)
are never rewritten in place.  This module reads them, applies explicit,
auditable cleaning rules, and writes organized artifacts:

    data/clean/games.parquet        one row per game, winner/loser orientation
    data/clean/team_games.parquet   long format: one row per team per game
    data/clean/quarantine.csv       every dropped row with a reason code
    data/clean/cleaning_report.json counts of each rule fired, per file

Cleaning rules (no smoothing, no imputation, no rating construction):
  * duplicate game ids / duplicate keys -> keep first, quarantine the rest
  * completed games with missing or negative scores -> quarantine
  * completed 0-0 rows -> cancelled/postponed sources artifact -> quarantine
  * genuine ties at D3 (nonzero equal scores) -> flagged, kept for audit,
    excluded from win/loss oriented outputs (there is no winner to orient)
  * missing division labels -> inferred from a team-id lookup learned from
    unambiguous rows; otherwise left null and flagged ``division_unknown``
  * team names trimmed; timestamps coerced to tz-aware UTC
  * market history: non-finite spreads quarantined, duplicates collapsed to
    the latest observation per (game_id, provider)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REASON_DUPLICATE_KEY = "duplicate_key"
REASON_MISSING_SCORE = "completed_missing_score"
REASON_NEGATIVE_SCORE = "negative_score"
REASON_ZERO_ZERO = "completed_zero_zero"
REASON_TIE = "tie_no_winner"
REASON_UNPARSEABLE_TIMESTAMP = "unparseable_timestamp"
REASON_NONFINITE_SPREAD = "nonfinite_spread"
REASON_SAME_TEAM_SIDES = "same_team_both_sides"

DIVISION_ORDER = {"fbs": 0, "fcs": 1, "ii": 2, "iii": 3}


def _utc(series):
    return pd.to_datetime(series, utc=True, errors="coerce")


def _truthy(series):
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def build_division_lookup(frames):
    """Learn team_id -> division from rows where both sides are labelled."""
    pairs = []
    for df in frames:
        if "home_division" not in df or "away_division" not in df:
            continue
        for id_col, div_col in (("home_id", "home_division"), ("away_id", "away_division")):
            sub = df[[id_col, div_col]].dropna()
            pairs.append(sub.rename(columns={id_col: "team_id", div_col: "division"}))
    if not pairs:
        return {}
    lookup = pd.concat(pairs, ignore_index=True)
    lookup["division"] = lookup.division.astype(str).str.strip().str.lower()
    # A team id must map consistently; conflicts are dropped (left unknown).
    counts = lookup.groupby("team_id").division.nunique()
    ok = counts[counts == 1].index
    lookup = lookup[lookup.team_id.isin(ok)].drop_duplicates("team_id")
    return dict(zip(lookup.team_id, lookup.division, strict=True))


def clean_schedule_frame(df, source, lookup):
    """Return (clean_rows, quarantined_rows) for one raw schedule frame."""
    df = df.copy()
    df["source_file"] = source
    df["_row"] = range(len(df))
    for col in ("home_team", "away_team", "venue"):
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    df["start_date_utc"] = _utc(df.get("start_date"))
    df["completed_flag"] = _truthy(df.get("completed"))
    df["neutral_flag"] = _truthy(df.get("neutral_site"))
    for col in ("home_points", "away_points"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    quarantine = []

    def fire(mask, reason):
        nonlocal df
        hit = df[mask]
        if len(hit):
            q = hit.copy()
            q["reason"] = reason
            quarantine.append(q)
            df = df[~mask]

    bad_time = df.start_date_utc.isna()
    fire(bad_time, REASON_UNPARSEABLE_TIMESTAMP)
    fire(df.home_id.eq(df.away_id), REASON_SAME_TEAM_SIDES)
    dup = df.duplicated("game_id", keep="first")
    fire(dup, REASON_DUPLICATE_KEY)
    fire(df.completed_flag & (df.home_points.isna() | df.away_points.isna()), REASON_MISSING_SCORE)
    fire(df.completed_flag & ((df.home_points < 0) | (df.away_points < 0)), REASON_NEGATIVE_SCORE)
    zero_zero = df.completed_flag & df.home_points.eq(0) & df.away_points.eq(0)
    fire(zero_zero, REASON_ZERO_ZERO)

    df["is_tie"] = df.completed_flag & df.home_points.eq(df.away_points)
    df["result_usable"] = df.completed_flag & ~df.is_tie

    # Division inference: fill only where raw label was missing.
    for side in ("home", "away"):
        col = f"{side}_division"
        if col not in df:
            df[col] = pd.NA
        df[col] = df[col].astype("string").str.strip().str.lower()
        missing = df[col].isna() | df[col].eq("")
        inferred = df.loc[missing, f"{side}_id"].map(lookup)
        df.loc[missing, col] = inferred
        df[f"{side}_division_inferred"] = missing & df[col].notna()
    df["division_unknown"] = df.home_division.isna() | df.away_division.isna()

    df["fbs_game"] = df.home_division.eq("fbs") & df.away_division.eq("fbs")
    df["winner_known"] = df.result_usable
    keep_cols = [
        "game_id",
        "season",
        "week",
        "season_type",
        "start_date_utc",
        "completed_flag",
        "result_usable",
        "is_tie",
        "neutral_flag",
        "conference_game",
        "venue",
        "attendance",
        "home_id",
        "home_team",
        "home_division",
        "home_conference",
        "home_points",
        "away_id",
        "away_team",
        "away_division",
        "away_conference",
        "away_points",
        "home_division_inferred",
        "away_division_inferred",
        "division_unknown",
        "fbs_game",
        "winner_known",
        "source_file",
        "_row",
    ]
    out = df[[c for c in keep_cols if c in df.columns]]
    if quarantine:
        quar = pd.concat(quarantine, ignore_index=True)
    else:
        quar = pd.DataFrame(columns=list(df.columns) + ["reason"])
    return out, quar


def organize_games(clean, ties_to_quarantine=True):
    """Winner/loser orientation plus margin/total columns (exact, not smoothed)."""
    quar_extra = None
    work = clean
    if ties_to_quarantine and clean.is_tie.any():
        ties = clean[clean.is_tie].copy()
        ties["reason"] = REASON_TIE
        quar_extra = ties.drop(columns=[c for c in ties.columns if c == "reason"], errors="ignore")
        quar_extra["reason"] = REASON_TIE
        work = clean[~clean.is_tie]

    g = work.copy()
    won = g.result_usable & (g.home_points > g.away_points)
    g["winner_id"] = g.home_id.where(won, g.away_id.where(g.result_usable, pd.NA))
    g["loser_id"] = g.away_id.where(won, g.home_id.where(g.result_usable, pd.NA))
    g["winner_score"] = g.home_points.where(won, g.away_points.where(g.result_usable, pd.NA))
    g["loser_score"] = g.away_points.where(won, g.home_points.where(g.result_usable, pd.NA))
    g["margin"] = (g.winner_score - g.loser_score).astype("Float64")
    g["total_points"] = (g.winner_score + g.loser_score).astype("Float64")
    g["home_win"] = won.astype("boolean")
    g = g.sort_values(["start_date_utc", "game_id"]).reset_index(drop=True)
    return g, quar_extra


def organize_team_games(games):
    """Long format: one row per team per game with points_for/against."""
    home = games.assign(
        team_id=games.home_id,
        team=games.home_team,
        division=games.home_division,
        venue_side="home",
        points_for=games.home_points,
        points_against=games.away_points,
    )
    away = games.assign(
        team_id=games.away_id,
        team=games.away_team,
        division=games.away_division,
        venue_side="away",
        points_for=games.away_points,
        points_against=games.home_points,
    )
    cols = [
        "game_id",
        "season",
        "week",
        "season_type",
        "start_date_utc",
        "team_id",
        "team",
        "division",
        "venue_side",
        "points_for",
        "points_against",
        "result_usable",
        "fbs_game",
    ]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)
    long["margin_for"] = long.points_for - long.points_against
    long["won"] = (long.result_usable & long.margin_for.gt(0)).astype("boolean")
    return long.sort_values(["start_date_utc", "game_id", "venue_side"]).reset_index(drop=True)


def clean_market_history(df):
    """Dedupe to latest observation per (game_id, provider); drop bad spreads."""
    df = df.copy()
    df["observed_at_utc"] = _utc(df.observed_at)
    df["home_spread"] = pd.to_numeric(df.home_spread, errors="coerce")
    bad = df.home_spread.isna() | ~df.home_spread.map(math.isfinite)
    quarantined = df[bad | df.observed_at_utc.isna()].copy()
    quarantined["reason"] = REASON_NONFINITE_SPREAD
    keep = df[~bad & df.observed_at_utc.notna()]
    keep = keep.sort_values("observed_at_utc").drop_duplicates(["game_id", "provider"], keep="last")
    return keep.reset_index(drop=True), quarantined.reset_index(drop=True)


def clean_adv_summaries(df):
    """Drop duplicate (game_id, pos_team_id) rows keeping the last."""
    df = df.copy()
    dup_mask = df.duplicated(["game_id", "pos_team_id"], keep="last")
    quarantined = df[dup_mask].copy()
    quarantined["reason"] = REASON_DUPLICATE_KEY
    return df[~dup_mask].reset_index(drop=True), quarantined.reset_index(drop=True)


def clean_all(data_root, out_dir=None):
    """Read raw files, apply every rule, write organized artifacts + report."""
    data_root = Path(data_root)
    out_dir = Path(out_dir) if out_dir else data_root / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    sched_paths = sorted(data_root.glob("schedule_*.csv")) + sorted(data_root.glob("schedule_*.parquet"))
    if not sched_paths:
        raise FileNotFoundError(f"No schedule files in {data_root}; run the fetch command")
    raw_frames = []
    for p in sched_paths:
        frame = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        frame["source_file"] = p.name
        raw_frames.append(frame)
    lookup = build_division_lookup(raw_frames)

    cleaned, quarantined = [], []
    per_file = {}
    for p, frame in zip(sched_paths, raw_frames, strict=True):
        out, quar = clean_schedule_frame(frame, p.name, lookup)
        cleaned.append(out)
        quarantined.append(quar)
        per_file[p.name] = {
            "raw_rows": int(len(frame)),
            "kept_rows": int(len(out)),
            "quarantined_rows": int(len(quar)),
            "ties_kept_for_audit": int(out.is_tie.sum()) if len(out) else 0,
        }
    clean = pd.concat(cleaned, ignore_index=True)
    games, tie_quar = organize_games(clean)
    team_games = organize_team_games(games)

    quar_frames = [q for q in quarantined if len(q)]
    if tie_quar is not None and len(tie_quar):
        quar_frames.append(tie_quar)
    quarantine = (
        pd.concat(quar_frames, ignore_index=True) if quar_frames else pd.DataFrame(columns=["reason"])
    )

    market_q = adv_q = None
    market_path = data_root / "market_history.csv"
    if market_path.exists():
        market, market_q = clean_market_history(pd.read_csv(market_path))
        market.to_parquet(out_dir / "market.parquet", index=False)
    adv_paths = sorted(data_root.glob("adv_team_*.parquet")) + sorted(data_root.glob("adv_team_*.csv"))
    if adv_paths:
        adv = pd.concat(
            [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in adv_paths],
            ignore_index=True,
        )
        adv, adv_q = clean_adv_summaries(adv)
        adv.to_parquet(out_dir / "adv_team.parquet", index=False)

    games.to_parquet(out_dir / "games.parquet", index=False)
    team_games.to_parquet(out_dir / "team_games.parquet", index=False)
    quarantine.to_csv(out_dir / "quarantine.csv", index=False)

    reasons = quarantine.reason.value_counts().to_dict() if "reason" in quarantine else {}
    report = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "files": per_file,
        "games_rows": int(len(games)),
        "team_games_rows": int(len(team_games)),
        "ties_excluded_from_orientation": int((quarantine.reason == REASON_TIE).sum())
        if "reason" in quarantine
        else 0,
        "divisions_inferred": int((clean.home_division_inferred | clean.away_division_inferred).sum()),
        "division_unknown_rows": int(clean.division_unknown.sum()),
        "quarantine_reasons": {str(k): int(v) for k, v in reasons.items()},
    }
    if market_q is not None:
        report["market_quarantined"] = int(len(market_q))
    if adv_q is not None:
        report["adv_quarantined"] = int(len(adv_q))
    (out_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    return {"games": games, "team_games": team_games, "quarantine": quarantine, "report": report}
