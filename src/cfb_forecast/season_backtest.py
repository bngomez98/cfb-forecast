"""Reconstruct and grade weekly forecasts across a season."""
from pathlib import Path
import json
import pandas as pd
from .data import load_data, timestamp
from .forecast import forecast


def run_season_backtest(data_root, model_root, out_root, season, draws=50000, lead_hours=48):
    data_root = Path(data_root)
    out_root = Path(out_root) / f"season-{int(season)}"
    schedules, _, _ = load_data(data_root)
    season_games = schedules[(schedules.season == int(season)) & schedules.fbs_game].copy()
    weeks = sorted(season_games.week.dropna().astype(int).unique())
    reports = []
    all_games = []
    for week in weeks:
        games = season_games[season_games.week == week].sort_values("start_date")
        if games.empty:
            continue
        first_kickoff = games.start_date.min()
        last_kickoff = games.start_date.max()
        as_of = first_kickoff - pd.Timedelta(hours=lead_hours)
        destination = out_root / f"week-{week}"
        predictions = forecast(
            data_root=data_root,
            model_root=model_root,
            out_root=destination,
            as_of=as_of.isoformat(),
            through=last_kickoff.isoformat(),
            season=int(season),
            draws=draws,
            target_game_ids=games.game_id.astype(int).tolist(),
        )
        actual = games[["game_id", "completed", "home_points", "away_points"]].copy()
        graded = predictions.merge(actual, on="game_id", how="left")
        graded = graded[graded.completed & graded.home_points.notna() & graded.away_points.notna()].copy()
        if graded.empty:
            continue
        graded["actual_home_win"] = (graded.home_points > graded.away_points).astype(int)
        graded["predicted_home_win"] = (graded.home_win_probability >= 0.5).astype(int)
        graded["correct"] = graded.actual_home_win == graded.predicted_home_win
        graded["brier"] = (graded.home_win_probability - graded.actual_home_win) ** 2
        graded["score_mae"] = (abs(graded.expected_home_score - graded.home_points) + abs(graded.expected_away_score - graded.away_points)) / 2
        graded["margin_mae"] = abs(graded.projected_home_margin - (graded.home_points - graded.away_points))
        graded["total_mae"] = abs(graded.projected_total - (graded.home_points + graded.away_points))
        graded["week"] = int(week)
        graded.to_csv(destination / "graded.csv", index=False)
        all_games.append(graded)
        reports.append({"season": int(season), "week": int(week), "games": int(len(graded)), "winner_accuracy": float(graded.correct.mean()), "brier_score": float(graded.brier.mean()), "score_mae": float(graded.score_mae.mean()), "margin_mae": float(graded.margin_mae.mean()), "total_mae": float(graded.total_mae.mean()), "as_of": as_of.isoformat()})
    if not all_games:
        raise ValueError(f"No completed games were available to grade for season {season}")
    combined = pd.concat(all_games, ignore_index=True)
    summary = {"season": int(season), "weeks": len(reports), "games": int(len(combined)), "winner_accuracy": float(combined.correct.mean()), "brier_score": float(combined.brier.mean()), "score_mae": float(combined.score_mae.mean()), "margin_mae": float(combined.margin_mae.mean()), "total_mae": float(combined.total_mae.mean())}
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(out_root / "weekly_metrics.csv", index=False)
    combined.to_csv(out_root / "graded_games.csv", index=False)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return pd.DataFrame(reports), summary
