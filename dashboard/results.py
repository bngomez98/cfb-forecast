"""Results and accuracy: saved forecasts graded against final scores."""
from html import escape
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from cfb_forecast.paths import artifacts_root

from board import ZONE, _time

SUNDAY = "Sunday forecasts checked against final scores"
BACKTEST = "Rebuilt forecasts for past weeks"
BANDS = [(0.5, 0.6, "50–60%"), (0.6, 0.7, "60–70%"), (0.7, 0.8, "70–80%"), (0.8, 0.9, "80–90%"), (0.9, 1.01, "90% or more")]


@st.cache_data(ttl=60)
def _read(paths, modified):
    return [pd.read_csv(path) for path in paths]


def _csvs(paths):
    paths = tuple(str(p) for p in paths)
    return _read(paths, tuple(Path(p).stat().st_mtime_ns for p in paths))


def _week_from_run(run_id):
    match = re.search(r"week-(\d+)", str(run_id))
    return int(match.group(1)) if match else None


def _sunday_runs(root):
    """Forecasts saved on each Sunday, graded by the weekly validation step."""
    paths = sorted(root.glob("runs/*/validation.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for path, frame in zip(paths, _csvs(paths)):
        if frame.empty:
            continue
        frame = frame.copy()
        run_id = frame.run_id.iloc[0] if "run_id" in frame else path.parent.name
        frame["week_label"] = f"Week {_week_from_run(run_id)}" if _week_from_run(run_id) else str(run_id)
        frame["week_order"] = _week_from_run(run_id) or 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _backtest(root):
    """Weekly forecasts rebuilt from the data available before each week, then graded."""
    paths = sorted(root.glob("backtest/season-*/graded_games.csv"))
    if not paths:
        return pd.DataFrame(), None
    path = max(paths, key=lambda p: int(re.search(r"season-(\d+)", str(p)).group(1)))
    frame = _csvs([path])[0].copy()
    if frame.empty:
        return frame, None
    post = frame.get("season_type", pd.Series("regular", index=frame.index)).astype(str).str.lower().eq("postseason")
    frame["week_label"] = ["Bowls" if p else f"Week {int(w)}" for p, w in zip(post, frame.week)]
    frame["week_order"] = frame.week.astype(int) + post.astype(int) * 100
    return frame, int(re.search(r"season-(\d+)", str(path)).group(1))


def load_results():
    """Prefer the Sunday forecasts that were actually published; fall back to the rebuilt season."""
    root = artifacts_root()
    sunday = _sunday_runs(root)
    if not sunday.empty:
        frame, source, season = sunday, SUNDAY, None
    else:
        frame, season = _backtest(root)
        source = BACKTEST
    if frame.empty:
        return frame, None, None
    frame = frame.copy()
    frame["home_win_probability"] = pd.to_numeric(frame.home_win_probability, errors="coerce")
    frame["actual_home_win"] = (frame.home_points > frame.away_points).astype(int)
    frame["correct"] = (frame.home_win_probability >= 0.5).astype(int) == frame.actual_home_win
    frame["pick_chance"] = frame.home_win_probability.where(frame.home_win_probability >= 0.5, 1 - frame.home_win_probability)
    margin = frame.get("projected_home_margin")
    if margin is None:
        margin = frame.expected_home_score - frame.expected_away_score
    frame["margin_miss"] = (margin - (frame.home_points - frame.away_points)).abs()
    frame["brier"] = (frame.home_win_probability - frame.actual_home_win) ** 2
    chance = frame.home_win_probability.clip(1e-9, 1 - 1e-9)
    frame["log_loss"] = -(frame.actual_home_win * np.log(chance) + (1 - frame.actual_home_win) * np.log(1 - chance))
    if "kickoff" in frame:
        frame["kickoff"] = pd.to_datetime(frame.kickoff, utc=True, errors="coerce")
    return frame, source, season


def season_accuracy():
    """Share of winners picked correctly so far, for the summary on other pages."""
    frame, _, _ = load_results()
    if frame.empty:
        return None
    return float(frame.correct.mean()), int(len(frame))


def _tiles(frame):
    right = int(frame.correct.sum())
    items = [
        ("Winners picked correctly", f"{frame.correct.mean():.1%}", f"{right} of {len(frame)} games"),
        ("Average miss on the margin", f"{frame.margin_miss.mean():.1f} pts", "How far the predicted margin was from the real one"),
        ("Brier score", f"{frame.brier.mean():.3f}", "Grades the win chances. Lower is better; always guessing 50% scores 0.250"),
        ("Weeks graded", str(frame.week_label.nunique()), f"{len(frame)} finished games"),
    ]
    st.html('<div class="summary">' + "".join(
        f'<div class="summary-item"><span>{escape(label)}</span><strong>{escape(value)}</strong><small>{escape(note)}</small></div>'
        for label, value, note in items
    ) + "</div>")


def _weekly(frame):
    weekly = (
        frame.groupby(["week_order", "week_label"], sort=True)
        .agg(games=("correct", "size"), right=("correct", "sum"), accuracy=("correct", "mean"), margin=("margin_miss", "mean"), brier=("brier", "mean"), log_loss=("log_loss", "mean"))
        .reset_index()
    )
    st.html('<h2 class="section">Accuracy by week</h2>')
    st.bar_chart(weekly.set_index("week_label")[["accuracy"]].rename(columns={"accuracy": "Winners picked correctly"}) * 100, y_label="Percent right", x_label="", color="#7dd8c0", height=260)
    st.html('<h2 class="section">Win-chance scores by week</h2>'
            '<p class="lede">Both scores grade the win chances themselves, not just the picks, and lower is better. '
            'Saying 50% for every game would score 0.250 on Brier and 0.693 on log loss.</p>')
    st.line_chart(
        weekly.set_index("week_label")[["brier", "log_loss"]].rename(columns={"brier": "Brier score", "log_loss": "Log loss"}),
        y_label="Score (lower is better)", x_label="", color=["#7dd8c0", "#e3be78"], height=240,
    )
    table = pd.DataFrame({
        "Week": weekly.week_label,
        "Games": weekly.games,
        "Picked right": weekly.right.astype(int),
        "Accuracy": weekly.accuracy * 100,
        "Average margin miss (pts)": weekly.margin.round(1),
        "Brier score": weekly.brier.round(3),
        "Log loss": weekly.log_loss.round(3),
    })
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Accuracy": st.column_config.ProgressColumn("Accuracy", format="%.1f%%", min_value=0, max_value=100)},
    )


def _confidence(frame):
    st.html('<h2 class="section">Do the win chances hold up?</h2>'
            '<p class="lede">Games are grouped by how sure the forecast was about its pick. '
            'If the chances are honest, picks made at 70–80% should come true about 75% of the time. '
            'Small groups swing by luck, so each group gets the range that luck alone would produce 95% of the time. '
            'A result outside that range is a real sign the chances are off.</p>')
    rows = []
    for low, high, label in BANDS:
        band = frame[(frame.pick_chance >= low) & (frame.pick_chance < high)]
        if band.empty:
            continue
        expected = band.pick_chance.mean()
        spread = 1.96 * np.sqrt((band.pick_chance * (1 - band.pick_chance)).sum()) / len(band)
        came_true = band.correct.mean()
        rows.append({
            "How sure the forecast was": label,
            "Games": len(band),
            "Average chance given": expected * 100,
            "Picks that came true": came_true * 100,
            "Range luck allows": f"{max(expected - spread, 0):.0%} to {min(expected + spread, 1):.0%}",
            "Holds up": "Yes" if abs(came_true - expected) <= spread else "No",
        })
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Average chance given": st.column_config.NumberColumn(format="%.0f%%"),
            "Picks that came true": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        },
    )


def _game_card(game):
    home = float(game.home_win_probability)
    pick_home = home >= 0.5
    pick = game.home_team if pick_home else game.away_team
    verdict = "Right" if game.correct else "Wrong"
    rows = [
        ("", game.away_team, int(game.away_points), 1 - home, not pick_home, game.away_points > game.home_points),
        ("at ", game.home_team, int(game.home_points), home, pick_home, game.home_points > game.away_points),
    ]
    body = "".join(
        f'<div class="row{" picked" if picked else ""}"><span class="team"><em>{escape(prefix)}</em>{escape(str(team))}{" ✓" if won else ""}</span>'
        f'<span class="num">{points}</span><span class="num">{chance:.0%}</span></div>'
        for prefix, team, points, chance, picked, won in rows
    )
    when = _time(game.kickoff) if "kickoff" in game._fields and pd.notna(game.kickoff) else ""
    return (
        f'<article class="card"><header><span>{when}</span><span class="verdict {verdict.lower()}">{verdict}: picked {escape(str(pick))}</span></header>'
        f'<div class="row head"><span>Team</span><span>Final</span><span>Our chance</span></div>{body}</article>'
    )


def _games(frame):
    st.html('<h2 class="section">Every graded game</h2>')
    order = frame.drop_duplicates("week_label").sort_values("week_order").week_label.tolist()
    left, right = st.columns([1, 1])
    week = left.selectbox("Week", order, index=len(order) - 1)
    show = right.segmented_control("Show", ["All", "Right", "Wrong"], default="All", label_visibility="hidden")
    view = frame[frame.week_label == week]
    if show == "Right":
        view = view[view.correct]
    elif show == "Wrong":
        view = view[~view.correct]
    if "kickoff" in view:
        view = view.sort_values("kickoff")
    st.html(f'<p class="lede">{int(view.correct.sum())} right and {int((~view.correct).sum())} wrong shown.</p>'
            '<div class="cards">' + "".join(_game_card(game) for game in view.itertuples(index=False)) + "</div>")


def results_page():
    frame, source, season = load_results()
    title = f"{season} results and accuracy" if season else "Results and accuracy"
    st.html(f'<div class="intro"><h1>{escape(title)}</h1></div>')
    if frame.empty:
        st.info("No forecasts have been graded yet. After each Sunday run, last week's forecasts are checked against the final scores and appear here.")
        return
    if source == SUNDAY:
        note = "These are the forecasts published each Sunday, checked against the final scores."
    else:
        note = ("No graded Sunday forecasts are saved yet, so each week of the season was forecast again using only the data "
                "available before that week's first game, then checked against the final scores.")
    st.html(f'<p class="updated">{escape(note)}</p>')
    _tiles(frame)
    _weekly(frame)
    _confidence(frame)
    _games(frame)
