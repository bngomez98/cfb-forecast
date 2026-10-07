from pathlib import Path
import ast
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CFB Forecast", page_icon="🏈", layout="wide")

ARTIFACT_ROOT = Path("/app/artifacts")
DATA_ROOT = Path("/app/data")
LATEST = ARTIFACT_ROOT / "latest-weekly"

@st.cache_data(ttl=300)
def load_csv(path: str):
    return pd.read_csv(path)

@st.cache_data(ttl=300)
def load_schedule(root: str):
    paths = sorted(Path(root).glob("schedule_*.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        required = {"game_id", "completed", "home_team", "away_team", "home_points", "away_points"}
        if required.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    schedule = pd.concat(frames, ignore_index=True)
    schedule["game_id"] = pd.to_numeric(schedule["game_id"], errors="coerce")
    schedule["home_points"] = pd.to_numeric(schedule["home_points"], errors="coerce")
    schedule["away_points"] = pd.to_numeric(schedule["away_points"], errors="coerce")
    schedule["completed"] = schedule["completed"].astype(str).str.lower().isin(["true", "1"])
    return schedule.drop_duplicates("game_id")

@st.cache_data(ttl=300)
def load_live_results(root: str):
    paths = sorted(Path(root).glob("results/*/results.csv"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    results = pd.concat(frames, ignore_index=True)
    results["game_id"] = pd.to_numeric(results["game_id"], errors="coerce")
    results["home_points"] = pd.to_numeric(results["home_points"], errors="coerce")
    results["away_points"] = pd.to_numeric(results["away_points"], errors="coerce")
    results["is_final"] = results["is_final"].astype(str).str.lower().isin(["true", "1"])
    return results.sort_values("retrieved_at").drop_duplicates("game_id", keep="last")


def probability_label(row):
    return row.home_team if row.home_win_probability >= 0.5 else row.away_team


def load_manifest(path):
    try:
        value = json.loads(path.read_text())
        return value[0] if isinstance(value, list) else value
    except (OSError, json.JSONDecodeError, IndexError):
        return {}


def parse_interval(value):
    try:
        parsed = ast.literal_eval(value)
        return f"[{float(parsed[0]):.0f}, {float(parsed[1]):.0f}]"
    except (ValueError, SyntaxError, TypeError, IndexError):
        return value


def forecast_directories():
    paths = []
    paths.extend(sorted(ARTIFACT_ROOT.glob("weekly/*")))
    paths.extend(sorted(ARTIFACT_ROOT.glob("forecast_*/")))
    return [path for path in paths if (path / "forecasts.csv").exists()]


def grade_forecast(predictions, schedule):
    if schedule.empty:
        return pd.DataFrame()
    actual = schedule[["game_id", "completed", "home_points", "away_points"]].copy()
    graded = predictions.merge(actual, on="game_id", how="inner")
    graded = graded[graded.completed & graded.home_points.notna() & graded.away_points.notna()].copy()
    if graded.empty:
        return graded
    graded["actual_home_margin"] = graded.home_points - graded.away_points
    graded["actual_total"] = graded.home_points + graded.away_points
    graded["actual_home_win"] = (graded.home_points > graded.away_points).astype(int)
    graded["predicted_home_win"] = (graded.home_win_probability >= 0.5).astype(int)
    graded["correct"] = graded.actual_home_win == graded.predicted_home_win
    graded["brier"] = (graded.home_win_probability - graded.actual_home_win) ** 2
    graded["home_score_error"] = graded.expected_home_score - graded.home_points
    graded["away_score_error"] = graded.expected_away_score - graded.away_points
    graded["margin_error"] = graded.projected_home_margin - graded.actual_home_margin
    graded["total_error"] = graded.projected_total - graded.actual_total
    return graded

st.title("CFB Forecast Dashboard")
st.caption("Weekly predictions and historical results backtesting")

forecast_path = LATEST / "forecasts.csv"
if not forecast_path.exists():
    st.error(f"No forecast found at {forecast_path}")
    st.info("Run the weekly forecast first, then refresh this page.")
    st.stop()

page = st.sidebar.radio("View", ["Current forecast", "Historical backtest"])
schedule = load_schedule(str(DATA_ROOT))
live_results = load_live_results(str(ARTIFACT_ROOT))
if not live_results.empty:
    live_results["completed"] = live_results["is_final"]
    schedule = pd.concat([schedule, live_results[["game_id", "completed", "home_points", "away_points"]]], ignore_index=True).drop_duplicates("game_id", keep="last")

if page == "Historical backtest":
    st.header("Historical backtest against actual results")
    directories = forecast_directories()
    if not directories:
        st.warning("No prior forecast artifacts were found.")
        st.stop()
    labels = {str(path): f"{path.name} ({load_manifest(path / 'weekly_manifest.json').get('as_of', '')[:10]})" if (path / "weekly_manifest.json").exists() else path.name for path in directories}
    selected_path = st.selectbox("Forecast run", list(labels), format_func=lambda value: labels[value])
    predictions = load_csv(str(Path(selected_path) / "forecasts.csv"))
    graded = grade_forecast(predictions, schedule)
    if graded.empty:
        st.warning("No completed actual results match this forecast run yet.")
        st.stop()

    accuracy = graded.correct.mean()
    brier = graded.brier.mean()
    score_mae = (graded.home_score_error.abs() + graded.away_score_error.abs()).mean() / 2
    margin_mae = graded.margin_error.abs().mean()
    total_mae = graded.total_error.abs().mean()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Games graded", len(graded))
    m2.metric("Winner accuracy", f"{accuracy:.1%}")
    m3.metric("Brier score", f"{brier:.3f}")
    m4.metric("Score MAE", f"{score_mae:.1f}")
    m5.metric("Margin MAE", f"{margin_mae:.1f}")
    st.metric("Total MAE", f"{total_mae:.1f}")

    display = graded[[
        "kickoff", "away_team", "home_team", "expected_away_score", "expected_home_score",
        "away_points", "home_points", "home_win_probability", "predicted_home_win",
        "actual_home_win", "correct", "margin_error", "total_error",
    ]].copy()
    display.columns = ["Kickoff", "Away", "Home", "Pred away", "Pred home", "Actual away", "Actual home", "Home win probability", "Predicted home win", "Actual home win", "Correct", "Margin error", "Total error"]
    display["Home win probability"] = display["Home win probability"].map(lambda x: f"{x:.1%}")
    st.subheader("Game results")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Prediction versus actual score")
    chart = graded[["away_team", "home_team", "expected_home_score", "expected_away_score", "home_points", "away_points"]].copy()
    chart["matchup"] = chart.away_team + " at " + chart.home_team
    chart = chart.set_index("matchup")[["expected_away_score", "away_points", "expected_home_score", "home_points"]]
    chart.columns = ["Pred away", "Actual away", "Pred home", "Actual home"]
    st.bar_chart(chart, y_label="Points")
    with st.expander("Backtest definition"):
        st.write("Completed results are joined by immutable game_id. Winner accuracy uses predicted home win probability >= 50%. Brier score grades the home-win probability. Score, margin, and total errors are mean absolute errors.")
    st.stop()

# Current forecast view
try:
    df = load_csv(str(forecast_path))
except Exception as exc:
    st.error(f"Could not load forecasts: {exc}")
    st.stop()
if df.empty:
    st.warning("The latest forecast contains no games.")
    st.stop()

manifest = load_manifest(LATEST / "weekly_manifest.json")
with st.sidebar:
    st.header("Filters")
    team_options = sorted(set(df.home_team) | set(df.away_team))
    selected_teams = st.multiselect("Teams", team_options)
    min_confidence = st.slider("Minimum win probability", 0.50, 1.00, 0.50, 0.01)
    sort_by = st.selectbox("Sort games by", ["Kickoff", "Confidence", "Projected margin", "Projected total"])

view = df.copy()
view["winner"] = view.apply(probability_label, axis=1)
view["winner_probability"] = view.apply(lambda r: r.home_win_probability if r.winner == r.home_team else 1 - r.home_win_probability, axis=1)
if selected_teams:
    view = view[view.home_team.isin(selected_teams) | view.away_team.isin(selected_teams)]
view = view[view.winner_probability >= min_confidence]
if sort_by == "Kickoff":
    view = view.sort_values("kickoff")
elif sort_by == "Confidence":
    view = view.sort_values("winner_probability", ascending=False)
elif sort_by == "Projected margin":
    view = view.sort_values("projected_home_margin", key=lambda s: s.abs(), ascending=False)
else:
    view = view.sort_values("projected_total", ascending=False)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Games", len(view))
m2.metric("Strongest pick", f"{view.winner_probability.max():.1%}" if len(view) else "—")
m3.metric("Average projected total", f"{view.projected_total.mean():.1f}" if len(view) else "—")
m4.metric("Forecast week", str(manifest.get("week", "—")))
st.subheader("Projected games")
if view.empty:
    st.info("No games match the selected filters.")
else:
    display = view[["kickoff", "away_team", "home_team", "expected_away_score", "expected_home_score", "projected_home_margin", "projected_total", "winner", "winner_probability", "margin_interval_80", "total_interval_80"]].copy()
    display.columns = ["Kickoff", "Away", "Home", "Away score", "Home score", "Home margin", "Total", "Pick", "Pick probability", "Margin 80% interval", "Total 80% interval"]
    display["Pick probability"] = display["Pick probability"].map(lambda x: f"{x:.1%}")
    for column in ["Away score", "Home score", "Home margin", "Total"]:
        display[column] = display[column].map(lambda x: f"{x:.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
st.subheader("Win probability")
if not view.empty:
    chart = view[["away_team", "home_team", "home_win_probability"]].copy()
    chart["home_win_probability"] *= 100
    chart["matchup"] = chart.away_team + " at " + chart.home_team
    st.bar_chart(chart.set_index("matchup")["home_win_probability"], y_label="Home win probability (%)")
st.subheader("Score outlook")
if not view.empty:
    score_chart = view[["away_team", "home_team", "expected_away_score", "expected_home_score"]].copy()
    score_chart["matchup"] = score_chart.away_team + " at " + score_chart.home_team
    score_chart = score_chart.set_index("matchup")[["expected_away_score", "expected_home_score"]]
    score_chart.columns = ["Away score", "Home score"]
    st.bar_chart(score_chart, y_label="Expected points")
with st.expander("Run metadata"):
    st.json(manifest)
    st.write(f"Source: `{forecast_path}`")
