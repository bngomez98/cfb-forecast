from pathlib import Path
import ast
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CFB Forecast", page_icon="🏈", layout="wide")

ARTIFACT_ROOT = Path("/app/artifacts")
LATEST = ARTIFACT_ROOT / "latest-weekly"

@st.cache_data(ttl=300)
def load_forecasts(path: str):
    return pd.read_csv(path)


def parse_interval(value):
    try:
        parsed = ast.literal_eval(value)
        return float(parsed[0]), float(parsed[1])
    except (ValueError, SyntaxError, TypeError, IndexError):
        return None, None


def probability_label(row):
    if row.home_win_probability >= 0.5:
        return row.home_team
    return row.away_team

st.title("CFB Forecast Dashboard")
st.caption("Weekly predictions from the latest reproducible forecast run")

forecast_path = LATEST / "forecasts.csv"
if not forecast_path.exists():
    st.error(f"No forecast found at {forecast_path}")
    st.info("Run the weekly forecast first, then refresh this page.")
    st.stop()

try:
    df = load_forecasts(str(forecast_path))
except Exception as exc:
    st.error(f"Could not load forecasts: {exc}")
    st.stop()

if df.empty:
    st.warning("The latest forecast contains no games.")
    st.stop()

manifest_path = LATEST / "weekly_manifest.json"
manifest = {}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text())[0]
    except (json.JSONDecodeError, IndexError):
        manifest = {}

with st.sidebar:
    st.header("Filters")
    team_options = sorted(set(df.home_team) | set(df.away_team))
    selected_teams = st.multiselect("Teams", team_options)
    min_confidence = st.slider("Minimum win probability", 0.50, 1.00, 0.50, 0.01)
    sort_by = st.selectbox("Sort games by", ["Kickoff", "Confidence", "Projected margin", "Projected total"])

view = df.copy()
view["winner"] = view.apply(probability_label, axis=1)
view["confidence"] = view[["home_win_probability", "simulated_home_win_probability"]].max(axis=1)
view["confidence"] = view["confidence"].clip(lower=0.5)
view["winner_probability"] = view.apply(
    lambda r: r.home_win_probability if r.winner == r.home_team else 1 - r.home_win_probability,
    axis=1,
)
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
    display = view[[
        "kickoff", "away_team", "home_team", "expected_away_score", "expected_home_score",
        "projected_home_margin", "projected_total", "winner", "winner_probability",
        "margin_interval_80", "total_interval_80",
    ]].copy()
    display.columns = [
        "Kickoff", "Away", "Home", "Away score", "Home score", "Home margin",
        "Total", "Pick", "Pick probability", "Margin 80% interval", "Total 80% interval",
    ]
    display["Pick probability"] = display["Pick probability"].map(lambda x: f"{x:.1%}")
    for column in ["Away score", "Home score", "Home margin", "Total"]:
        display[column] = display[column].map(lambda x: f"{x:.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader("Win probability")
if not view.empty:
    chart = view[["away_team", "home_team", "home_win_probability"]].copy()
    chart["home_win_probability"] = chart.home_win_probability * 100
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
