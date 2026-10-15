"""FieldQuant desk backed exclusively by saved application runs."""

from pathlib import Path
import json
import pandas as pd
import streamlit as st
from cfb_forecast.paths import artifacts_root
from desk import shell, overview, render_pipeline, historical, sources, training_root

ROOT = artifacts_root()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


@st.cache_data(ttl=60)
def read_csv(path, modified):
    return pd.read_csv(path)


def table(path):
    path = Path(path)
    return read_csv(str(path), path.stat().st_mtime_ns)


def main():
    latest = ROOT / "latest"
    manifest = read_json(latest / "run_manifest.json", {})
    page = shell(ROOT, manifest)
    if page == "Overview":
        overview(ROOT, manifest)
        return
    if page == "Calculation pipeline":
        render_pipeline(ROOT)
        return
    if page == "Historical training":
        historical()
        return
    if page == "Sources":
        sources(ROOT, latest)
        return
    if page == "Forecast log":
        board_path = latest / "forecast_board.json"
        board = read_json(board_path, {})
        rows = pd.DataFrame(board.get("forecasts", []))
        st.subheader("Published forecast log")
        if rows.empty:
            st.info("No forecast log has been published yet.")
            return
        st.caption(f"Run {board.get('run_id', 'unknown')} · {board.get('games', len(rows))} games · {board.get('as_of', '')}")
        st.dataframe(
            rows[[column for column in ["kickoff", "away_team", "home_team", "expected_away_score", "expected_home_score", "winner", "confidence"] if column in rows.columns]],
            hide_index=True,
            width="stretch",
        )
        return
    forecasts = table(latest / "forecasts.csv") if (latest / "forecasts.csv").exists() else pd.DataFrame()
    if page in {"Forecast desk", "Game inputs"} and forecasts.empty:
        st.info("No upcoming forecasts in the latest completed run. Historical training, source receipts and pipeline evidence remain available in the sidebar.")
        return
    if page == "Forecast desk":
        st.subheader("Upcoming FBS games")
        selected = st.multiselect(
            "Teams", sorted(set(forecasts.home_team) | set(forecasts.away_team))
        )
        view = forecasts.copy()
        if selected:
            view = view[view.home_team.isin(selected) | view.away_team.isin(selected)]
        view["winner"] = view.apply(
            lambda r: r.home_team if r.home_win_probability >= 0.5 else r.away_team,
            axis=1,
        )
        view["confidence"] = view.home_win_probability.where(
            view.home_win_probability >= 0.5, 1 - view.home_win_probability
        )
        a, b, c = st.columns(3)
        a.metric("FBS games", len(view))
        b.metric(
            "Games with complete inputs",
            int(view.inputs_complete.sum()) if "inputs_complete" in view else 0,
        )
        c.metric(
            "Eligible moneyline candidates",
            int((view.moneyline_candidate != "none").sum()) if "moneyline_candidate" in view else 0,
        )
        columns = [
            "kickoff",
            "away_team",
            "home_team",
            "winner",
            "confidence",
            "expected_away_score",
            "expected_home_score",
        ]
        st.dataframe(
            view[[c for c in columns if c in view]],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Score expectations come from the fitted distribution. They are predictions, never reported final results."
        )
        st.download_button(
            "Download predictions",
            view.to_csv(index=False),
            "fbs-predictions.csv",
            "text/csv",
        )
    elif page == "Lines":
        view = forecasts.copy()
        columns = [
            "kickoff",
            "away_team",
            "home_team",
            "model_spread",
            "market_spread",
            "home_spread",
            "home_moneyline",
            "away_moneyline",
            "home_ml_decimal",
            "away_ml_decimal",
            "projected_home_margin",
        ]
        st.subheader("Spread and moneyline")
        st.dataframe(view[[c for c in columns if c in view]], hide_index=True, width="stretch")
        st.caption("Model spread is the forecast margin in sportsbook sign. Fair American moneylines come from the same win probability.")
        priced = view.dropna(subset=[c for c in ["home_ml_decimal", "away_ml_decimal"] if c in view])
        if "home_ml_decimal" in priced and "away_ml_decimal" in priced and len(priced):
            st.subheader("Recorded book prices")
            columns = [
                "away_team",
                "home_team",
                "home_win_probability",
                "market_home_win_probability_no_vig",
                "home_ml_decimal",
                "away_ml_decimal",
                "home_moneyline_edge",
                "home_moneyline_expected_net_per_unit",
                "away_moneyline_expected_net_per_unit",
                "hypothetical_bankroll_fraction",
                "market_observed_at",
                "market_provider",
            ]
            st.dataframe(priced[[c for c in columns if c in priced]], hide_index=True, width="stretch")
            st.caption("Hypothetical exposure uses quarter Kelly, a 1% game cap, and a 5% slate cap. No wagers are placed.")
    elif page == "Statistics":
        st.subheader("Recorded team and player rates")
        st.caption("Same-season and prior-season facts stored after each forecast run. Missing files stay missing.")
        try:
            from contextlib import closing
            from cfb_forecast.database import connect
            from cfb_forecast.paths import database_path
            with closing(connect(database_path())) as conn:
                names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "team_game_facts" in names:
                    st.dataframe(pd.read_sql_query("SELECT * FROM team_game_facts ORDER BY season DESC, week DESC LIMIT 200", conn), hide_index=True, width="stretch")
                else:
                    st.info("team_game_facts is created on the next forecast run that reads parquet history.")
                if "player_game_facts" in names:
                    st.dataframe(pd.read_sql_query("SELECT * FROM player_game_facts ORDER BY season DESC, yards DESC LIMIT 200", conn), hide_index=True, width="stretch")
        except Exception as exc:
            st.warning(str(exc))
    elif page == "Methodology":
        st.subheader("One forecasting model")
        st.write("The hierarchical team state sets the score location from completed games. Same-season and prior-season points and play rates enter that forecast before kickoff. Weather, rest, injuries, and a recorded line enter as known-future information when they exist. Expected scores are the model means. The winner is the side with home win probability at least 50 percent. Each Sunday run is validated against finished games before the next week.")
        st.write("This is a weekly point-in-time forecast for FBS games, not a live player-tracking engine and not a betting instruction.")
    elif page == "Weekly validation":
        report = read_json(latest / "previous_week_validation.json", {})
        batches = report.get("runs", [])
        if not batches:
            st.info(
                "There are no prior saved runs to validate. Previous predictions will be scored after real final results arrive."
            )
            return
        st.subheader("Previous forecasts against actual results")
        st.dataframe(
            pd.DataFrame(batches).drop(
                columns=["market_probability_metrics"], errors="ignore"
            ),
            hide_index=True,
            width="stretch",
        )
        choice = st.selectbox("Forecast run", [r["run_id"] for r in batches])
        path = ROOT / "runs" / choice
        if (path / "validation.csv").exists():
            st.dataframe(
                table(path / "validation.csv"),
                hide_index=True,
                width="stretch",
            )
            bins = table(path / "validation_calibration.csv")
            st.subheader("Probability calibration")
            st.line_chart(bins.set_index("mean_probability")[["observed_win_rate"]])
            st.dataframe(bins, hide_index=True)
            st.caption(
                "Log loss and Brier score grade the probability. Betting results use saved prices and candidates; they are hypothetical returns, not executed wagers."
            )
    elif page == "Game inputs":
        records = read_json(latest / "inputs" / "game_inputs.json", [])
        if not records:
            st.info("No game inputs are available for this run.")
            return
        choices = {
            int(r.game_id): f"{r.away_team} at {r.home_team}"
            for r in forecasts.itertuples()
        }
        gid = st.selectbox("Game", list(choices), format_func=choices.get)
        record = next(r for r in records if r["game_id"] == gid)
        st.subheader("Weather and venue")
        st.json(
            {
                "weather": record["weather"],
                "venue": record["venue"],
                "retrieved_at": record["retrieved_at"],
                "source": record["source_url"],
            }
        )
        for side in ["away", "home"]:
            team = record["teams"][side]
            st.subheader(f"{side.title()} team · player and team inputs")
            st.write(
                f"Lineup: {team['lineup_status']} · Injury coverage: {team['injury_status']}"
            )
            with st.expander("Team statistics"):
                st.dataframe(
                    pd.DataFrame(
                        list(team["statistics"].items()), columns=["statistic", "value"]
                    ),
                    hide_index=True,
                )
            with st.expander("Player statistics and leaders"):
                st.json(team["leaders"])
            with st.expander("Roster, reported injuries, and depth chart"):
                st.json(
                    {
                        "depthchart": team["depthchart"],
                        "reported_injuries": team["reported_injuries"],
                        "stale_injury_reports": team["stale_injury_reports"],
                        "roster": team["roster"],
                    }
                )
        st.subheader("Calculation inputs")
        features = table(latest / "inputs" / "features.csv")
        st.dataframe(features[features.game_id == gid].T.astype(str), width="stretch")
        prediction = forecasts[forecasts.game_id == gid].iloc[0]
        st.write(
            f"Context logit contribution: {prediction.context_logit_contribution:.6f}"
        )
        st.caption(
            "Context coefficients are learned from saved pregame features and later outcomes, then compared on a later week. An unavailable injury report is never treated as a healthy lineup."
        )
    elif page == "Model research":
        st.subheader("Model selection and measured validation")
        st.json(read_json(latest / "context_model_selection.json", {}))
        st.json(read_json(training_root() / "winner_model_selection.json", {}))
        path = training_root() / "metrics.csv"
        if path.exists():
            st.dataframe(table(path), hide_index=True, width="stretch")
        st.caption(
            "Historical comparisons guide later deployment. Calibration and robustness experiments write separate candidates and do not silently replace the current model."
        )


if __name__ == "__main__":
    main()
