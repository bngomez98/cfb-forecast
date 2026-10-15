"""Read-only presentation components for saved pipeline evidence."""
from html import escape
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from cfb_forecast.paths import data_root, model_root
from cfb_forecast.pipeline_status import STAGES

PAGES = ["Overview", "Forecast desk", "Lines", "Statistics", "Methodology", "Forecast log", "Calculation pipeline", "Historical training", "Weekly validation", "Game inputs", "Model research", "Sources", "Players & teams", "Web sources"]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def shell(root, manifest):
    st.set_page_config(page_title="FieldQuant | College Football Intelligence", layout="wide")
    st.html(Path(__file__).with_name("theme.css").read_text())
    with st.sidebar:
        st.html('<div class="brand"><span class="brand-mark">FQ</span><div>FieldQuant<small>FOOTBALL INTELLIGENCE</small></div></div>')
        st.caption("WORKSPACE / FBS")
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("DATA, NOT GUESSWORK")
        st.write("Real sources. Traceable forecasts.")
        st.caption("Historical training and Sunday slate analysis. No simulated live score samples.")
        with st.expander("Runtime directories"):
            st.text(f"Data: {data_root()}\nArtifacts: {root}\nModel: {model_root()}")
        if st.button("Refresh saved data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("Refresh reads saved artifacts; it does not start a model run.")
    season = manifest.get("season", "2026")
    status = str(manifest.get("status", "awaiting first run")).replace("_", " ")
    st.html(f'<div class="topline"><span>RESEARCH DESK <span class="slash">/</span> COLLEGE FOOTBALL</span><span class="status-pill">{escape(status.upper())}</span></div>')
    headings = {"Overview": ("An edge built on evidence.", "From historical play-by-play to the next kickoff. Every forecast has a paper trail."), "Calculation pipeline": ("Inside the forecast.", "Seven connected stages. Inspect the evidence behind every calculation."), "Historical training": ("The foundation of every forecast.", "Real season archives, chronological validation and traceable model inputs.")}
    title, subtitle = headings.get(page, (page, "Source-backed analysis. Predictions and actual outcomes always remain separate."))
    st.html(f'<div class="page-heading"><div class="eyebrow">FIELDQUANT / {escape(str(season))} SEASON</div><h1>{escape(title)}</h1><p class="subtitle">{escape(subtitle)}</p></div>')
    return page


def training_root():
    return model_root().parent


def pipeline(root):
    candidates = list((root / "work").glob("*/pipeline.json"))
    latest = root / "latest" / "pipeline.json"
    if latest.exists():
        candidates.append(latest)
    if not candidates:
        return {"status": "not run", "stages": [{"id": key, "name": name, "description": desc, "status": "not run"} for key, name, desc in STAGES]}
    path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    report = read_json(path, {})
    report["evidence_path"] = str(path)
    return report


def render_pipeline(root, compact=False):
    report = pipeline(root)
    if report.get("error"):
        st.error(report["error"])
    if report.get("evidence_path"):
        st.caption(f"Execution evidence: {report['evidence_path']}")
    for i, stage in enumerate(report.get("stages", []), 1):
        status = stage.get("status", "not run")
        color = "teal" if status == "completed" else "amber" if status in {"failed", "running"} else "muted"
        st.html(f'<div class="stage"><span class="stage-number">{i:02d}</span><div class="stage-body"><strong>{escape(stage["name"])}</strong><p>{escape(stage["description"])}</p></div><span class="stage-status {color}">{escape(status.upper())}</span></div>')
        if not compact and stage.get("started_at"):
            st.caption(f"Started {stage['started_at']} · Finished {stage.get('completed_at', '—')}")
    if not compact:
        st.info("Completed means the stage executed, not that every provider returned complete data. Missing injuries, lineups or prices remain unavailable and may suppress recommendations.")


def overview(root, manifest):
    train = training_root()
    inputs = read_json(train / "training_inputs.json", {})
    model = read_json(model_root() / "model.json", {})
    rows = inputs.get("seasons", [])
    metrics = pd.read_csv(train / "metrics.csv") if (train / "metrics.csv").exists() else pd.DataFrame()
    selected = metrics[metrics.model.eq(model.get("variant"))] if "model" in metrics else pd.DataFrame()
    cols = st.columns(4)
    values = [("HISTORICAL SEASONS", str(len(rows)) if rows else "—", "Verified training inputs"), ("TRAINING ARCHIVE", f"{sum(r['games'] for r in rows):,}" if rows else "—", "Completed FBS matchups"), ("HELD-OUT ACCURACY", f"{selected.iloc[0].accuracy:.1%}" if len(selected) else "—", "Historical evaluation, not live accuracy"), ("UPCOMING SLATE", str(manifest.get("games", "—")), "Saved run · FBS versus FBS")]
    for col, (label, value, note) in zip(cols, values):
        with col:
            st.html(f'<div class="metric-card"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>')
    st.write("")
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        st.subheader("From source to signal")
        st.caption("THE CALCULATION PIPELINE")
        render_pipeline(root, compact=True)
    with right:
        st.subheader("Model snapshot")
        variant = model.get("variant", "Not trained")
        st.html(f'<div class="model-card"><span class="eyebrow">SELECTED SCORE HEAD</span><h2>{escape(str(variant).upper())}</h2><p>Opponent-adjusted team states.<br>Chronological model selection.<br>Direct probability integration.</p></div>')
        st.write("Training through", model.get("training_through_season", "Not available"))
        st.write("Calibration season", model.get("calibration_season", "Not available"))
        if len(selected):
            a, b = st.columns(2)
            a.metric("Log loss", f"{selected.iloc[0].log_loss:.3f}")
            b.metric("Brier score", f"{selected.iloc[0].brier:.3f}")
        st.divider()
        st.caption("LATEST PUBLICATION")
        st.write(manifest.get("run_id", "No completed run"))
        st.caption(manifest.get("as_of", "Forecasts appear after the pipeline publishes a run."))
        st.info("Research only. Exposure figures are hypothetical; no wagers are placed.")
    st.divider()
    st.subheader("A transparent data stack")
    for col, name, description in zip(st.columns(3), ["SportsDataverse", "ESPN", "Football DB / CFBTrack"], ["Schedules, final scores and advanced team Parquet archives.", "Pregame weather, rosters, injuries and displayed market quotes.", "Supplementary team and player tables, when verified and available."]):
        with col:
            st.markdown(f"**{name}**")
            st.caption(description)
    st.caption("Provider availability is reported in Sources. A configured source is not a guarantee of coverage.")


def historical():
    root = training_root()
    report = read_json(root / "training_inputs.json", {})
    if not report:
        st.info("No verified training-input report has been saved yet.")
        return
    st.success("Historical game and team joins verified before training.")
    rows = pd.DataFrame(report["seasons"])
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(report["missing_policy"])
    st.subheader("Chronology is a model constraint")
    st.write(report["chronology"])
    st.caption("Historical archives may contain later corrections; this is not a reconstruction of original publication timestamps.")
    st.subheader("Source files used in this training run")
    files = pd.DataFrame(report["source_files"])
    st.dataframe(files, hide_index=True, width="stretch")
    st.download_button("Download training evidence", json.dumps(report, indent=2), "training-inputs.json", "application/json")


def sources(root, latest):
    manifest = read_json(data_root() / "source_manifest.json", {"files": {}})
    st.subheader("Historical source receipts")
    entries = [{"file": name, **meta} for name, meta in manifest.get("files", {}).items()]
    if entries:
        st.dataframe(pd.DataFrame(entries)[["file", "url", "downloaded_at", "bytes", "sha256"]], hide_index=True, width="stretch")
    else:
        st.info("No historical downloads have been recorded.")
    st.subheader("Pregame provider coverage")
    receipts = read_json(latest / "inputs" / "provider_status.json", [])
    if not receipts:
        st.info("No pregame provider report is available for the latest completed run.")
    for receipt in receipts:
        with st.expander(f"{receipt['provider']} · {receipt['status']}"):
            st.json(receipt)
    st.caption("An empty injury response is unavailable coverage, not evidence of a healthy roster. A roster is not a confirmed starting lineup.")


def changes(latest):
    """What moved since the previous daily run (written next to every daily forecast)."""
    path = Path(latest) / "changes.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    moved = table.dropna(subset=["win_probability_change"])
    st.subheader("What changed since the last run")
    if moved.empty:
        st.caption("This is the first daily run, or no earlier run covers these games.")
        return
    st.caption(f"Compared with the run at {str(moved.prior_as_of.iloc[0])[:16]} UTC. Win probability is the home team's.")
    top = moved.head(10)[["kickoff", "away_team", "home_team", "prior_home_win_probability", "home_win_probability", "win_probability_change", "margin_change"]].copy()
    for column in ["prior_home_win_probability", "home_win_probability"]:
        top[column] = top[column].map(lambda x: f"{x:.1%}")
    top["win_probability_change"] = top["win_probability_change"].map(lambda x: f"{100 * x:+.1f} pts")
    top["margin_change"] = top["margin_change"].map(lambda x: f"{x:+.1f}")
    top.columns = ["Kickoff", "Away", "Home", "Before", "Now", "Change", "Margin change"]
    st.dataframe(top, hide_index=True, width="stretch")


def players_teams(latest):
    teams_path, players_path = Path(latest) / "team_ratings.csv", Path(latest) / "player_ratings.csv"
    if not teams_path.exists():
        st.info("No ratings yet. They are written with every daily forecast run once player aggregates exist (`cfb-forecast player-aggregates`).")
        return
    st.caption("Standardized units: positive offense means more yards per play than the league after adjusting for the defenses faced; positive defense means fewer allowed. "
               "Ratings are recency-weighted and partially pooled player to team to league.")
    teams = pd.read_csv(teams_path)
    st.subheader("Team ratings")
    st.dataframe(teams.rename(columns={"team": "Team", "pass_offense": "Pass offense", "rush_offense": "Rush offense", "player_offense": "Offense",
                                       "player_defense": "Defense", "net_rating": "Net"}).drop(columns="team_id", errors="ignore").round(3), hide_index=True, width="stretch")
    if players_path.exists():
        players = pd.read_csv(players_path)
        st.subheader("Player ratings")
        a, b, c = st.columns(3)
        role = a.selectbox("Role", ["pass", "rush"])
        team = b.selectbox("Team", ["All"] + sorted(players.team.dropna().unique()))
        active = c.checkbox("Currently used (recent plays)", value=True)
        view = players[players.role == role]
        if team != "All":
            view = view[view.team == team]
        if active:
            view = view[view.usage_share > 0]
        st.dataframe(view.rename(columns={"player_name": "Player", "team": "Team", "rating": "Rating", "usage_share": "Usage share", "plays": "Plays",
                                          "recent_plays": "Recent plays", "games": "Games"})[["Player", "Team", "Rating", "Usage share", "Plays", "Recent plays", "Games"]].round(3),
                     hide_index=True, width="stretch")


def web_sources():
    root = data_root() / "sources"
    report = read_json(root / "collections" / "latest.json", {})
    if not report:
        st.info("No web collection yet. Run `cfb-forecast sources-collect --season <year>` or the source pipeline.")
        return
    st.caption(f"Last collection {report.get('started_at', '')} · season {report.get('season', '')}")
    status = pd.DataFrame([{"Source": n, "Status": v.get("status"), "Kind": v.get("kind"), "Results": v.get("results", 0), "Picks": v.get("picks", 0),
                            "Lines": v.get("lines", 0), "Team stats": v.get("team_stats", 0), "Detail": v.get("error", "")} for n, v in report.get("sources", {}).items()])
    st.dataframe(status, hide_index=True, width="stretch")
    cross = root / "results_crosscheck.csv"
    if cross.exists():
        table = pd.read_csv(cross)
        conflicts = table[(table.sources_agree == False) | (table.schedule_agrees == False)]  # noqa: E712
        a, b = st.columns(2)
        a.metric("Games with external finals", len(table))
        b.metric("Score conflicts", len(conflicts))
        if len(conflicts):
            st.dataframe(conflicts[["kickoff", "away_team", "home_team", "source_scores", "schedule_score"]], hide_index=True, width="stretch")
    picks_path = root / "picks.csv"
    if picks_path.exists():
        picks = pd.read_csv(picks_path).sort_values("retrieved_at").drop_duplicates(["source", "game_id"], keep="last")
        upcoming = picks[pd.to_datetime(picks.kickoff, utc=True) > pd.Timestamp.now(tz="UTC")]
        st.subheader("External predictions for upcoming games (reference only)")
        st.dataframe(upcoming[["source", "kickoff", "away_team", "home_team", "pick_team", "pick_home_win_prob", "pick_away_score", "pick_home_score", "pick_home_spread"]],
                     hide_index=True, width="stretch")
