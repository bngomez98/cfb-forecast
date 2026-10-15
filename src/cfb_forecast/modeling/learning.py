"""Chronological score/PBP forecasting with immutable evidence and continual reinforcement."""
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from ..config import StateConfig
from ..ingest.data import load_data, timestamp, sha256 as file_hash
from ..storage.database import connect
from ..monitoring.data_watchdog import check_data
from ..evaluation.evaluate import metrics
from ..ingest.features import observations
from .model import ScoreHead, load_model, write_model
from .ratings import build_features
from .simulate import win_probability

SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_versions (
 version TEXT PRIMARY KEY, created_at TEXT NOT NULL, cutoff TEXT NOT NULL,
 status TEXT NOT NULL, model_path TEXT NOT NULL, report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_examples (
 version TEXT NOT NULL REFERENCES learning_versions(version),
 game_id INTEGER NOT NULL, season INTEGER NOT NULL, forecast_cutoff TEXT NOT NULL,
 result_available_at TEXT NOT NULL, latest_feature_result TEXT,
 features_json TEXT NOT NULL, observations_json TEXT NOT NULL,
 PRIMARY KEY(version, game_id)
);
CREATE INDEX IF NOT EXISTS training_examples_season ON training_examples(season);
"""


def validate_in_r(frame, folder):
    source = Path(__file__).with_name("validate_learning.R")
    executable = shutil.which("Rscript")
    frame.to_csv(folder / "gate_predictions.csv", index=False)
    if not executable:
        if os.environ.get("CFB_REQUIRE_R") == "1":
            raise RuntimeError("Rscript is required for independent learning validation")
        return {"status": "unavailable", "reason": "Rscript is not installed"}
    subprocess.run([executable, str(source), str(folder / "gate_predictions.csv"), str(folder / "r_metrics.csv")], check=True, timeout=60, capture_output=True, text=True)
    independent = pd.read_csv(folder / "r_metrics.csv").iloc[0].to_dict()
    expected = metrics(frame.home_win, frame.candidate_p)
    for name in ("log_loss", "brier"):
        if not np.isclose(independent[name], expected[name], atol=1e-9):
            raise ValueError(f"R/Python validation mismatch: {name}")
    return {"status": "verified", **independent}


def learn(data_root, base_root, output_root, database, as_of):
    at = timestamp(as_of)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    watchdog = check_data(data_root, database, required=("schedule_*.csv", "adv_team_*.parquet", "source_manifest.json"))
    if watchdog["status"] == "error":
        raise ValueError(f"Data watchdog blocked training: {watchdog}")
    base_root = Path(base_root).resolve()
    original_root = base_root
    original_hash = file_hash(base_root / "model.json")
    base = load_model(base_root)
    previous_report = None
    with closing(connect(database)) as conn:
        conn.executescript(SCHEMA)
        for row in conn.execute("SELECT report_json FROM learning_versions WHERE status='promoted' ORDER BY created_at DESC"):
            previous = json.loads(row[0])
            if previous.get("base_model_sha256") == original_hash and timestamp(previous["as_of"]) <= at:
                champion = Path(previous["model_path"])
                if file_hash(champion / "model.json") != previous["model_sha256"]:
                    raise ValueError("Promoted champion checksum mismatch")
                previous_report = previous
                break
    schedules, summaries, paths = load_data(data_root)
    games = schedules[schedules.fbs_game & schedules.result_usable & (schedules.result_available_at < at) & (schedules.season >= base["warmup_season"])].copy().reset_index(drop=True)
    if games.empty:
        raise ValueError("No completed prior games available for training")
    if previous_report is not None:
        new_results = int((games.result_available_at >= timestamp(previous_report["as_of"])).sum())
        if new_results < 30:
            return previous_report
        base_root = Path(previous_report["model_path"])
        base = load_model(base_root)
    else:
        new_results = int((games.season > base["training_through_season"]).sum())
        if new_results < 30:
            return {
                "status": "insufficient_new_results",
                "model_path": str(original_root),
                "as_of": at.isoformat(),
                "new_games": new_results,
            }
    sources = {p.name: file_hash(p) for p in paths}
    identity = {"algorithm": 3, "base": file_hash(Path(base_root) / "model.json"), "sources": sources, "games": sorted(games.game_id.astype(int).tolist())}
    version = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = output / version
    with closing(connect(database)) as conn:
        conn.executescript(SCHEMA)
        existing = conn.execute("SELECT report_json FROM learning_versions WHERE version=?", (version,)).fetchone()
        if existing:
            report = json.loads(existing[0])
            artifact = Path(report["model_path"]) / "model.json"
            if not artifact.exists() or file_hash(artifact) != report["model_sha256"]:
                raise ValueError("Registered learning model is missing or altered")
            if timestamp(report["as_of"]) > at:
                raise ValueError("Cached learning model was fitted after the requested cutoff")
            return report
    Y = observations(games, summaries)
    F, cutoffs, latest = build_features(games, Y, base["center"], base["scale"], StateConfig(**base["state_config"]), warmup=base["warmup_season"])
    eligible = games.season.ge(base["train_start_season"]).to_numpy()
    # Evaluation history begins with games that were not used by the original head.
    gate_pool = eligible & games.season.gt(base["training_through_season"]).to_numpy()
    if base.get("adaptive_training_cutoff"):
        gate_pool &= (cutoffs > timestamp(base["adaptive_training_cutoff"])).to_numpy()
    gate_dates = sorted(cutoffs[gate_pool].unique())
    if eligible.sum() < 50:
        return {"status": "insufficient_training_history", "training_games": int(eligible.sum()), "required_games": 50, "model_path": str(base_root), "watchdog": watchdog}
    boundary = gate_dates[len(gate_dates) // 2] if len(gate_dates) >= 2 else cutoffs[eligible].max()
    train = eligible & (games.result_available_at < boundary).to_numpy()
    gate = gate_pool & (cutoffs >= boundary).to_numpy()
    if train.sum() < 50:
        train = eligible.copy()
    if gate.sum() < 30:
        gate = gate_pool if gate_pool.sum() else eligible
    folder.mkdir(parents=True, exist_ok=True)
    # Persist the exact features, then fit from SQL, not a second mutable file read.
    feature_rows = []
    for i in np.flatnonzero(eligible):
        values = np.where(np.isfinite(Y[i]), Y[i], np.nan)
        feature_rows.append((version, int(games.iloc[i].game_id), int(games.iloc[i].season), cutoffs.iloc[i].isoformat(), games.iloc[i].result_available_at.isoformat(), latest[i].isoformat() if latest[i] is not None else None, json.dumps(F[i].tolist(), allow_nan=False), json.dumps([[float(x) if np.isfinite(x) else None for x in side] for side in values], allow_nan=False)))
    with closing(connect(database)) as conn:
        conn.execute("INSERT INTO learning_versions VALUES(?,?,?,?,?,?)", (version, datetime.now(timezone.utc).isoformat(), at.isoformat(), "training", str(folder / "model"), "{}"))
        conn.executemany("INSERT INTO training_examples VALUES(?,?,?,?,?,?,?,?)", feature_rows)
        stored = {row["game_id"]: row for row in conn.execute("SELECT * FROM training_examples WHERE version=?", (version,))}
        for i in np.flatnonzero(eligible):
            row = stored[int(games.iloc[i].game_id)]
            F[i] = np.asarray(json.loads(row["features_json"]), dtype=float)
            Y[i] = np.asarray(json.loads(row["observations_json"]), dtype=float)
        labels = (Y[gate, 0, 0] > Y[gate, 1, 0]).astype(int)
        incumbent_mu = ScoreHead.from_dict(base["head"]).predict(F[gate])
        incumbent_p = win_probability(incumbent_mu, base["score_covariance"])
        incumbent = metrics(labels, incumbent_p)
        # Continual reinforcement weights recent completed games more heavily while
        # retaining the complete historical record in every refit.
        age_days = (at - pd.to_datetime(games.result_available_at, utc=True)).dt.total_seconds() / 86400.0
        decay = float(base.get("reinforcement_decay", 0.002))
        reinforcement_weights = np.exp(-decay * np.maximum(age_days, 0.0))
        candidate = ScoreHead(base["variant"], base["head"]["alpha"]).fit(F, Y, train, reinforcement_weights)
        candidate_mu = candidate.predict(F[gate])
        candidate_p = win_probability(candidate_mu, base["score_covariance"])
        evaluated = metrics(labels, candidate_p)
        predictions = games.loc[gate, ["game_id", "season"]].copy()
        predictions["home_win"] = labels
        predictions["candidate_p"] = candidate_p
        predictions["incumbent_p"] = incumbent_p
        r_validation = (
            validate_in_r(predictions, folder)
            if len(predictions) >= 30 and predictions.home_win.nunique() == 2
            else {"status": "not_applicable", "reason": "insufficient outcome variation for independent score calculation"}
        )
        # Retain the incumbent unless there is independent evidence to promote.
        gate_independent = not bool((gate & train).any())
        adopted = gate_independent and int(gate.sum()) >= 30 and evaluated["log_loss"] <= incumbent["log_loss"]
        if adopted:
            # The deployed head is continuously reinforced from every result available
            # at this cutoff; the decay keeps recent team form influential without
            # discarding long-run evidence.
            production = dict(base)
            production["head"] = ScoreHead(base["variant"], base["head"]["alpha"]).fit(F, Y, eligible, reinforcement_weights).to_dict()
            production["reinforcement_decay"] = decay
            production["adaptive_training_cutoff"] = at.isoformat()
            production["adaptive_latest_result"] = games.result_available_at.max().isoformat()
            production["learning_version"] = version
            write_model(folder / "model", production)
            model_dir = folder / "model"
        else:
            model_dir = Path(base_root)
        report = {"version": version, "base_model_sha256": original_hash, "status": "promoted" if adopted else "retained", "gate_independent": gate_independent, "as_of": at.isoformat(), "model_path": str(model_dir), "model_sha256": file_hash(model_dir / "model.json"), "training_games": int(train.sum()), "refit_games": int(eligible.sum()), "reinforcement_decay": decay, "reinforcement_mode": "exponentially_weighted_continual_refit", "gate_games": int(gate.sum()), "gate_start": timestamp(boundary).isoformat(), "latest_result": games.result_available_at.max().isoformat(), "incumbent": incumbent, "candidate": evaluated, "r_validation": r_validation, "watchdog": watchdog, "source_hashes": sources, "probability_layer": "joint_gaussian_score_distribution", "limitations": ["Historical Parquet statistics may be retrospectively revised", "Continual reinforcement reflects information available at each cutoff", "Retraining updates the statistical estimate but cannot eliminate future uncertainty"]}
        if base.get("win_strength_model"):
            report["strength_model_note"] = "A strength model is saved with the base model for comparison only; it does not set win chances."
        conn.execute("UPDATE learning_versions SET status=?,model_path=?,report_json=? WHERE version=?", (report["status"], report["model_path"], json.dumps(report), version))
        conn.commit()
    (folder / "learning_report.json").write_text(json.dumps(report, indent=2))
    return report
