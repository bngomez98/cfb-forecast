import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cfb_forecast.storage.database import connect
from cfb_forecast.storage.evidence import freeze_forecast, verify_evidence
from cfb_forecast.evaluation.evidence import settle_evidence, gaussian_crps
from cfb_forecast.ingest.data import normalize_schedule
from cfb_forecast.modeling.forecast import _forecast_known
from cfb_forecast.modeling.forecast_model import KNOWN_COLUMNS, MARKET_COLUMNS


def batch():
    return pd.DataFrame([dict(game_id=7, season=2026, home_team="Home", away_team="Away", kickoff="2026-10-10T19:00Z", as_of="2026-10-09T19:00Z", expected_home_score=30.123456789, expected_away_score=20., home_win_probability=.75, margin_standard_deviation=12., total_standard_deviation=15., forecast_mode="football")])


def frozen(tmp_path):
    conn = connect(tmp_path / "test.sqlite")
    run = tmp_path / "run"
    run.mkdir()
    frame = batch()
    frame.to_csv(run / "forecasts.csv", index=False)
    (run / "state_snapshot.npz").write_bytes(b"exact saved state")
    freeze_forecast(conn, "run", frame, run, tmp_path / "evidence")
    return conn, run, frame


def test_frozen_forecast_retry_preserves_full_precision(tmp_path):
    conn, run, frame = frozen(tmp_path)
    freeze_forecast(conn, "run", frame, run, tmp_path / "evidence")
    row = conn.execute("SELECT * FROM forecast_evidence").fetchone()
    assert json.loads(row["payload_json"])["expected_home_score"] == 30.123456789
    assert conn.execute("SELECT count(*) FROM forecast_evidence").fetchone()[0] == 1
    assert verify_evidence(conn, "run")["input_files"] == 2
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM forecast_evidence")
    conn.close()


def test_conflicts_and_late_batch_failure_do_not_modify_frozen_rows(tmp_path):
    conn, run, frame = frozen(tmp_path)
    changed = pd.concat([frame.assign(game_id=8), frame.assign(expected_home_score=50.)])
    with pytest.raises(ValueError, match="changed"):
        freeze_forecast(conn, "run", changed, run, tmp_path / "evidence")
    assert conn.execute("SELECT count(*) FROM forecast_evidence").fetchone()[0] == 1
    (run / "state_snapshot.npz").write_bytes(b"changed state")
    with pytest.raises(ValueError, match="input changed"):
        freeze_forecast(conn, "run", frame, run, tmp_path / "evidence")
    conn.close()


def test_missing_and_corrupted_evidence_are_detected(tmp_path):
    conn, _, _ = frozen(tmp_path)
    artifact = conn.execute("SELECT artifact_path FROM forecast_inputs LIMIT 1").fetchone()[0]
    from pathlib import Path
    Path(artifact).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_evidence(conn, "run")
    conn.close()


def test_settlement_keeps_original_and_versions_corrected_outcomes(tmp_path):
    conn, _, frame = frozen(tmp_path)
    schedule = normalize_schedule(pd.DataFrame([dict(game_id=7, season=2026, week=6, start_date="2026-10-10T19:00Z", completed=True, neutral_site=False, home_id=1, away_id=2, home_team="Home", away_team="Away", home_points=28, away_points=21, fbs_game=True)]))
    _, pending = settle_evidence(conn, "run", schedule, "2026-10-10T20:00Z")
    assert pending["pending_games"] == 1
    result, report = settle_evidence(conn, "run", schedule, "2026-10-12T00:00Z")
    assert report["accuracy"] == 1
    assert result.iloc[0].margin_error == pytest.approx(30.123456789 - 20 - 7)
    assert result.iloc[0].brier == .0625
    settle_evidence(conn, "run", schedule, "2026-10-13T00:00Z")
    assert conn.execute("SELECT count(*) FROM forecast_evaluations").fetchone()[0] == 1
    schedule.loc[0, "home_points"] = 31
    settle_evidence(conn, "run", schedule, "2026-10-14T00:00Z")
    assert conn.execute("SELECT count(*) FROM forecast_evaluations").fetchone()[0] == 2
    original = json.loads(conn.execute("SELECT payload_json FROM forecast_evidence").fetchone()[0])
    assert original["expected_home_score"] == frame.iloc[0].expected_home_score
    conn.close()


def test_football_mode_excludes_quotes_but_retains_rest():
    frame = pd.DataFrame([dict(home_spread=-7., total_line=50., home_ml_decimal=1.5, away_ml_decimal=3., home_rest_days=7.)])
    football = _forecast_known(frame, "football")
    conditioned = _forecast_known(frame, "market_conditioned")
    for column in MARKET_COLUMNS:
        assert np.isnan(football[0, KNOWN_COLUMNS.index(column)])
        assert np.isfinite(conditioned[0, KNOWN_COLUMNS.index(column)])
    assert football[0, KNOWN_COLUMNS.index("home_rest_days")] == 7
    assert frame.iloc[0].home_spread == -7
    with pytest.raises(ValueError, match="forecast_mode"):
        _forecast_known(frame, "invalid")


def test_crps_known_normal_value():
    assert gaussian_crps(0, 1, 0) == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi))
    assert gaussian_crps(3, 2, 3) == pytest.approx(2 * gaussian_crps(0, 1, 0))
    assert gaussian_crps(0, 1, 3) > gaussian_crps(0, 1, 0)


def test_team_totals_use_the_same_joint_score_distribution():
    from cfb_forecast.modeling.analytic import score_summary
    result = score_summary([28., 21.], [[100., 20.], [20., 64.]], home_total_line=28., away_total_line=20.5)
    assert result["home_total_over_probability"] == pytest.approx(norm.sf(.5 / 10))
    assert result["home_total_under_probability"] == pytest.approx(norm.cdf(-.5 / 10))
    assert result["home_total_push_probability"] > 0
    assert result["away_total_push_probability"] == pytest.approx(0)
    for side in ("home", "away"):
        assert sum(result[f"{side}_total_{event}_probability"] for event in ("over", "under", "push")) == pytest.approx(1)
    assert result["home_score_standard_deviation"] == 10


def test_publisher_conflict_cannot_replace_latest_board(tmp_path, monkeypatch):
    from cfb_forecast.pipeline.publisher import publish_forecast_update
    monkeypatch.delenv("CFB_SMTP_HOST", raising=False)
    monkeypatch.delenv("DD_API_KEY", raising=False)
    run = tmp_path / "run"
    run.mkdir()
    batch().to_csv(run / "forecasts.csv", index=False)
    kwargs = dict(artifacts=tmp_path / "artifacts", database=tmp_path / "store.sqlite", web_json=tmp_path / "board.json", run_id="original")
    publish_forecast_update(run, **kwargs)
    original = (tmp_path / "board.json").read_bytes()
    batch().assign(expected_home_score=99).to_csv(run / "forecasts.csv", index=False)
    with pytest.raises(ValueError, match="changed"):
        publish_forecast_update(run, **kwargs)
    assert (tmp_path / "board.json").read_bytes() == original


def test_evidence_can_be_verified_after_moving_a_run_bundle(tmp_path):
    import shutil
    conn, _, _ = frozen(tmp_path)
    shutil.move(tmp_path / "evidence", tmp_path / "restored-evidence")
    assert verify_evidence(conn, "run", tmp_path / "restored-evidence")["status"] == "verified"
    conn.close()
