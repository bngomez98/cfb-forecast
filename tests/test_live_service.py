"""Small deterministic fixtures verify behavior; production never imports these rows."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from cfb_forecast.modeling.analytic import score_summary
from cfb_forecast.modeling.context_model import feature_columns
from cfb_forecast.ingest.data import normalize_schedule, sha256
from test_model_artifact_validation import model_value
from cfb_forecast.odds.decision import american_to_decimal, moneyline_diagnostics
from cfb_forecast.modeling.forecast import (
    archive_market,
    SnapshotReplayRequired,
    advance_forecast_snapshot,
)
from cfb_forecast.modeling.probability import (
    apply_calibration,
    fit_calibration,
    audit_probabilities,
    stress_calibration,
    forecast_probability,
)
from cfb_forecast.pipeline.sunday import slate_window, select_slate
from cfb_forecast.evaluation.validation import validate_runs


def games():
    return normalize_schedule(
        pd.DataFrame(
            [
                dict(
                    game_id=1,
                    season=2026,
                    week=7,
                    start_date="2026-10-17T16:00Z",
                    completed=False,
                    neutral_site=False,
                    home_id=1,
                    away_id=2,
                    home_team="A",
                    away_team="B",
                    home_points=None,
                    away_points=None,
                    fbs_game=True,
                ),
                dict(
                    game_id=2,
                    season=2026,
                    week=8,
                    start_date="2026-10-24T16:00Z",
                    completed=False,
                    neutral_site=False,
                    home_id=1,
                    away_id=2,
                    home_team="A",
                    away_team="B",
                    home_points=None,
                    away_points=None,
                    fbs_game=True,
                ),
                dict(
                    game_id=3,
                    season=2026,
                    week=7,
                    start_date="2026-10-17T16:00Z",
                    completed=False,
                    neutral_site=False,
                    home_id=3,
                    away_id=4,
                    home_team="C",
                    away_team="D",
                    home_points=None,
                    away_points=None,
                    fbs_game=False,
                ),
            ]
        )
    )


def test_week_selection_excludes_other_weeks(monkeypatch, tmp_path):
    import cfb_forecast.pipeline.weekly as weekly

    monkeypatch.setattr(weekly, "load_data", lambda root: (games(), None, None))

    def calculate(**kw):
        assert kw["target_game_ids"] == [1]
        Path(kw["out_root"]).mkdir(parents=True, exist_ok=True)
        return pd.DataFrame({"game_id": [1], "season": [2026], "home_team": ["A"], "away_team": ["B"],
                             "kickoff": ["2026-10-17T16:00Z"], "as_of": [kw["as_of"]],
                             "home_win_probability": [0.6], "expected_home_score": [28.0],
                             "expected_away_score": [24.0], "projected_home_margin": [4.0],
                             "projected_total": [52.0]})

    monkeypatch.setattr(weekly, "forecast", calculate)
    monkeypatch.setattr("cfb_forecast.pipeline.publisher.publish_forecast_update", lambda *a, **kw: {"status": "skipped"})
    _, report = weekly.run_weekly_forecast(
        tmp_path, tmp_path, tmp_path, 2026, as_of="2026-10-12T12:00Z"
    )
    assert report["games"] == 1


def test_sunday_selects_next_calendar_week_and_only_fbs():
    start, end = slate_window("2026-10-11T15:00Z")
    assert start == pd.Timestamp("2026-10-12T05:00Z")
    assert end == pd.Timestamp("2026-10-19T05:00Z")
    assert select_slate(games(), 2026, "2026-10-11T15:00Z").game_id.tolist() == [1]
    start, _ = slate_window("2026-11-01T16:00Z")
    assert start == pd.Timestamp("2026-11-02T06:00Z")


def test_market_archive_integrity_and_absent_source(tmp_path):
    store = tmp_path / "quotes.csv"
    store.write_text("game_id,home_spread\n1,-3\n")
    output = tmp_path / "out"
    output.mkdir()
    meta = archive_market(store, output)
    assert sha256(output / meta["market_file"]) == meta["market_sha256"]
    store.write_text("changed")
    assert sha256(output / meta["market_file"]) == meta["market_sha256"]
    assert archive_market(tmp_path / "absent.csv", output) == {
        "market_file": None,
        "market_sha256": None,
    }


def test_changed_source_requests_replay_before_loading_snapshot(tmp_path):
    prev, data, model = [tmp_path / x for x in ["prev", "data", "model"]]
    for p in [prev, data, model]:
        p.mkdir()
    (model / "model.json").write_text(
        json.dumps({**model_value(), "training_through_season": 2025})
    )
    (data / "schedule_2026.csv").write_text("changed")
    manifest = {
        "as_of": "2026-10-12T10:00Z",
        "model_sha256": sha256(model / "model.json"),
        "source_hashes": {"schedule_2026.csv": "old"},
    }
    (prev / "forecast_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SnapshotReplayRequired, match="full replay"):
        advance_forecast_snapshot(
            prev,
            data,
            model,
            tmp_path / "out",
            "2026-10-12T11:00Z",
            "2026-10-19T00:00Z",
            2026,
        )


def test_direct_probabilities_cover_push_and_total_sum():
    r = score_summary([28, 21], [[100, 10], [10, 100]], home_spread=-7, total_line=49)
    assert r["distribution_method"] == "analytic_gaussian"
    assert "simulation_draws" not in r
    assert r["home_cover_probability"] + r["away_cover_probability"] + r[
        "spread_push_probability"
    ] == pytest.approx(1)
    assert r["over_probability"] + r["under_probability"] + r[
        "total_push_probability"
    ] == pytest.approx(1)


def test_no_vig_price_ev_and_exposure_are_separate():
    h, a = american_to_decimal(-105), american_to_decimal(-115)
    result = moneyline_diagnostics(0.67, h, a)
    assert result["market_home_win_probability_no_vig"] == pytest.approx(
        0.4891657638136511
    )
    assert result["home_moneyline_expected_net_per_unit"] == pytest.approx(0.67 * h - 1)
    assert result["hypothetical_bankroll_fraction"] == 0.01
    # A no-vig edge is not sufficient if the offered price still has negative EV.
    assert (
        moneyline_diagnostics(0.52, 1.8, 1.8, min_edge=0.01)["moneyline_candidate"]
        == "none"
    )
    assert moneyline_diagnostics(0.67)["moneyline_status"] == "missing_two_sided_prices"
    with pytest.raises(ValueError):
        american_to_decimal(0)


def audit_fixture(tmp_path):
    n = 120
    f = pd.DataFrame(
        {
            "game_id": np.arange(2 * n),
            "season": np.repeat([2024, 2025], n),
            "score_p": np.tile(np.linspace(0.1, 0.9, n), 2),
            "home_win": np.tile(np.arange(n) % 2, 2),
            "start_date": np.repeat(["2024-10-05T16:00Z", "2025-10-04T16:00Z"], n),
            "forecast_cutoff": np.repeat(["2024-09-30T00:00Z", "2025-09-29T00:00Z"], n),
        }
    )
    path = tmp_path / "data.csv"
    f.to_csv(path, index=False)
    return f, path


def test_calibration_respects_prior_seasons_and_future_labels(tmp_path):
    f, path = audit_fixture(tmp_path)
    audit_probabilities(path, tmp_path / "first", min_games=100)
    first = pd.read_csv(tmp_path / "first/predictions.csv")
    f.loc[f.season == 2025, "home_win"] = 1 - f.loc[f.season == 2025, "home_win"]
    f.to_csv(path, index=False)
    audit_probabilities(path, tmp_path / "second", min_games=100)
    second = pd.read_csv(tmp_path / "second/predictions.csv")
    np.testing.assert_allclose(
        first.calibrated_probability, second.calibrated_probability
    )
    assert (
        first.loc[first.season == 2024, "calibration_status"]
        .eq("insufficient_prior_data")
        .all()
    )
    with pytest.raises(ValueError, match="before the forecast season"):
        forecast_probability(
            0.6,
            {"win_probability_calibration": {"training_through_season": 2026}},
            2026,
        )


def test_constrained_fitting_and_stress_are_bounded():
    p = np.linspace(0.1, 0.9, 120)
    y = np.arange(120) % 2
    spec = fit_calibration(p, y, 2025, max_change=0.05)
    assert np.max(np.abs(apply_calibration(p, spec) - p)) <= 0.050001
    stress = stress_calibration(p, y, spec, radius=0.1)
    assert stress["coefficient_delta_norm"] <= 0.100001
    assert stress["stress_log_loss"] >= stress["baseline_log_loss"]


def test_probability_audit_command_writes_reports(tmp_path, monkeypatch, capsys):
    import sys
    from cfb_forecast.cli import main

    _, path = audit_fixture(tmp_path)
    out = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", [
        "cfb-forecast", "probability-audit", "--input", str(path), "--out", str(out),
        "--max-change", "0.05", "--stress-radius", "0.1",
    ])
    main()
    summary = json.loads((out / "summary.json").read_text())
    assert json.loads(capsys.readouterr().out)["games"] == summary["games"] == 240
    assert summary["production_changed"] is False
    assert summary["max_probability_change"] == 0.05
    assert [row["season"] for row in summary["stress_tests"]] == [2025]
    for name in ("predictions.csv", "metrics.csv", "calibration.csv", "candidate_calibrator.json"):
        assert (out / name).exists()


def test_context_features_exclude_outcomes_and_betting_prices():
    frame = pd.DataFrame(
        {
            "home_points": [21],
            "home_win": [1],
            "home_ml_decimal": [1.9],
            "home_passing.passingYards": [400],
            "weather_temperature_f": [60],
            "weather_status": ["available"],
        }
    )
    assert feature_columns(frame) == [
        "home_passing.passingYards",
        "weather_temperature_f",
    ]


def test_validation_does_not_rewrite_predictions_and_grades_actual_prices(tmp_path):
    path = tmp_path / "run1"
    path.mkdir()
    pred = pd.DataFrame(
        [
            dict(
                game_id=1,
                as_of="2026-10-12T12:00Z",
                kickoff="2026-10-17T16:00Z",
                home_win_probability=0.7,
                projected_home_margin=7.0,
                projected_total=49.0,
                moneyline_candidate="home",
                home_ml_decimal=2.0,
            )
        ]
    )
    pred.to_csv(path / "forecasts.csv", index=False)
    original = sha256(path / "forecasts.csv")
    (path / "run_manifest.json").write_text(json.dumps({"prediction_sha256": original}))
    schedule = games()
    schedule.loc[schedule.game_id == 1, ["completed", "home_points", "away_points"]] = [
        True,
        28,
        21,
    ]
    schedule = normalize_schedule(schedule)
    result, report = validate_runs(tmp_path, schedule, "2026-10-18T15:00Z")
    assert result.home_win.tolist() == [1]
    assert report["runs"][0]["hypothetical_profit_units"] == 1.0
    assert sha256(path / "forecasts.csv") == original
    (path / "forecasts.csv").write_text("edited")
    with pytest.raises(ValueError, match="changed"):
        validate_runs(tmp_path, schedule, "2026-10-18T15:00Z")


def test_named_sources_require_verified_season_and_preserve_identity():
    from cfb_forecast.ingest.sources import TableParser, named_provider_features

    html = '<table><tr><th>Rank</th><th>Team</th><th>Conf</th><th>Games</th><th>Total Yards / Game</th></tr><tr><td>1</td><td><a href="/teams/194-ohio-state">Ohio State</a></td><td>Big Ten</td><td>6</td><td>503.5</td></tr></table>'
    parser = TableParser()
    parser.feed(html)
    source = {
        "provider": "cfbtrack_team_stats",
        "season_verified": False,
        "tables": parser.tables,
        "table_links": parser.table_links,
        "table_fragments": parser.table_fragments,
        "url": "https://cfbtrack.com/stats/team-season-leaders",
    }
    assert named_provider_features([source], 194, [])[0] == {}
    source["season_verified"] = True
    features, evidence = named_provider_features([source], 194, [])
    assert features["cfbtrack.total_yards_per_game"] == 503.5
    assert evidence[0]["team_id"] == 194
    assert named_provider_features([source], 84, [])[0] == {}


def test_stale_or_future_injury_reports_cannot_enter_live_inputs():
    from cfb_forecast.ingest.sources import current_injuries

    reports = [
        {"date": "2020-11-21T18:31Z", "status": "Out"},
        {"date": "2026-10-20T12:00Z", "status": "Out"},
        {"date": "2026-10-11T12:00Z", "status": "Questionable"},
    ]
    current, rejected = current_injuries(reports, "2026-10-12T12:00Z")
    assert current == [reports[2]]
    assert len(rejected) == 2


def test_sunday_collects_inputs_before_validation_and_inference_and_preserves_run(
    monkeypatch, tmp_path
):
    import cfb_forecast.pipeline.sunday as service

    events = []
    monkeypatch.setattr(service, "utc_now", lambda: "2026-10-12T12:00:00Z")
    monkeypatch.setattr(service, "load_data", lambda root: (games(), None, []))
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.json").write_text(
        json.dumps({**model_value(), "training_through_season": 2025})
    )

    def collect(games, season, root):
        events.append("inputs")
        root.mkdir(parents=True)
        features = pd.DataFrame({"game_id": [1], "inputs_complete": [False]})
        features.to_csv(root / "features.csv", index=False)
        return features, []

    def prices(*args):
        events.append("prices")
        return 0

    def validate(*args):
        events.append("validate")
        return pd.DataFrame(), {"runs": [], "settled_games": 0}

    def infer(data, model, out, at, end, season, **kwargs):
        events.append("forecast")
        assert kwargs["analytic"]
        assert (out / "inputs/features.csv").exists()
        return pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "as_of": str(at),
                    "kickoff": "2026-10-17T16:00Z",
                    "home_win_probability": 0.6,
                    "raw_home_win_probability": 0.6,
                }
            ]
        )

    monkeypatch.setattr(service, "collect_game_inputs", collect)
    monkeypatch.setattr(service, "capture_prices", prices)
    monkeypatch.setattr("cfb_forecast.pipeline.database_update.update_database", lambda *a, **kw: {"status": "updated"})
    monkeypatch.setattr("cfb_forecast.ingest.cfbd_market.capture_cfbd_prices", lambda *a, **kw: 0)
    monkeypatch.setattr(service, "validate_runs", validate)
    monkeypatch.setattr(service, "forecast", infer)
    import cfb_forecast.modeling.learning as learning
    import cfb_forecast.storage.database as database
    def train(*args):
        events.append("learn")
        return {"status": "retained", "model_path": str(model)}
    monkeypatch.setattr(learning, "learn", train)
    monkeypatch.setattr(database, "persist_completed_run", lambda *args: events.append("persist"))
    report = service.sunday_run(
        tmp_path / "data", tmp_path / "artifacts", 2026, model_root=model, refresh=False
    )
    assert events == ["inputs", "prices", "validate", "learn", "forecast", "persist"]
    prediction = tmp_path / "artifacts/runs" / report["run_id"] / "forecasts.csv"
    before = sha256(prediction)
    assert report["input_hashes"]["features.csv"]
    repeated = service.sunday_run(
        tmp_path / "data", tmp_path / "artifacts", 2026, model_root=model, refresh=False
    )
    assert repeated["status"] == "already_completed"
    assert sha256(prediction) == before
    assert len(events) == 6


def test_strength_ratings_do_not_use_future_final_results():
    from cfb_forecast.modeling.strength import strength_margin

    schedule = games()
    schedule.loc[
        schedule.game_id.isin([1, 2]), ["completed", "home_points", "away_points"]
    ] = [True, 28, 21]
    schedule = normalize_schedule(schedule)
    target = schedule[schedule.game_id == 2]
    original = strength_margin(schedule, target, "2026-10-20T12:00Z")
    changed = schedule.copy()
    changed.loc[changed.game_id == 2, ["home_points", "away_points"]] = [99, 0]
    np.testing.assert_array_equal(
        original, strength_margin(changed, target, "2026-10-20T12:00Z")
    )
