import json
from datetime import date

from cfb_forecast import cli
from cfb_forecast.status import collect_status, current_season, format_status


def test_season_rolls_over_in_summer():
    assert current_season(date(2026, 10, 16)) == 2026
    assert current_season(date(2027, 1, 20)) == 2026
    assert current_season(date(2027, 8, 1)) == 2027


def test_status_points_to_one_next_command(tmp_path, monkeypatch):
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(tmp_path / "art"))
    report = collect_status(2026)
    assert not report["ready"]
    assert report["next_step"].startswith("cfb-forecast update --season 2026")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("a")
    model = tmp_path / "art" / "training" / "model"
    model.mkdir(parents=True)
    (model / "m.json").write_text("{}")
    latest = tmp_path / "art" / "latest"
    latest.mkdir()
    (latest / "run_manifest.json").write_text(json.dumps({"status": "completed", "season": 2026, "games": 3}))
    (tmp_path / "art" / "cfb_forecast.sqlite").write_text("")
    report = collect_status(2026)
    assert report["ready"] and "3 games" in format_status(report)


def test_bare_command_shows_status(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(tmp_path / "a"))
    monkeypatch.setattr("sys.argv", ["cfb-forecast"])
    cli.main()
    assert "Next:" in capsys.readouterr().out
