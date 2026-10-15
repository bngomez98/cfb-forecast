import json

import pandas as pd

from cfb_forecast.db_cli import main


def season_files(path):
    pd.DataFrame([dict(game_id=1, season=2026, week=1, start_date="2026-09-05T16:00Z", completed=True, neutral_site=False,
                       home_id=1, away_id=2, home_team="A", away_team="B", home_points=30, away_points=20, fbs_game=True)]
                 ).to_csv(path / "schedule_2026.csv", index=False)
    pd.DataFrame([dict(game_id=1, pos_team_id=t, yards_per_play=5.0, yards_per_pass=7.0, yards_per_rush=4.0, first_downs_created_rate=.3)
                  for t in (1, 2)]).to_parquet(path / "adv_team_2026.parquet", index=False)


def test_init_update_status_and_query(tmp_path, capsys):
    season_files(tmp_path)
    db = str(tmp_path / "f.sqlite")
    assert main(["--database", db, "init"]) == 0
    assert main(["--database", db, "update", "--season", "2026", "--data", str(tmp_path), "--no-collect"]) == 0
    capsys.readouterr()
    assert main(["--database", db, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["games"] == 1
    assert main(["--database", db, "query", "SELECT home_team, home_points FROM games"]) == 0
    assert capsys.readouterr().out.splitlines() == ["home_team\thome_points", "A\t30.0"]


def test_query_is_read_only(tmp_path, capsys):
    db = str(tmp_path / "f.sqlite")
    main(["--database", db, "init"])
    assert main(["--database", db, "query", "DELETE FROM games"]) == 1
    assert "SQL error" in capsys.readouterr().err
