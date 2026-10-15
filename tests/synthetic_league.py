"""A small synthetic league with known team strengths, written into the real tables (test use only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build(conn, teams=8, seasons=(2023, 2024, 2025), weeks=7, observed_weeks_last=4, seed=3):
    """Round-robin weeks; the last season only has results through ``observed_weeks_last``.

    Returns the true offense and defense strengths so tests can check recovery.
    """
    rng = np.random.default_rng(seed)
    off = rng.normal(0, 0.12, teams)       # true pass/rush/success quality of each offense
    dfn = rng.normal(0, 0.12, teams)
    qb_q = rng.normal(0, 0.1, teams)
    games, plays, derived, drives, players = [], [], [], [], []
    gid, pid = 1000, 0
    team_ids = list(range(1, teams + 1))
    for p in team_ids:
        players += [(f"qb{p}", "QB"), (f"rb{p}a", "RB"), (f"rb{p}b", "RB")] + [(f"wr{p}{k}", "REC") for k in range(4)]
    for yi, season in enumerate(seasons):
        order = np.array(team_ids)
        for week in range(1, weeks + 1):
            rng.shuffle(order)
            for k in range(teams // 2):
                home, away = int(order[2 * k]), int(order[2 * k + 1])
                start = pd.Timestamp(f"{season}-09-01", tz="UTC") + pd.Timedelta(days=7 * (week - 1), hours=16)
                hidden = yi == len(seasons) - 1 and week > observed_weeks_last
                hp = ap = None
                per_team_points = {}
                for off_t, def_t, side in ((home, away, 1), (away, home, 0)):
                    n = int(rng.integers(36, 52))
                    is_pass = rng.random(n) < 0.5
                    mu_pass = 0.10 + off[off_t - 1] - dfn[def_t - 1] + qb_q[off_t - 1] + 0.02 * side
                    mu_rush = 0.00 + 0.6 * (off[off_t - 1] - dfn[def_t - 1])
                    epa = np.where(is_pass, rng.normal(mu_pass, 1.1, n), rng.normal(mu_rush, 1.0, n))
                    p_succ = 1 / (1 + np.exp(-(-0.3 + 2.0 * (off[off_t - 1] - dfn[def_t - 1]))))
                    success = rng.random(n) < p_succ
                    sack = is_pass & (rng.random(n) < 0.07)
                    down = rng.integers(1, 5, n)
                    pid0 = pid
                    for i in range(n):
                        pid += 1
                        plays.append(dict(play_id=str(pid), game_id=gid, drive_id=f"{gid}-{off_t}-{i // 6}", offense_team_id=off_t, defense_team_id=def_t,
                                          scrimmage_play=1, is_pass=int(is_pass[i]), is_rush=int(not is_pass[i]), sack=int(sack[i]), qb_hurry=int(rng.random() < 0.1),
                                          early_down=int(down[i] <= 2), late_down=int(down[i] > 2), red_zone=int(rng.random() < 0.12), goal_to_go=int(rng.random() < 0.04),
                                          stuffed_run=int((not is_pass[i]) and rng.random() < 0.15), tackle_for_loss=int(rng.random() < 0.07),
                                          short_yardage_attempt=int(rng.random() < 0.06), short_yardage_success=int(rng.random() < 0.7),
                                          scoring_opportunity=int(rng.random() < 0.2), first_down=int(rng.random() < 0.25), touchdown=0, interception=int(is_pass[i] and rng.random() < 0.025),
                                          fumble_lost=int(rng.random() < 0.01), completion=int(is_pass[i] and rng.random() < 0.6), penalty=0, down=int(down[i]),
                                          punt=0, field_goal_attempt=0, kickoff=0, touchback=0, yards_to_goal=float(rng.integers(5, 95)),
                                          passer_id=f"qb{off_t}" if is_pass[i] else None,
                                          rusher_id=None if is_pass[i] else (f"rb{off_t}a" if rng.random() < 0.7 else f"rb{off_t}b"),
                                          receiver_id=f"wr{off_t}{int(rng.integers(0, 4))}" if is_pass[i] else None,
                                          air_yards=float(rng.integers(-2, 30)) if is_pass[i] else None, yards_after_catch=float(rng.integers(0, 12)),
                                          source_id="sportsdataverse", available_at=str(start)))
                        derived.append((str(pid), "cfbfastR_ep", "synthetic", None, None, float(epa[i]), 0.6, float(rng.normal(0, 0.2)), int(success[i]),
                                        int(rng.random() < 0.08), 0, "synthetic", str(start)))
                    # drives with outcomes depending on the matchup
                    strength = off[off_t - 1] - dfn[def_t - 1] + qb_q[off_t - 1]
                    z = np.array([0.0, -1.0 + 3 * strength, -2.3 + 1.5 * strength, -3.0, -2.4 - 2 * strength, -2.8, -3.5, -5.0])
                    probs = np.exp(z) / np.exp(z).sum()
                    classes = ["punt", "touchdown", "field_goal", "missed_field_goal", "turnover", "turnover_on_downs", "end_of_period", "rare_score"]
                    pts_total = 0
                    for i in range(8):
                        c = classes[rng.choice(8, p=probs)]
                        pts = {"touchdown": rng.choice([6, 7, 8], p=[.05, .9, .05]), "field_goal": 3, "rare_score": 0}.get(c, 0)
                        pts_total += int(pts)
                        drives.append(dict(drive_id=f"{gid}-{off_t}-{i}", game_id=gid, team_id=off_t, outcome_class=c, points=float(pts), opponent_points=0.0,
                                           result=c, plays=6, source_id="sportsdataverse", available_at=str(start)))
                    per_team_points[off_t] = pts_total
                hp, ap = per_team_points[home], per_team_points[away]
                games.append(dict(game_id=gid, season=season, week=week, start_date=str(start), completed=int(not hidden), neutral_site=0, fbs_game=1,
                                  result_usable=int(not hidden), result_available_at=str(start + pd.Timedelta(hours=8)), home_id=home, away_id=away,
                                  home_team=f"T{home}", away_team=f"T{away}", home_points=None if hidden else float(hp), away_points=None if hidden else float(ap)))
                if hidden:           # hidden games have no results and no plays yet
                    plays = [p for p in plays if p["game_id"] != gid]
                    drives = [d for d in drives if d["game_id"] != gid]
                gid += 1
    keep = {p["play_id"] for p in plays}
    derived = [d for d in derived if d[0] in keep]
    pd.DataFrame(games).to_sql("games", conn, if_exists="append", index=False)
    pd.DataFrame(plays).to_sql("plays", conn, if_exists="append", index=False)
    pd.DataFrame(drives).to_sql("drives", conn, if_exists="append", index=False)
    pd.DataFrame(players, columns=["player_id", "position"]).to_sql("players", conn, if_exists="append", index=False)
    conn.executemany("""INSERT INTO derived_play_observations(play_id,derivation_model,derivation_model_version,expected_points_before,expected_points_after,epa,
        expected_completion_probability,completion_over_expected,success_indicator,explosive_play_indicator,negative_play_indicator,input_definition,
        calculation_timestamp) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", derived)
    conn.commit()
    return dict(offense=off, defense=dfn, qb=qb_q)
