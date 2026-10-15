"""Specification of the football system being modeled.

Everything here describes *football*: which time-varying state representations exist
for teams, units and players, and which observed quantities each of them explains.
The joint model (``football_model``) is built from this specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------------------ state representations
TEAM_GROUPS = {
    "OFF": ["pass_efficiency", "rush_efficiency", "success_creation", "pass_explosiveness", "rush_explosiveness",
            "early_down_efficiency", "late_down_efficiency", "short_yardage_efficiency", "red_zone_efficiency",
            "goal_to_go_efficiency", "drive_sustainability", "field_position_conversion", "sack_avoidance",
            "passing_turnover_tendency", "rushing_ball_security", "penalty_tendency", "tempo", "fourth_down_execution",
            "scoring_opportunity_creation", "scoring_opportunity_conversion"],
    "DEF": ["pass_suppression", "rush_suppression", "success_suppression", "explosive_pass_prevention",
            "explosive_rush_prevention", "early_down_defense", "late_down_defense", "short_yardage_defense",
            "red_zone_defense", "goal_line_defense", "drive_termination", "pressure_creation", "sack_creation",
            "interception_creation", "fumble_creation", "penalty_tendency", "field_position_control", "tempo_influence"],
    # offensive-line continuity needs lineup data no source provides; it is not estimated.
    "OL": ["pass_protection", "rush_blocking", "penalty_tendency"],
    "FRONT": ["pass_pressure", "rush_disruption", "sack_creation", "tackle_for_loss_creation"],
    "ST": ["field_goal_ability", "kickoff_ability", "punt_ability", "kick_coverage", "punt_coverage", "return_efficiency"],
}
# Scramble and designed-run efficiency cannot be separated in play-by-play; they share one component.
PLAYER_GROUPS = {
    "QB": ["passing_efficiency", "completion_quality", "short_pass_efficiency", "intermediate_pass_efficiency",
           "deep_pass_efficiency", "interception_tendency", "sack_avoidance", "rushing_efficiency", "ball_security",
           "usage_capacity"],
    "RB": ["rushing_efficiency", "explosive_run_ability", "short_yardage_ability", "receiving_efficiency",
           "ball_security", "usage_capacity"],
    "REC": ["target_efficiency", "catch_efficiency", "deep_receiving_efficiency", "yards_after_catch_efficiency",
            "red_zone_receiving_efficiency", "usage_capacity"],
}
# Season-transition covariates R (observed roster/coaching changes).
TEAM_COVARIATES = ["returning_usage_offense", "qb_change", "returning_production_ppa"]
PLAYER_COVARIATES = ["prior_usage_share"]

TEAM_OFFSETS: dict[str, int] = {}
_k = 0
for _g, _c in TEAM_GROUPS.items():
    TEAM_OFFSETS[_g] = _k
    _k += len(_c)
K_TEAM = _k                                       # columns of the stacked team state vector


def col(group: str, comp: str) -> int:
    return TEAM_OFFSETS[group] + TEAM_GROUPS[group].index(comp)


# ------------------------------------------------------------------ observation streams
@dataclass
class StreamSpec:
    name: str
    kind: str                                      # "N" mean of a continuous play quantity, "B" binomial rate
    i: list = field(default_factory=list)          # (group, component, sign) terms of the offense/first team
    j: list = field(default_factory=list)          # (group, component, sign) terms of the defense/second team
    players: list = field(default_factory=list)    # (player_group, component, role) additive player effects
    matchup: bool = False                          # carries the O^T W D interaction
    loading: bool = False                          # free positive loading on the team terms
    home: bool = True
    key: bool = False                              # feeds the possession-outcome model


def _s(name, kind, i=(), j=(), players=(), **kw):
    return StreamSpec(name, kind, list(i), list(j), list(players), **kw)


STREAMS = [
    _s("pass_epa", "N", [("OFF", "pass_efficiency", 1)], [("DEF", "pass_suppression", -1)],
       [("QB", "passing_efficiency", "dropback"), ("REC", "target_efficiency", "target")], matchup=True, key=True),
    _s("rush_epa", "N", [("OFF", "rush_efficiency", 1)], [("DEF", "rush_suppression", -1)],
       [("QB", "rushing_efficiency", "carry_qb"), ("RB", "rushing_efficiency", "carry_rb")], matchup=True, key=True),
    _s("success_rate", "B", [("OFF", "success_creation", 1)], [("DEF", "success_suppression", -1)], matchup=True, key=True),
    _s("pass_explosive", "B", [("OFF", "pass_explosiveness", 1)], [("DEF", "explosive_pass_prevention", -1)],
       [("REC", "deep_receiving_efficiency", "target")], matchup=True, key=True),
    _s("rush_explosive", "B", [("OFF", "rush_explosiveness", 1)], [("DEF", "explosive_rush_prevention", -1)],
       [("RB", "explosive_run_ability", "carry_rb")], matchup=True, key=True),
    _s("early_down_epa", "N", [("OFF", "early_down_efficiency", 1)], [("DEF", "early_down_defense", -1)], matchup=True),
    _s("late_down_epa", "N", [("OFF", "late_down_efficiency", 1)], [("DEF", "late_down_defense", -1)], matchup=True),
    _s("short_yardage_success", "B", [("OFF", "short_yardage_efficiency", 1)], [("DEF", "short_yardage_defense", -1)],
       [("RB", "short_yardage_ability", "carry_rb")], matchup=True),
    _s("red_zone_epa", "N", [("OFF", "red_zone_efficiency", 1)], [("DEF", "red_zone_defense", -1)],
       [("REC", "red_zone_receiving_efficiency", "target")], matchup=True, key=True),
    _s("goal_to_go_epa", "N", [("OFF", "goal_to_go_efficiency", 1)], [("DEF", "goal_line_defense", -1)], matchup=True),
    _s("first_down_rate", "B", [("OFF", "drive_sustainability", 1)], [("DEF", "drive_termination", -1)], matchup=True, key=True),
    _s("sack_avoid", "B", [("OFF", "sack_avoidance", 1), ("OL", "pass_protection", 1)],
       [("DEF", "sack_creation", -1), ("FRONT", "sack_creation", -1)], [("QB", "sack_avoidance", "dropback")], matchup=True, key=True),
    _s("hurry_avoid", "B", [("OL", "pass_protection", 1)], [("DEF", "pressure_creation", -1), ("FRONT", "pass_pressure", -1)]),
    _s("stuff_avoid", "B", [("OL", "rush_blocking", 1)], [("FRONT", "rush_disruption", -1)]),
    _s("tfl_avoid", "B", [("OL", "rush_blocking", 1)], [("FRONT", "tackle_for_loss_creation", -1)]),
    _s("pass_int", "B", [("OFF", "passing_turnover_tendency", 1)], [("DEF", "interception_creation", -1)],
       [("QB", "interception_tendency", "dropback")], matchup=True, key=True),
    _s("rush_fumble_safe", "B", [("OFF", "rushing_ball_security", 1)], [("DEF", "fumble_creation", -1)],
       [("QB", "ball_security", "carry_qb"), ("RB", "ball_security", "carry_rb")], key=True),
    _s("offense_penalty_rate", "B", [("OFF", "penalty_tendency", 1), ("OL", "penalty_tendency", 1)], []),
    _s("defense_penalty_rate", "B", [], [("DEF", "penalty_tendency", 1)]),
    _s("tempo_plays", "N", [("OFF", "tempo", 1)], [("DEF", "tempo_influence", -1)], key=True),
    _s("fourth_down_conversion", "B", [("OFF", "fourth_down_execution", 1)], [("DEF", "late_down_defense", -1)]),
    _s("scoring_opp_created", "B", [("OFF", "scoring_opportunity_creation", 1)], [("DEF", "drive_termination", -1)], key=True),
    _s("scoring_opp_converted", "B", [("OFF", "scoring_opportunity_conversion", 1)], [("DEF", "red_zone_defense", -1)], key=True),
    _s("field_position_points", "N", [("OFF", "field_position_conversion", 1)], [("DEF", "field_position_control", -1)]),
    _s("opponent_drive_start", "N", [], [("DEF", "field_position_control", 1)]),
    _s("pass_epa_short", "N", [("OFF", "pass_efficiency", 1)], [("DEF", "pass_suppression", -1)],
       [("QB", "short_pass_efficiency", "dropback")], loading=True),
    _s("pass_epa_intermediate", "N", [("OFF", "pass_efficiency", 1)], [("DEF", "pass_suppression", -1)],
       [("QB", "intermediate_pass_efficiency", "dropback")], loading=True),
    _s("pass_epa_deep", "N", [("OFF", "pass_explosiveness", 1)], [("DEF", "explosive_pass_prevention", -1)],
       [("QB", "deep_pass_efficiency", "dropback"), ("REC", "deep_receiving_efficiency", "target")], loading=True),
    _s("completion_over_expected", "N", [], [("DEF", "pass_suppression", -1)],
       [("QB", "completion_quality", "dropback"), ("REC", "catch_efficiency", "target")], loading=True),
    _s("yards_after_catch", "N", [], [("DEF", "pass_suppression", -1)],
       [("REC", "yards_after_catch_efficiency", "target")], loading=True),
    _s("rb_target_epa", "N", [], [("DEF", "pass_suppression", -1)], [("RB", "receiving_efficiency", "rb_target")], loading=True),
    _s("fg_make", "B", [("ST", "field_goal_ability", 1)], [], key=True, loading=True, home=False),
    _s("kickoff_touchback", "B", [], [("ST", "kickoff_ability", 1)], loading=True, home=False),
    _s("kickoff_yards", "N", [], [("ST", "kickoff_ability", 1)], loading=True, home=False),
    _s("punt_yards", "N", [("ST", "punt_ability", 1)], [], loading=True, home=False),
    _s("kick_return_yards", "N", [("ST", "return_efficiency", 1)], [("ST", "kick_coverage", -1)], loading=True),
    _s("punt_return_yards", "N", [("ST", "punt_coverage", -1)], [("ST", "return_efficiency", 1)], loading=True),
]
STREAM_INDEX = {s.name: n for n, s in enumerate(STREAMS)}
KEY_STREAMS = [s.name for s in STREAMS if s.key]

# Possession outcomes; "punt" is the reference class.
OUTCOMES = ["punt", "touchdown", "field_goal", "missed_field_goal", "turnover", "turnover_on_downs", "end_of_period", "rare_score"]
TD_POINTS = [6, 7, 8]
RARE_POINTS = [0, 2, 6, 7, 8]

WEEKS = 16                                         # weeks per season in the time index
SOURCES = ["sportsdataverse", "cfbd"]
RATING_SOURCES = ["sp", "fpi", "elo", "srs"]


def period_index(season_index: int, week: int) -> int:
    return season_index * WEEKS + min(max(int(week), 1), WEEKS) - 1
