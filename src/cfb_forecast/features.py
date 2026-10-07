"""Past-game observations and deterministic optional raw-play cleaning."""
import numpy as np
import pandas as pd
from .config import METRICS
from .data import require_columns

def clean_plays(plays):
    cols = ["game_id", "game_play_number", "pos_team_id", "def_pos_team_id", "down", "distance", "statYardage", "rush", "pass", "sack", "kneel_down", "penalty_no_play", "period", "start.pos_score_diff"]
    require_columns(plays, cols, "raw plays")
    p = plays.drop_duplicates(["game_id", "game_play_number"]).copy()
    flags = ["rush", "pass", "sack", "kneel_down", "penalty_no_play"]
    for c in flags:
        if not p[c].dropna().isin([True, False, 0, 1]).all():
            raise ValueError(f"Invalid raw-play flag {c}")
        p[c] = p[c].fillna(False).astype(bool)
    p = p[(p.rush | p["pass"] | p.sack) & ~p.kneel_down & ~p.penalty_no_play & p.down.between(1,4) & p.distance.between(1,99) & p.period.between(1,4) & p["start.pos_score_diff"].abs().le(21) & p.statYardage.notna()].copy()
    p["pass_play"] = p["pass"] | p.sack
    threshold = p.distance * np.select([p.down.eq(1), p.down.eq(2)], [0.5, 0.7], default=1.)
    p["success"] = p.statYardage.ge(threshold).astype(float)
    p["explosive"] = p.statYardage.ge(np.where(p.pass_play,20,10)).astype(float)
    p["clipped_yards"] = p.statYardage.clip(-20,80)
    return p

def aggregate_plays(plays):
    p = clean_plays(plays)
    return p.groupby(["game_id", "pos_team_id", "def_pos_team_id"], as_index=False).agg(plays=("game_play_number", "size"), success_rate=("success", "mean"), explosive_rate=("explosive", "mean"), yards_per_play=("clipped_yards", "mean"))

def observations(games, summaries):
    """No provider EPA, win probability, final-season ratings or betting inputs."""
    Y = np.full((len(games), 2, len(METRICS)), np.nan)
    Y[:,0,0] = games.home_points.to_numpy()
    Y[:,1,0] = games.away_points.to_numpy()
    if len(summaries):
        z = summaries.set_index(["game_id", "pos_team_id"])
        for side, team in enumerate(("home_id", "away_id")):
            ix = pd.MultiIndex.from_arrays([games.game_id, games[team]])
            Y[:,side,1:] = z.reindex(ix)[list(METRICS[1:])].to_numpy(dtype=float)
    Y[:,:,1] = np.clip(Y[:,:,1],0,15)
    Y[:,:,2] = np.clip(Y[:,:,2],0,20)
    Y[:,:,3] = np.clip(Y[:,:,3],-2,15)
    Y[:,:,4] = np.clip(Y[:,:,4],0,1)
    return Y

def normalization(Y, mask):
    center = np.nanmean(Y[mask], axis=(0,1))
    scale = np.nanstd(Y[mask], axis=(0,1))
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale < 1e-8).any():
        raise ValueError("Warmup data must contain variable observations for every metric")
    return center, scale

def feature_columns(variant):
    # Layout: offense(5), defense(5), offense-defense(5), products(5),
    # score-state variance, PBP variance, signed venue, early-season,
    # offense pass-rush difference, defense pass-rush difference.
    if variant == "score":
        return [0,5,10,20,22,23]
    if variant == "pbp":
        return list(range(26))
    raise ValueError(f"Unknown variant {variant}")
