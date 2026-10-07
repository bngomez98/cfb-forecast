"""Discrete forecast distributions derived from reproducible joint score simulations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def probability_tables(scores: np.ndarray, game_id: int) -> dict[str, pd.DataFrame]:
    """Return normalized joint-score, margin, and total PMFs for one game."""
    scores = np.asarray(scores, dtype=int)
    if scores.ndim != 2 or scores.shape[1] != 2 or len(scores) == 0 or (scores < 0).any():
        raise ValueError("scores must be a non-empty n x 2 matrix of nonnegative integers")
    n = len(scores)
    joint = (
        pd.DataFrame({"home_score": scores[:, 0], "away_score": scores[:, 1]})
        .value_counts(sort=False)
        .rename("count")
        .reset_index()
    )
    joint.insert(0, "game_id", int(game_id))
    joint["probability"] = joint["count"] / n

    margin = (
        pd.Series(scores[:, 0] - scores[:, 1], name="margin")
        .value_counts(sort=False)
        .rename("count")
        .reset_index()
        .sort_values("margin")
        .reset_index(drop=True)
    )
    margin.insert(0, "game_id", int(game_id))
    margin["probability"] = margin["count"] / n

    total = (
        pd.Series(scores[:, 0] + scores[:, 1], name="total")
        .value_counts(sort=False)
        .rename("count")
        .reset_index()
        .sort_values("total")
        .reset_index(drop=True)
    )
    total.insert(0, "game_id", int(game_id))
    total["probability"] = total["count"] / n
    return {"joint": joint, "margin": margin, "total": total}


def discrete_crps(values, probabilities, observed: int) -> float:
    """CRPS for an integer-valued distribution using its exact discrete CDF."""
    values = np.asarray(values, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(values) == 0 or len(values) != len(probabilities) or (probabilities < 0).any():
        raise ValueError("invalid discrete distribution")
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("distribution probabilities must sum to a positive finite value")
    probabilities = probabilities / total
    lo = int(min(values.min(), observed))
    hi = int(max(values.max(), observed))
    grid = np.arange(lo, hi + 1)
    order = np.argsort(values)
    v = values[order]
    p = probabilities[order]
    cdf_at = np.cumsum(p)
    idx = np.searchsorted(v, grid, side="right") - 1
    f = np.where(idx >= 0, cdf_at[np.maximum(idx, 0)], 0.0)
    h = (grid >= int(observed)).astype(float)
    return float(np.sum((f - h) ** 2))


def distribution_summary(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    margin = scores[:, 0] - scores[:, 1]
    total = scores.sum(axis=1)
    corr = np.corrcoef(scores[:, 0], scores[:, 1])[0, 1] if len(scores) > 1 else np.nan
    return {
        "simulated_margin_mean": float(margin.mean()),
        "simulated_margin_sd": float(margin.std(ddof=1)),
        "simulated_total_mean": float(total.mean()),
        "simulated_total_sd": float(total.std(ddof=1)),
        "simulated_score_correlation": float(corr),
        "home_score_median": float(np.median(scores[:, 0])),
        "away_score_median": float(np.median(scores[:, 1])),
        "margin_median": float(np.median(margin)),
        "total_median": float(np.median(total)),
    }


def normal_crps(mean, sd, observed):
    """Closed-form CRPS for a Normal predictive distribution."""
    from scipy.stats import norm
    mean=np.asarray(mean,dtype=float); observed=np.asarray(observed,dtype=float); sd=np.asarray(sd,dtype=float)
    if np.any(sd<=0) or not np.isfinite(sd).all():
        raise ValueError("Normal CRPS requires positive finite scale")
    z=(observed-mean)/sd
    return sd*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))


def validate_heldout_distribution(predictions, model_by_variant):
    """Evaluate the saved Gaussian score distribution on frozen held-out games."""
    from scipy.stats import norm
    f=predictions.copy(); rows=[]; z80=float(norm.ppf(.9))
    for variant,model in model_by_variant.items():
        cov=np.asarray(model["score_covariance"],dtype=float)
        vm=float(cov[0,0]+cov[1,1]-2*cov[0,1]); vt=float(cov[0,0]+cov[1,1]+2*cov[0,1])
        sdm, sdt=np.sqrt(vm),np.sqrt(vt)
        mm=f[f"{variant}_home_margin"].to_numpy(float); tm=f[f"{variant}_total"].to_numpy(float)
        ym=f.actual_home_margin.to_numpy(float); yt=f.actual_total.to_numpy(float)
        hmu=f[f"{variant}_home_score"].to_numpy(float); amu=f[f"{variant}_away_score"].to_numpy(float)
        yh=f.home_points.to_numpy(float); ya=f.away_points.to_numpy(float)
        hsd,asd=np.sqrt(cov[0,0]),np.sqrt(cov[1,1])
        rows.append({
            "variant":variant,"games":int(len(f)),
            "margin_crps":float(normal_crps(mm,sdm,ym).mean()),
            "total_crps":float(normal_crps(tm,sdt,yt).mean()),
            "team_score_crps":float(np.r_[normal_crps(hmu,hsd,yh),normal_crps(amu,asd,ya)].mean()),
            "margin_interval_80_coverage":float(((ym>=mm-z80*sdm)&(ym<=mm+z80*sdm)).mean()),
            "total_interval_80_coverage":float(((yt>=tm-z80*sdt)&(yt<=tm+z80*sdt)).mean()),
            "team_score_interval_80_coverage":float((((yh>=hmu-z80*hsd)&(yh<=hmu+z80*hsd)).mean()+((ya>=amu-z80*asd)&(ya<=amu+z80*asd)).mean())/2),
            "margin_sd":float(sdm),"total_sd":float(sdt),"score_correlation":float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])),
        })
    return pd.DataFrame(rows)
