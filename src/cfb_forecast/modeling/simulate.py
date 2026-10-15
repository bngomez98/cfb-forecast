"""Joint Gaussian score approximation, with bounded integer-score simulations.

Simulation is a score-distribution approximation, not a drive/rules engine.
Continuous margin probabilities are used in historical probability evaluation.
"""
import hashlib
import numpy as np
from scipy.stats import norm
from .distribution import distribution_summary


def checked_covariance(cov):
    cov=np.asarray(cov,dtype=float)
    if cov.shape!=(2,2) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min()<=0:
        raise ValueError("Score covariance must be a finite positive-definite 2x2 matrix")
    return cov


def win_probability(mu,cov):
    cov=checked_covariance(cov)
    sd=np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1])
    mu=np.asarray(mu,dtype=float)
    return np.clip(norm.cdf((mu[...,0]-mu[...,1])/sd),1e-9,1-1e-9)


def game_seed(game_id,seed=20261002):
    return int.from_bytes(hashlib.sha256(f"{seed}:{game_id}".encode()).digest()[:8],"little")


def draw_score_samples(mu,cov,game_id,draws=50000,seed=20261002):
    if draws<1000:
        raise ValueError("Use at least 1000 simulation draws")
    mu=np.asarray(mu,dtype=float)
    if mu.shape!=(2,) or not np.isfinite(mu).all() or (mu<0).any():
        raise ValueError("Expected scores must be two nonnegative finite values")
    cov=checked_covariance(cov)
    rng=np.random.default_rng(game_seed(game_id,seed))
    scores=np.maximum(np.rint(rng.multivariate_normal(mu,cov,size=draws)),0).astype(int)
    # Approximate overtime for tied simulated regulation scores; all final draws have a winner.
    tied=np.flatnonzero(scores[:,0]==scores[:,1])
    winners=rng.integers(0,2,len(tied)); scores[tied,winners]+=rng.choice([3,7],len(tied),p=[.6,.4])
    return scores


def summarize_score_samples(scores,mu,cov,game_id,seed=20261002,home_spread=None,total_line=None,home_decimal=None,away_decimal=None):
    scores=np.asarray(scores,dtype=int); mu=np.asarray(mu,dtype=float); cov=checked_covariance(cov)
    margin=scores[:,0]-scores[:,1]; total=scores.sum(axis=1)
    result={"expected_home_score":float(mu[0]),"expected_away_score":float(mu[1]),"projected_home_margin":float(mu[0]-mu[1]),"projected_total":float(mu.sum()),"home_win_probability":float(win_probability(mu,cov)),"simulated_home_win_probability":float(np.mean(margin>0)),"simulation_draws":int(len(scores)),"simulation_seed":game_seed(game_id,seed),"margin_interval_80":np.quantile(margin,[.1,.9]).tolist(),"total_interval_80":np.quantile(total,[.1,.9]).tolist(),"home_score_interval_80":np.quantile(scores[:,0],[.1,.9]).tolist(),"away_score_interval_80":np.quantile(scores[:,1],[.1,.9]).tolist(),"simulated_mean_home_score":float(scores[:,0].mean()),"simulated_mean_away_score":float(scores[:,1].mean()),**distribution_summary(scores)}
    if home_spread is not None:
        adjusted=margin+float(home_spread)
        result.update(home_spread=float(home_spread),home_cover_probability=float(np.mean(adjusted>0)),away_cover_probability=float(np.mean(adjusted<0)),spread_push_probability=float(np.mean(adjusted==0)))
        for side,odds in [("home",home_decimal),("away",away_decimal)]:
            if odds is not None:
                if not np.isfinite(odds) or odds<=1:
                    raise ValueError("Decimal odds must exceed 1")
                pwin=result[f"{side}_cover_probability"]; plose=result[f"{'away' if side=='home' else 'home'}_cover_probability"]
                result[f"{side}_spread_expected_net_per_unit"]=pwin*(odds-1)-plose
    if total_line is not None:
        result.update(total_line=float(total_line),over_probability=float(np.mean(total>total_line)),under_probability=float(np.mean(total<total_line)),total_push_probability=float(np.mean(total==total_line)))
    return result


def simulate_scores(mu,cov,game_id,draws=50000,seed=20261002,home_spread=None,total_line=None,home_decimal=None,away_decimal=None,return_samples=False):
    scores=draw_score_samples(mu,cov,game_id,draws=draws,seed=seed)
    result=summarize_score_samples(scores,mu,cov,game_id,seed,home_spread,total_line,home_decimal,away_decimal)
    return (result,scores) if return_samples else result
