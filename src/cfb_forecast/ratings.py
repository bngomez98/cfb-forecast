"""Exact conditional Kalman updates for additive offense/defense states.

Independent metric-specific Gaussian state models share opponent identities.
Defense strength is subtracted: positive defense means fewer points/yards allowed.
Hyperparameters are fixed for an entire evaluated season.
"""
import numpy as np
import pandas as pd
from .config import METRICS
from .data import timestamp

class DynamicRatings:
    def __init__(self, teams, center, scale, config):
        self.teams = [int(t) for t in teams]
        self.index = {t:i for i,t in enumerate(self.teams)}
        self.n = len(teams); self.dim = 2*self.n
        self.center = np.asarray(center); self.scale = np.asarray(scale)
        self.config = config
        self.mean = np.zeros((len(METRICS), self.dim))
        self.cov = np.tile(np.eye(self.dim)*config.prior_variance,(len(METRICS),1,1))
        self.clock = None; self.latest_update = None

    def advance(self, at):
        at = timestamp(at)
        if self.clock is not None:
            if at < self.clock:
                raise ValueError("State clock cannot run backward")
            weeks = (at-self.clock).total_seconds()/604800
            # Long offseason gaps get a separate season reset, not weekly diffusion.
            if 0 < weeks < 15:
                rho = self.config.weekly_carry**weeks
                self.mean *= rho; self.cov *= rho*rho
                self.cov[:,np.arange(self.dim),np.arange(self.dim)] += self.config.process_variance*weeks
        self.clock = at

    def offseason(self, at):
        self.advance(at)
        rho = self.config.offseason_carry
        self.mean *= rho; self.cov *= rho*rho
        self.cov[:,np.arange(self.dim),np.arange(self.dim)] += self.config.offseason_variance

    def features(self, home, away, neutral, early=False):
        hi, ai = self.index[int(home)], self.index[int(away)]
        pairs = [(hi,self.n+ai),(ai,self.n+hi)]
        venue = 0 if neutral else 1
        F = np.zeros((2,26))
        for side,(oi,di) in enumerate(pairs):
            o,d = self.mean[:,oi],self.mean[:,di]
            variances = self.cov[:,oi,oi]+self.cov[:,di,di]-2*self.cov[:,oi,di]
            F[side] = np.r_[o,d,o-d,o*d,variances[0],variances[1:].mean(),venue*(1 if side==0 else -1),int(early),o[2]-o[3],d[2]-d[3]]
        return F

    def update(self, home, away, neutral, values, available_at):
        at = timestamp(available_at)
        self.advance(at)
        hi,ai = self.index[int(home)],self.index[int(away)]
        pairs = [(hi,self.n+ai),(ai,self.n+hi)]
        venue = 0 if neutral else 1
        for j in range(len(METRICS)):
            sides = np.flatnonzero(np.isfinite(values[:,j]))
            if not len(sides):
                continue
            U = np.stack([self.cov[j,:,pairs[s][0]]-self.cov[j,:,pairs[s][1]] for s in sides],axis=1)
            S = np.stack([U[pairs[s][0]]-U[pairs[s][1]] for s in sides])
            R = np.full_like(S,self.config.observation_correlation); np.fill_diagonal(R,1.)
            residual = []
            for s in sides:
                oi,di = pairs[s]
                adj = venue*self.config.home_margin/2/self.scale[0]*(1 if s==0 else -1) if j==0 else 0.
                residual.append((values[s,j]-self.center[j])/self.scale[j]-adj-(self.mean[j,oi]-self.mean[j,di]))
            K = np.linalg.solve(S+R,U.T).T
            self.mean[j] += K @ np.asarray(residual)
            self.cov[j] -= K @ U.T
            self.cov[j] = (self.cov[j]+self.cov[j].T)/2
        self.latest_update = at

def cutoff_for_games(games, explicit=None):
    # Weekly Monday 00:00 UTC; optional verified archived forecast timestamps.
    dates = games.start_date
    cutoffs = dates.dt.normalize()-pd.to_timedelta(dates.dt.weekday,unit="D")
    cutoffs = cutoffs.where(cutoffs < dates,cutoffs-pd.Timedelta(days=7))
    if explicit:
        cutoffs = pd.Series([timestamp(explicit[int(gid)]) if int(gid) in explicit else cut for gid,cut in zip(games.game_id,cutoffs)], index=games.index)
    if (cutoffs >= games.start_date).any():
        raise ValueError("A forecast cutoff must be before kickoff")
    return cutoffs

def build_features(games, Y, center, scale, config, explicit=None, warmup=2021):
    teams = sorted(set(games.home_id)|set(games.away_id))
    states = DynamicRatings(teams,center,scale,config)
    F = np.full((len(games),2,26),np.nan)
    cutoffs = cutoff_for_games(games,explicit)
    events = []
    records = list(games.itertuples(index=False))
    for k,r in enumerate(records):
        if int(r.season)>warmup:
            events.append((cutoffs.iloc[k].value,1,k))
        events.append((r.result_available_at.value,2,k))
    for year,g in games.groupby("season"):
        events.append(((g.start_date.min()-pd.Timedelta(days=21)).value,0,int(year)))
    latest = [None]*len(games)
    for tm,typ,k in sorted(events):
        at = pd.Timestamp(tm,tz="UTC")
        if typ==0:
            states.offseason(at); continue
        r=records[k]
        states.advance(at)
        if typ==1:
            if states.latest_update is not None and states.latest_update >= at:
                raise AssertionError("Result at or after forecast cutoff entered ratings")
            F[k] = states.features(r.home_id,r.away_id,r.neutral_site,int(r.week)<=4)
            latest[k] = states.latest_update
        else:
            states.update(r.home_id,r.away_id,r.neutral_site,Y[k],at)
    return F,cutoffs,latest
