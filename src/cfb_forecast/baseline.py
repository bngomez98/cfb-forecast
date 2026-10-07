"""Independent chronological Elo baseline with prior-season calibration."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .ratings import cutoff_for_games
from .evaluate import metrics

def elo_features(games,explicit=None):
    rating={int(t):1500. for t in set(games.home_id)|set(games.away_id)}
    cutoffs=cutoff_for_games(games,explicit); records=list(games.itertuples(index=False))
    events=[]; F=np.zeros((len(games),2))
    for k,r in enumerate(records):
        events.extend([(cutoffs.iloc[k].value,1,k),(r.result_available_at.value,2,k)])
    for year,g in games.groupby("season"):
        events.append(((g.start_date.min()-pd.Timedelta(days=21)).value,0,int(year)))
    for _,typ,k in sorted(events):
        if typ==0:
            rating={t:1500.+.67*(v-1500.) for t,v in rating.items()}; continue
        r=records[k]; h,a=int(r.home_id),int(r.away_id); venue=float(not r.neutral_site)
        if typ==1:
            F[k]=[venue,(rating[h]-rating[a])/400]
        else:
            p=1/(1+10**(-(rating[h]-rating[a]+55*venue)/400))
            delta=20*(float(r.home_points>r.away_points)-p)
            rating[h]+=delta; rating[a]-=delta
    return F

def evaluate_elo(games,train,val,final_train,test,explicit=None):
    F=elo_features(games,explicit); y=(games.home_points>games.away_points).to_numpy(dtype=int)
    def fit(mask,C):
        return make_pipeline(StandardScaler(),LogisticRegression(C=C,fit_intercept=False,max_iter=1000)).fit(np.r_[F[mask],-F[mask]],np.r_[y[mask],1-y[mask]])
    trials=[]
    for C in [.01,.1,1.,10.]:
        model=fit(train,C); p=model.predict_proba(F[val])[:,1]
        trials.append({"C":C,**metrics(y[val],p)})
    chosen=min(trials,key=lambda r:r["log_loss"])
    p=fit(final_train,chosen["C"]).predict_proba(F[test])[:,1]
    return p,trials,chosen
