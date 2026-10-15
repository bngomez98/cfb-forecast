import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss

def metrics(y,p,margin=None,actual_margin=None,total=None,actual_total=None,mu=None,actual_scores=None):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if len(y)==0 or len(y)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Invalid prediction/outcome population")
    r={"games":int(len(y)),"correct":int(((p>=.5)==y).sum()),"accuracy":float(((p>=.5)==y).mean()),"log_loss":float(log_loss(y,np.clip(p,1e-9,1-1e-9),labels=[0,1])),"brier":float(brier_score_loss(y,p))}
    if margin is not None:
        e=np.asarray(margin)-np.asarray(actual_margin); r.update(margin_mae=float(np.abs(e).mean()),margin_rmse=float(np.sqrt(np.mean(e*e))),margin_bias=float(e.mean()))
    if total is not None:
        r["total_mae"]=float(np.abs(np.asarray(total)-np.asarray(actual_total)).mean())
    if mu is not None:
        r["team_score_mae"]=float(np.abs(np.asarray(mu)-np.asarray(actual_scores)).mean())
    return r

def spread_regression(actual_margin, point_spread):
    """Home margin on the market's implied home margin. An unbiased line has intercept 0 and slope 1."""
    spread = np.asarray(point_spread, dtype=float)
    margin = np.asarray(actual_margin, dtype=float)
    if len(spread) < 8 or spread.shape != margin.shape or not np.isfinite(spread).all() or not np.isfinite(margin).all():
        raise ValueError("Spread regression needs at least eight finite margins and lines")
    implied = -spread
    design = np.column_stack([np.ones(len(implied)), implied])
    intercept, slope = np.linalg.lstsq(design, margin, rcond=None)[0]
    return {"games": int(len(margin)), "intercept": float(intercept), "slope": float(slope)}


def calibration(y,p):
    """Observed win rate in each tenth of probability, with the range chance alone allows.

    If the probabilities are honest, a bin's win count is a sum of independent
    coin flips with those probabilities, so its standard error is
    sqrt(sum p(1-p)) / games. A gap more than 1.96 standard errors from zero is
    outside the 95% range and is flagged.
    """
    f=pd.DataFrame({"y":np.asarray(y,dtype=int),"p":np.asarray(p,dtype=float)}); f["bin"]=pd.cut(f.p,np.linspace(0,1,11),include_lowest=True)
    f["variance"]=f.p*(1-f.p)
    out=f.groupby("bin",observed=True).agg(games=("y","size"),mean_probability=("p","mean"),observed_win_rate=("y","mean"),variance=("variance","sum")).reset_index()
    out["gap"]=out.observed_win_rate-out.mean_probability
    out["gap_standard_error"]=np.sqrt(out.variance)/out.games
    out["outside_95_range"]=out.gap.abs()>1.96*out.gap_standard_error
    return out.drop(columns="variance")


def calibration_test(y,p):
    """Spiegelhalter's z-test: one overall check that the probabilities match the outcomes.

    z is near zero for honest probabilities; |z| above 1.96 rejects calibration at the 5% level.
    """
    from scipy.stats import norm
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)
    if len(y)==0 or len(y)!=len(p):
        raise ValueError("Invalid prediction/outcome population")
    spread=np.sum((1-2*p)**2*p*(1-p))
    z=float(np.sum((y-p)*(1-2*p))/np.sqrt(spread)) if spread>0 else 0.0
    bins=calibration(y,p)
    return {"games":int(len(y)),"spiegelhalter_z":z,"p_value":float(2*norm.sf(abs(z))),"calibrated_at_5_percent":bool(abs(z)<=1.96),"bins_outside_95_range":int(bins.outside_95_range.sum()),"bins":int(len(bins))}

def paired_bootstrap(frame,left,right,draws=3000,seed=20261002):
    f=frame.copy(); y=f.home_win.to_numpy()
    def losses(col):
        p=np.clip(f[col].to_numpy(),1e-9,1-1e-9)
        return -(y*np.log(p)+(1-y)*np.log1p(-p))
    f["delta"]=losses(left)-losses(right)
    f["block"]=pd.to_datetime(f.start_date,utc=True).dt.strftime("%G-%V")
    blocks=f.groupby("block").delta.agg(["sum","count"])
    if len(blocks)<2:
        raise ValueError("At least two weekly blocks required")
    ix=np.random.default_rng(seed).integers(0,len(blocks),size=(draws,len(blocks)))
    dist=blocks["sum"].to_numpy()[ix].sum(1)/blocks["count"].to_numpy()[ix].sum(1)
    return {"comparison":f"{left} minus {right}","games":len(f),"blocks":len(blocks),"delta_log_loss":float(f.delta.mean()),"ci95_low":float(np.quantile(dist,.025)),"ci95_high":float(np.quantile(dist,.975)),"bootstrap_draws":draws,"seed":seed,"note":"Season-week blocks omit some cross-week repeated-team dependence and model-selection uncertainty"}
