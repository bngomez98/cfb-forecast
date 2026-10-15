"""Closed-form scoring of the Gaussian predictive distribution of scores, margins and totals."""
from __future__ import annotations

import numpy as np
import pandas as pd


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
