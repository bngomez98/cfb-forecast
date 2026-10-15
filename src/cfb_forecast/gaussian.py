"""Gaussian score-location helpers shared by the forecast, evaluation and learning code."""
import numpy as np
from scipy.stats import norm


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
