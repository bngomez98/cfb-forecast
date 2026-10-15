"""Regularized score heads; calibration covariance estimated on prior validation."""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .features import feature_columns

class ScoreHead:
    def __init__(self, variant="score", alpha=10., n_metrics=None):
        self.variant=variant; self.alpha=float(alpha); self.cols=feature_columns(variant, n_metrics)
        self.scaler=StandardScaler(); self.regressor=Ridge(alpha=self.alpha)

    def fit(self,F,Y,mask,sample_weight=None):
        x=F[mask][:,:,self.cols].reshape(-1,len(self.cols))
        y=Y[mask,:,0].reshape(-1)
        if len(y)<100 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Need at least 50 completed games with finite pregame features to fit")
        weights=None
        if sample_weight is not None:
            game_weights=np.asarray(sample_weight,dtype=float)[mask]
            if len(game_weights)!=int(mask.sum()) or not np.isfinite(game_weights).all() or (game_weights<=0).any():
                raise ValueError("sample_weight must contain one positive finite value per game")
            weights=np.repeat(game_weights,2)
        self.regressor.fit(self.scaler.fit_transform(x),y,sample_weight=weights)
        return self

    def predict(self,F):
        x=F[:,:,self.cols].reshape(-1,len(self.cols))
        if not np.isfinite(x).all():
            raise ValueError("Missing or invalid pregame features")
        mu=self.regressor.predict(self.scaler.transform(x)).reshape(-1,2)
        return np.maximum(mu,0.)

    def to_dict(self):
        return {"variant":self.variant,"alpha":self.alpha,"cols":self.cols,"mean":self.scaler.mean_.tolist(),"scale":self.scaler.scale_.tolist(),"coef":self.regressor.coef_.tolist(),"intercept":float(self.regressor.intercept_)}

    @classmethod
    def from_dict(cls,value):
        head=cls(value["variant"],value["alpha"])
        if head.cols!=value["cols"]:
            raise ValueError("Feature schema incompatible with saved model")
        head.scaler.mean_=np.array(value["mean"]); head.scaler.scale_=np.array(value["scale"])
        head.scaler.var_=head.scaler.scale_**2; head.scaler.n_features_in_=len(head.cols)
        head.regressor.coef_=np.array(value["coef"]); head.regressor.intercept_=value["intercept"]; head.regressor.n_features_in_=len(head.cols)
        return head

def residual_covariance(actual, predicted):
    residual=actual-predicted
    if len(residual)<50:
        raise ValueError("At least 50 out-of-fit calibration games are required")
    cov=np.cov(residual.T)
    values,vectors=np.linalg.eigh(cov)
    return (vectors*np.maximum(values,1.))@vectors.T

def write_model(path, model):
    Path(path).mkdir(parents=True,exist_ok=True)
    (Path(path)/"model.json").write_text(json.dumps(model,indent=2))

def load_model(path):
    value=json.loads((Path(path)/"model.json").read_text())
    if value.get("schema_version")!=1:
        raise ValueError("Unsupported model schema")
    return value
