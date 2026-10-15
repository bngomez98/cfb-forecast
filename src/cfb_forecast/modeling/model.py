"""Regularized score heads; calibration covariance estimated on prior validation."""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from ..ingest.features import feature_columns


def _finite_array(value, shape, name, positive=False):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and contain only finite values")
    if positive and (array <= 0).any():
        raise ValueError(f"{name} must contain only positive values")
    return array


def _finite_scalar(value, name, positive=False):
    array = _finite_array(value, (), name)
    number = float(array)
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


class ScoreHead:
    def __init__(self, variant="score", alpha=10.):
        self.variant=variant; self.alpha=float(alpha); self.cols=feature_columns(variant)
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
        if not isinstance(value, dict):
            raise ValueError("Saved score head must be an object")
        required = {"variant", "alpha", "cols", "mean", "scale", "coef", "intercept"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Saved score head missing fields: {sorted(missing)}")
        alpha = _finite_scalar(value["alpha"], "head.alpha", positive=True)
        head = cls(value["variant"], alpha)
        if head.cols != value["cols"]:
            raise ValueError("Feature schema incompatible with saved model")
        shape = (len(head.cols),)
        mean = _finite_array(value["mean"], shape, "head.mean")
        scale = _finite_array(value["scale"], shape, "head.scale", positive=True)
        coef = _finite_array(value["coef"], shape, "head.coef")
        intercept = _finite_scalar(value["intercept"], "head.intercept")
        head.scaler.mean_ = mean.copy(); head.scaler.scale_ = scale.copy()
        head.scaler.var_ = scale**2; head.scaler.n_features_in_ = len(head.cols)
        head.regressor.coef_ = coef.copy(); head.regressor.intercept_ = intercept
        head.regressor.n_features_in_ = len(head.cols)
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
    if not isinstance(value, dict) or value.get("schema_version")!=1:
        raise ValueError("Unsupported model schema")
    required = {"variant", "metric_names", "center", "scale", "head", "score_covariance"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Saved model missing fields: {sorted(missing)}")
    if not isinstance(value["metric_names"], list) or not value["metric_names"]:
        raise ValueError("metric_names must be a nonempty list")
    shape = (len(value["metric_names"]),)
    _finite_array(value["center"], shape, "center")
    _finite_array(value["scale"], shape, "scale", positive=True)
    head = ScoreHead.from_dict(value["head"])
    if head.variant != value["variant"]:
        raise ValueError("Model variant does not match saved score head")
    cov = _finite_array(value["score_covariance"], (2, 2), "score_covariance")
    if not np.allclose(cov, cov.T, rtol=0, atol=1e-10):
        raise ValueError("score_covariance must be symmetric")
    if np.linalg.eigvalsh((cov + cov.T) / 2).min() <= 0:
        raise ValueError("score_covariance must be positive definite")
    return value
