import copy
import json
import numpy as np
import pytest
from cfb_forecast.modeling.model import ScoreHead, load_model


def head_value():
    return {"variant": "score", "alpha": 10.0, "cols": [0, 5, 10, 20, 22, 23],
            "mean": [0.0] * 6, "scale": [1.0] * 6,
            "coef": [1.0, -1.0, 2.0, -0.1, 1.5, 0.3], "intercept": 27.0}


def model_value():
    return {"schema_version": 1, "variant": "score",
            "metric_names": ["points", "yards_per_play", "yards_per_pass",
                             "yards_per_rush", "first_downs_created_rate"],
            "center": [27.0, 6.0, 7.0, 5.0, 0.25], "scale": [14.0, 1.5, 2.6, 1.6, 0.05],
            "head": head_value(), "score_covariance": [[144.0, 3.5], [3.5, 125.0]],
            "training_through_season": 2024}


def load_value(tmp_path, value):
    (tmp_path / "model.json").write_text(json.dumps(value))
    return load_model(tmp_path)


@pytest.mark.parametrize("variant", ["score", "pbp"])
def test_fitted_head_roundtrip_preserves_predictions(variant):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 2, 26))
    outcomes = rng.normal(27, 7, size=(80, 2, 5))
    head = ScoreHead(variant).fit(features, outcomes, np.ones(80, dtype=bool))
    restored = ScoreHead.from_dict(head.to_dict())
    np.testing.assert_allclose(restored.predict(features), head.predict(features), rtol=0, atol=0)


@pytest.mark.parametrize("field,bad", [
    ("mean", [0.0] * 5), ("coef", [1.0] * 7), ("scale", [[1.0] * 6]),
    ("scale", [0.0] + [1.0] * 5), ("scale", [-1.0] + [1.0] * 5),
    ("mean", [float("nan")] + [0.0] * 5),
    ("coef", [float("inf")] + [1.0] * 5),
    ("intercept", float("nan")), ("intercept", [27.0]),
    ("alpha", 0.0), ("alpha", -1.0), ("alpha", float("inf")),
])
def test_invalid_head_rejected(field, bad):
    value = head_value(); value[field] = bad
    with pytest.raises(ValueError, match=field):
        ScoreHead.from_dict(value)


def test_missing_head_field_rejected():
    value = head_value(); del value["coef"]
    with pytest.raises(ValueError, match="missing fields"):
        ScoreHead.from_dict(value)


@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_nonobject_head_rejected(value):
    with pytest.raises(ValueError, match="object"):
        ScoreHead.from_dict(value)


@pytest.mark.parametrize("cov", [
    [[1, 2], [2, 1]], [[1, 2], [0, 1]], [[1, 0], [0, 0]],
    [[1, 0]], [[float("nan"), 0], [0, 1]],
])
def test_invalid_covariance_rejected_at_load(tmp_path, cov):
    value = model_value(); value["score_covariance"] = cov
    with pytest.raises(ValueError, match="score_covariance"):
        load_value(tmp_path, value)


@pytest.mark.parametrize("field,bad", [
    ("center", [0.0]), ("scale", [0.0] * 5),
    ("scale", [float("inf")] * 5), ("metric_names", []),
])
def test_invalid_normalization_rejected(tmp_path, field, bad):
    value = model_value(); value[field] = bad
    with pytest.raises(ValueError, match=field):
        load_value(tmp_path, value)


def test_variant_mismatch_rejected(tmp_path):
    value = model_value(); value["variant"] = "pbp"
    with pytest.raises(ValueError, match="variant"):
        load_value(tmp_path, value)


def test_missing_artifact_field_rejected(tmp_path):
    value = model_value(); del value["head"]
    with pytest.raises(ValueError, match="missing fields"):
        load_value(tmp_path, value)


@pytest.mark.parametrize("value", [[], {"schema_version": 2}])
def test_unsupported_schema_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="schema"):
        load_value(tmp_path, value)


def test_valid_heldout_artifact_unchanged_and_roundoff_allowed(tmp_path):
    value = model_value(); value["score_covariance"][1][0] += 1e-14
    original = copy.deepcopy(value)
    assert load_value(tmp_path, value) == original
