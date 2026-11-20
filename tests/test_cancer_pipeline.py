"""Tests for binnedInc parsing, feature prep, and county risk prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features import (
    DEMO_FEATURES,
    TARGET_COL,
    parse_binned_inc,
    prepare_frame,
)
from src.predict import load_model, predict
from src.train import METRICS_PATH, MODEL_PATH, train

ROOT = Path(__file__).resolve().parents[1]
CANCER = ROOT / "dataset" / "cancer_reg.csv"
HOUSEHOLD = ROOT / "dataset" / "avg-household-size.csv"


@pytest.fixture(scope="module")
def trained_metrics():
    assert CANCER.is_file() and HOUSEHOLD.is_file()
    metrics = train()
    assert MODEL_PATH.is_file()
    assert METRICS_PATH.is_file()
    assert metrics["best"]["r2"] >= 0.35
    return metrics


@pytest.fixture(scope="module")
def artifact(trained_metrics):
    return load_model(MODEL_PATH)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("(61494.5, 125635]", (61494.5 + 125635) / 2),
        ("[42724.4, 45201]", (42724.4 + 45201) / 2),
        (" (48021.6, 51046.4] ", (48021.6 + 51046.4) / 2),
        ("not-an-interval", float("nan")),
        (None, float("nan")),
    ],
)
def test_parse_binned_inc(raw, expected):
    got = parse_binned_inc(raw)
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_prepare_frame_transform():
    X, y, features = prepare_frame()
    assert TARGET_COL not in X.columns
    assert "binnedinc_mid" in features
    assert len(X) >= 2500
    assert len(features) >= 10
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0
    assert set(DEMO_FEATURES).issubset(set(features))


def test_metrics_include_six_models(trained_metrics):
    names = {m["name"] for m in trained_metrics["models"]}
    assert names == {
        "Linear",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "RandomForest",
        "GradientBoosting",
    }
    for row in trained_metrics["models"]:
        assert row["cv_rmse"] > 0
        assert row["test_rmse"] > 0
        assert "mae" in row and "r2" in row
    assert trained_metrics["best_model"] in names


def test_model_file_loads(artifact):
    assert "model" in artifact
    assert hasattr(artifact["model"], "predict")
    assert artifact.get("national_median_deathrate", 0) > 0
    assert len(artifact.get("presets") or []) == 3


def test_prediction_output_shape(artifact):
    demo = artifact["demo_features"]
    defaults = artifact["feature_defaults"]
    features = {c: float(defaults[c]) for c in demo}
    result = predict(features, artifact=artifact)
    assert result["predicted_deathrate"] > 0
    assert "national_median_deathrate" in result
    assert "percentile_vs_us_counties" in result
    assert 0 <= result["percentile_vs_us_counties"] <= 100
    assert result["risk_band"]
    assert len(result["key_risk_drivers"]) >= 1


def test_higher_poverty_raises_risk(artifact):
    defaults = artifact["feature_defaults"]
    demo = artifact["demo_features"]
    base = {c: float(defaults[c]) for c in demo}
    rich = dict(base)
    poor = dict(base)
    rich["povertypercent"] = 5.0
    rich["medincome"] = 90000.0
    poor["povertypercent"] = 35.0
    poor["medincome"] = 25000.0
    low = predict(rich, artifact=artifact)
    high = predict(poor, artifact=artifact)
    assert high["predicted_deathrate"] > low["predicted_deathrate"]
