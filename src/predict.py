"""Single-county cancer death-rate inference vs US national median."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import DEMO_FEATURES, TARGET_COL
from src.train import MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model at {path}. Run: python -m src.train")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Unexpected artifact format in {path}")
    return artifact


def _complete_features(
    features: dict[str, float],
    artifact: dict[str, Any],
) -> dict[str, float]:
    columns = list(artifact["features"])
    defaults = dict(artifact.get("feature_defaults") or {})
    row: dict[str, float] = {}
    missing_required: list[str] = []
    for col in columns:
        if col in features and features[col] is not None:
            row[col] = float(features[col])
        elif col in defaults:
            row[col] = float(defaults[col])
        else:
            missing_required.append(col)
    if missing_required:
        raise ValueError(f"Missing features with no defaults: {missing_required}")
    return row


def _percentile_vs_reference(value: float, reference: np.ndarray) -> float:
    """Empirical percentile of ``value`` within ``reference`` (0–100)."""
    ref = np.asarray(reference, dtype=float)
    if ref.size == 0:
        return float("nan")
    return float(100.0 * np.mean(ref <= value))


def predict(
    features: dict[str, float],
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict county ``target_deathrate`` and compare to the US median."""
    art = artifact or load_model()
    completed = _complete_features(features, art)
    columns = list(art["features"])
    frame = pd.DataFrame([completed], columns=columns)
    raw = float(np.asarray(art["model"].predict(frame), dtype=float).ravel()[0])
    prediction = float(max(0.0, raw))

    national_median = float(art.get("national_median_deathrate", float("nan")))
    reference = np.asarray(art.get("y_full") or art.get("y_train") or [], dtype=float)
    percentile = _percentile_vs_reference(prediction, reference)

    importances = dict(art.get("feature_importance") or {})
    demo_cols = list(art.get("demo_features") or DEMO_FEATURES)
    drivers = sorted(
        ((c, importances.get(c, 0.0), completed[c]) for c in demo_cols if c in completed),
        key=lambda t: t[1],
        reverse=True,
    )[:5]

    vs_median = prediction - national_median
    if vs_median > 5:
        risk_band = "Above US median"
    elif vs_median < -5:
        risk_band = "Below US median"
    else:
        risk_band = "Near US median"

    return {
        "predicted_deathrate": round(prediction, 2),
        "predicted_deathrate_raw": raw,
        "national_median_deathrate": round(national_median, 2),
        "vs_national_median": round(vs_median, 2),
        "percentile_vs_us_counties": round(percentile, 1),
        "risk_band": risk_band,
        "model_name": art.get("model_name"),
        "target": art.get("target", TARGET_COL),
        "features_used": completed,
        "key_risk_drivers": [
            {"feature": name, "importance": round(imp, 4), "value": round(val, 3)}
            for name, imp, val in drivers
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict county cancer death rate from socioeconomic indicators."
    )
    parser.add_argument("--incidencerate", type=float, required=True)
    parser.add_argument("--medincome", type=float, required=True)
    parser.add_argument("--povertypercent", type=float, required=True)
    parser.add_argument("--pctprivatecoverage", type=float, required=True)
    parser.add_argument("--medianage", type=float, required=True)
    parser.add_argument("--pctbachdeg25_over", type=float, required=True)
    parser.add_argument("--pctpubliccoveragealone", type=float, default=None)
    parser.add_argument("--pctemployed16_over", type=float, default=None)
    parser.add_argument("--pctunemployed16_over", type=float, default=None)
    parser.add_argument("--pcths25_over", type=float, default=None)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    features = {
        "incidencerate": args.incidencerate,
        "medincome": args.medincome,
        "povertypercent": args.povertypercent,
        "pctprivatecoverage": args.pctprivatecoverage,
        "medianage": args.medianage,
        "pctbachdeg25_over": args.pctbachdeg25_over,
    }
    for key in (
        "pctpubliccoveragealone",
        "pctemployed16_over",
        "pctunemployed16_over",
        "pcths25_over",
    ):
        val = getattr(args, key)
        if val is not None:
            features[key] = val

    result = predict(features, artifact=load_model(args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
