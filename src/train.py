"""Train and benchmark six regressors; export pipeline, metrics, and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (
    DEMO_FEATURES,
    TARGET_COL,
    feature_defaults,
    prepare_frame,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
MODEL_PATH = ARTIFACTS / "cancer_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
COMPARISON_PATH = ASSETS / "model_comparison.png"
IMPORTANCE_PATH = ASSETS / "feature_importance.png"

RANDOM_STATE = 42
CV_FOLDS = 5
# zlib compression keeps the RF artifact ~5MB (vs ~17MB uncompressed) for
# faster Space / clone cold-start without changing predictions.
JOBLIB_COMPRESS = 3


def build_models() -> dict[str, Any]:
    """Six-model portfolio benchmark (linear + tree ensembles)."""
    return {
        "Linear": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "Lasso": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.1, max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "ElasticNet": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.1,
                        l1_ratio=0.5,
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test), dtype=float).ravel()
    return {
        "name": name,
        "cv_rmse": float(-cv_scores.mean()),
        "cv_rmse_std": float(cv_scores.std()),
        "test_rmse": _rmse(y_test, preds),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "y_pred": preds.tolist(),
        "model": model,
    }


def _feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    if isinstance(model, Pipeline):
        est = model.named_steps.get("model", model[-1])
        if hasattr(est, "coef_"):
            coef = np.abs(np.asarray(est.coef_, dtype=float).ravel())
            total = float(coef.sum()) or 1.0
            return {f: float(v / total) for f, v in zip(feature_names, coef)}
        model = est

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float).ravel()
        return {f: float(v) for f, v in zip(feature_names, imp)}
    return {f: 0.0 for f in feature_names}


def _save_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    names = [r["name"] for r in rows]
    test_rmse = [r["test_rmse"] for r in rows]
    cv_rmse = [r["cv_rmse"] for r in rows]
    r2 = [r["r2"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width / 2, cv_rmse, width, label="CV RMSE", color="#1b4965")
    axes[0].bar(x + width / 2, test_rmse, width, label="Test RMSE", color="#5fa8d3")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("RMSE (deaths / 100k)")
    axes[0].set_title("Model RMSE comparison")
    axes[0].legend()

    axes[1].bar(x, r2, color="#2a6f97")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("R²")
    axes[1].set_title("Held-out R²")
    axes[1].set_ylim(0, max(0.05, max(r2) * 1.15))

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_importance(importances: dict[str, float], path: Path) -> None:
    names = list(importances.keys())
    values = [importances[n] for n in names]
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(x=values, y=names, ax=ax, color="#2a6f97", orient="h")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature importance (best model)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(
    cancer_path: str | Path | None = None,
    household_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cancer_path is not None:
        kwargs["cancer_path"] = cancer_path
    if household_path is not None:
        kwargs["household_path"] = household_path

    X, y, feature_names = prepare_frame(**kwargs)
    y_arr = y.to_numpy(dtype=float)
    national_median = float(np.median(y_arr))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=test_size, random_state=random_state
    )

    results: list[dict[str, Any]] = []
    for name, model in build_models().items():
        results.append(_evaluate(name, model, X_train, y_train, X_test, y_test))

    # Prefer higher R², then lower test RMSE.
    best = max(results, key=lambda r: (r["r2"], -r["test_rmse"], -r["mae"]))
    best_model = best["model"]
    importances = _feature_importance(best_model, feature_names)
    defaults = feature_defaults(X_train)

    # Three Gradio presets from the held-out split, ranked by *predicted* risk
    # so the explorer demos show clear Below / Near / Above median behavior.
    test_preds = np.asarray(best_model.predict(X_test), dtype=float).ravel()
    order = np.argsort(test_preds)
    pick_pos = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    presets: list[dict[str, Any]] = []
    labels = ["Lower-risk county", "Median-risk county", "Higher-risk county"]
    for label, pos in zip(labels, pick_pos):
        row = X_test.iloc[pos]
        presets.append(
            {
                "label": label,
                "actual_deathrate": float(y_test[pos]),
                "predicted_deathrate": float(test_preds[pos]),
                "values": {c: float(row[c]) for c in feature_names},
            }
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": best_model,
        "model_name": best["name"],
        "features": feature_names,
        "demo_features": [c for c in DEMO_FEATURES if c in feature_names],
        "target": TARGET_COL,
        "feature_importance": importances,
        "feature_defaults": defaults,
        "national_median_deathrate": national_median,
        "presets": presets,
        # Full death-rate distribution for percentile comparisons in predict().
        "y_full": y_arr.tolist(),
    }
    joblib.dump(artifact, MODEL_PATH, compress=JOBLIB_COMPRESS)

    summary_rows = [
        {
            "name": r["name"],
            "cv_rmse": r["cv_rmse"],
            "cv_rmse_std": r["cv_rmse_std"],
            "test_rmse": r["test_rmse"],
            "mae": r["mae"],
            "r2": r["r2"],
        }
        for r in results
    ]
    _save_comparison(summary_rows, COMPARISON_PATH)
    _save_importance(importances, IMPORTANCE_PATH)

    metrics: dict[str, Any] = {
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_names,
        "demo_features": artifact["demo_features"],
        "test_size": test_size,
        "random_state": random_state,
        "cv_folds": CV_FOLDS,
        "national_median_deathrate": national_median,
        "best_model": best["name"],
        "best": {
            "name": best["name"],
            "cv_rmse": best["cv_rmse"],
            "test_rmse": best["test_rmse"],
            "mae": best["mae"],
            "r2": best["r2"],
        },
        "models": summary_rows,
        "feature_importance": importances,
        "presets": [
            {
                "label": p["label"],
                "actual_deathrate": p["actual_deathrate"],
                "demo_values": {c: p["values"][c] for c in artifact["demo_features"]},
            }
            for p in presets
        ],
        "artifacts": {
            "model": str(MODEL_PATH.relative_to(ROOT)),
            "model_comparison": str(COMPARISON_PATH.relative_to(ROOT)),
            "feature_importance": str(IMPORTANCE_PATH.relative_to(ROOT)),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train()
    best = metrics["best"]
    print(
        f"Best={best['name']} | "
        f"R²={best['r2']:.4f} MAE={best['mae']:.4f} "
        f"Test RMSE={best['test_rmse']:.4f} CV RMSE={best['cv_rmse']:.4f}"
    )
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")
    for m in metrics["models"]:
        print(
            f"  {m['name']}: R²={m['r2']:.4f} "
            f"MAE={m['mae']:.4f} Test RMSE={m['test_rmse']:.4f} "
            f"CV RMSE={m['cv_rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
