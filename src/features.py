"""Feature engineering for US county cancer death-rate regression."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANCER = ROOT / "dataset" / "cancer_reg.csv"
DEFAULT_HOUSEHOLD = ROOT / "dataset" / "avg-household-size.csv"

TARGET_COL = "target_deathrate"

# Dropped as near-leakage / not socioeconomic inputs for the portfolio demo.
LEAKAGE_COLS = ("avganncount", "avgdeathsperyear")

# Socioeconomic / health indicators kept after correlation screening.
FEATURE_CANDIDATES: tuple[str, ...] = (
    "incidencerate",
    "medincome",
    "povertypercent",
    "pctbachdeg25_over",
    "pctpubliccoveragealone",
    "pctemployed16_over",
    "pcths25_over",
    "pctpubliccoverage",
    "pctprivatecoverage",
    "pctunemployed16_over",
    "pctprivatecoveragealone",
    "medianage",
    "percentmarried",
    "pctmarriedhouseholds",
    "birthrate",
    "pctbachdeg18_24",
    "pctempprivcoverage",
    "avghouseholdsize",
    "binnedinc_mid",
)

# Interactive Gradio controls (remaining features filled with train medians).
DEMO_FEATURES: tuple[str, ...] = (
    "incidencerate",
    "medincome",
    "povertypercent",
    "pctprivatecoverage",
    "medianage",
    "pctbachdeg25_over",
    "pctpubliccoveragealone",
    "pctemployed16_over",
    "pctunemployed16_over",
    "pcths25_over",
)

_BINNED_RE = re.compile(
    r"^\s*[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]\s*$"
)


def parse_binned_inc(value: object) -> float:
    """Parse a `binnedinc` interval string into its midpoint.

    Accepts forms like ``(61494.5, 125635]`` or ``[42724.4, 45201]``.
    Returns ``nan`` when the value is missing or unparsable.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    match = _BINNED_RE.match(text)
    if not match:
        return float("nan")
    low = float(match.group(1))
    high = float(match.group(2))
    return (low + high) / 2.0


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def load_merged(
    cancer_path: str | Path = DEFAULT_CANCER,
    household_path: str | Path = DEFAULT_HOUSEHOLD,
) -> pd.DataFrame:
    """Load cancer + household CSVs and left-merge on ``geography``."""
    cancer_path = Path(cancer_path)
    household_path = Path(household_path)
    if not cancer_path.is_file():
        raise FileNotFoundError(f"Missing cancer dataset at {cancer_path}")
    if not household_path.is_file():
        raise FileNotFoundError(f"Missing household dataset at {household_path}")

    cancer = _normalize_columns(pd.read_csv(cancer_path))
    household = _normalize_columns(pd.read_csv(household_path))
    if "geography" not in cancer.columns:
        raise ValueError(f"{cancer_path} missing geography column")
    if "geography" not in household.columns or "avghouseholdsize" not in household.columns:
        raise ValueError(f"{household_path} must include geography and avghouseholdsize")

    merged = cancer.merge(
        household[["geography", "avghouseholdsize"]],
        on="geography",
        how="left",
    )
    return merged


def _drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with missing fraction above ``threshold`` (e.g. pctsomecol18_24)."""
    frac = df.isna().mean()
    drop = [c for c in df.columns if frac[c] > threshold and c != TARGET_COL]
    return df.drop(columns=drop, errors="ignore")


def _clip_outliers(
    df: pd.DataFrame,
    columns: Iterable[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        lo = float(series.quantile(lower_q))
        hi = float(series.quantile(upper_q))
        out[col] = series.clip(lower=lo, upper=hi)
    return out


def prepare_frame(
    df: pd.DataFrame | None = None,
    *,
    cancer_path: str | Path = DEFAULT_CANCER,
    household_path: str | Path = DEFAULT_HOUSEHOLD,
    feature_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build a modeling-ready frame: parse bins, impute, clip, select features.

    Returns ``(X, y, feature_names)``.
    """
    raw = df if df is not None else load_merged(cancer_path, household_path)
    raw = _normalize_columns(raw)
    if TARGET_COL not in raw.columns:
        raise ValueError(f"Missing target column {TARGET_COL}")

    work = raw.copy()
    if "binnedinc" in work.columns:
        work["binnedinc_mid"] = work["binnedinc"].map(parse_binned_inc)
    work = work.drop(columns=list(LEAKAGE_COLS), errors="ignore")
    work = _drop_high_missing(work, threshold=0.5)

    wanted = list(feature_cols) if feature_cols is not None else list(FEATURE_CANDIDATES)
    available = [c for c in wanted if c in work.columns]
    if not available:
        raise ValueError("No candidate feature columns present after cleaning")

    # Median-impute remaining gaps (pctemployed16_over, pctprivatecoveragealone, …).
    for col in available + [TARGET_COL]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    medians = work[available].median(numeric_only=True)
    work[available] = work[available].fillna(medians)

    work = _clip_outliers(work, available)
    work = work.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    X = work[available].astype(float)
    y = work[TARGET_COL].astype(float)
    return X, y, available


def feature_defaults(X: pd.DataFrame) -> dict[str, float]:
    """Training-set medians used to fill non-interactive demo inputs."""
    return {c: float(X[c].median()) for c in X.columns}
