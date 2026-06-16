"""
quality.py
Quality sub-block, 30% weight in the fundamental score.

Normalizes five quality metrics against the S&P 500 universe, combines their
z-scores into a composite raw score and normalizes that composite again against
the equivalent universe distribution.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.fundamental.engine.ratio_profiles import SP500_VALIDATED, RatioProfile
from app.services.fundamental.engine.sanitizer import sanitize_quality

logger = logging.getLogger(__name__)

# Weights/columns of the sp500_validated profile (the active profile is injected
# via `profile` in compute_quality_score).
_WEIGHTS: dict[str, float] = {
    "gp_a": 0.30,
    "roa": 0.25,
    "roe": 0.20,
    "operating_margin": 0.15,
    "fcf_ni": 0.10,
}

_UNIVERSE_COL: dict[str, str] = {
    "gp_a": "gp_a",
    "roa": "returnOnAssets",
    "roe": "returnOnEquity",
    "operating_margin": "operatingMargins",
    "fcf_ni": "fcf_ni",
}

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
_UNIVERSE_RAW_CACHE: dict[tuple, pd.Series] = {}


def _universe_key(universe_df: pd.DataFrame) -> object:
    return universe_df.attrs.get("universe_name", id(universe_df))


def _winsorize(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return series
    return series.clip(
        lower=float(clean.quantile(WINSOR_LOWER)),
        upper=float(clean.quantile(WINSOR_UPPER)),
    )


def _z_score(value: float, distribution: pd.Series) -> float | None:
    clean = distribution.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None
    std = float(clean.std(ddof=0))
    if std == 0.0:
        return None
    return float((value - float(clean.mean())) / std)


def _z_to_score(z: float, k: float = 1.2) -> float:
    x = float(np.clip(k * z, -60.0, 60.0))
    return float(10.0 / (1.0 + np.exp(-x)))


def _signal(score: float) -> str:
    if score > 6.5:
        return "alta calidad"
    if score < 4.0:
        return "baja calidad"
    return "calidad media"


def _normalize_variable(
    value: float | None,
    universe_df: pd.DataFrame,
    universe_col: str,
) -> float | None:
    if value is None:
        return None
    if universe_col not in universe_df.columns:
        logger.warning("Column '%s' not found in universe_df.", universe_col)
        return None
    wins = _winsorize(universe_df[universe_col])
    clean = wins.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    value_clipped = float(np.clip(value, float(clean.min()), float(clean.max())))
    return _z_score(value_clipped, wins)


def _combine_z_scores(
    z_values: dict[str, float | None],
    weights: dict[str, float],
) -> float | None:
    total_weight = 0.0
    weighted_sum = 0.0
    for var, z_val in z_values.items():
        if z_val is not None:
            weighted_sum += weights[var] * z_val
            total_weight += weights[var]
    return weighted_sum / total_weight if total_weight > 0.0 else None


def _quality_raw_from_values(
    values: dict[str, float | None],
    universe_df: pd.DataFrame,
    weights: dict[str, float],
    columns: dict[str, str],
    fcf_ni_zeroed: bool = False,
    neutral_flags: set[str] | None = None,
) -> tuple[float | None, dict[str, float | None]]:
    """
    Compute the composite quality raw score by iterating over the profile variables.

    For sp500_validated the variables/columns/order match the previous version,
    so the result is identical. `fcf_ni` gets z=0 when fcf_ni_zeroed (NI<0 or an
    extreme ratio).
    """
    z_values: dict[str, float | None] = {}
    z_detail: dict[str, float | None] = {}
    neutral_flags = neutral_flags or set()
    for var in weights:
        if var == "roe" and "roe_no_data" in neutral_flags:
            z = 0.0
        elif var == "fcf_ni" and fcf_ni_zeroed:
            z = 0.0
        else:
            z = _normalize_variable(values.get(var), universe_df, columns[var])
        z_values[var] = z
        z_detail[f"z_{var}"] = z
    return _combine_z_scores(z_values, weights), z_detail


def _universe_raw_scores(
    universe_df: pd.DataFrame,
    weights: dict[str, float],
    columns: dict[str, str],
) -> pd.Series:
    cache_key = (_universe_key(universe_df), tuple(weights.items()))
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    if universe_df.empty:
        return pd.Series(dtype=float)

    def row_score(row: pd.Series) -> float | None:
        values = {var: row.get(col) for var, col in columns.items()}
        raw, _ = _quality_raw_from_values(
            values,
            universe_df,
            weights,
            columns,
            fcf_ni_zeroed=False,
        )
        return raw

    raw_scores = universe_df.apply(row_score, axis=1)
    raw_scores = raw_scores.replace([np.inf, -np.inf], np.nan).dropna()
    _UNIVERSE_RAW_CACHE[cache_key] = raw_scores
    return raw_scores


def _normalize_composite(raw_score: float | None, universe_raw: pd.Series) -> float | None:
    if raw_score is None or universe_raw.empty:
        return None
    wins = _winsorize(universe_raw)
    clean = wins.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    # raw_score is not clipped to the winsorized range: doing so collapses all
    # outliers to the same P1/P99 value, producing artificial score clusters.
    # The z-score of an out-of-range value will simply be more extreme, which is
    # the correct behavior (a genuinely atypical company).
    return _z_score(raw_score, wins)


def compute_quality_score(
    company_data: dict[str, Any],
    universe_df: pd.DataFrame,
    profile: RatioProfile = SP500_VALIDATED,
) -> dict[str, Any]:
    """Compute a company's quality score (0-10)."""
    result: dict[str, Any] = {
        "sub_bloque": "calidad",
        "ticker": company_data.get("ticker"),
        "sector": company_data.get("sector"),
        "data_flags": (
            list(company_data.get("data_flags") or []) if profile.name != "sp500_validated" else []
        ),
    }

    ratios = sanitize_quality(company_data)
    fcf_ni_zeroed = ratios.fcf_ni_zeroed

    weights = profile.quality_weights
    columns = profile.quality_columns

    # Company values per profile variable. The 5 standard ones come from the
    # sanitizer; the new ones (e.g. roce) are read from company_data.
    base_values = {
        "gp_a": ratios.gp_a,
        "roa": ratios.roa,
        "roe": ratios.roe,
        "operating_margin": ratios.operating_margin,
        "fcf_ni": ratios.fcf_ni,
    }
    values = {
        var: (base_values[var] if var in base_values else company_data.get(var)) for var in weights
    }

    result.update({k: v for k, v in base_values.items()})
    if "roce" in weights:
        result["roce"] = values.get("roce")
    result["fcf_ni_zeroed"] = fcf_ni_zeroed
    result["fcf_ni_extreme"] = ratios.fcf_ni_extreme

    raw_score, z_detail = _quality_raw_from_values(
        values,
        universe_df,
        weights,
        columns,
        fcf_ni_zeroed=fcf_ni_zeroed,
        neutral_flags=(
            set(company_data.get("data_flags") or [])
            if profile.name != "sp500_validated"
            else set()
        ),
    )
    result.update(z_detail)
    result["z_combined"] = raw_score
    result["quality_raw_score"] = raw_score

    quality_z_score = _normalize_composite(
        raw_score, _universe_raw_scores(universe_df, weights, columns)
    )
    result["quality_z_score"] = quality_z_score

    if quality_z_score is None:
        score: float | None = None
        signal = "sin_datos"
    else:
        score = _z_to_score(quality_z_score)
        signal = _signal(score)

    result["score"] = score
    result["signal"] = signal
    result["quality_summary"] = _build_summary(result)
    return result


def _build_summary(r: dict[str, Any]) -> str:
    ticker = r.get("ticker", "?")
    score = r.get("score")
    signal = r.get("signal", "?")
    score_str = f"{score:.2f}/10" if score is not None else "N/A"

    def fmt(v: float | None, pct: bool = False) -> str:
        if v is None:
            return "N/A"
        return f"{v * 100:.1f}%" if pct else f"{v:.4f}"

    def z_fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    fcf_ni_note = (
        " [neutralizado: |FCF/NI|>10x]"
        if r.get("fcf_ni_extreme")
        else " [neutralizado NI<0]"
        if r.get("fcf_ni_zeroed")
        else f" (z={z_fmt(r.get('z_fcf_ni'))}, x0.10)"
    )
    components = (
        f"GP/A={fmt(r.get('gp_a'))} (z={z_fmt(r.get('z_gp_a'))}, x0.30), "
        f"ROA={fmt(r.get('roa'), pct=True)} (z={z_fmt(r.get('z_roa'))}, x0.25), "
        f"ROE={fmt(r.get('roe'), pct=True)} (z={z_fmt(r.get('z_roe'))}, x0.20), "
        f"Mg.Op={fmt(r.get('operating_margin'), pct=True)} "
        f"(z={z_fmt(r.get('z_operating_margin'))}, x0.15), "
        f"FCF/NI={fmt(r.get('fcf_ni'))}{fcf_ni_note}"
    )

    label = {
        "alta calidad": "presenta metricas de alta calidad",
        "calidad media": "muestra una calidad de negocio media",
        "baja calidad": "registra indicadores de baja calidad",
        "sin_datos": "calidad no evaluable por datos insuficientes",
    }.get(signal, signal)

    return f"{ticker} {label} - score de calidad {score_str} ({signal}). Componentes: {components}."
