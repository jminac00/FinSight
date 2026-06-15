"""Quality sub-block, 30% weight in the fundamental score.

Normalizes five quality metrics against the S&P 500 universe, combines their z-scores into a
composite raw score, and normalizes that composite again against the equivalent universe
distribution.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.fundamental.engine.sanitizer import sanitize_quality

logger = logging.getLogger(__name__)

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
_UNIVERSE_RAW_CACHE: dict[int, pd.Series] = {}


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


def _combine_z_scores(z_values: dict[str, float | None]) -> float | None:
    total_weight = 0.0
    weighted_sum = 0.0
    for var, z_val in z_values.items():
        if z_val is not None:
            weighted_sum += _WEIGHTS[var] * z_val
            total_weight += _WEIGHTS[var]
    return weighted_sum / total_weight if total_weight > 0.0 else None


def _quality_raw_from_values(
    gp_a: float | None,
    roa: float | None,
    roe: float | None,
    operating_margin: float | None,
    fcf_ni: float | None,
    universe_df: pd.DataFrame,
    fcf_ni_zeroed: bool = False,
) -> tuple[float | None, dict[str, float | None]]:
    z_gp_a = _normalize_variable(gp_a, universe_df, _UNIVERSE_COL["gp_a"])
    z_roa = _normalize_variable(roa, universe_df, _UNIVERSE_COL["roa"])
    z_roe = _normalize_variable(roe, universe_df, _UNIVERSE_COL["roe"])
    z_om = _normalize_variable(operating_margin, universe_df, _UNIVERSE_COL["operating_margin"])
    z_fcf_ni = (
        0.0 if fcf_ni_zeroed else _normalize_variable(fcf_ni, universe_df, _UNIVERSE_COL["fcf_ni"])
    )
    z_values = {
        "gp_a": z_gp_a,
        "roa": z_roa,
        "roe": z_roe,
        "operating_margin": z_om,
        "fcf_ni": z_fcf_ni,
    }
    return _combine_z_scores(z_values), {
        "z_gp_a": z_gp_a,
        "z_roa": z_roa,
        "z_roe": z_roe,
        "z_operating_margin": z_om,
        "z_fcf_ni": z_fcf_ni,
    }


def _universe_raw_scores(universe_df: pd.DataFrame) -> pd.Series:
    cache_key = id(universe_df)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    if universe_df.empty:
        return pd.Series(dtype=float)

    def row_score(row: pd.Series) -> float | None:
        raw, _ = _quality_raw_from_values(
            row.get("gp_a"),
            row.get("returnOnAssets"),
            row.get("returnOnEquity"),
            row.get("operatingMargins"),
            row.get("fcf_ni"),
            universe_df,
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
    # raw_score is NOT clipped to the winsorized range: doing so collapses all outliers to the
    # same P1/P99 value, producing artificial score clusters. The z-score of an out-of-range
    # value will simply be more extreme, which is the correct behavior (genuinely atypical company).
    return _z_score(raw_score, wins)


def compute_quality_score(
    company_data: dict[str, Any],
    universe_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute the quality score (0-10) of a company."""
    result: dict[str, Any] = {
        "sub_bloque": "calidad",
        "ticker": company_data.get("ticker"),
        "sector": company_data.get("sector"),
    }

    ratios = sanitize_quality(company_data)
    gp_a = ratios.gp_a
    roa = ratios.roa
    roe = ratios.roe
    operating_margin = ratios.operating_margin
    fcf_ni = ratios.fcf_ni
    fcf_ni_zeroed = ratios.fcf_ni_zeroed

    result.update(
        {
            "gp_a": gp_a,
            "roa": roa,
            "roe": roe,
            "operating_margin": operating_margin,
            "fcf_ni": fcf_ni,
        }
    )
    result["fcf_ni_zeroed"] = fcf_ni_zeroed
    result["fcf_ni_extreme"] = ratios.fcf_ni_extreme

    raw_score, z_detail = _quality_raw_from_values(
        gp_a,
        roa,
        roe,
        operating_margin,
        fcf_ni,
        universe_df,
        fcf_ni_zeroed=fcf_ni_zeroed,
    )
    result.update(z_detail)
    result["z_combined"] = raw_score
    result["quality_raw_score"] = raw_score

    quality_z_score = _normalize_composite(raw_score, _universe_raw_scores(universe_df))
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
