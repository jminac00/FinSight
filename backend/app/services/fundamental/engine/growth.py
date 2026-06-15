"""Growth sub-block, 20% weight in the fundamental score.

Variables and weights:
    Revenue CAGR 3y           30%  direct
    FCF CAGR 3y               25%  direct; neutral if not computable
    Gross margin delta        20%  direct
    Asset Growth              15%  inverted at the z-score level
    Share Dilution            10%  inverted at the z-score level

Methodology:
    1. Normalize each variable against its natural universe distribution.
    2. Negate the z-score of asset_growth and share_dilution, not the raw value.
    3. Combine z-scores into growth_raw_score.
    4. Normalize the composite again against the universe raw scores.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.fundamental.engine.sanitizer import sanitize_growth

logger = logging.getLogger(__name__)

_WEIGHTS: dict[str, float] = {
    "revenue_trend_ols": 0.30,
    "fcf_trend_ols": 0.25,
    "delta_gross_margin": 0.20,
    "asset_growth_inv": 0.15,
    "share_dilution_inv": 0.10,
}

_UNIVERSE_COL: dict[str, str | None] = {
    "revenue_trend_ols": "revenue_trend_ols",
    "fcf_trend_ols": "fcf_trend_ols",
    "delta_gross_margin": "delta_gross_margin",
    "asset_growth": "asset_growth",
    "share_dilution": "share_dilution",
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
        return "crecimiento solido"
    if score < 4.0:
        return "debil o ineficiente"
    return "moderado"


def _normalize_variable(
    value: float | None,
    universe_df: pd.DataFrame,
    universe_col: str | None,
) -> float | None:
    """Winsorization + clip + z-score of a variable against its natural distribution."""
    if value is None or universe_col is None:
        return None
    if universe_col not in universe_df.columns:
        logger.debug("Column '%s' not found in universe_df.", universe_col)
        return None
    wins = _winsorize(universe_df[universe_col])
    clean = wins.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    value_clipped = float(np.clip(value, float(clean.min()), float(clean.max())))
    return _z_score(value_clipped, wins)


def _combine_z_scores(z_map: dict[str, float | None]) -> float | None:
    total_weight = 0.0
    weighted_sum = 0.0
    for var, z_val in z_map.items():
        if z_val is not None:
            weighted_sum += _WEIGHTS[var] * z_val
            total_weight += _WEIGHTS[var]
    return weighted_sum / total_weight if total_weight > 0.0 else None


def _growth_raw_from_values(
    rev_cagr: float | None,
    fcf_cagr: float | None,
    delta_gm: float | None,
    asset_gr: float | None,
    dilution: float | None,
    universe_df: pd.DataFrame,
    fcf_zeroed: bool = False,
) -> tuple[float | None, dict[str, float | None]]:
    # OLS trend columns; fallback to CAGR columns if universe lacks OLS
    rev_col = (
        "revenue_trend_ols" if "revenue_trend_ols" in universe_df.columns else "revenue_cagr_3y"
    )
    fcf_col = "fcf_trend_ols" if "fcf_trend_ols" in universe_df.columns else "fcf_cagr_3y"

    z_rev = _normalize_variable(rev_cagr, universe_df, _UNIVERSE_COL.get(rev_col, rev_col))
    z_dgm = _normalize_variable(delta_gm, universe_df, _UNIVERSE_COL["delta_gross_margin"])

    z_asset_growth = _normalize_variable(asset_gr, universe_df, _UNIVERSE_COL["asset_growth"])
    z_asset_growth_inv = -z_asset_growth if z_asset_growth is not None else None

    z_share_dilution = _normalize_variable(dilution, universe_df, _UNIVERSE_COL["share_dilution"])
    z_share_dilution_inv = -z_share_dilution if z_share_dilution is not None else None

    z_fcf = (
        0.0
        if fcf_zeroed
        else _normalize_variable(fcf_cagr, universe_df, _UNIVERSE_COL.get(fcf_col, fcf_col))
    )

    z_map = {
        "revenue_trend_ols": z_rev,
        "fcf_trend_ols": z_fcf,
        "delta_gross_margin": z_dgm,
        "asset_growth_inv": z_asset_growth_inv,
        "share_dilution_inv": z_share_dilution_inv,
    }
    details = {
        "z_revenue_trend": z_rev,
        "z_fcf_trend": z_fcf,
        "z_delta_gross_margin": z_dgm,
        "z_asset_growth": z_asset_growth,
        "z_asset_growth_inv": z_asset_growth_inv,
        "z_share_dilution": z_share_dilution,
        "z_share_dilution_inv": z_share_dilution_inv,
    }
    return _combine_z_scores(z_map), details


def _universe_raw_scores(universe_df: pd.DataFrame) -> pd.Series:
    cache_key = id(universe_df)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    if universe_df.empty:
        return pd.Series(dtype=float)

    use_ols = "revenue_trend_ols" in universe_df.columns

    def row_score(row: pd.Series) -> float | None:
        rev_val = row.get("revenue_trend_ols") if use_ols else row.get("revenue_cagr_3y")
        fcf_val = row.get("fcf_trend_ols") if use_ols else row.get("fcf_cagr_3y")
        raw, _ = _growth_raw_from_values(
            rev_val,
            fcf_val,
            row.get("delta_gross_margin"),
            row.get("asset_growth"),
            row.get("share_dilution"),
            universe_df,
            fcf_zeroed=False,
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
    return _z_score(raw_score, wins)


def compute_growth_score(
    company_data: dict[str, Any],
    universe_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute the growth score (0-10) without breaking the public API."""
    result: dict[str, Any] = {
        "sub_bloque": "crecimiento",
        "ticker": company_data.get("ticker"),
        "sector": company_data.get("sector"),
    }

    ratios = sanitize_growth(company_data)
    rev_cagr = ratios.revenue_cagr_3y
    fcf_cagr = ratios.fcf_cagr_3y
    delta_gm = ratios.delta_gross_margin
    asset_gr = ratios.asset_growth
    dilution = ratios.share_dilution
    fcf_cagr_zeroed = ratios.fcf_cagr_zeroed

    result.update(
        {
            "revenue_trend_ols": rev_cagr,
            "fcf_trend_ols": fcf_cagr,
            "delta_gross_margin": delta_gm,
            "asset_growth": asset_gr,
            "share_dilution": dilution,
        }
    )
    result["fcf_trend_zeroed"] = fcf_cagr_zeroed
    result["fcf_cagr_base_negative"] = ratios.fcf_cagr_base_negative
    result["revenue_cagr_outlier"] = ratios.revenue_cagr_outlier

    raw_score, z_details = _growth_raw_from_values(
        rev_cagr,
        fcf_cagr,
        delta_gm,
        asset_gr,
        dilution,
        universe_df,
        fcf_zeroed=fcf_cagr_zeroed,
    )
    result.update(z_details)
    result["z_combined"] = raw_score
    result["growth_raw_score"] = raw_score

    growth_z_score = _normalize_composite(raw_score, _universe_raw_scores(universe_df))
    result["growth_z_score"] = growth_z_score

    if growth_z_score is None:
        score: float | None = None
        signal = "sin_datos"
    else:
        score = _z_to_score(growth_z_score)
        signal = _signal(score)

    result["score"] = score
    result["signal"] = signal
    result["growth_summary"] = _build_summary(result)
    return result


def _build_summary(r: dict[str, Any]) -> str:
    ticker = r.get("ticker", "?")
    score = r.get("score")
    signal = r.get("signal", "?")
    score_str = f"{score:.2f}/10" if score is not None else "N/A"

    def pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    def z_fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    def trend_fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    if r.get("fcf_cagr_base_negative"):
        fcf_note = " [neutralizado: CAGR base extrema]"
    elif r.get("fcf_trend_zeroed"):
        fcf_note = " [neutro]"
    else:
        fcf_note = f" (z={z_fmt(r.get('z_fcf_trend'))})"
    rev_note = (
        " [neutralizado: CAGR>100%]"
        if r.get("revenue_cagr_outlier")
        else f" (z={z_fmt(r.get('z_revenue_trend'))})"
    )
    components = (
        f"Rev.Trend={trend_fmt(r.get('revenue_trend_ols'))}{rev_note} (x0.30), "
        f"FCF.Trend={trend_fmt(r.get('fcf_trend_ols'))}{fcf_note} (x0.25), "
        f"dMg.Bruto={pct(r.get('delta_gross_margin'))} "
        f"(z={z_fmt(r.get('z_delta_gross_margin'))}, x0.20), "
        f"AssetGr={pct(r.get('asset_growth'))} inv_z={z_fmt(r.get('z_asset_growth_inv'))} (x0.15), "
        f"Dilucion={pct(r.get('share_dilution'))} "
        f"inv_z={z_fmt(r.get('z_share_dilution_inv'))} (x0.10)"
    )

    label = {
        "crecimiento solido": "muestra un crecimiento de alta calidad",
        "moderado": "presenta un crecimiento moderado",
        "debil o ineficiente": "registra un crecimiento debil o ineficiente",
        "sin_datos": "crecimiento no evaluable por datos insuficientes",
    }.get(signal, signal)

    return (
        f"{ticker} {label} - score de crecimiento {score_str} ({signal}). "
        f"Componentes: {components}."
    )
