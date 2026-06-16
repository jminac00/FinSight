"""
valuation.py
Valuation sub-block — 30% weight in the fundamental score.

Evaluates whether a company is undervalued or overvalued by comparing 4 return
yields against the S&P 500 universe and against its GICS sector.

Pipeline:
    company yields
    → weighted raw score (yields floored to 0)
    → p1-p99 winsorization over the universe distribution
    → z-score vs S&P 500 universe (30%) + z-score vs GICS sector (70%)
    → 0-10 rescaling: clip(5 + (z/3)·5, 0, 10)
    → qualitative signal + summary for the LLM

KNOWN LIMITATIONS:
  - Book Yield with negative equity: when equity < 0 (e.g. companies with
    aggressive buybacks like AAPL), book_value_total is negative and book_yield is
    floored to 0 in data_fetcher.py. This means the valuation sub-block does not
    penalize negative equity from this angle. The penalty comes from the solvency
    sub-block (inverted D/E, z_de = -3.0 when equity < 0). The design is
    intentional: negative equity from aggressive buybacks does not indicate
    business deterioration per se, so it should not penalize relative valuation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.fundamental.engine.data_quality import (
    VALUATION_DISPLAY_NAMES,
    VALUATION_PLAUSIBILITY_LIMITS,
    evaluate_valuation_data_quality,
    is_component_valid,
)
from app.services.fundamental.engine.ratio_profiles import SP500_VALIDATED, RatioProfile
from app.services.fundamental.engine.sanitizer import sanitize_valuation

logger = logging.getLogger(__name__)

# Weights of the 4 yields in the raw score (sp500_validated profile; the active
# profile is injected into compute_valuation_score via `profile`).
_WEIGHTS = {
    "ebitda_yield": 0.40,
    "earnings_yield": 0.25,
    "fcf_yield": 0.25,
    "book_yield": 0.10,
}

_DISPLAY_NAMES = VALUATION_DISPLAY_NAMES
_PLAUSIBILITY_LIMITS = VALUATION_PLAUSIBILITY_LIMITS


def _universe_key(universe_df: pd.DataFrame) -> object:
    """Stable per-universe cache key (avoids the latent id() bug)."""
    return universe_df.attrs.get("universe_name", id(universe_df))


WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
Z_CLIP = 3.0
MIN_SECTOR_SIZE = 10  # minimum number of companies to use sector normalization
_UNIVERSE_RAW_CACHE: dict[tuple, pd.Series] = {}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _is_component_plausible(name: str, value: float | None) -> bool:
    """True if the yield is finite, non-negative and does not exceed the economic threshold."""
    return is_component_valid(name, value)


def _compute_raw_score(values: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """
    Weighted mean of valid yields, reweighting only active components.
    Requires at least two valid components to avoid a single ratio dominating.
    """
    valid = {k: float(values[k]) for k in weights if _is_component_plausible(k, values.get(k))}
    if len(valid) < 2:
        return None
    total_weight = sum(weights[k] for k in valid)
    if total_weight <= 0.0:
        return None
    return float(sum((weights[k] / total_weight) * valid[k] for k in valid))


def _universe_raw_scores(universe_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Compute valuation raw scores for the universe using the same active
    components and excluding implausible cells before reweighting.
    """
    cache_key = (
        _universe_key(universe_df),
        tuple(weights.items()),
        tuple(sorted(_PLAUSIBILITY_LIMITS.items())),
    )
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    weighted = pd.Series(0.0, index=universe_df.index, dtype=float)
    weight_sum = pd.Series(0.0, index=universe_df.index, dtype=float)
    component_count = pd.Series(0, index=universe_df.index, dtype=int)
    for k, w in weights.items():
        series = universe_df.get(k, pd.Series(np.nan, index=universe_df.index))
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = clean.ge(0.0) & clean.le(_PLAUSIBILITY_LIMITS.get(k, np.inf))
        weighted = weighted.add(w * clean.where(valid, 0.0), fill_value=0.0)
        weight_sum = weight_sum.add(w * valid.astype(float), fill_value=0.0)
        component_count = component_count.add(valid.astype(int), fill_value=0).astype(int)

    raw = (weighted / weight_sum.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    raw = raw[component_count >= min(2, len(weights))]
    _UNIVERSE_RAW_CACHE[cache_key] = raw
    return raw


def _winsorize(series: pd.Series) -> pd.Series:
    """p1-p99 winsorization over the universe distribution."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return series
    return series.clip(
        lower=float(clean.quantile(WINSOR_LOWER)),
        upper=float(clean.quantile(WINSOR_UPPER)),
    )


def _z_score(value: float, distribution: pd.Series) -> float | None:
    """Z-score of a value against a given distribution."""
    clean = distribution.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None
    std = float(clean.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float((value - float(clean.mean())) / std)


def _z_score_clipped(value: float, winsorized_distribution: pd.Series) -> float | None:
    """Z-score of the raw score, clipping value to the winsorized range and z to +/-3."""
    clean = winsorized_distribution.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    clipped_value = float(np.clip(value, float(clean.min()), float(clean.max())))
    z = _z_score(clipped_value, winsorized_distribution)
    return float(np.clip(z, -Z_CLIP, Z_CLIP)) if z is not None else None


def _z_to_score(z: float, k: float = 1.2) -> float:
    """Convert a z-score to a 0-10 score via a sigmoid (k=1.2)."""
    z = float(np.clip(z, -Z_CLIP, Z_CLIP))
    x = float(np.clip(k * z, -60.0, 60.0))
    return float(10.0 / (1.0 + np.exp(-x)))


def _signal(score: float) -> str:
    if score > 6.5:
        return "infravalorada"
    if score < 4.0:
        return "sobrevalorada"
    return "neutral"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_valuation_score(
    company_data: dict[str, Any],
    universe_df: pd.DataFrame,
    profile: RatioProfile = SP500_VALIDATED,
) -> dict[str, Any]:
    """
    Compute a company's valuation score (0-10).

    Args:
        company_data: Output of get_company_data().
        universe_df:  Output of load_universe() with yield columns.

    Returns:
        Dictionary with keys:
            sub_bloque, ticker, sector,
            ebitda_yield, earnings_yield, fcf_yield, book_yield,   # raw
            ebitda_yield_eff, earnings_yield_eff,                   # floored
            fcf_yield_eff, book_yield_eff,
            raw_score,
            z_sp500, z_sector, z_final, score_sp500, score_sector, sector_size,
            score, signal, valuation_summary
    """
    result: dict[str, Any] = {
        "sub_bloque": "valoracion",
        "ticker": company_data.get("ticker"),
        "sector": company_data.get("sector"),
    }

    # ---- 1. Sanitized yields + outlier flags ----
    ratios = sanitize_valuation(company_data)

    result["ebitda_yield"] = company_data.get("ebitda_yield")
    result["earnings_yield"] = company_data.get("earnings_yield")
    result["fcf_yield"] = company_data.get("fcf_yield")
    result["book_yield"] = company_data.get("book_yield")
    result["ebitda_yield_floored"] = ratios.ebitda_yield_floored
    result["ev_negative"] = ratios.ev_negative
    result["fcf_yield_capped"] = ratios.fcf_yield_capped
    result["fcf_yield_raw"] = ratios.fcf_yield_raw
    result["book_yield_outlier"] = ratios.book_yield_outlier
    result["book_yield_raw"] = ratios.book_yield_raw

    weights = profile.valuation_weights
    has_book = "book_yield" in weights

    result["price_currency"] = company_data.get("price_currency")
    result["market_cap_currency"] = company_data.get("market_cap_currency")
    result["financial_currency"] = company_data.get("financial_currency")
    result["ticker_currency"] = company_data.get("ticker_currency")
    result["name"] = company_data.get("name")
    result["country"] = company_data.get("country")
    result["market"] = company_data.get("market")
    result["exchange"] = company_data.get("exchange")
    result["is_adr"] = bool(company_data.get("is_adr"))
    result["is_foreign_company"] = bool(company_data.get("is_foreign_company"))

    ey_eff = ratios.ebitda_yield
    ery_eff = ratios.earnings_yield
    fy_eff = ratios.fcf_yield  # floored to 0, not yet p95-capped
    by_eff = ratios.book_yield  # floored to 0, not yet p95-capped

    # Apply a 95th-percentile cap from the universe when the flags require it.
    # Done here (not in sanitizer) because the universe is only available in valuation.
    if ratios.fcf_yield_capped and "fcf_yield" in universe_df.columns:
        _univ_fy = universe_df["fcf_yield"].fillna(0.0).clip(lower=0.0)
        _p95_fy = float(_univ_fy.quantile(0.95))
        fy_eff = max(min(ratios.fcf_yield_raw, _p95_fy), 0.0)

    if has_book and ratios.book_yield_outlier and "book_yield" in universe_df.columns:
        _univ_by = universe_df["book_yield"].fillna(0.0).clip(lower=0.0)
        _p95_by = float(_univ_by.quantile(0.95))
        by_eff = max(min(ratios.book_yield_raw, _p95_by), 0.0)

    result["ebitda_yield_eff"] = ey_eff
    result["earnings_yield_eff"] = ery_eff
    result["fcf_yield_eff"] = fy_eff
    if has_book:
        result["book_yield_eff"] = by_eff

    candidate_values: dict[str, float | None] = {
        "ebitda_yield": ratios.ebitda_yield
        if company_data.get("ebitda_yield") is not None
        else None,
        "earnings_yield": ratios.earnings_yield
        if company_data.get("earnings_yield") is not None
        else None,
        "fcf_yield": ratios.fcf_yield_raw if company_data.get("fcf_yield") is not None else None,
    }
    if has_book:
        candidate_values["book_yield"] = (
            ratios.book_yield_raw if company_data.get("book_yield") is not None else None
        )

    quality = evaluate_valuation_data_quality(company_data, candidate_values, weights)
    valid_components = quality["valid_components"]
    effective_values = quality["effective_values"]

    ey_eff = effective_values.get("ebitda_yield")
    ery_eff = effective_values.get("earnings_yield")
    fy_eff = effective_values.get("fcf_yield")
    by_eff = effective_values.get("book_yield")

    result["ebitda_yield_eff"] = ey_eff
    result["earnings_yield_eff"] = ery_eff
    result["fcf_yield_eff"] = fy_eff
    if has_book:
        result["book_yield_eff"] = by_eff
    result["valid_valuation_components"] = valid_components
    result["invalid_valuation_components"] = quality["invalid_components"]
    result["valuation_warnings"] = quality["warnings"]
    result["neutralized_blocks"] = quality["neutralized_blocks"]
    result["valuation_data_quality"] = quality["data_quality"]

    # ---- 2. Company raw score ----
    values = {"ebitda_yield": ey_eff, "earnings_yield": ery_eff, "fcf_yield": fy_eff}
    if has_book:
        values["book_yield"] = by_eff
    active_weights = {k: weights[k] for k in valid_components}
    raw = _compute_raw_score(values, active_weights)
    result["raw_score"] = raw

    if raw is None:
        result.update(
            {
                "z_sp500": None,
                "score_sp500": None,
                "sector_size": 0,
                "z_sector": None,
                "score_sector": None,
                "z_final": None,
                "score": 5.0,
                "signal": "neutral",
            }
        )
        result["valuation_summary"] = _build_summary(result)
        return result

    # ---- 3. Universe distribution (raw scores + winsorization) ----
    universe_scores = _universe_raw_scores(universe_df, active_weights)
    universe_scores_wins = _winsorize(universe_scores)

    # ---- 4. Z-score vs S&P 500 ----
    z_sp500 = _z_score_clipped(raw, universe_scores_wins)
    result["z_sp500"] = z_sp500
    score_sp500 = _z_to_score(z_sp500) if z_sp500 is not None else None
    result["score_sp500"] = score_sp500

    # ---- 5. Z-score vs GICS sector ----
    sector = company_data.get("sector")
    z_sector: float | None = None
    sector_size: int = 0

    if sector and "sector" in universe_df.columns:
        mask = universe_df["sector"] == sector
        sector_size = int(mask.sum())

        if sector_size >= MIN_SECTOR_SIZE:
            sector_wins = _winsorize(universe_scores[mask])
            z_sector = _z_score_clipped(raw, sector_wins)
        else:
            logger.info(
                "Sector '%s' with only %d companies; skipping sector normalization.",
                sector,
                sector_size,
            )

    result["sector_size"] = sector_size
    result["z_sector"] = z_sector
    score_sector = _z_to_score(z_sector) if z_sector is not None else None
    result["score_sector"] = score_sector

    # ---- 6. Combined z (30% universe + 70% sector) ----
    if z_sp500 is None:
        z_final: float | None = None
    elif z_sector is not None:
        z_final = 0.30 * z_sp500 + 0.70 * z_sector
    else:
        z_final = z_sp500  # fallback: not enough sector data
    if z_final is not None:
        z_final = float(np.clip(z_final, -Z_CLIP, Z_CLIP))

    result["z_final"] = z_final

    # ---- 7. Final 0-10 score and signal ----
    if score_sp500 is None:
        score: float | None = None
        signal = "sin_datos"
    elif score_sector is not None:
        score = 0.30 * score_sp500 + 0.70 * score_sector
        signal = _signal(score)
    else:
        score = score_sp500
        signal = _signal(score)

    result["score"] = score
    result["signal"] = signal

    # ---- 8. Summary for the LLM ----
    result["valuation_summary"] = _build_summary(result)

    return result


# ---------------------------------------------------------------------------
# Summary construction
# ---------------------------------------------------------------------------


def _build_summary_legacy_unused(r: dict[str, Any]) -> str:
    """Concise valuation text to be consumed by the LLM."""
    ticker = r.get("ticker", "?")
    score = r.get("score")
    signal = r.get("signal", "?")
    sector = r.get("sector") or "sector desconocido"

    score_str = f"{score:.2f}/10" if score is not None else "N/A"

    def pct(v: float | None) -> str:
        return f"{(v or 0.0) * 100:.2f}%"

    yields_desc = (
        f"EBITDA Yield {pct(r.get('ebitda_yield_eff'))} (×0.40), "
        f"Earnings Yield {pct(r.get('earnings_yield_eff'))} (×0.25), "
        f"FCF Yield {pct(r.get('fcf_yield_eff'))} (×0.25), "
        f"Book Yield {pct(r.get('book_yield_eff'))} (×0.10)."
    )

    z_sp = r.get("z_sp500")
    z_sec = r.get("z_sector")
    if z_sp is not None and z_sec is not None:
        z_desc = f"Z vs S&P500={z_sp:.2f} (30%) + Z vs {sector}={z_sec:.2f} (70%)."
    elif z_sp is not None:
        z_desc = f"Z vs S&P500={z_sp:.2f} (normalización solo vs universo)."
    else:
        z_desc = "Z no calculable."

    label = {
        "infravalorada": "cotiza por debajo de su valor relativo al mercado",
        "sobrevalorada": "cotiza por encima de su valor relativo al mercado",
        "neutral": "presenta una valoración en línea con el mercado",
        "sin_datos": "valoración no calculable por datos insuficientes",
    }.get(signal, signal)

    # Data-quality warnings
    warnings: list[str] = []
    if r.get("ev_negative"):
        warnings.append("EV negativo: EBITDA Yield forzado a 0")
    if r.get("fcf_yield_capped"):
        warnings.append(
            f"[AVISO: FCF Yield > 40% (raw={r.get('fcf_yield_raw', 0) * 100:.1f}%), "
            f"posible efecto extraordinario. Valor ajustado al p95 del universo "
            f"para normalización.]"
        )
    if r.get("book_yield_outlier"):
        warnings.append(
            f"[AVISO: Book Yield > 2000% (raw={r.get('book_yield_raw', 0) * 100:.0f}%), "
            f"posible error de datos. Valor ajustado al p95 del universo para normalización.]"
        )
    warnings_str = " ".join(warnings)

    return (
        f"{ticker} {label} — score de valoración {score_str} ({signal}). "
        f"Componentes: {yields_desc} "
        f"Normalización: {z_desc}" + (f" {warnings_str}" if warnings_str else "")
    )


def _build_summary(r: dict[str, Any]) -> str:
    """Valuation summary based on valid components and data quality."""
    ticker = r.get("ticker", "?")
    score = r.get("score")
    signal = r.get("signal", "?")
    sector = r.get("sector") or "sector desconocido"
    score_str = f"{score:.2f}/10" if score is not None else "N/A"

    def pct(v: float | None) -> str:
        return "invalid" if v is None else f"{v * 100:.2f}%"

    valid_components = set(r.get("valid_valuation_components") or [])
    component_parts: list[str] = []
    for name, default_weight in _WEIGHTS.items():
        eff_key = f"{name}_eff"
        if eff_key not in r:
            continue
        status = "" if name in valid_components else " [excluido]"
        component_parts.append(
            f"{_DISPLAY_NAMES.get(name, name)} {pct(r.get(eff_key))} "
            f"(x{default_weight:.2f}){status}"
        )
    yields_desc = ", ".join(component_parts) + "."

    z_sp = r.get("z_sp500")
    z_sec = r.get("z_sector")
    if z_sp is not None and z_sec is not None:
        z_desc = f"Z vs S&P500={z_sp:.2f} (30%) + Z vs {sector}={z_sec:.2f} (70%)."
    elif z_sp is not None:
        z_desc = f"Z vs S&P500={z_sp:.2f} (normalizacion solo vs universo)."
    else:
        z_desc = "Z no calculable."

    label = {
        "infravalorada": "cotiza por debajo de su valor relativo al mercado",
        "sobrevalorada": "cotiza por encima de su valor relativo al mercado",
        "neutral": "presenta una valoracion neutral por datos insuficientes o mixtos",
        "sin_datos": "valoracion no calculable por datos insuficientes",
    }.get(signal, signal)

    warnings_str = " ".join(str(w) for w in (r.get("valuation_warnings") or []))
    quality = r.get("valuation_data_quality", "N/A")
    valid_n = len(r.get("valid_valuation_components") or [])

    return (
        f"{ticker} {label} - score de valoracion {score_str} ({signal}). "
        f"Calidad datos: {quality}; componentes validos={valid_n}. "
        f"Componentes: {yields_desc} Normalizacion: {z_desc}"
        + (f" Avisos: {warnings_str}" if warnings_str else "")
    )
