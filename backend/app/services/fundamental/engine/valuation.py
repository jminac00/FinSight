"""Valuation sub-block — 30% weight in the fundamental score.

Assesses whether a company is under- or over-valued by comparing 4 return yields against the
S&P 500 universe and against its GICS sector.

Pipeline:
    company yields
    → weighted raw score (yields floored to 0)
    → winsorization p1-p99 over the universe distribution
    → z-score vs S&P 500 universe (30%) + z-score vs GICS sector (70%)
    → 0-10 rescale via sigmoid (k=1.2)
    → qualitative signal + LLM summary

KNOWN LIMITATIONS:
  - Book Yield with negative equity: when equity < 0 (e.g. companies with aggressive buybacks
    like AAPL), book_value_total is negative and book_yield is set to 0 by the floor applied in
    data_fetcher.py. This means the valuation sub-block does not penalize negative equity from
    this angle. The penalty comes from the solvency sub-block (inverted D/E, z_de = -3.0 when
    equity < 0). This is intentional: negative equity caused by aggressive buybacks does not
    signal business deterioration per se, so it should not penalize relative valuation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.fundamental.engine.sanitizer import sanitize_valuation

logger = logging.getLogger(__name__)

# Weights of the 4 yields in the raw score
_WEIGHTS = {
    "ebitda_yield": 0.40,
    "earnings_yield": 0.25,
    "fcf_yield": 0.25,
    "book_yield": 0.10,
}

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
MIN_SECTOR_SIZE = 10  # minimum companies required to use sector normalization
_UNIVERSE_RAW_CACHE: dict[int, pd.Series] = {}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _compute_raw_score(
    ebitda_yield: float | None,
    earnings_yield: float | None,
    fcf_yield: float | None,
    book_yield: float | None,
) -> float:
    """Weighted average of the 4 yields; each component floored to 0."""
    ey = max(ebitda_yield or 0.0, 0.0)
    ery = max(earnings_yield or 0.0, 0.0)
    fy = max(fcf_yield or 0.0, 0.0)
    by = max(book_yield or 0.0, 0.0)
    return (
        _WEIGHTS["ebitda_yield"] * ey
        + _WEIGHTS["earnings_yield"] * ery
        + _WEIGHTS["fcf_yield"] * fy
        + _WEIGHTS["book_yield"] * by
    )


def _universe_raw_scores(universe_df: pd.DataFrame) -> pd.Series:
    """Compute the valuation raw score for each row of the universe."""
    cache_key = id(universe_df)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    def col(name: str) -> pd.Series:
        return (
            universe_df.get(name, pd.Series(0.0, index=universe_df.index))
            .fillna(0.0)
            .clip(lower=0.0)
        )

    raw = (
        _WEIGHTS["ebitda_yield"] * col("ebitda_yield")
        + _WEIGHTS["earnings_yield"] * col("earnings_yield")
        + _WEIGHTS["fcf_yield"] * col("fcf_yield")
        + _WEIGHTS["book_yield"] * col("book_yield")
    )
    _UNIVERSE_RAW_CACHE[cache_key] = raw
    return raw


def _winsorize(series: pd.Series) -> pd.Series:
    """Winsorization p1-p99 over the universe distribution."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return series
    return series.clip(
        lower=float(clean.quantile(WINSOR_LOWER)),
        upper=float(clean.quantile(WINSOR_UPPER)),
    )


def _z_score(value: float, distribution: pd.Series) -> float | None:
    """Z-score of a scalar value against a given distribution."""
    clean = distribution.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None
    std = float(clean.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float((value - float(clean.mean())) / std)


def _z_score_clipped(value: float, winsorized_distribution: pd.Series) -> float | None:
    """Z-score of the raw score against the winsorized distribution (value not clipped)."""
    clean = winsorized_distribution.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    return _z_score(value, winsorized_distribution)


def _z_to_score(z: float, k: float = 1.2) -> float:
    """Convert z-score to a 0-10 score via sigmoid (k=1.2)."""
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
) -> dict[str, Any]:
    """Compute the valuation score (0-10) of a company.

    Args:
        company_data: Output of get_company_data().
        universe_df:  Output of load_universe() with yield columns.

    Returns:
        Dict with keys:
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

    ey_eff = ratios.ebitda_yield
    ery_eff = ratios.earnings_yield
    fy_eff = ratios.fcf_yield  # floored to 0, not yet p95-capped
    by_eff = ratios.book_yield  # floored to 0, not yet p95-capped

    # Apply the p95 universe cap when the flags indicate so.
    # Done here (not in sanitizer) because the universe is only available in valuation.
    if ratios.fcf_yield_capped and "fcf_yield" in universe_df.columns:
        _univ_fy = universe_df["fcf_yield"].fillna(0.0).clip(lower=0.0)
        _p95_fy = float(_univ_fy.quantile(0.95))
        fy_eff = max(min(ratios.fcf_yield_raw, _p95_fy), 0.0)

    if ratios.book_yield_outlier and "book_yield" in universe_df.columns:
        _univ_by = universe_df["book_yield"].fillna(0.0).clip(lower=0.0)
        _p95_by = float(_univ_by.quantile(0.95))
        by_eff = max(min(ratios.book_yield_raw, _p95_by), 0.0)

    result["ebitda_yield_eff"] = ey_eff
    result["earnings_yield_eff"] = ery_eff
    result["fcf_yield_eff"] = fy_eff
    result["book_yield_eff"] = by_eff

    # ---- 2. Company raw score ----
    raw = _compute_raw_score(ey_eff, ery_eff, fy_eff, by_eff)
    result["raw_score"] = raw

    # ---- 3. Universe distribution (raw scores + winsorization) ----
    universe_scores = _universe_raw_scores(universe_df)
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
                "Sector '%s' has only %d companies; skipping sector normalization.",
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

    # ---- 8. LLM summary ----
    result["valuation_summary"] = _build_summary(result)

    return result


# ---------------------------------------------------------------------------
# Summary construction
# ---------------------------------------------------------------------------


def _build_summary(r: dict[str, Any]) -> str:
    """Concise valuation text to be consumed by the LLM (kept in Spanish)."""
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
            f"posible efecto extraordinario. "
            f"Valor ajustado al p95 del universo para normalización.]"
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
