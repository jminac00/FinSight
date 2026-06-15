"""Final combination of the fundamental analysis.

Takes the results of the four sub-blocks and combines them into a final 0-10 score with a
qualitative signal and a narrative summary for the LLM.

Weights:
    Valuation   30%
    Quality     30%
    Growth      20%
    Solvency    20%

If a sub-block has no score (insufficient data) its weight is redistributed proportionally
among the available ones so the result stays comparable.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.services.fundamental.engine.data_fetcher import get_company_data
from app.services.fundamental.engine.growth import compute_growth_score
from app.services.fundamental.engine.quality import compute_quality_score
from app.services.fundamental.engine.solvency import compute_solvency_score
from app.services.fundamental.engine.valuation import compute_valuation_score

logger = logging.getLogger(__name__)

# Nominal weight of each sub-block
_BLOCK_WEIGHTS: dict[str, float] = {
    "valoracion": 0.30,
    "calidad": 0.30,
    "crecimiento": 0.20,
    "solvencia": 0.20,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _overall_signal(score: float) -> str:
    if score >= 7.0:
        return "muy positivo"
    if score >= 5.5:
        return "positivo"
    if score >= 4.5:
        return "neutral"
    if score >= 3.0:
        return "negativo"
    return "muy negativo"


def _score_label(score: float | None) -> str:
    """Convert a numeric score into a textual label for the summary."""
    if score is None:
        return "sin datos"
    if score >= 7.0:
        return "fuerte"
    if score >= 5.5:
        return "bueno"
    if score >= 4.5:
        return "neutral"
    if score >= 3.0:
        return "débil"
    return "muy débil"


# ---------------------------------------------------------------------------
# Sub-block combination
# ---------------------------------------------------------------------------


def combine_fundamental_scores(
    val_result: dict[str, Any],
    qual_result: dict[str, Any],
    grow_result: dict[str, Any],
    solv_result: dict[str, Any],
) -> dict[str, Any]:
    """Combine the four sub-blocks into the final fundamental score.

    Args:
        val_result:  Output of compute_valuation_score().
        qual_result: Output of compute_quality_score().
        grow_result: Output of compute_growth_score().
        solv_result: Output of compute_solvency_score().

    Returns:
        Dict with keys:
            ticker, sector,
            score_valoracion, score_calidad,
            score_crecimiento, score_solvencia,   # sub-scores
            weights_used,                          # effective weights applied
            score_final, signal_final,
            sub_signals,                           # signal of each sub-block
            fundamental_summary                    # text for the LLM
    """
    ticker = val_result.get("ticker") or qual_result.get("ticker")
    sector = val_result.get("sector") or qual_result.get("sector")

    sub_scores: dict[str, float | None] = {
        "valoracion": val_result.get("score"),
        "calidad": qual_result.get("score"),
        "crecimiento": grow_result.get("score"),
        "solvencia": solv_result.get("score"),
    }

    sub_signals: dict[str, str] = {
        "valoracion": val_result.get("signal", "sin_datos"),
        "calidad": qual_result.get("signal", "sin_datos"),
        "crecimiento": grow_result.get("signal", "sin_datos"),
        "solvencia": solv_result.get("signal", "sin_datos"),
    }

    # ---- Redistribute weights if any sub-block has no score ----
    total_weight = sum(_BLOCK_WEIGHTS[k] for k, v in sub_scores.items() if v is not None)

    if total_weight == 0.0:
        score_final: float | None = None
        weights_used = {k: 0.0 for k in _BLOCK_WEIGHTS}
    else:
        weights_used = {
            k: (_BLOCK_WEIGHTS[k] / total_weight if sub_scores[k] is not None else 0.0)
            for k in _BLOCK_WEIGHTS
        }
        score_final = sum(
            weights_used[k] * sub_scores[k] for k in _BLOCK_WEIGHTS if sub_scores[k] is not None
        )

    signal_final = _overall_signal(score_final) if score_final is not None else "sin_datos"

    result: dict[str, Any] = {
        "ticker": ticker,
        "sector": sector,
        "score_valoracion": sub_scores["valoracion"],
        "score_calidad": sub_scores["calidad"],
        "score_crecimiento": sub_scores["crecimiento"],
        "score_solvencia": sub_scores["solvencia"],
        "weights_used": weights_used,
        "score_final": round(score_final, 2) if score_final is not None else None,
        "signal_final": signal_final,
        "sub_signals": sub_signals,
        "valuation_detail": val_result,
        "quality_detail": qual_result,
        "growth_detail": grow_result,
        "solvency_detail": solv_result,
        "fundamental_summary": "",  # filled in below
    }

    result["fundamental_summary"] = _build_summary(
        result, val_result, qual_result, grow_result, solv_result
    )

    return result


# ---------------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------------


def _build_summary(
    r: dict[str, Any],
    val: dict[str, Any],
    qual: dict[str, Any],
    grow: dict[str, Any],
    solv: dict[str, Any],
) -> str:
    """Build a narrative summary of the full fundamental analysis (kept in Spanish).
    Mentions the strengths and weaknesses of each sub-block.
    """
    ticker = r["ticker"] or "?"
    score_final = r["score_final"]
    signal = r["signal_final"]
    sector = r["sector"] or "sector desconocido"

    score_str = f"{score_final:.2f}/10" if score_final is not None else "N/A"

    def sc(v: float | None) -> str:
        return f"{v:.1f}" if v is not None else "N/A"

    # Detect strengths (score >= 6.5) and weaknesses (score <= 4) per sub-block
    strengths: list[str] = []
    weaknesses: list[str] = []

    block_info = [
        (
            "Valoración",
            r["score_valoracion"],
            val.get("signal", ""),
            val.get("valuation_summary", ""),
        ),
        ("Calidad", r["score_calidad"], qual.get("signal", ""), qual.get("quality_summary", "")),
        (
            "Crecimiento",
            r["score_crecimiento"],
            grow.get("signal", ""),
            grow.get("growth_summary", ""),
        ),
        (
            "Solvencia",
            r["score_solvencia"],
            solv.get("signal", ""),
            solv.get("solvency_summary", ""),
        ),
    ]

    for block_name, block_score, _block_signal, _ in block_info:
        if block_score is not None:
            if block_score >= 6.5:
                strengths.append(f"{block_name} ({block_score:.1f}/10)")
            elif block_score <= 4.0:
                weaknesses.append(f"{block_name} ({block_score:.1f}/10)")

    strengths_str = ", ".join(strengths) if strengths else "ninguno destacable"
    weaknesses_str = ", ".join(weaknesses) if weaknesses else "ninguno reseñable"

    # Sub-score table
    table = (
        f"Valoración {sc(r['score_valoracion'])}/10 "
        f"({r['sub_signals']['valoracion']}) | "
        f"Calidad {sc(r['score_calidad'])}/10 "
        f"({r['sub_signals']['calidad']}) | "
        f"Crecimiento {sc(r['score_crecimiento'])}/10 "
        f"({r['sub_signals']['crecimiento']}) | "
        f"Solvencia {sc(r['score_solvencia'])}/10 "
        f"({r['sub_signals']['solvencia']})"
    )

    return (
        f"[ANÁLISIS FUNDAMENTAL — {ticker} | {sector}]\n"
        f"Score final: {score_str} — Señal: {signal.upper()}\n"
        f"Sub-bloques: {table}\n"
        f"Puntos fuertes: {strengths_str}.\n"
        f"Puntos débiles: {weaknesses_str}.\n"
        f"Detalle valoración: {val.get('valuation_summary', 'N/A')}\n"
        f"Detalle calidad: {qual.get('quality_summary', 'N/A')}\n"
        f"Detalle crecimiento: {grow.get('growth_summary', 'N/A')}\n"
        f"Detalle solvencia: {solv.get('solvency_summary', 'N/A')}"
    )


# ---------------------------------------------------------------------------
# Full pipeline: ticker → final score
# ---------------------------------------------------------------------------


def analyze_ticker(
    ticker_str: str,
    universe_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run the full fundamental pipeline for a ticker.

    1. Download company data (get_company_data)
    2. Compute the 4 sub-blocks
    3. Combine into the final score

    Args:
        ticker_str:   Stock symbol (e.g. "AAPL").
        universe_df:  Pre-loaded S&P 500 universe used for cross-sectional normalization.

    Returns:
        Dict with the full result (sub-scores + final score + summary).
    """
    logger.info("Downloading data for %s...", ticker_str)
    company_data = get_company_data(ticker_str)

    logger.info("Computing sub-blocks...")
    val_result = compute_valuation_score(company_data, universe_df)
    qual_result = compute_quality_score(company_data, universe_df)
    grow_result = compute_growth_score(company_data, universe_df)
    solv_result = compute_solvency_score(company_data, universe_df)

    return combine_fundamental_scores(val_result, qual_result, grow_result, solv_result)
