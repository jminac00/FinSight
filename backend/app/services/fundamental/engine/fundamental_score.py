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
from app.services.fundamental.engine.ratio_profiles import SP500_VALIDATED, get_profile
from app.services.fundamental.engine.solvency import compute_solvency_score
from app.services.fundamental.engine.universe_manager import get_universe_manager
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


# Candidate indicators per sub-block for the degradation report. Only those PRESENT in the detail
# are counted (so roce/roe and ratios excluded by profile/methodology are not counted as missing).
_DEGRADE_CANDIDATES: dict[str, list[str]] = {
    "valoracion": ["ebitda_yield_eff", "earnings_yield_eff", "fcf_yield_eff", "book_yield_eff"],
    "calidad": ["z_gp_a", "z_roa", "z_roe", "z_roce", "z_operating_margin", "z_fcf_ni"],
    "crecimiento": [
        "z_revenue_trend",
        "z_fcf_trend",
        "z_delta_gross_margin",
        "z_asset_growth_inv",
        "z_share_dilution_inv",
    ],
    "solvencia": [
        "z_dn_ebitda",
        "z_interest_coverage",
        "z_current_ratio",
        "z_debt_to_equity",
        "z_cfo_debt",
    ],
}

# Human-readable notes for data flags (product-facing, kept in Spanish).
_FLAG_NOTES: dict[str, str] = {
    "roe_no_data": "ROE sin equity fiable: z_ROE=0",
    "de_no_data": "D/E sin equity fiable: z_D/E=0",
    "gross_profit_no_data": "Gross profit historico no calculable: z_dMgBruto=0",
    "coverage_bank_no_data": "Banco sin interest expense de deuda fiable: z_CobInt=0",
}


def _collect_data_flags(*details: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for detail in details:
        for flag in detail.get("data_flags") or []:
            if flag not in flags:
                flags.append(flag)
    return flags


def _degradation(detail: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    """Graceful-degradation report for a sub-block: how many applicable indicators have a value
    vs the applicable total, with a human-readable note.
    """
    applicable = [k for k in candidates if k in detail]
    available = [k for k in applicable if detail.get(k) is not None]
    missing = [k for k in applicable if detail.get(k) is None]
    n_app, n_av = len(applicable), len(available)
    note = (
        ""
        if (n_app == 0 or n_av == n_app)
        else f"score calculado con {n_av}/{n_app} indicadores disponibles"
    )
    return {
        "indicators_total": n_app,
        "indicators_available": n_av,
        "missing": missing,
        "degraded": bool(note),
        "note": note,
    }


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
        Dict with ticker, sector, the four sub-scores (under both `score_*` and `*_score` keys),
        weights_used, score_final/fundamental_score, signal, sub_signals, per-block details,
        data-quality fields, degradation report and the narrative fundamental_summary.
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
        "fundamental_score": round(score_final, 2) if score_final is not None else None,
        "fundamental_signal": signal_final,
        "score_valoracion": sub_scores["valoracion"],
        "score_calidad": sub_scores["calidad"],
        "score_crecimiento": sub_scores["crecimiento"],
        "score_solvencia": sub_scores["solvencia"],
        "valuation_score": sub_scores["valoracion"],
        "quality_score": sub_scores["calidad"],
        "growth_score": sub_scores["crecimiento"],
        "solvency_score": sub_scores["solvencia"],
        "weights_used": weights_used,
        "score_final": round(score_final, 2) if score_final is not None else None,
        "signal_final": signal_final,
        "sub_signals": sub_signals,
        "valuation_detail": val_result,
        "quality_detail": qual_result,
        "growth_detail": grow_result,
        "solvency_detail": solv_result,
        "valuation_data_quality": val_result.get("valuation_data_quality"),
        "valuation_warnings": val_result.get("valuation_warnings", []),
        "valid_valuation_components": val_result.get("valid_valuation_components", []),
        "fundamental_summary": "",  # filled in below
    }

    result["degradation"] = {
        "valoracion": _degradation(val_result, _DEGRADE_CANDIDATES["valoracion"]),
        "calidad": _degradation(qual_result, _DEGRADE_CANDIDATES["calidad"]),
        "crecimiento": _degradation(grow_result, _DEGRADE_CANDIDATES["crecimiento"]),
        "solvencia": _degradation(solv_result, _DEGRADE_CANDIDATES["solvencia"]),
    }

    flags = _collect_data_flags(val_result, qual_result, grow_result, solv_result)
    if val_result.get("valuation_data_quality") == "No fiable":
        flags.append("valuation_low_quality")
    result["data_flags"] = ";".join(flags) if flags else "ok"
    notes = [_FLAG_NOTES[f] for f in flags if f in _FLAG_NOTES]
    notes.extend(str(w) for w in (val_result.get("valuation_warnings") or []))
    result["score_note"] = "; ".join(notes)

    result["fundamental_summary"] = _build_summary(
        result, val_result, qual_result, grow_result, solv_result
    )

    return result


# ---------------------------------------------------------------------------
# Narrative summary (kept in Spanish — product content fed to the LLM)
# ---------------------------------------------------------------------------


def _build_summary(
    r: dict[str, Any],
    val: dict[str, Any],
    qual: dict[str, Any],
    grow: dict[str, Any],
    solv: dict[str, Any],
) -> str:
    """Build a narrative summary of the full fundamental analysis, mentioning the strengths and
    weaknesses of each sub-block.
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
    universe_df: pd.DataFrame | None = None,
    universe: str = "sp500",
) -> dict[str, Any]:
    """Run the full fundamental pipeline for a ticker.

    1. Download company data (get_company_data)
    2. Compute the 4 sub-blocks
    3. Combine into the final score

    Args:
        ticker_str:  Stock symbol (e.g. "AAPL").
        universe_df: Pre-loaded universe; if None the UniverseManager selects it.
        universe:    Reference universe for normalization ("sp500" by default). Only used to pick
            the ratio profile when universe_df is provided.

    Returns:
        Dict with the full result (sub-scores + final score + summary).
    """
    mgr = get_universe_manager()
    if universe_df is None:
        logger.info("Selecting universe '%s'...", universe)
        cfg = mgr.get_config(universe)
        universe_df = cfg.universe_df
        profile = get_profile(cfg.ratio_profile)
    else:
        profile = (
            get_profile(mgr.profile_name(universe))
            if universe in ("sp500", "msci_world")
            else SP500_VALIDATED
        )

    logger.info("Downloading data for %s (profile %s)...", ticker_str, profile.name)
    company_data = get_company_data(
        ticker_str,
        apply_c6_fallbacks=(profile.name != "sp500_validated"),
    )

    logger.info("Computing sub-blocks...")
    val_result = compute_valuation_score(company_data, universe_df, profile=profile)
    qual_result = compute_quality_score(company_data, universe_df, profile=profile)
    grow_result = compute_growth_score(company_data, universe_df, profile=profile)
    solv_result = compute_solvency_score(company_data, universe_df, profile=profile)

    return combine_fundamental_scores(val_result, qual_result, grow_result, solv_result)
