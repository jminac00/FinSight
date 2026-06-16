"""Routing recommendations between ADR/alternative listings.

Does not change the ticker automatically: it reports whether a more reliable alternative for
fundamentals exists and lets the caller decide whether to rerun. Product-facing reason strings
are kept in Spanish on purpose.
"""

from __future__ import annotations

from typing import Any

from app.services.fundamental.engine.data_quality import (
    DATA_QUALITY_OK,
    DATA_QUALITY_REVIEW,
    DATA_QUALITY_UNRELIABLE,
)

_ROUTING_GROUPS: dict[str, dict[str, Any]] = {
    "ASML": {
        "tickers": ["ASML.AS", "ASML"],
        "preferred": "ASML.AS",
        "reason": (
            "ASML parece cotizar en USD mientras sus estados financieros estan en EUR. "
            "ASML.AS mantiene moneda de cotizacion y estados financieros comparables."
        ),
    },
    "NOVO": {
        "tickers": ["NOVO-B.CO", "NVO"],
        "preferred": "NOVO-B.CO",
        "reason": (
            "NVO es ADR/listing US con precio en USD y estados financieros en DKK. "
            "NOVO-B.CO es mas coherente para fundamentales."
        ),
    },
    "TOYOTA": {
        "tickers": ["7203.T", "TM"],
        "preferred": "7203.T",
        "reason": (
            "TM es ADR con precio/market cap en USD y estados financieros en JPY. "
            "7203.T evita la mezcla de monedas en valoracion."
        ),
    },
    "BHP": {
        "tickers": ["BHP.AX", "BHP"],
        "preferred": "BHP",
        "reason": (
            "BHP.AX cotiza en AUD pero yfinance reporta estados financieros en USD. "
            "BHP mantiene precio/market cap y estados financieros en USD en esta fuente."
        ),
    },
}

_TICKER_TO_GROUP = {
    ticker: group_name for group_name, cfg in _ROUTING_GROUPS.items() for ticker in cfg["tickers"]
}

_QUALITY_RANK = {
    DATA_QUALITY_UNRELIABLE: 0,
    DATA_QUALITY_REVIEW: 1,
    DATA_QUALITY_OK: 2,
    None: -1,
}


def _valuation_quality(result: dict[str, Any] | None) -> str | None:
    if not result:
        return None
    detail = result.get("valuation_detail") or result
    return detail.get("valuation_data_quality")


def _quality_rank(result: dict[str, Any] | None) -> int:
    return _QUALITY_RANK.get(_valuation_quality(result), -1)


def recommend_ticker(
    input_ticker: str,
    original_result: dict[str, Any] | None = None,
    candidate_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recommend an alternative ticker if it improves fundamental comparability.

    Args:
        input_ticker: ticker entered by the user.
        original_result: fundamental result of the original ticker, if already computed.
        candidate_results: fundamental results per alternative ticker, if the caller computed
            them. If provided, preference is by observed quality over the static map.
    """
    ticker = input_ticker.strip().upper()
    group_name = _TICKER_TO_GROUP.get(ticker)
    if group_name is None:
        return {
            "input_ticker": ticker,
            "preferred_ticker": ticker,
            "reason": "No hay alternativa ADR/local registrada.",
            "confidence": "baja",
            "alternative_tickers": [],
            "whether_rerun_is_recommended": False,
        }

    cfg = _ROUTING_GROUPS[group_name]
    alternatives = [t for t in cfg["tickers"] if t != ticker]
    preferred = cfg["preferred"]
    reason = cfg["reason"]
    confidence = "alta"

    if candidate_results:
        scored = sorted(
            cfg["tickers"],
            key=lambda t: (_quality_rank(candidate_results.get(t)), t == preferred),
            reverse=True,
        )
        observed_best = scored[0]
        if _quality_rank(candidate_results.get(observed_best)) > _quality_rank(original_result):
            preferred = observed_best
            reason = (
                f"{observed_best} presenta mayor calidad de datos de valoracion "
                f"({_valuation_quality(candidate_results.get(observed_best))}) que {ticker} "
                f"({_valuation_quality(original_result)})."
            )
        confidence = "alta" if _quality_rank(candidate_results.get(preferred)) >= 1 else "media"

    rerun = preferred != ticker
    if (
        original_result is not None
        and _valuation_quality(original_result) == DATA_QUALITY_OK
        and not candidate_results
    ):
        rerun = preferred != ticker and ticker != cfg["preferred"]

    return {
        "input_ticker": ticker,
        "preferred_ticker": preferred,
        "reason": reason,
        "confidence": confidence,
        "alternative_tickers": alternatives,
        "whether_rerun_is_recommended": rerun,
    }


def registered_route_tickers() -> dict[str, list[str]]:
    """Return registered groups for tests/documentation."""
    return {name: list(cfg["tickers"]) for name, cfg in _ROUTING_GROUPS.items()}
