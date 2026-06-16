"""Unit tests for TechnicalService: engine→result mapping, caching and error handling.

The universe manager, the engine entry point and the LLM are all mocked, so no network or snapshot
is needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.technical import TechnicalResult
from app.services.technical.service import (
    TechnicalAnalysisError,
    TechnicalService,
    TechnicalUniverseNotReadyError,
)

_CLOSES = object()  # opaque payload; the mocked engine ignores it

_RAW = {
    "ticker": "AAPL",
    "technical_score": 6.42,
    "technical_signal": "neutral",
    "signal": "neutral",
    "individual_scores": {
        "momentum": 7.1,
        "trend": 6.0,
        "risk_stability": 5.5,
        "confirmation": 6.8,
    },
    "weights": {"momentum": 0.35, "trend": 0.30, "risk_stability": 0.20, "confirmation": 0.15},
    "effective_weights": {
        "momentum": 0.35,
        "trend": 0.30,
        "risk_stability": 0.20,
        "confirmation": 0.15,
    },
    "data_completeness": 1.0,
    "score_reliable": True,
    "summary": "AAPL: Technical Score 6.42/10 (neutral).",
    "blocks": {
        "momentum": {"normalized_score_0_10": 7.1, "summary": "momentum detail"},
        "trend": {"trend_score_0_10": 6.0, "summary": "trend detail"},
        "risk_stability": {"risk_stability_score_0_10": 5.5, "summary": "risk detail"},
        "confirmation": {"confirmation_score_0_10": 6.8, "summary": "confirmation detail"},
    },
    "errors": {},
}


def _make_service(
    raw: dict | None = None,
    universe: str = "sp500",
    not_ready: bool = False,
) -> tuple[TechnicalService, MagicMock, MagicMock]:
    manager = MagicMock()
    manager.resolve_universe.return_value = universe
    if not_ready:
        manager.load.side_effect = TechnicalUniverseNotReadyError("snapshot missing")
    else:
        manager.load.return_value = {"closes": _CLOSES, "ohlcv": None}

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="  Análisis técnico en español.  ")

    analyze_fn = MagicMock(return_value=raw if raw is not None else _RAW)

    service = TechnicalService(
        llm_service=llm, cache_ttl=86400, manager=manager, analyze_fn=analyze_fn
    )
    return service, llm, analyze_fn


async def test_analyze_maps_engine_result() -> None:
    service, llm, analyze_fn = _make_service()

    result = await service.analyze("AAPL", mode="domestic")

    assert isinstance(result, TechnicalResult)
    assert result.score == 6.42
    assert result.signal == "neutral"
    assert result.block_scores.momentum == 7.1
    assert result.block_scores.confirmation == 6.8
    assert result.indicators["universe"] == "sp500"
    assert result.llm_analysis == "Análisis técnico en español."
    llm.complete.assert_awaited_once()
    analyze_fn.assert_called_once_with(
        "AAPL", "sp500", universe_closes=_CLOSES, universe_ohlcv=None
    )


async def test_result_is_cached_per_universe_ticker() -> None:
    service, llm, analyze_fn = _make_service()

    await service.analyze("AAPL")
    await service.analyze("AAPL")

    analyze_fn.assert_called_once()
    llm.complete.assert_awaited_once()


async def test_force_refresh_recomputes() -> None:
    service, _, analyze_fn = _make_service()

    await service.analyze("AAPL")
    await service.analyze("AAPL", force_refresh=True)

    assert analyze_fn.call_count == 2


async def test_missing_score_raises_analysis_error() -> None:
    service, _, _ = _make_service(raw={**_RAW, "technical_score": None})

    with pytest.raises(TechnicalAnalysisError):
        await service.analyze("AAPL")


async def test_universe_not_ready_propagates() -> None:
    service, _, _ = _make_service(not_ready=True)

    with pytest.raises(TechnicalUniverseNotReadyError):
        await service.analyze("AAPL")
