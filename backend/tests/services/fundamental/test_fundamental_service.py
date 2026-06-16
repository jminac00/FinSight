"""Tests for FundamentalService orchestration (no network, fake engine and LLM)."""

import pandas as pd
import pytest

from app.models.fundamental import FundamentalResult
from app.services.fundamental.service import (
    FundamentalAnalysisError,
    FundamentalService,
    UniverseNotReadyError,
)


def _raw_result(score_final: float | None = 6.5) -> dict:
    return {
        "ticker": "AAPL",
        "sector": "Technology",
        "score_valoracion": 5.0,
        "score_calidad": 8.0,
        "score_crecimiento": 7.0,
        "score_solvencia": 6.0,
        "weights_used": {"valoracion": 0.3, "calidad": 0.3, "crecimiento": 0.2, "solvencia": 0.2},
        "score_final": score_final,
        "signal_final": "positivo" if score_final is not None else "sin_datos",
        "sub_signals": {
            "valoracion": "neutral",
            "calidad": "alta calidad",
            "crecimiento": "moderado",
            "solvencia": "moderada",
        },
        "valuation_detail": {"ebitda_yield": 0.08, "earnings_yield": 0.05, "fcf_yield": 0.04},
        "quality_detail": {"roe": 0.35, "roa": 0.18, "operating_margin": 0.30, "gp_a": 0.45},
        "growth_detail": {},
        "solvency_detail": {"dn_ebitda": 1.2, "current_ratio": 1.5, "debt_to_equity": 1.2},
        "fundamental_summary": "[ANÁLISIS FUNDAMENTAL — AAPL] Score final: 6.50/10",
    }


class _FakeManager:
    """Stand-in for UniverseManager: routes by mode and serves a fake universe DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        ready: bool = True,
        universe: str = "sp500",
    ) -> None:
        self._df = df if df is not None else pd.DataFrame({"ticker": ["AAPL"]})
        self._ready = ready
        self._universe = universe
        self.resolve_calls: list[tuple[str, str]] = []

    def resolve_universe(self, ticker: str, mode: str = "auto") -> str:
        self.resolve_calls.append((ticker, mode))
        return self._universe

    def get_universe_df(self, universe: str = "sp500") -> pd.DataFrame:
        if not self._ready:
            raise FileNotFoundError("universe not built yet")
        return self._df


class _FakeAnalyze:
    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self.calls: list[tuple[str, str]] = []

    def __call__(self, ticker: str, universe_df: pd.DataFrame, universe: str) -> dict:
        self.calls.append((ticker, universe))
        return {**self._raw, "ticker": ticker}


def _make_service(mock_llm, *, raw=None, ready=True, universe="sp500", manager=None):
    analyze = _FakeAnalyze(raw if raw is not None else _raw_result())
    manager = manager or _FakeManager(ready=ready, universe=universe)
    service = FundamentalService(
        llm_service=mock_llm,
        cache_ttl=1800,
        manager=manager,
        analyze_fn=analyze,
    )
    return service, analyze, manager


async def test_analyze_returns_result_and_calls_llm(mock_llm):
    service, analyze, _ = _make_service(mock_llm)

    result = await service.analyze("AAPL")

    assert isinstance(result, FundamentalResult)
    assert result.score == 6.5
    assert result.llm_analysis == "Respuesta de prueba del LLM."
    # The deterministic summary must be the LLM user prompt
    assert mock_llm.complete.await_args.kwargs["user_prompt"].startswith("[ANÁLISIS FUNDAMENTAL")
    # Metrics expose the four sub-scores, plus universe and mode
    assert result.metrics["scores"]["valoracion"] == 5.0
    assert result.metrics["scores"]["solvencia"] == 6.0
    assert result.metrics["universe"] == "sp500"
    assert result.metrics["mode"] == "auto"
    assert analyze.calls == [("AAPL", "sp500")]


async def test_result_is_cached_per_universe_ticker(mock_llm):
    service, analyze, _ = _make_service(mock_llm)

    await service.analyze("AAPL")
    await service.analyze("AAPL")

    assert analyze.calls == [("AAPL", "sp500")]  # second call served from cache
    assert mock_llm.complete.await_count == 1


async def test_force_refresh_bypasses_cache(mock_llm):
    service, analyze, _ = _make_service(mock_llm)

    await service.analyze("AAPL")
    await service.analyze("AAPL", force_refresh=True)

    assert analyze.calls == [("AAPL", "sp500"), ("AAPL", "sp500")]


async def test_mode_routes_to_resolved_universe(mock_llm):
    service, analyze, manager = _make_service(mock_llm, universe="msci_world")

    result = await service.analyze("ASML.AS", mode="global")

    assert manager.resolve_calls == [("ASML.AS", "global")]
    assert result.metrics["universe"] == "msci_world"
    assert analyze.calls == [("ASML.AS", "msci_world")]


async def test_universe_not_ready_propagates(mock_llm):
    service, analyze, _ = _make_service(mock_llm, ready=False)

    with pytest.raises(UniverseNotReadyError):
        await service.analyze("AAPL")

    assert analyze.calls == []


async def test_missing_final_score_raises(mock_llm):
    service, _, _ = _make_service(mock_llm, raw=_raw_result(score_final=None))

    with pytest.raises(FundamentalAnalysisError):
        await service.analyze("AAPL")
