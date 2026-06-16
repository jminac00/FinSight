"""Fundamental analysis service (multi-universe).

Wraps the vendored scoring engine (see engine/): it resolves the reference universe for a ticker
(S&P 500 or MSCI World, per the requested mode), normalizes the ticker against it, and turns the
deterministic summary into a natural-language analysis in Spanish via the LLM. Results are cached
per (universe, ticker) with TTL = CACHE_TTL_FUNDAMENTAL.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from cachetools import TTLCache

from app.llm.base import LLMService
from app.models.fundamental import FundamentalResult
from app.services.fundamental.engine.fundamental_score import analyze_ticker
from app.services.fundamental.engine.ticker_routing import recommend_ticker
from app.services.fundamental.engine.universe_manager import UniverseManager

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = 512

_SYSTEM_PROMPT = """You are a fundamental equity analyst writing for retail investors with no \
finance background.

You receive a structured fundamental analysis of a company: a 0-10 overall score, four \
sub-scores (valuation, quality, growth, solvency) and supporting metrics. Turn it into a clear, \
concise explanation.

Rules:
- Write in SPANISH, in plain language a non-expert can follow.
- 3-5 sentences. Explain what the score means and the company's main strengths and weaknesses.
- If the data quality is not "OK", briefly warn that the result should be read with caution.
- Do NOT give investment advice nor buy/sell recommendations.
- Do NOT invent data beyond what is provided.
- Return plain text only (no markdown, no JSON)."""


class FundamentalError(Exception):
    """Base error for the fundamental analysis pipeline."""


class UniverseNotReadyError(FundamentalError):
    """Raised when a required universe snapshot is not available."""


class FundamentalAnalysisError(FundamentalError):
    """Raised when the engine cannot produce a usable score for the ticker."""


def _build_metrics(raw: dict[str, Any], universe: str, mode: str, ticker: str) -> dict[str, Any]:
    """Extract a curated, frontend-friendly metrics dict from the engine result."""
    val = raw.get("valuation_detail") or {}
    qual = raw.get("quality_detail") or {}
    solv = raw.get("solvency_detail") or {}
    return {
        "universe": universe,
        "mode": mode,
        "scores": {
            "valoracion": raw.get("score_valoracion"),
            "calidad": raw.get("score_calidad"),
            "crecimiento": raw.get("score_crecimiento"),
            "solvencia": raw.get("score_solvencia"),
        },
        "sub_signals": raw.get("sub_signals", {}),
        "weights_used": raw.get("weights_used", {}),
        "sector": raw.get("sector"),
        "valuation_data_quality": raw.get("valuation_data_quality"),
        "data_flags": raw.get("data_flags"),
        "degradation": raw.get("degradation", {}),
        "ratios": {
            "ebitda_yield": val.get("ebitda_yield"),
            "earnings_yield": val.get("earnings_yield"),
            "fcf_yield": val.get("fcf_yield"),
            "roe": qual.get("roe"),
            "roa": qual.get("roa"),
            "roce": qual.get("roce"),
            "operating_margin": qual.get("operating_margin"),
            "gp_a": qual.get("gp_a"),
            "dn_ebitda": solv.get("dn_ebitda"),
            "current_ratio": solv.get("current_ratio"),
            "debt_to_equity": solv.get("debt_to_equity"),
        },
        "routing": recommend_ticker(ticker, original_result=raw),
    }


class FundamentalService:
    """Orchestrates the fundamental analysis for a single ticker across universes.

    Flow: resolve universe (by mode) → engine scoring → LLM explanation in Spanish.
    """

    def __init__(
        self,
        llm_service: LLMService,
        cache_ttl: int,
        manager: UniverseManager | None = None,
        analyze_fn: Callable[[str, pd.DataFrame, str], dict[str, Any]] = analyze_ticker,
    ) -> None:
        """Args:
        llm_service: LLM provider used to render the analysis in Spanish.
        cache_ttl: Seconds a result stays cached per (universe, ticker).
        manager: Universe manager (its own instance so a refresh can drop it via cache_clear).
        analyze_fn: Engine entry point; injectable for testing.
        """
        self._llm = llm_service
        self._manager = manager or UniverseManager()
        self._analyze_fn = analyze_fn
        self._cache: TTLCache[tuple[str, str], FundamentalResult] = TTLCache(
            maxsize=_CACHE_MAX_ENTRIES, ttl=cache_ttl
        )

    async def analyze(
        self, ticker: str, mode: str = "auto", force_refresh: bool = False
    ) -> FundamentalResult:
        """Run the fundamental analysis for a stock ticker.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL', 'ASML.AS').
            mode: 'auto' (S&P 500 if in the index, else MSCI World), 'domestic' (S&P 500) or
                'global' (MSCI World).
            force_refresh: Skip the cached result and recompute.

        Returns:
            FundamentalResult with score, metrics, Spanish LLM analysis and cache timestamp.

        Raises:
            UniverseNotReadyError: If the resolved universe snapshot is unavailable.
            FundamentalAnalysisError: If the engine cannot produce a usable score.
        """
        universe = self._manager.resolve_universe(ticker, mode)
        cache_key = (universe, ticker)
        if not force_refresh and cache_key in self._cache:
            logger.info("Fundamental cache hit for %s (%s)", ticker, universe)
            return self._cache[cache_key]

        try:
            universe_df = self._manager.get_universe_df(universe)
        except FileNotFoundError as exc:
            raise UniverseNotReadyError(str(exc)) from exc

        raw = await asyncio.to_thread(self._analyze_fn, ticker, universe_df, universe)

        score_final = raw.get("score_final")
        if score_final is None:
            raise FundamentalAnalysisError(
                f"Insufficient fundamental data to score ticker {ticker}"
            )

        summary = raw.get("fundamental_summary") or ""
        llm_analysis = await self._llm.complete(system_prompt=_SYSTEM_PROMPT, user_prompt=summary)

        result = FundamentalResult(
            score=float(score_final),
            metrics=_build_metrics(raw, universe, mode, ticker),
            llm_analysis=llm_analysis.strip(),
            cached_at=datetime.now(tz=UTC),
        )
        self._cache[cache_key] = result
        return result
