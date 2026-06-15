"""Fundamental analysis service.

Wraps the vendored scoring engine (see engine/): it normalizes a single ticker against the
precomputed S&P 500 universe, then turns the deterministic summary into a natural-language
analysis in Spanish via the LLM. Results are cached per ticker (TTL = CACHE_TTL_FUNDAMENTAL).
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from cachetools import TTLCache

from app.llm.base import LLMService
from app.models.fundamental import FundamentalResult
from app.services.fundamental.engine.scoring import analyze_ticker
from app.services.fundamental.engine.universe import DEFAULT_UNIVERSE_PATH, load_universe

logger = logging.getLogger(__name__)

_CACHE_MAX_TICKERS = 256

_SYSTEM_PROMPT = """You are a fundamental equity analyst writing for retail investors with no \
finance background.

You receive a structured fundamental analysis of a company: a 0-10 overall score, four \
sub-scores (valuation, quality, growth, solvency) and supporting metrics. Turn it into a clear, \
concise explanation.

Rules:
- Write in SPANISH, in plain language a non-expert can follow.
- 3-5 sentences. Explain what the score means and the company's main strengths and weaknesses.
- Do NOT give investment advice nor buy/sell recommendations.
- Do NOT invent data beyond what is provided.
- Return plain text only (no markdown, no JSON)."""


class FundamentalError(Exception):
    """Base error for the fundamental analysis pipeline."""


class UnknownTickerError(FundamentalError):
    """Raised when the ticker does not match any listed symbol."""


class UniverseNotReadyError(FundamentalError):
    """Raised when the S&P 500 universe snapshot has not been built yet."""


class FundamentalAnalysisError(FundamentalError):
    """Raised when the engine cannot produce a usable score for the ticker."""


class _TickerExistence(Protocol):
    """Minimal interface for the ticker existence check (see sentiment.TickerValidator)."""

    async def exists(self, ticker: str) -> bool: ...


class UniverseProvider:
    """Loads and caches the S&P 500 universe snapshot in memory.

    The snapshot is an offline/scheduler-built CSV; it is never built on demand here (that would
    block a request for minutes), so a missing file raises UniverseNotReadyError.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Args:
        path: CSV path; defaults to the engine package location.
        """
        self._path = path or DEFAULT_UNIVERSE_PATH
        self._cache: pd.DataFrame | None = None

    def get(self) -> pd.DataFrame:
        """Return the universe DataFrame, reading the CSV once and caching it.

        Raises:
            UniverseNotReadyError: If the snapshot CSV does not exist yet.
        """
        if self._cache is None:
            if not self._path.exists():
                raise UniverseNotReadyError(
                    f"Fundamental universe snapshot not found at {self._path}"
                )
            self._cache = load_universe(self._path)
        return self._cache

    def invalidate(self) -> None:
        """Drop the in-memory snapshot so the next get() re-reads the CSV."""
        self._cache = None


def _build_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract a curated, frontend-friendly metrics dict from the engine result."""
    val = raw.get("valuation_detail") or {}
    qual = raw.get("quality_detail") or {}
    solv = raw.get("solvency_detail") or {}
    return {
        "scores": {
            "valoracion": raw.get("score_valoracion"),
            "calidad": raw.get("score_calidad"),
            "crecimiento": raw.get("score_crecimiento"),
            "solvencia": raw.get("score_solvencia"),
        },
        "sub_signals": raw.get("sub_signals", {}),
        "weights_used": raw.get("weights_used", {}),
        "sector": raw.get("sector"),
        "ratios": {
            "ebitda_yield": val.get("ebitda_yield"),
            "earnings_yield": val.get("earnings_yield"),
            "fcf_yield": val.get("fcf_yield"),
            "roe": qual.get("roe"),
            "roa": qual.get("roa"),
            "operating_margin": qual.get("operating_margin"),
            "gp_a": qual.get("gp_a"),
            "dn_ebitda": solv.get("dn_ebitda"),
            "current_ratio": solv.get("current_ratio"),
            "debt_to_equity": solv.get("debt_to_equity"),
        },
    }


class FundamentalService:
    """Orchestrates the fundamental analysis for a single ticker.

    Flow: ticker existence check → universe snapshot → engine scoring → LLM explanation (Spanish).
    """

    def __init__(
        self,
        universe_provider: UniverseProvider,
        llm_service: LLMService,
        ticker_validator: _TickerExistence,
        cache_ttl: int,
        analyze_fn: Callable[[str, pd.DataFrame], dict[str, Any]] = analyze_ticker,
    ) -> None:
        """Args:
        universe_provider: Provides the cached S&P 500 universe snapshot.
        llm_service: LLM provider used to render the analysis in Spanish.
        ticker_validator: Existence check for the requested ticker (fail-open).
        cache_ttl: Seconds a result stays cached per ticker.
        analyze_fn: Engine entry point; injectable for testing.
        """
        self._universe = universe_provider
        self._llm = llm_service
        self._ticker_validator = ticker_validator
        self._analyze_fn = analyze_fn
        self._cache: TTLCache[str, FundamentalResult] = TTLCache(
            maxsize=_CACHE_MAX_TICKERS, ttl=cache_ttl
        )

    async def analyze(self, ticker: str, force_refresh: bool = False) -> FundamentalResult:
        """Run the fundamental analysis for a stock ticker.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL').
            force_refresh: Skip the cached result and recompute.

        Returns:
            FundamentalResult with score, metrics, Spanish LLM analysis and cache timestamp.

        Raises:
            UnknownTickerError: If the ticker does not match any listed symbol.
            UniverseNotReadyError: If the universe snapshot has not been built yet.
            FundamentalAnalysisError: If the engine cannot produce a usable score.
        """
        if not force_refresh and ticker in self._cache:
            logger.info("Fundamental cache hit for %s", ticker)
            return self._cache[ticker]

        if not await self._ticker_validator.exists(ticker):
            raise UnknownTickerError(f"Ticker {ticker} does not match any listed symbol")

        universe_df = self._universe.get()

        raw = await asyncio.to_thread(self._analyze_fn, ticker, universe_df)

        score_final = raw.get("score_final")
        if score_final is None:
            raise FundamentalAnalysisError(
                f"Insufficient fundamental data to score ticker {ticker}"
            )

        summary = raw.get("fundamental_summary") or ""
        llm_analysis = await self._llm.complete(system_prompt=_SYSTEM_PROMPT, user_prompt=summary)

        result = FundamentalResult(
            score=float(score_final),
            metrics=_build_metrics(raw),
            llm_analysis=llm_analysis.strip(),
            cached_at=datetime.now(tz=UTC),
        )
        self._cache[ticker] = result
        return result
