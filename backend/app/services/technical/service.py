"""Technical analysis service (multi-universe).

Wraps the vendored technical engine (see engine/): it resolves the reference universe for a ticker
(S&P 500 or MSCI World, per the requested mode), loads the frozen universe snapshot, normalizes the
ticker against it and turns the deterministic summary into a natural-language analysis in Spanish
via the LLM. Results are cached per (universe, ticker) with TTL = CACHE_TTL_TECHNICAL.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from cachetools import TTLCache

from app.llm.base import LLMService
from app.models.technical import TechnicalBlockScores, TechnicalResult
from app.services.technical.engine.technical_score import compute_technical_score
from app.services.technical.engine.universe_manager import (
    TechnicalUniverseManager,
    TechnicalUniverseNotReadyError,
    get_technical_universe_manager,
)

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = 512

_SYSTEM_PROMPT = """You are a technical analyst writing for retail investors with no finance \
background.

You receive a structured technical analysis of a stock: a 0-10 overall score, a signal \
(alcista/bajista/neutral) and four blocks (momentum, trend, risk/stability, confirmation) with \
their own scores and notes. Turn it into a clear, concise explanation.

Rules:
- Write in SPANISH, in plain language a non-expert can follow.
- 3-5 sentences. Explain what the score and signal mean and the main technical strengths and \
weaknesses.
- Do NOT give investment advice nor buy/sell recommendations.
- Do NOT invent data beyond what is provided.
- Return plain text only (no markdown, no JSON)."""


class TechnicalError(Exception):
    """Base error for the technical analysis pipeline."""


class TechnicalAnalysisError(TechnicalError):
    """Raised when the engine cannot produce a usable score for the ticker."""


def _build_indicators(raw: dict[str, Any], universe: str) -> dict[str, Any]:
    """Extract a frontend-friendly indicators dict from the engine result."""
    return {
        "universe": universe,
        "data_completeness": raw.get("data_completeness"),
        "score_reliable": raw.get("score_reliable"),
        "weights": raw.get("weights"),
        "effective_weights": raw.get("effective_weights"),
        "blocks": raw.get("blocks"),
        "errors": raw.get("errors"),
    }


def _build_user_prompt(raw: dict[str, Any]) -> str:
    """Compose the LLM input from the overall summary and each block's summary."""
    block_summaries = [
        block["summary"]
        for block in (raw.get("blocks") or {}).values()
        if block and block.get("summary")
    ]
    return raw.get("summary", "") + "\n\n" + "\n".join(block_summaries)


class TechnicalService:
    """Orchestrates the technical analysis for a single ticker across universes.

    Flow: resolve universe (by mode) → load frozen snapshot → engine scoring → LLM explanation in
    Spanish. Results are cached per (universe, ticker).
    """

    def __init__(
        self,
        llm_service: LLMService,
        cache_ttl: int,
        manager: TechnicalUniverseManager | None = None,
        analyze_fn: Callable[..., dict[str, Any]] = compute_technical_score,
    ) -> None:
        """Args:
        llm_service: LLM provider used to render the analysis in Spanish.
        cache_ttl: Seconds a result stays cached per (universe, ticker).
        manager: Technical universe manager (defaults to the shared instance).
        analyze_fn: Engine entry point; injectable for testing.
        """
        self._llm = llm_service
        self._manager = manager or get_technical_universe_manager()
        self._analyze_fn = analyze_fn
        self._cache: TTLCache[tuple[str, str], TechnicalResult] = TTLCache(
            maxsize=_CACHE_MAX_ENTRIES, ttl=cache_ttl
        )

    async def analyze(
        self, ticker: str, mode: str = "auto", force_refresh: bool = False
    ) -> TechnicalResult:
        """Run the technical analysis for a stock ticker.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL', 'ASML.AS').
            mode: 'auto' (S&P 500 if in the index, else MSCI World), 'domestic' (S&P 500) or
                'global' (MSCI World).
            force_refresh: Skip the cached result and recompute.

        Returns:
            TechnicalResult with score, signal, per-block scores, indicators, Spanish LLM analysis
            and the calculation timestamp.

        Raises:
            TechnicalUniverseNotReadyError: If the resolved universe snapshot is unavailable.
            TechnicalAnalysisError: If the engine cannot produce a usable score.
        """
        universe = self._manager.resolve_universe(ticker, mode)
        cache_key = (universe, ticker)
        if not force_refresh and cache_key in self._cache:
            logger.info("Technical cache hit for %s (%s)", ticker, universe)
            return self._cache[cache_key]

        payload = self._manager.load(universe)  # raises TechnicalUniverseNotReadyError

        raw = await asyncio.to_thread(
            self._analyze_fn,
            ticker,
            universe,
            universe_closes=payload["closes"],
            universe_ohlcv=payload["ohlcv"],
        )

        score = raw.get("technical_score")
        if score is None:
            raise TechnicalAnalysisError(f"Insufficient technical data to score ticker {ticker}")

        llm_analysis = await self._llm.complete(
            system_prompt=_SYSTEM_PROMPT, user_prompt=_build_user_prompt(raw)
        )

        scores = raw.get("individual_scores", {})
        result = TechnicalResult(
            score=float(score),
            signal=raw["signal"],
            block_scores=TechnicalBlockScores(
                momentum=scores.get("momentum"),
                trend=scores.get("trend"),
                risk_stability=scores.get("risk_stability"),
                confirmation=scores.get("confirmation"),
            ),
            indicators=_build_indicators(raw, universe),
            llm_analysis=llm_analysis.strip(),
            calculated_at=datetime.now(tz=UTC),
        )
        self._cache[cache_key] = result
        return result


__all__ = ["TechnicalAnalysisError", "TechnicalService", "TechnicalUniverseNotReadyError"]
