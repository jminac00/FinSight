import logging

from app.models.fundamental import FundamentalResult

logger = logging.getLogger(__name__)


class FundamentalService:
    """Fetches fundamental metrics and generates a natural-language analysis via LLM.

    Flow: Yahoo Finance / Finnhub → metrics dict → LLM explanation in Spanish.
    Results are cached until the next market session (TTL = CACHE_TTL_FUNDAMENTAL).
    """

    async def analyze(self, ticker: str) -> FundamentalResult:
        """Run the fundamental analysis for a stock ticker.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL').

        Returns:
            FundamentalResult with score, raw metrics dict, LLM analysis
            in Spanish and the cache timestamp.
        """
        # TODO: fetch fundamentals from yfinance (PER, ROE, EV/EBITDA, margins, FCF, debt)
        # TODO: build system + user prompt with the raw metrics
        # TODO: call get_llm_service().complete(system_prompt, user_prompt)
        # TODO: compute composite score (0-10) from raw metrics
        # TODO: return FundamentalResult; cache result with CACHE_TTL_FUNDAMENTAL
        raise NotImplementedError("FundamentalService.analyze is not yet implemented")
