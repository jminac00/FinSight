"""Ticker existence validation via Finnhub symbol lookup.

Prevents nonexistent tickers from reaching keyword-based news search, which
could otherwise feed the LLM random articles and produce a plausible-looking
analysis of a company that does not exist.
"""

import asyncio
import logging

import finnhub

logger = logging.getLogger(__name__)


class TickerValidator:
    """Checks that a ticker exists on a known exchange, with in-memory caching."""

    def __init__(self, client: finnhub.Client) -> None:
        """Args:
        client: Shared Finnhub client (see app.core.finnhub).
        """
        self._client = client
        self._cache: dict[str, bool] = {}

    async def exists(self, ticker: str) -> bool:
        """Return True if the ticker matches a listed symbol exactly.

        Fail-open: if the lookup itself fails (rate limit, network), the
        analysis proceeds — validation protects quality, not availability.
        Fail-open results are not cached so the next call retries the lookup.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        try:
            response = await asyncio.to_thread(self._client.symbol_lookup, ticker)
        except Exception as exc:
            logger.warning("Ticker lookup failed for %s (%s) — failing open", ticker, exc)
            return True

        # symbol_lookup is a fuzzy search; require an exact symbol match.
        found = any(item.get("symbol") == ticker for item in response.get("result") or [])
        self._cache[ticker] = found
        if not found:
            logger.info("Ticker %s not found in symbol lookup", ticker)
        return found
