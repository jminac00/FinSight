"""Company/symbol search backed by Finnhub symbol_lookup, with TTL caching.

symbol_lookup is a fuzzy search that matches both ticker symbols and company
names, which lets a user find a stock by either. Results are cached per query
to protect the shared Finnhub quota during autocomplete.
"""

import asyncio
import logging

import finnhub
from cachetools import TTLCache

from app.models.search import SymbolMatch

logger = logging.getLogger(__name__)

_CACHE_MAX_QUERIES = 512
_MAX_RESULTS = 15


class SymbolSearchService:
    """Resolve company names or tickers to listed symbols via Finnhub."""

    def __init__(
        self,
        client: finnhub.Client,
        cache_ttl: int,
        max_results: int = _MAX_RESULTS,
    ) -> None:
        """Args:
        client: Shared Finnhub client (see app.core.finnhub).
        cache_ttl: Seconds to cache each query's results.
        max_results: Maximum number of matches to return.
        """
        self._client = client
        self._cache: TTLCache[str, list[SymbolMatch]] = TTLCache(
            maxsize=_CACHE_MAX_QUERIES, ttl=cache_ttl
        )
        self._max_results = max_results

    async def search(self, query: str) -> list[SymbolMatch]:
        """Return listed symbols matching the query by ticker or company name.

        Fail-soft: an empty query or a provider error yields an empty list so
        the autocomplete degrades gracefully instead of surfacing an error.
        """
        normalized = query.strip()
        if not normalized:
            return []

        key = normalized.upper()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            response = await asyncio.to_thread(self._client.symbol_lookup, normalized)
        except Exception as exc:  # noqa: BLE001 - provider errors must not break autocomplete
            logger.warning("Symbol lookup failed for %r (%s)", normalized, exc)
            return []

        matches = self._rank(self._map(response), key)
        self._cache[key] = matches
        return matches

    @staticmethod
    def _map(response: dict) -> list[SymbolMatch]:
        raw = response.get("result") or []
        return [
            SymbolMatch(
                symbol=item.get("symbol", ""),
                description=item.get("description", ""),
                type=item.get("type", ""),
                display_symbol=item.get("displaySymbol") or item.get("symbol", ""),
            )
            for item in raw
            if item.get("symbol")
        ]

    def _rank(self, matches: list[SymbolMatch], key: str) -> list[SymbolMatch]:
        # Surface exact and prefix ticker matches first for a natural autocomplete order.
        matches.sort(key=lambda m: (m.symbol.upper() != key, not m.symbol.upper().startswith(key)))
        return matches[: self._max_results]
