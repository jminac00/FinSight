"""Fallback chain over news providers (Chain of Responsibility).

Provider coverage depends on the requested ticker (e.g. Finnhub only covers
North American companies), so the chain resolves coverage gaps at runtime:
the first provider returning a non-empty result wins. See ADR-0005.
"""

import logging

from app.services.sentiment.news.base import (
    NewsArticle,
    NewsProvider,
    NewsProviderError,
    NewsQuotaError,
)

logger = logging.getLogger(__name__)


class NewsProviderChain(NewsProvider):
    """Tries each provider in order and returns the first non-empty result."""

    def __init__(self, providers: list[NewsProvider]) -> None:
        """Args:
        providers: Ordered chain links; the first one is the primary provider.
        """
        self._providers = providers

    async def fetch_news(self, ticker: str) -> list[NewsArticle]:
        """Return news from the first provider in the chain with results.

        An empty result (no coverage for the ticker) or a provider failure
        falls through to the next link. If no link yields articles, a quota
        error encountered along the way is propagated (the caller can signal
        "try again later"); otherwise an empty list is returned.
        """
        quota_error: NewsQuotaError | None = None
        for provider in self._providers:
            name = type(provider).__name__
            try:
                articles = await provider.fetch_news(ticker)
            except NewsQuotaError as exc:
                logger.warning("%s quota exhausted for %s, trying next provider", name, ticker)
                quota_error = exc
                continue
            except NewsProviderError as exc:
                logger.warning("%s failed for %s (%s), trying next provider", name, ticker, exc)
                continue
            if articles:
                return articles
            logger.info("%s returned no articles for %s, trying next provider", name, ticker)

        if quota_error is not None:
            raise quota_error
        return []
