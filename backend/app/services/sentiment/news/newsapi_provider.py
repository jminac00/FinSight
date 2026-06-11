"""NewsAPI.org provider: keyword search over the /v2/everything endpoint.

Free-tier limitations: articles are delayed 24 hours and the quota is
100 requests/day, so this provider acts as the universal fallback behind
Finnhub in the provider chain.
"""

import logging

import httpx

from app.services.sentiment.news.base import (
    NewsArticle,
    NewsProvider,
    NewsProviderError,
    NewsQuotaError,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://newsapi.org/v2/everything"
_TIMEOUT_SECONDS = 15.0


class NewsAPIProvider(NewsProvider):
    """Fetches recent English-language news for a ticker via NewsAPI.org."""

    def __init__(
        self,
        api_key: str,
        max_articles: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args:
        api_key: NewsAPI key (from settings, never hardcoded).
        max_articles: Maximum number of articles to return per query.
        transport: Optional httpx transport, injectable for testing.
        """
        self._api_key = api_key
        self._max_articles = max_articles
        self._transport = transport

    async def fetch_news(self, ticker: str) -> list[NewsArticle]:
        """Return the most recent news articles mentioning the ticker."""
        params = {
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": str(self._max_articles),
        }
        headers = {"X-Api-Key": self._api_key}

        async with httpx.AsyncClient(transport=self._transport, timeout=_TIMEOUT_SECONDS) as client:
            try:
                response = await client.get(_BASE_URL, params=params, headers=headers)
            except httpx.HTTPError as exc:
                raise NewsProviderError(f"NewsAPI request failed: {exc}") from exc

        if response.status_code == 429:
            logger.warning("NewsAPI daily quota exhausted for ticker %s", ticker)
            raise NewsQuotaError("NewsAPI daily request quota exhausted")
        if response.status_code != 200:
            raise NewsProviderError(
                f"NewsAPI returned HTTP {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise NewsProviderError("NewsAPI returned a non-JSON response") from exc

        articles = payload.get("articles", [])[: self._max_articles]
        return [
            NewsArticle(
                title=a.get("title") or "",
                url=a.get("url") or "",
                source=(a.get("source") or {}).get("name") or "",
                published_at=a.get("publishedAt") or "",
                description=a.get("description") or "",
            )
            for a in articles
        ]
