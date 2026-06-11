"""Finnhub provider: per-ticker company news with no publication delay.

Free tier allows 60 requests/min but only covers North American companies,
hence the fallback chain behind it.
"""

import asyncio
import logging
from datetime import UTC, date, datetime, timedelta

import finnhub

from app.services.sentiment.news.base import (
    NewsArticle,
    NewsProvider,
    NewsProviderError,
    NewsQuotaError,
)

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 7


class FinnhubNewsProvider(NewsProvider):
    """Fetches recent company news for a ticker via the Finnhub API."""

    def __init__(self, client: finnhub.Client, max_articles: int) -> None:
        """Args:
        client: Shared Finnhub client (see app.core.finnhub).
        max_articles: Maximum number of articles to return per query.
        """
        self._client = client
        self._max_articles = max_articles

    async def fetch_news(self, ticker: str) -> list[NewsArticle]:
        """Return company news from the last week, newest first.

        The finnhub client is synchronous, so the call runs in a worker thread.
        """
        today = date.today()
        try:
            raw = await asyncio.to_thread(
                self._client.company_news,
                ticker,
                _from=(today - timedelta(days=_LOOKBACK_DAYS)).isoformat(),
                to=today.isoformat(),
            )
        except finnhub.FinnhubAPIException as exc:
            if exc.status_code == 429:
                logger.warning("Finnhub rate limit exhausted for ticker %s", ticker)
                raise NewsQuotaError("Finnhub rate limit exhausted") from exc
            raise NewsProviderError(f"Finnhub returned an error: {exc}") from exc
        except Exception as exc:
            raise NewsProviderError(f"Finnhub request failed: {exc}") from exc

        return [
            NewsArticle(
                title=a.get("headline") or "",
                url=a.get("url") or "",
                source=a.get("source") or "",
                published_at=_unix_to_iso(a.get("datetime")),
                description=a.get("summary") or "",
            )
            for a in raw[: self._max_articles]
        ]


def _unix_to_iso(timestamp: int | None) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
