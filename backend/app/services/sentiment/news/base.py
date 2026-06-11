"""Abstract interface and shared types for news providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class NewsProviderError(Exception):
    """Raised when a news provider returns an error or is unreachable."""


class NewsQuotaError(NewsProviderError):
    """Raised when a news provider's request quota or rate limit is exhausted."""


@dataclass
class NewsArticle:
    """A news article in provider-agnostic form."""

    title: str
    url: str
    source: str
    published_at: str
    description: str


class NewsProvider(ABC):
    """Abstract interface for all news providers.

    The sentiment module fetches news exclusively through this interface so
    that adding or reordering providers requires no changes in business logic.
    """

    @abstractmethod
    async def fetch_news(self, ticker: str) -> list[NewsArticle]:
        """Return the most recent news articles for the ticker, newest first.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL').

        Returns:
            Up to the provider's configured maximum of articles; empty list
            when the provider has no coverage or no recent news for the ticker.

        Raises:
            NewsQuotaError: If the provider's quota or rate limit is exhausted.
            NewsProviderError: On any other provider failure.
        """
