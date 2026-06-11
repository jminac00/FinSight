"""Factory for the configured news provider chain."""

import logging
from functools import lru_cache

from app.core.config import get_settings
from app.core.finnhub import get_finnhub_client
from app.services.sentiment.news.base import NewsProvider
from app.services.sentiment.news.chain import NewsProviderChain
from app.services.sentiment.news.finnhub_provider import FinnhubNewsProvider
from app.services.sentiment.news.newsapi_provider import NewsAPIProvider

logger = logging.getLogger(__name__)


@lru_cache
def get_news_provider() -> NewsProvider:
    """Return the singleton news provider chain ordered via NEWS_PROVIDER."""
    settings = get_settings()

    finnhub_provider = FinnhubNewsProvider(
        client=get_finnhub_client(),
        max_articles=settings.max_news_articles,
    )
    newsapi_provider = NewsAPIProvider(
        api_key=settings.newsapi_key,
        max_articles=settings.max_news_articles,
    )

    if settings.news_provider == "finnhub":
        providers: list[NewsProvider] = [finnhub_provider, newsapi_provider]
    elif settings.news_provider == "newsapi":
        providers = [newsapi_provider, finnhub_provider]
    else:
        raise ValueError(
            f"Unknown NEWS_PROVIDER: {settings.news_provider!r}. Must be 'finnhub' or 'newsapi'."
        )

    logger.info("News provider chain: %s", " → ".join(type(p).__name__ for p in providers))
    return NewsProviderChain(providers)
