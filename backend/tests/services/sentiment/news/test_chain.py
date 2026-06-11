"""Tests for the news provider fallback chain."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.sentiment.news.base import (
    NewsArticle,
    NewsProviderError,
    NewsQuotaError,
)
from app.services.sentiment.news.chain import NewsProviderChain

_ARTICLE = NewsArticle(
    title="Title",
    url="https://example.com/news/1",
    source="Source",
    published_at="2026-06-11T12:00:00Z",
    description="Description",
)


def _provider(result=None, error: Exception | None = None) -> MagicMock:
    provider = MagicMock()
    if error is not None:
        provider.fetch_news = AsyncMock(side_effect=error)
    else:
        provider.fetch_news = AsyncMock(return_value=result if result is not None else [])
    return provider


async def test_primary_with_results_does_not_touch_fallback():
    primary = _provider(result=[_ARTICLE])
    fallback = _provider(result=[_ARTICLE])
    chain = NewsProviderChain([primary, fallback])

    articles = await chain.fetch_news("AAPL")

    assert articles == [_ARTICLE]
    primary.fetch_news.assert_awaited_once_with("AAPL")
    fallback.fetch_news.assert_not_awaited()


async def test_empty_primary_falls_through_to_next():
    primary = _provider(result=[])  # e.g. non-US ticker not covered by Finnhub
    fallback = _provider(result=[_ARTICLE])
    chain = NewsProviderChain([primary, fallback])

    articles = await chain.fetch_news("ASML")

    assert articles == [_ARTICLE]
    fallback.fetch_news.assert_awaited_once_with("ASML")


async def test_provider_error_falls_through_to_next():
    primary = _provider(error=NewsProviderError("boom"))
    fallback = _provider(result=[_ARTICLE])
    chain = NewsProviderChain([primary, fallback])

    articles = await chain.fetch_news("AAPL")

    assert articles == [_ARTICLE]


async def test_quota_error_falls_through_to_next():
    primary = _provider(error=NewsQuotaError("rate limited"))
    fallback = _provider(result=[_ARTICLE])
    chain = NewsProviderChain([primary, fallback])

    articles = await chain.fetch_news("AAPL")

    assert articles == [_ARTICLE]


async def test_all_empty_returns_empty_list():
    chain = NewsProviderChain([_provider(result=[]), _provider(result=[])])

    assert await chain.fetch_news("ZZZZ") == []


async def test_quota_error_propagated_when_no_link_succeeds():
    primary = _provider(error=NewsQuotaError("quota exhausted"))
    fallback = _provider(result=[])
    chain = NewsProviderChain([primary, fallback])

    with pytest.raises(NewsQuotaError):
        await chain.fetch_news("AAPL")


async def test_all_provider_errors_without_quota_returns_empty():
    chain = NewsProviderChain(
        [_provider(error=NewsProviderError("a")), _provider(error=NewsProviderError("b"))]
    )

    assert await chain.fetch_news("AAPL") == []
