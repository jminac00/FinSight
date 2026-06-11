"""Tests for the news provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.sentiment.news import factory
from app.services.sentiment.news.chain import NewsProviderChain
from app.services.sentiment.news.finnhub_provider import FinnhubNewsProvider
from app.services.sentiment.news.newsapi_provider import NewsAPIProvider


def _fake_settings(news_provider: str) -> MagicMock:
    return MagicMock(
        news_provider=news_provider,
        newsapi_key="test-key",
        max_news_articles=10,
    )


def _build(news_provider: str) -> NewsProviderChain:
    factory.get_news_provider.cache_clear()
    with (
        patch.object(factory, "get_settings", return_value=_fake_settings(news_provider)),
        patch.object(factory, "get_finnhub_client", return_value=MagicMock()),
    ):
        provider = factory.get_news_provider()
    factory.get_news_provider.cache_clear()
    return provider


def test_finnhub_head_by_default():
    chain = _build("finnhub")

    assert isinstance(chain, NewsProviderChain)
    assert isinstance(chain._providers[0], FinnhubNewsProvider)
    assert isinstance(chain._providers[1], NewsAPIProvider)


def test_newsapi_head_when_configured():
    chain = _build("newsapi")

    assert isinstance(chain._providers[0], NewsAPIProvider)
    assert isinstance(chain._providers[1], FinnhubNewsProvider)


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError):
        _build("bogus")


def test_factory_returns_singleton():
    factory.get_news_provider.cache_clear()
    with (
        patch.object(factory, "get_settings", return_value=_fake_settings("finnhub")),
        patch.object(factory, "get_finnhub_client", return_value=MagicMock()),
    ):
        first = factory.get_news_provider()
        second = factory.get_news_provider()
    factory.get_news_provider.cache_clear()

    assert first is second
