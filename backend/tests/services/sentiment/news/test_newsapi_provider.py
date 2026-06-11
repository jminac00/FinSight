"""Tests for the NewsAPI provider. All HTTP traffic is mocked via httpx.MockTransport."""

import httpx
import pytest

from app.services.sentiment.news.base import NewsProviderError, NewsQuotaError
from app.services.sentiment.news.newsapi_provider import NewsAPIProvider


def _article(i: int) -> dict:
    return {
        "source": {"id": None, "name": f"Source {i}"},
        "author": "Author",
        "title": f"Title {i}",
        "description": f"Description {i}",
        "url": f"https://example.com/news/{i}",
        "urlToImage": None,
        "publishedAt": f"2026-06-1{i}T12:00:00Z",
        "content": f"Content {i}",
    }


def _ok_response(articles: list[dict]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"status": "ok", "totalResults": len(articles), "articles": articles},
    )


def _make_provider(handler, max_articles: int = 10) -> NewsAPIProvider:
    transport = httpx.MockTransport(handler)
    return NewsAPIProvider(api_key="test-key", max_articles=max_articles, transport=transport)


async def test_fetch_news_returns_articles():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return _ok_response([_article(1), _article(2)])

    provider = _make_provider(handler)
    articles = await provider.fetch_news("AAPL")

    assert len(articles) == 2
    first = articles[0]
    assert first.title == "Title 1"
    assert first.url == "https://example.com/news/1"
    assert first.source == "Source 1"
    assert first.published_at == "2026-06-11T12:00:00Z"
    assert first.description == "Description 1"

    request = captured["request"]
    assert request.headers["X-Api-Key"] == "test-key"
    params = dict(httpx.QueryParams(request.url.query))
    assert params["q"] == "AAPL"
    assert params["pageSize"] == "10"


async def test_fetch_news_caps_articles_to_max():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response([_article(i) for i in range(1, 6)])

    provider = _make_provider(handler, max_articles=3)
    articles = await provider.fetch_news("AAPL")

    assert len(articles) == 3


async def test_fetch_news_quota_exhausted_raises_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "status": "error",
                "code": "rateLimited",
                "message": "You have made too many requests recently.",
            },
        )

    provider = _make_provider(handler)
    with pytest.raises(NewsQuotaError):
        await provider.fetch_news("AAPL")


async def test_fetch_news_http_error_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"status": "error", "code": "apiKeyInvalid", "message": "Bad key."},
        )

    provider = _make_provider(handler)
    with pytest.raises(NewsProviderError):
        await provider.fetch_news("AAPL")


async def test_fetch_news_network_error_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = _make_provider(handler)
    with pytest.raises(NewsProviderError):
        await provider.fetch_news("AAPL")


async def test_fetch_news_empty_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response([])

    provider = _make_provider(handler)
    articles = await provider.fetch_news("AAPL")

    assert articles == []


async def test_fetch_news_invalid_json_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    provider = _make_provider(handler)
    with pytest.raises(NewsProviderError):
        await provider.fetch_news("AAPL")
