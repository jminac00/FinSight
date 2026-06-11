"""Tests for the NewsAPI client. All HTTP traffic is mocked via httpx.MockTransport."""

import httpx
import pytest

from app.services.sentiment.news_client import (
    NewsAPIClient,
    NewsAPIError,
    NewsAPIQuotaError,
)


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


def _make_client(handler, max_articles: int = 10) -> NewsAPIClient:
    transport = httpx.MockTransport(handler)
    return NewsAPIClient(api_key="test-key", max_articles=max_articles, transport=transport)


async def test_fetch_news_returns_articles():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return _ok_response([_article(1), _article(2)])

    client = _make_client(handler)
    articles = await client.fetch_news("AAPL")

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

    client = _make_client(handler, max_articles=3)
    articles = await client.fetch_news("AAPL")

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

    client = _make_client(handler)
    with pytest.raises(NewsAPIQuotaError):
        await client.fetch_news("AAPL")


async def test_fetch_news_http_error_raises_newsapi_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"status": "error", "code": "apiKeyInvalid", "message": "Bad key."},
        )

    client = _make_client(handler)
    with pytest.raises(NewsAPIError):
        await client.fetch_news("AAPL")


async def test_fetch_news_network_error_raises_newsapi_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _make_client(handler)
    with pytest.raises(NewsAPIError):
        await client.fetch_news("AAPL")


async def test_fetch_news_empty_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response([])

    client = _make_client(handler)
    articles = await client.fetch_news("AAPL")

    assert articles == []


async def test_fetch_news_invalid_json_raises_newsapi_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = _make_client(handler)
    with pytest.raises(NewsAPIError):
        await client.fetch_news("AAPL")
