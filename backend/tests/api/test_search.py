"""Contract tests for the symbol search endpoint. The search service is mocked."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.search import get_symbol_search_service
from app.main import app
from app.models.search import SymbolMatch

_MATCHES = [
    SymbolMatch(symbol="AAPL", description="APPLE INC", type="Common Stock", display_symbol="AAPL"),
    SymbolMatch(
        symbol="APLE",
        description="APPLE HOSPITALITY REIT INC",
        type="Common Stock",
        display_symbol="APLE",
    ),
]


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.search = AsyncMock(return_value=_MATCHES)
    app.dependency_overrides[get_symbol_search_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_symbol_search_service, None)


def test_search_returns_matches(client, mock_service):
    response = client.get("/api/v1/search?q=apple")

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "apple"
    assert body["results"][0]["symbol"] == "AAPL"
    assert body["results"][0]["description"] == "APPLE INC"
    mock_service.search.assert_awaited_once_with("apple")


def test_search_without_query_returns_422(client, mock_service):
    response = client.get("/api/v1/search")

    assert response.status_code == 422
    mock_service.search.assert_not_awaited()


def test_search_with_empty_query_returns_422(client, mock_service):
    response = client.get("/api/v1/search?q=")

    assert response.status_code == 422
    mock_service.search.assert_not_awaited()
