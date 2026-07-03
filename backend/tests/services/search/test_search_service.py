"""Unit tests for the symbol search service (Finnhub-backed, cached)."""

from unittest.mock import MagicMock

from app.services.search.service import SymbolSearchService


def _lookup_response() -> dict:
    return {
        "count": 3,
        "result": [
            {
                "description": "APPLE HOSPITALITY REIT INC",
                "displaySymbol": "APLE",
                "symbol": "APLE",
                "type": "Common Stock",
            },
            {
                "description": "APPLE INC",
                "displaySymbol": "AAPL",
                "symbol": "AAPL",
                "type": "Common Stock",
            },
            {
                "description": "no symbol, should be dropped",
                "displaySymbol": "",
                "symbol": "",
                "type": "",
            },
        ],
    }


def _make_service(response: dict | None = None, ttl: int = 3600):
    client = MagicMock()
    client.symbol_lookup.return_value = response if response is not None else _lookup_response()
    return SymbolSearchService(client=client, cache_ttl=ttl), client


async def test_search_maps_finnhub_fields():
    service, _ = _make_service()

    results = await service.search("apple")

    by_symbol = {r.symbol: r for r in results}
    assert "AAPL" in by_symbol
    assert by_symbol["AAPL"].description == "APPLE INC"
    assert by_symbol["AAPL"].type == "Common Stock"
    assert by_symbol["AAPL"].display_symbol == "AAPL"


async def test_search_drops_entries_without_symbol():
    service, _ = _make_service()

    results = await service.search("apple")

    assert all(r.symbol for r in results)


async def test_exact_symbol_match_is_ranked_first():
    service, _ = _make_service()

    results = await service.search("AAPL")

    assert results[0].symbol == "AAPL"


async def test_empty_query_skips_the_lookup():
    service, client = _make_service()

    results = await service.search("   ")

    assert results == []
    client.symbol_lookup.assert_not_called()


async def test_repeated_query_is_cached_case_insensitively():
    service, client = _make_service()

    await service.search("apple")
    await service.search("APPLE")

    client.symbol_lookup.assert_called_once()


async def test_provider_failure_returns_empty_list():
    service, client = _make_service()
    client.symbol_lookup.side_effect = RuntimeError("network down")

    results = await service.search("apple")

    assert results == []
