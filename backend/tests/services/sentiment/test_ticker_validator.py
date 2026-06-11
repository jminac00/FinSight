"""Tests for ticker existence validation via Finnhub symbol lookup."""

from unittest.mock import MagicMock

from app.services.sentiment.ticker_validator import TickerValidator


def _lookup_response(symbols: list[str]) -> dict:
    return {
        "count": len(symbols),
        "result": [
            {"symbol": s, "displaySymbol": s, "description": f"{s} Inc", "type": "Common Stock"}
            for s in symbols
        ],
    }


async def test_exact_symbol_match_exists():
    client = MagicMock()
    client.symbol_lookup.return_value = _lookup_response(["AAPL", "AAPL.MX"])
    validator = TickerValidator(client=client)

    assert await validator.exists("AAPL") is True


async def test_fuzzy_only_matches_do_not_exist():
    client = MagicMock()
    # symbol_lookup is a fuzzy search: "ZZZZ" may return unrelated near-matches
    client.symbol_lookup.return_value = _lookup_response(["ZZZ.BE", "ZZZAP"])
    validator = TickerValidator(client=client)

    assert await validator.exists("ZZZZ") is False


async def test_no_results_do_not_exist():
    client = MagicMock()
    client.symbol_lookup.return_value = {"count": 0, "result": []}
    validator = TickerValidator(client=client)

    assert await validator.exists("QQXYZ") is False


async def test_lookup_failure_fails_open():
    client = MagicMock()
    client.symbol_lookup.side_effect = ConnectionError("network down")
    validator = TickerValidator(client=client)

    assert await validator.exists("AAPL") is True


async def test_result_is_cached_per_ticker():
    client = MagicMock()
    client.symbol_lookup.return_value = _lookup_response(["AAPL"])
    validator = TickerValidator(client=client)

    await validator.exists("AAPL")
    await validator.exists("AAPL")

    client.symbol_lookup.assert_called_once()


async def test_fail_open_result_is_not_cached():
    client = MagicMock()
    client.symbol_lookup.side_effect = [ConnectionError("down"), _lookup_response([])]
    validator = TickerValidator(client=client)

    assert await validator.exists("QQXYZ") is True  # fail-open
    assert await validator.exists("QQXYZ") is False  # retried, real answer cached now
    assert client.symbol_lookup.call_count == 2
