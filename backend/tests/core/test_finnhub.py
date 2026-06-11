"""Tests for the shared Finnhub client singleton."""

from unittest.mock import MagicMock, patch

from app.core import finnhub as finnhub_core


def test_get_finnhub_client_is_singleton():
    finnhub_core.get_finnhub_client.cache_clear()
    fake_client = MagicMock()

    with patch.object(finnhub_core.finnhub, "Client", return_value=fake_client) as factory:
        first = finnhub_core.get_finnhub_client()
        second = finnhub_core.get_finnhub_client()

    assert first is second is fake_client
    factory.assert_called_once()
    finnhub_core.get_finnhub_client.cache_clear()
