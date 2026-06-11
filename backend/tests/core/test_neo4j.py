"""Tests for the managed Neo4j driver lifecycle."""

from unittest.mock import MagicMock, patch

from app.core import neo4j as neo4j_core


def test_get_neo4j_driver_is_singleton():
    neo4j_core.get_neo4j_driver.cache_clear()
    fake_driver = MagicMock()

    with patch.object(neo4j_core.GraphDatabase, "driver", return_value=fake_driver) as factory:
        first = neo4j_core.get_neo4j_driver()
        second = neo4j_core.get_neo4j_driver()

    assert first is second is fake_driver
    factory.assert_called_once()
    neo4j_core.close_neo4j_driver()


def test_close_neo4j_driver_closes_and_resets():
    neo4j_core.get_neo4j_driver.cache_clear()
    fake_driver = MagicMock()

    with patch.object(neo4j_core.GraphDatabase, "driver", return_value=fake_driver):
        neo4j_core.get_neo4j_driver()
        neo4j_core.close_neo4j_driver()

    fake_driver.close.assert_called_once()
    assert neo4j_core.get_neo4j_driver.cache_info().currsize == 0


def test_close_neo4j_driver_noop_when_never_created():
    neo4j_core.get_neo4j_driver.cache_clear()
    # Must not raise nor create a driver as a side effect.
    neo4j_core.close_neo4j_driver()
    assert neo4j_core.get_neo4j_driver.cache_info().currsize == 0
