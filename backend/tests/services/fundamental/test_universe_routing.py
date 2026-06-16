"""Tests for universe routing (auto/domestic/global) without touching the network or CSVs."""

import pandas as pd

from app.services.fundamental.engine.universe_manager import UniverseConfig, UniverseManager


def _manager_with_sp500(tickers: list[str]) -> UniverseManager:
    """Build a manager with a pre-seeded S&P 500 config (skips load_universe / CSV)."""
    manager = UniverseManager()
    df = pd.DataFrame({"ticker": tickers})
    df.attrs["universe_name"] = "sp500"
    manager._cache["sp500"] = UniverseConfig(
        name="sp500",
        tickers=tickers,
        stats={},
        ratio_profile="sp500_validated",
        fx_convert=False,
        universe_df=df,
    )
    return manager


def test_auto_routes_sp500_member_to_domestic():
    manager = _manager_with_sp500(["AAPL", "MSFT", "BRK-B"])
    assert manager.resolve_universe("AAPL", "auto") == "sp500"


def test_auto_routes_non_member_to_global():
    manager = _manager_with_sp500(["AAPL", "MSFT"])
    assert manager.resolve_universe("ASML.AS", "auto") == "msci_world"


def test_global_mode_always_msci_world():
    manager = _manager_with_sp500(["AAPL"])
    assert manager.resolve_universe("AAPL", "global") == "msci_world"


def test_domestic_mode_always_sp500():
    manager = _manager_with_sp500(["AAPL"])
    assert manager.resolve_universe("ASML.AS", "domestic") == "sp500"


def test_resolve_portfolio_mixed_auto_forces_global():
    manager = _manager_with_sp500(["AAPL", "MSFT"])
    routed = manager.resolve_portfolio(["AAPL", "ASML.AS"], "auto")
    assert set(routed.values()) == {"msci_world"}


def test_resolve_portfolio_all_domestic_stays_sp500():
    manager = _manager_with_sp500(["AAPL", "MSFT"])
    routed = manager.resolve_portfolio(["AAPL", "MSFT"], "auto")
    assert routed == {"AAPL": "sp500", "MSFT": "sp500"}
