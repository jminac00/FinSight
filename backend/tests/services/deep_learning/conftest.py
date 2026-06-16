import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_ohlc():
    """Factory for synthetic, OHLC-coherent daily price frames (date-indexed)."""

    def _make(n: int = 400, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.0, 0.02, n)
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = close * (1.0 + rng.normal(0.0, 0.005, n))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
        dates = pd.bdate_range("2015-01-01", periods=n)
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)

    return _make
