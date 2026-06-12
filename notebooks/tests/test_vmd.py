"""Tests for benchmark.vmd: causal decomposition and disk cache."""

import numpy as np
import pytest

from benchmark.vmd import causal_vmd_modes

LOOKBACK = 10
HORIZON = 2
DECOMP_LEN = 64
K = 3


def _signal(n: int = 160) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    return 100.0 + 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 20) + 0.5 * np.sin(2 * np.pi * t / 5)


def test_shapes() -> None:
    close = _signal()
    modes = causal_vmd_modes(close, lookback=LOOKBACK, horizon=HORIZON, decomp_len=DECOMP_LEN, k=K)
    t_start = DECOMP_LEN - 1
    n_expected = len(close) - HORIZON - t_start
    assert modes.shape == (n_expected, LOOKBACK, K)
    assert np.isfinite(modes).all()


def test_causality() -> None:
    close = _signal()
    modes = causal_vmd_modes(close, lookback=LOOKBACK, horizon=HORIZON, decomp_len=DECOMP_LEN, k=K)

    # Corrupt every value after the first sample's last-known day t.
    t = DECOMP_LEN - 1
    close2 = close.copy()
    close2[t + 1 :] += 50.0
    modes2 = causal_vmd_modes(close2, lookback=LOOKBACK, horizon=HORIZON, decomp_len=DECOMP_LEN, k=K)

    np.testing.assert_allclose(modes[0], modes2[0], atol=1e-10)


def test_cache_roundtrip(tmp_path) -> None:
    close = _signal()
    kwargs = dict(lookback=LOOKBACK, horizon=HORIZON, decomp_len=DECOMP_LEN, k=K, cache_dir=tmp_path, ticker="TEST")
    first = causal_vmd_modes(close, **kwargs)
    assert any(tmp_path.iterdir()), "expected a cache file on disk"
    second = causal_vmd_modes(close, **kwargs)
    np.testing.assert_array_equal(first, second)


def test_rejects_odd_decomp_len() -> None:
    with pytest.raises(ValueError):
        causal_vmd_modes(_signal(), lookback=LOOKBACK, horizon=HORIZON, decomp_len=63, k=K)
