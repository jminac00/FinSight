"""Tests for benchmark.data: target alignment, causality and split hygiene."""

import numpy as np
import pandas as pd
import pytest

from benchmark.data import (
    WindowDataset,
    build_windows,
    chrono_split,
    engineered_features,
    future_return,
)


def _synthetic(n: int = 120) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Linear close series with 4 pseudo-OHLC feature columns."""
    close = np.linspace(100.0, 220.0, n)
    features = np.stack([close - 1.0, close + 1.0, close - 2.0, close], axis=1)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return features, close, dates


def test_future_return_alignment() -> None:
    close = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
    r = future_return(close, horizon=2)
    assert len(r) == 3
    assert r[0] == pytest.approx((104.0 - 100.0) / 100.0 * 100.0)
    assert r[2] == pytest.approx((108.0 - 104.0) / 104.0 * 100.0)


def test_build_windows_shapes_and_target() -> None:
    features, close, dates = _synthetic()
    lookback, horizon = 5, 3
    ds = build_windows(features, close, dates, lookback=lookback, horizon=horizon)

    n_expected = len(close) - horizon - (lookback - 1)
    assert ds.X.shape == (n_expected, lookback, features.shape[1])
    assert ds.y.shape == (n_expected,)
    assert len(ds.t_index) == n_expected

    # Sample i has last-known day t = lookback - 1 + i and target return t -> t+horizon.
    r_all = future_return(close, horizon)
    assert ds.y[0] == pytest.approx(r_all[lookback - 1])
    assert ds.t_index[0] == dates[lookback - 1]
    assert ds.y[-1] == pytest.approx(r_all[len(close) - horizon - 1])


def test_build_windows_respects_t_start() -> None:
    features, close, dates = _synthetic()
    ds = build_windows(features, close, dates, lookback=5, horizon=3, t_start=50)
    assert ds.t_index[0] == dates[50]
    assert len(ds.y) == len(close) - 3 - 50


def test_build_windows_is_causal() -> None:
    features, close, dates = _synthetic()
    lookback, horizon = 5, 3
    ds = build_windows(features, close, dates, lookback=lookback, horizon=horizon)

    # Corrupt everything after the first sample's last-known day t.
    t = lookback - 1
    features2, close2 = features.copy(), close.copy()
    features2[t + 1 :] *= 7.0
    close2[t + 1 :] *= 7.0
    ds2 = build_windows(features2, close2, dates, lookback=lookback, horizon=horizon)

    # Inputs at t must not change; the (future) target obviously does.
    np.testing.assert_allclose(ds.X[0], ds2.X[0])
    assert ds.y[0] != pytest.approx(ds2.y[0])


def test_build_windows_zscores_each_window() -> None:
    features, close, dates = _synthetic()
    ds = build_windows(features, close, dates, lookback=10, horizon=3)
    means = ds.X.mean(axis=1)
    stds = ds.X.std(axis=1)
    np.testing.assert_allclose(means, 0.0, atol=1e-5)
    np.testing.assert_allclose(stds, 1.0, atol=1e-4)


def test_chrono_split_gap() -> None:
    n, gap = 200, 10
    ds = WindowDataset(
        X=np.zeros((n, 4, 2), dtype=np.float32),
        y=np.arange(n, dtype=np.float32),
        t_index=pd.bdate_range("2020-01-01", periods=n),
    )
    train, val, test = chrono_split(ds, train_frac=0.70, val_frac=0.15, gap=gap)

    i1, i2 = int(n * 0.70), int(n * 0.85)
    assert train.y[-1] == i1 - 1
    assert val.y[0] == i1 + gap          # gap samples dropped at the boundary
    assert val.y[-1] == i2 - 1
    assert test.y[0] == i2 + gap
    assert test.y[-1] == n - 1


def test_engineered_features_values() -> None:
    # rows are [open, high, low, close]
    ohlc = np.array(
        [
            [10.0, 13.0, 9.0, 12.0],
            [12.0, 14.0, 11.0, 13.0],
            [15.0, 16.0, 13.0, 14.0],
        ]
    )
    feats = engineered_features(ohlc)
    assert feats.shape == (3, 3)

    rng, body, gap = feats[:, 0], feats[:, 1], feats[:, 2]
    # range = (high - low) / close
    np.testing.assert_allclose(rng, [4 / 12, 3 / 13, 3 / 14], rtol=1e-6)
    # body = (close - open) / close  (positive and negative directions)
    np.testing.assert_allclose(body, [2 / 12, 1 / 13, -1 / 14], rtol=1e-6)
    # gap = (open - prev_close) / prev_close, first day has no previous close
    np.testing.assert_allclose(gap, [0.0, (12 - 12) / 12, (15 - 13) / 13], rtol=1e-6)


def test_engineered_features_is_causal() -> None:
    features, _, _ = _synthetic()
    feats = engineered_features(features)

    t = 40
    features2 = features.copy()
    features2[t + 1 :] *= 5.0
    feats2 = engineered_features(features2)

    # Row t depends only on rows t and t-1, never on the future.
    np.testing.assert_allclose(feats[: t + 1], feats2[: t + 1])
