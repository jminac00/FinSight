import numpy as np
import pandas as pd

from app.services.deep_learning import preprocessing as pp
from app.services.deep_learning.recipe import load_frozen_recipe


def test_input_size_is_four_ohlc_plus_three_engineered():
    assert pp.INPUT_SIZE == 7


def test_future_return_alignment():
    close = np.array([100.0, 110.0, 121.0, 133.1], dtype=np.float64)
    # horizon 1: pct change day to day = +10% each step, for t in [0, len-1)
    out = pp.future_return(close, horizon=1)
    np.testing.assert_allclose(out, [10.0, 10.0, 10.0])


def test_engineered_features_formulas_and_causal_gap():
    ohlc = np.array(
        [
            [10.0, 12.0, 9.0, 11.0],
            [11.0, 13.0, 10.0, 12.0],
        ],
        dtype=np.float64,
    )
    feats = pp.engineered_features(ohlc)
    # range = (high-low)/close, body = (close-open)/close, gap = (open-prev_close)/prev_close
    np.testing.assert_allclose(feats[0], [(12 - 9) / 11, (11 - 10) / 11, 0.0])
    np.testing.assert_allclose(feats[1], [(13 - 10) / 12, (12 - 11) / 12, (11 - 11) / 11])
    assert feats[0, 2] == 0.0  # gap is 0 on day 0 (no previous close)


def test_build_windows_are_zscored_per_window():
    rng = np.random.default_rng(0)
    n = 60
    features = rng.normal(5.0, 3.0, size=(n, 7)).astype(np.float64)
    close = np.linspace(100, 130, n)
    dates = pd.bdate_range("2020-01-01", periods=n)
    ds = pp.build_windows(features, close, dates, lookback=24, horizon=10)
    # every window standardized along the time axis: mean ~ 0, std ~ 1 per feature
    means = ds.X.mean(axis=1)
    stds = ds.X.std(axis=1)
    np.testing.assert_allclose(means, np.zeros_like(means), atol=1e-4)
    np.testing.assert_allclose(stds, np.ones_like(stds), atol=1e-4)


def test_build_windows_are_causal():
    rng = np.random.default_rng(1)
    n = 80
    features = rng.normal(0.0, 1.0, size=(n, 7))
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2020-01-01", periods=n)
    lookback, horizon = 24, 10

    base = pp.build_windows(features, close, dates, lookback, horizon)

    # Mutate data strictly after a chosen day t0; samples ending at or before t0
    # (and whose target window t0+horizon is untouched) must not change.
    t0 = 40
    features2 = features.copy()
    close2 = close.copy()
    features2[t0 + 1 :] += 99.0
    close2[t0 + horizon + 1 :] += 99.0
    mutated = pp.build_windows(features2, close2, dates, lookback, horizon)

    n_safe = t0 - (lookback - 1) + 1  # samples with last day t in [lookback-1, t0]
    np.testing.assert_array_equal(base.X[:n_safe], mutated.X[:n_safe])
    np.testing.assert_array_equal(base.y[:n_safe], mutated.y[:n_safe])


def test_split_bounds_ratios_and_gap():
    tr, va, te = pp.split_bounds(100, train_frac=0.70, val_frac=0.15, gap=10)
    assert (tr.start, tr.stop) == (0, 70)
    assert (va.start, va.stop) == (80, 85)
    assert (te.start, te.stop) == (95, 100)


def test_make_dataset_shapes(make_ohlc):
    recipe = load_frozen_recipe()
    ds = pp.make_dataset(make_ohlc(n=300, seed=3), recipe)
    assert ds.X.ndim == 3
    assert ds.X.shape[1] == recipe.lookback
    assert ds.X.shape[2] == pp.INPUT_SIZE
    assert ds.X.shape[0] == ds.y.shape[0] == len(ds.t_index)
