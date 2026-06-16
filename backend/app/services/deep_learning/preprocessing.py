"""OHLC -> model-ready samples. The single source of truth for the feature
pipeline, shared by training and inference so both apply identical, causal
transformations (different preprocessing would invalidate predictions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.deep_learning.recipe import GRURecipe

FEATURES = ["open", "high", "low", "close"]
ENGINEERED = ["range", "body", "gap"]
INPUT_SIZE = len(FEATURES) + len(ENGINEERED)


class InsufficientHistoryError(ValueError):
    """Raised when a ticker has too little history to build a usable dataset."""


@dataclass
class WindowDataset:
    """Per-sample input windows with their future-return targets.

    X: (n_samples, lookback, INPUT_SIZE) z-scored per window.
    y: (n_samples,) percentage return between t and t+horizon.
    t_index: date of the last known day t of each sample.
    """

    X: np.ndarray
    y: np.ndarray
    t_index: pd.DatetimeIndex


def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a date-indexed OHLC frame (source-agnostic).

    Cleaning rules: keep FEATURES columns, drop NaNs and duplicated dates, sort
    chronologically, and drop OHLC-incoherent or non-positive rows.
    """
    out = df[FEATURES].dropna().sort_index()
    out = out[~out.index.duplicated(keep="first")]
    valid = (
        (out["low"] <= out[["open", "close"]].min(axis=1))
        & (out["high"] >= out[["open", "close"]].max(axis=1))
        & (out > 0).all(axis=1)
    )
    return out[valid]


def future_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """Percentage return between t and t+horizon, for t in [0, len-horizon)."""
    return (close[horizon:] - close[:-horizon]) / close[:-horizon] * 100.0


def engineered_features(ohlc: np.ndarray) -> np.ndarray:
    """Causal intraday features from OHLC columns [open, high, low, close].

    - range = (high - low) / close   intraday volatility proxy
    - body  = (close - open) / close  intraday direction
    - gap   = (open - prev_close) / prev_close  overnight gap (0 on day 0)

    Each row uses only data up to its own day t (gap uses the previous close),
    so the output is causal. Returns an array of shape (n, 3).
    """
    open_, high, low, close = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
    rng = (high - low) / close
    body = (close - open_) / close
    gap = np.zeros_like(close)
    gap[1:] = (open_[1:] - close[:-1]) / close[:-1]
    return np.stack([rng, body, gap], axis=1).astype(np.float32)


def build_windows(
    features: np.ndarray,
    close: np.ndarray,
    dates: pd.DatetimeIndex,
    lookback: int,
    horizon: int,
) -> WindowDataset:
    """Build stride-1 samples: window [t-lookback+1, t] -> return t -> t+horizon.

    Each window is z-scored with its own statistics, so every sample depends only
    on data up to its last known day t (causal by construction).
    """
    y_all = future_return(close, horizon)
    xs: list[np.ndarray] = []
    ts: list[pd.Timestamp] = []
    for t in range(lookback - 1, len(close) - horizon):
        window = features[t - lookback + 1 : t + 1].astype(np.float64)
        mu = window.mean(axis=0)
        sd = window.std(axis=0)
        sd[sd < 1e-8] = 1.0
        xs.append((window - mu) / sd)
        ts.append(dates[t])

    return WindowDataset(
        X=np.asarray(xs, dtype=np.float32).reshape(-1, lookback, features.shape[1]),
        y=y_all[lookback - 1 :].astype(np.float32),
        t_index=pd.DatetimeIndex(ts),
    )


def split_bounds(
    n: int, train_frac: float = 0.70, val_frac: float = 0.15, gap: int = 10
) -> tuple[slice, slice, slice]:
    """Chronological split boundaries with `gap` samples dropped at each edge.

    With stride-1 samples and an h-day target, the last h train samples have
    target windows overlapping the first validation days; dropping `gap`
    (>= horizon) samples at each boundary removes that overlap.
    """
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return slice(0, i1), slice(i1 + gap, i2), slice(i2 + gap, n)


def chrono_split(
    ds: WindowDataset, train_frac: float = 0.70, val_frac: float = 0.15, gap: int = 10
) -> tuple[WindowDataset, WindowDataset, WindowDataset]:
    """Split a WindowDataset chronologically using split_bounds."""

    def take(s: slice) -> WindowDataset:
        return WindowDataset(ds.X[s], ds.y[s], ds.t_index[s])

    tr, va, te = split_bounds(len(ds.y), train_frac, val_frac, gap)
    return take(tr), take(va), take(te)


def make_dataset(ohlc: pd.DataFrame, recipe: GRURecipe) -> WindowDataset:
    """Turn a cleaned, date-indexed OHLC frame into a windowed dataset.

    Raises InsufficientHistoryError if the history yields no samples.
    """
    arr = ohlc[FEATURES].to_numpy(dtype=np.float64)
    features = np.concatenate([arr, engineered_features(arr)], axis=1)
    ds = build_windows(features, arr[:, 3], ohlc.index, recipe.lookback, recipe.horizon)
    if len(ds.y) == 0:
        raise InsufficientHistoryError(
            f"need more than lookback+horizon={recipe.lookback + recipe.horizon} rows, "
            f"got {len(ohlc)}"
        )
    return ds
