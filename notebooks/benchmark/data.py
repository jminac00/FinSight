"""Data loading and leakage-free dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["open", "high", "low", "close"]


def load_ohlc(csv_path: Path) -> pd.DataFrame:
    """Load and clean one ticker CSV with columns ticker,date,open,high,low,close.

    Cleaning rules: drop NaNs, duplicated dates, OHLC-incoherent rows and
    non-positive prices. Returns a date-indexed frame with FEATURES columns.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")[FEATURES].dropna()
    df = df[~df.index.duplicated(keep="first")]
    valid = (
        (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df > 0).all(axis=1)
    )
    return df[valid]


@dataclass
class WindowDataset:
    """Per-sample input windows with their future-return targets.

    X: (n_samples, lookback, n_features) z-scored per window.
    y: (n_samples,) percentage return between t and t+horizon.
    t_index: date of the last known day t of each sample.
    """

    X: np.ndarray
    y: np.ndarray
    t_index: pd.DatetimeIndex


def future_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """Percentage return between t and t+horizon, for t in [0, len-horizon)."""
    return (close[horizon:] - close[:-horizon]) / close[:-horizon] * 100.0


def build_windows(
    features: np.ndarray,
    close: np.ndarray,
    dates: pd.DatetimeIndex,
    lookback: int,
    horizon: int,
    t_start: int | None = None,
) -> WindowDataset:
    """Build stride-1 samples: window [t-lookback+1, t] -> return t -> t+horizon.

    Each window is z-scored with its own statistics, so every sample depends
    only on data up to its last known day t (causal by construction).
    `t_start` overrides the first last-known day, e.g. to align datasets whose
    features need a longer warm-up (VMD decomposition window).
    """
    if t_start is None:
        t_start = lookback - 1
    if t_start < lookback - 1:
        raise ValueError(f"t_start={t_start} smaller than lookback-1={lookback - 1}")

    y_all = future_return(close, horizon)
    xs: list[np.ndarray] = []
    ts: list[pd.Timestamp] = []
    for t in range(t_start, len(close) - horizon):
        window = features[t - lookback + 1 : t + 1].astype(np.float64)
        mu = window.mean(axis=0)
        sd = window.std(axis=0)
        sd[sd < 1e-8] = 1.0
        xs.append((window - mu) / sd)
        ts.append(dates[t])

    return WindowDataset(
        X=np.asarray(xs, dtype=np.float32),
        y=y_all[t_start:].astype(np.float32),
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
