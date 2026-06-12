"""Statistical baselines: zero-return naive, momentum and walk-forward ARIMA."""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def naive_zero(n: int) -> np.ndarray:
    """Random-walk baseline: always predict a 0% return."""
    return np.zeros(n, dtype=np.float64)


def momentum(y_full: np.ndarray, start: int, end: int, horizon: int) -> np.ndarray:
    """Trailing-return baseline for stride-1 samples.

    With one sample per day, the target of sample i-horizon is the return
    between t-horizon and t — i.e. the trailing horizon-day return known at
    the prediction day t of sample i.
    """
    if start < horizon:
        raise ValueError(f"start={start} must be >= horizon={horizon}")
    return y_full[np.arange(start, end) - horizon].astype(np.float64)


def arima(
    log_close: np.ndarray,
    sample_ts: np.ndarray,
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    """Walk-forward ARIMA-with-drift forecasts of the horizon-day return.

    Fits once on data up to the first prediction day, then extends the fitted
    model observation by observation (no refit) — each forecast uses only data
    up to its own last known day t.
    """
    from statsmodels.tsa.arima.model import ARIMA

    sample_ts = np.asarray(sample_ts)
    preds = np.empty(len(sample_ts), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(log_close[: sample_ts[0] + 1], order=order, trend="t").fit()
        prev_t = sample_ts[0]
        for i, t in enumerate(sample_ts):
            if t > prev_t:
                res = res.extend(log_close[prev_t + 1 : t + 1])
                prev_t = t
            forecast = np.asarray(res.forecast(horizon))
            preds[i] = (np.exp(forecast[-1] - log_close[t]) - 1.0) * 100.0
            if (i + 1) % 500 == 0:
                logger.info("ARIMA walk-forward: %d/%d", i + 1, len(sample_ts))
    return preds
