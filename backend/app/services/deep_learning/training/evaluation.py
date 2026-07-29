"""Quality metrics on the 10-day return, matching the ModelMetrics contract."""

from __future__ import annotations

import numpy as np


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return rmse, mae (percentage points) and directional_accuracy ([0, 1])."""
    err = y_true - y_pred
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "directional_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
    }


def naive_rmse(y_true: np.ndarray) -> float:
    """Return the RMSE of the naive predictor: a zero return at every step.

    That baseline is the driftless random walk — assuming the price does not
    move over the horizon. A GRU that cannot beat it adds no predictive value,
    so this is the reference the quality gate scores models against.
    """
    return evaluate(y_true, np.zeros_like(y_true))["rmse"]
