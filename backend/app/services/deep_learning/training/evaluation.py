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
