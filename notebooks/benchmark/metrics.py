"""Evaluation metrics shared by every model in the benchmark."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def classify_trend(returns_pct: np.ndarray, threshold: float) -> np.ndarray:
    """Map percentage returns to classes: 0=down, 1=neutral, 2=up."""
    out = np.ones(len(returns_pct), dtype=int)
    out[returns_pct > threshold] = 2
    out[returns_pct < -threshold] = 0
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    """Regression and trend-classification metrics for one model.

    dir_acc is the sign-agreement rate; it is NaN for degenerate predictors
    that never take a side (e.g. the zero-return naive).
    """
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    pred_sign = np.sign(y_pred)
    dir_acc = float(np.mean(pred_sign == np.sign(y_true))) if pred_sign.any() else float("nan")

    cls_true = classify_trend(y_true, threshold)
    cls_pred = classify_trend(y_pred, threshold)
    cls_acc = float(np.mean(cls_true == cls_pred))
    f1 = float(f1_score(cls_true, cls_pred, average="macro", zero_division=0))

    return {"rmse": rmse, "mae": mae, "dir_acc": dir_acc, "cls_acc": cls_acc, "f1_macro": f1}
