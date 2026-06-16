"""Shared normalization helpers for the technical analysis blocks.

Main pipeline:
    raw score -> robust z-score (median/MAD) -> sigmoid -> 0-10 score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.technical.engine.config import (
    MIN_OBSERVATIONS,
    ROBUST_SIGMOID_K,
    SIGNAL_THRESHOLDS,
)


def compute_robust_zscore(
    value: float,
    universe_series: pd.Series,
    min_observations: int = MIN_OBSERVATIONS,
    fallback_to_std: bool = True,
) -> float | None:
    """Compute the robust z-score:

        robust_z = (x - median) / (1.4826 * MAD)

    MAD = median(|x_i - median|). The 1.4826 factor makes it comparable with the standard deviation
    on approximately normal distributions.

    If MAD is zero and ``fallback_to_std`` is True, use the classic z-score when the standard
    deviation is valid. If there is no valid dispersion either, return None.
    """
    clean = universe_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_observations or not np.isfinite(value):
        return None

    median = float(clean.median())
    mad = float((clean - median).abs().median())
    scale = 1.4826 * mad

    if np.isfinite(scale) and scale > 0.0:
        return float((value - median) / scale)

    if fallback_to_std:
        sigma = float(clean.std(ddof=0))
        if np.isfinite(sigma) and sigma > 0.0:
            return float((value - float(clean.mean())) / sigma)

    return None


def sigmoid_score_0_10(z: float, k: float = ROBUST_SIGMOID_K) -> float:
    """Map a robust z-score to a 0-10 score:

        score = 10 / (1 + exp(-k*z))

    The default k=0.9 separates companies above/below the median reasonably without saturating too
    quickly at 0 or 10.
    """
    if not np.isfinite(z):
        return float("nan")
    # Clip to ±5 to avoid saturation on extreme scores (9.993+).
    # With k=0.9 the maximum reachable value is 10/(1+exp(-4.5)) = 9.89.
    x = float(np.clip(k * z, -5.0, 5.0))
    return float(10.0 / (1.0 + np.exp(-x)))


def robust_sigmoid_normalize(
    value: float,
    universe_series: pd.Series,
    k: float = ROBUST_SIGMOID_K,
    min_observations: int = MIN_OBSERVATIONS,
) -> dict[str, float | str] | None:
    """Normalize ``value`` against ``universe_series`` with a robust z-score and a sigmoid.

    Returns:
        {
            "z_score": z,
            "score_0_10": score,
            "method": "robust_zscore_sigmoid",
            "k": k,
        }
    """
    z = compute_robust_zscore(value, universe_series, min_observations=min_observations)
    if z is None:
        return None

    score = sigmoid_score_0_10(z, k=k)
    if not np.isfinite(score):
        return None

    return {
        "z_score": z,
        "score_0_10": score,
        "method": "robust_zscore_sigmoid",
        "k": k,
    }


def assign_signal(
    score: float,
    thresholds: tuple[float, float] = SIGNAL_THRESHOLDS,
    labels: tuple[str, str, str] = ("bajista", "neutral", "alcista"),
) -> str:
    """Return the qualitative label for a 0-10 score.

    Labels are product-facing Spanish text returned through the API.
    """
    if score < thresholds[0]:
        return labels[0]
    if score > thresholds[1]:
        return labels[2]
    return labels[1]
