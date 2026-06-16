"""
normalization.py
Funciones compartidas de normalizacion para los bloques de analisis tecnico.

Pipeline principal:
    raw score -> z-score robusto (mediana/MAD) -> sigmoide -> score 0-10
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROBUST_SIGMOID_K = 0.9
MIN_OBSERVATIONS = 10


def compute_robust_zscore(
    value: float,
    universe_series: pd.Series,
    min_observations: int = MIN_OBSERVATIONS,
    fallback_to_std: bool = True,
) -> float | None:
    """
    Calcula z-score robusto:
        robust_z = (x - median) / (1.4826 * MAD)

    MAD = median(|x_i - median|). El factor 1.4826 lo hace comparable con la
    desviacion tipica en distribuciones aproximadamente normales.

    Si MAD es cero y fallback_to_std=True, usa z-score clasico si la desviacion
    tipica es valida. Si tampoco hay dispersion valida, devuelve None.
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
    """
    Convierte z-score robusto a score 0-10:
        score = 10 / (1 + exp(-k*z))

    El default k=0.9 (ROBUST_SIGMOID_K) separa razonablemente empresas por
    encima/debajo de la mediana sin saturar demasiado rapido en 0 o 10.
    """
    if not np.isfinite(z):
        return float("nan")
    # Clip a ±5 para evitar saturacion en scores extremos (9.993+).
    # Con k=0.9 el maximo alcanzable es 10/(1+exp(-4.5)) = 9.89.
    x = float(np.clip(k * z, -5.0, 5.0))
    return float(10.0 / (1.0 + np.exp(-x)))


def robust_sigmoid_normalize(
    value: float,
    universe_series: pd.Series,
    k: float = ROBUST_SIGMOID_K,
    min_observations: int = MIN_OBSERVATIONS,
) -> dict[str, float | str] | None:
    """
    Normaliza value contra universe_series con z-score robusto y sigmoide.

    Devuelve:
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


def diagnose_score_distribution(scores: pd.Series | list[float]) -> dict[str, float | int]:
    """
    Diagnostico rapido de dispersion de scores 0-10.
    """
    series = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "valid_scores": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "pct_between_4_and_6": 0.0,
            "pct_between_4_and_6_5": 0.0,
            "unique_scores_rounded_2": 0,
            "count_above_9_5": 0,
            "count_below_0_5": 0,
        }

    return {
        "valid_scores": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
        "pct_between_4_and_6": float(series.between(4.0, 6.0, inclusive="both").mean() * 100.0),
        "pct_between_4_and_6_5": float(series.between(4.0, 6.5, inclusive="both").mean() * 100.0),
        "unique_scores_rounded_2": int(series.round(2).nunique()),
        "count_above_9_5": int((series > 9.5).sum()),
        "count_below_0_5": int((series < 0.5).sum()),
    }


def assign_signal(
    score: float,
    thresholds: tuple[float, float] = (4.0, 6.5),
    labels: tuple[str, str, str] = ("bajista", "neutral", "alcista"),
) -> str:
    """
    Devuelve la etiqueta cualitativa para un score 0-10.
    """
    if score < thresholds[0]:
        return labels[0]
    if score > thresholds[1]:
        return labels[2]
    return labels[1]
