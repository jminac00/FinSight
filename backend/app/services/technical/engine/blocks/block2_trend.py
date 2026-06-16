"""Block 2: Trend — 30% weight in the Technical Score.

Evaluates the structural direction of price by combining:
    DMA   = (Close[-1] - MA200) / MA200
    T_reg = beta_126 * R2_126, on log prices

DMA and T_reg are normalized separately against the universe before being combined:
    Trend_Raw = 0.4 * z_DMA + 0.6 * z_T_reg

Then the shared normalization is applied to the composite score:
    raw -> robust z-score -> sigmoid -> 0-10 scale
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.services.technical.engine.config import (
    TREND_MA_SESSIONS,
    TREND_REGRESSION_WINDOW,
    TREND_SIGMOID_K,
)
from app.services.technical.engine.utils.normalization import (
    assign_signal,
    compute_robust_zscore,
    robust_sigmoid_normalize,
)

logger = logging.getLogger(__name__)

_UNIVERSE_COMPONENTS_CACHE: dict[object, pd.DataFrame] = {}
_UNIVERSE_RAW_CACHE: dict[object, pd.Series] = {}


def _universe_cache_key(universe_closes: pd.DataFrame) -> object:
    return universe_closes.attrs.get("universe_name", id(universe_closes))


def _trend_components(
    closes: pd.Series,
) -> tuple[float, float, float, float, float, float]:
    """Return (price, ma200, dma, beta, r2, t_reg)."""
    closes = closes.dropna()
    if len(closes) < TREND_MA_SESSIONS:
        raise ValueError(
            f"Only {len(closes)} sessions; {TREND_MA_SESSIONS} are required for MA200."
        )

    price = float(closes.iloc[-1])
    ma200 = float(closes.iloc[-TREND_MA_SESSIONS:].mean())
    if ma200 == 0 or not np.isfinite(ma200):
        raise ValueError("Invalid MA200.")
    dma = (price - ma200) / ma200

    n = min(TREND_REGRESSION_WINDOW, len(closes))
    log_p = np.log(closes.iloc[-n:].values.astype(float))
    t_arr = np.arange(n, dtype=float)
    slope, _, r_value, _, _ = stats.linregress(t_arr, log_p)
    beta = float(slope)
    r2 = float(r_value**2)
    t_reg = beta * r2

    return price, ma200, dma, beta, r2, t_reg


def _robust_z(value: float, universe_series: pd.Series) -> float | None:
    """Robust z-score of a component against its universe distribution."""
    return compute_robust_zscore(value, universe_series)


def _universe_components_trend(universe_closes: pd.DataFrame) -> pd.DataFrame:
    """Compute DMA and T_reg for the whole universe, vectorized."""
    cache_key = _universe_cache_key(universe_closes)
    if cache_key in _UNIVERSE_COMPONENTS_CACHE:
        return _UNIVERSE_COMPONENTS_CACHE[cache_key]

    if len(universe_closes) < TREND_MA_SESSIONS:
        return pd.DataFrame(columns=["dma", "t_reg", "beta", "r2"])

    valid = universe_closes.count() >= TREND_MA_SESSIONS
    closes = universe_closes.loc[:, valid]
    if closes.empty or len(closes) < TREND_REGRESSION_WINDOW:
        return pd.DataFrame(columns=["dma", "t_reg", "beta", "r2"])

    ma200 = closes.iloc[-TREND_MA_SESSIONS:].mean(axis=0).replace(0, np.nan)
    current = closes.iloc[-1]
    dma_univ = (current - ma200) / ma200

    log_p = np.log(closes.iloc[-TREND_REGRESSION_WINDOW:].values.astype(float))
    t_arr = np.arange(TREND_REGRESSION_WINDOW, dtype=float)
    t_c = t_arr - t_arr.mean()
    t_ss = float((t_c**2).sum())

    y_bar = log_p.mean(axis=0)
    y_c = log_p - y_bar
    y_ss = (y_c**2).sum(axis=0)
    cov_num = t_c @ y_c
    betas = cov_num / t_ss
    r2_arr = np.clip(cov_num**2 / (t_ss * y_ss + 1e-20), 0.0, 1.0)
    t_reg_arr = betas * r2_arr

    components = pd.DataFrame(
        {
            "dma": dma_univ.values,
            "t_reg": t_reg_arr,
            "beta": betas,
            "r2": r2_arr,
        },
        index=closes.columns,
    )
    components = components.replace([np.inf, -np.inf], np.nan).dropna(subset=["dma", "t_reg"])
    _UNIVERSE_COMPONENTS_CACHE[cache_key] = components
    return components


def _universe_raw_scores_trend(universe_closes: pd.DataFrame) -> pd.Series:
    """Universe Trend_Raw after normalizing DMA and T_reg separately."""
    cache_key = _universe_cache_key(universe_closes)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    components = _universe_components_trend(universe_closes)
    if components.empty:
        return pd.Series(dtype=float)

    z_dma = components["dma"].apply(lambda value: _robust_z(float(value), components["dma"]))
    z_t_reg = components["t_reg"].apply(lambda value: _robust_z(float(value), components["t_reg"]))
    raw = 0.4 * z_dma + 0.6 * z_t_reg
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    _UNIVERSE_RAW_CACHE[cache_key] = raw
    return raw


def compute_trend_block(
    ticker: str,
    universe_closes: pd.DataFrame,
) -> dict[str, Any] | None:
    """Compute the Trend block (0-10) for the given ticker.

    Returns None if the ticker has insufficient data or if the universe does not allow normalizing
    with enough dispersion.
    """
    if ticker not in universe_closes.columns:
        raise ValueError(f"Ticker '{ticker}' not found in universe_closes.")

    ticker_closes = universe_closes[ticker].dropna()
    if len(ticker_closes) < TREND_MA_SESSIONS:
        logger.warning(
            "%s: only %d sessions; Block 2 (Trend) returns None.",
            ticker,
            len(ticker_closes),
        )
        return None

    price, ma200, dma, beta, r2, t_reg = _trend_components(ticker_closes)

    components = _universe_components_trend(universe_closes)
    if components.empty:
        logger.warning(
            "Block 'trend' returns None for %s: empty universe components.",
            ticker,
        )
        return None

    dma_z = _robust_z(dma, components["dma"])
    t_reg_z = _robust_z(t_reg, components["t_reg"])
    if dma_z is None or t_reg_z is None:
        logger.warning(
            "Block 'trend' returns None for %s: insufficient internal normalization.",
            ticker,
        )
        return None

    raw_score = 0.4 * dma_z + 0.6 * t_reg_z
    universe_raw = _universe_raw_scores_trend(universe_closes)
    if universe_raw.empty:
        logger.warning(
            "Block 'trend' returns None for %s: empty composite distribution.",
            ticker,
        )
        return None

    norm = robust_sigmoid_normalize(raw_score, universe_raw, k=TREND_SIGMOID_K)
    if norm is None:
        logger.warning(
            "Block 'trend' returns None for %s: insufficient robust normalization.",
            ticker,
        )
        return None
    z = float(norm["z_score"])
    score = float(norm["score_0_10"])
    signal = assign_signal(score, labels=("bajista", "neutral", "alcista"))

    pos = "por encima" if dma > 0 else "por debajo"
    summary = (
        f"{ticker} cotiza {pos} de su MA200 (DMA={dma * 100:.1f}%, z_DMA={dma_z:.2f}). "
        f"Regresion {TREND_REGRESSION_WINDOW} sesiones: beta_diario={beta:.6f}, R2={r2:.3f}, "
        f"z_T_reg={t_reg_z:.2f}. Score tendencia {score:.2f}/10."
    )

    return {
        "price": price,
        "ma_200": ma200,
        "distance_to_ma200": dma,
        "regression_beta": beta,
        "regression_r2": r2,
        "regression_trend_component": t_reg,
        "dma_z_score": dma_z,
        "t_reg_z_score": t_reg_z,
        "trend_raw_score": raw_score,
        "trend_z_score": z,
        "trend_score_0_10": score,
        "normalization_method": norm["method"],
        "normalization_k": norm["k"],
        "signal": signal,
        "summary": summary,
    }
