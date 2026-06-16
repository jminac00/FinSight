"""
block2_trend.py
Bloque 2: Trend, peso 30% en el Technical Score.

Evalua la direccion estructural del precio combinando:
    DMA   = (Close[-1] - MA200) / MA200
    T_reg = beta_126 * R2_126, sobre log-precios

DMA y T_reg se normalizan por separado frente al universo antes de combinarse:
    Trend_Raw = 0.4 * z_DMA + 0.6 * z_T_reg

Despues se aplica la normalizacion comun al score compuesto:
    raw -> z-score robusto -> sigmoide -> escala 0-10
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.services.technical.engine.utils.normalization import (
    assign_signal,
    compute_robust_zscore,
    robust_sigmoid_normalize,
)

logger = logging.getLogger(__name__)

MIN_SESSIONS_MA200 = 200
WIN_REG = 126
_UNIVERSE_COMPONENTS_CACHE: dict[object, pd.DataFrame] = {}
_UNIVERSE_RAW_CACHE: dict[object, pd.Series] = {}


def _universe_cache_key(universe_closes: pd.DataFrame) -> object:
    return universe_closes.attrs.get("universe_name", id(universe_closes))


def _trend_components(
    closes: pd.Series,
) -> tuple[float, float, float, float, float, float]:
    """Devuelve (price, ma200, dma, beta, r2, t_reg)."""
    closes = closes.dropna()
    if len(closes) < MIN_SESSIONS_MA200:
        raise ValueError(
            f"Solo {len(closes)} sesiones; se necesitan {MIN_SESSIONS_MA200} para MA200."
        )

    price = float(closes.iloc[-1])
    ma200 = float(closes.iloc[-MIN_SESSIONS_MA200:].mean())
    if ma200 == 0 or not np.isfinite(ma200):
        raise ValueError("MA200 invalida.")
    dma = (price - ma200) / ma200

    n = min(WIN_REG, len(closes))
    log_p = np.log(closes.iloc[-n:].values.astype(float))
    t_arr = np.arange(n, dtype=float)
    slope, _, r_value, _, _ = stats.linregress(t_arr, log_p)
    beta = float(slope)
    r2 = float(r_value**2)
    t_reg = beta * r2

    return price, ma200, dma, beta, r2, t_reg


def _robust_z(value: float, universe_series: pd.Series) -> float | None:
    """Z-score robusto de un componente contra su distribucion del universo."""
    return compute_robust_zscore(value, universe_series)


def _universe_components_trend(universe_closes: pd.DataFrame) -> pd.DataFrame:
    """Calcula DMA y T_reg para todo el universo de forma vectorizada."""
    cache_key = _universe_cache_key(universe_closes)
    if cache_key in _UNIVERSE_COMPONENTS_CACHE:
        return _UNIVERSE_COMPONENTS_CACHE[cache_key]

    if len(universe_closes) < MIN_SESSIONS_MA200:
        return pd.DataFrame(columns=["dma", "t_reg", "beta", "r2"])

    valid = universe_closes.count() >= MIN_SESSIONS_MA200
    closes = universe_closes.loc[:, valid]
    if closes.empty or len(closes) < WIN_REG:
        return pd.DataFrame(columns=["dma", "t_reg", "beta", "r2"])

    ma200 = closes.iloc[-MIN_SESSIONS_MA200:].mean(axis=0).replace(0, np.nan)
    current = closes.iloc[-1]
    dma_univ = (current - ma200) / ma200

    log_p = np.log(closes.iloc[-WIN_REG:].values.astype(float))
    t_arr = np.arange(WIN_REG, dtype=float)
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
    """Trend_Raw del universo tras normalizar por separado DMA y T_reg."""
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
    """
    Calcula el bloque de Trend (0-10) para el ticker dado.

    Devuelve None si el ticker tiene datos insuficientes o si el universo no
    permite normalizar con dispersion suficiente.
    """
    if ticker not in universe_closes.columns:
        raise ValueError(f"Ticker '{ticker}' no encontrado en universe_closes.")

    ticker_closes = universe_closes[ticker].dropna()
    if len(ticker_closes) < MIN_SESSIONS_MA200:
        logger.warning(
            "%s: solo %d sesiones; Bloque 2 (Trend) devuelve None.",
            ticker,
            len(ticker_closes),
        )
        return None

    price, ma200, dma, beta, r2, t_reg = _trend_components(ticker_closes)

    components = _universe_components_trend(universe_closes)
    if components.empty:
        logger.warning(
            "Bloque 'trend' devuelve None para %s: componentes del universo vacios.",
            ticker,
        )
        return None

    dma_z = _robust_z(dma, components["dma"])
    t_reg_z = _robust_z(t_reg, components["t_reg"])
    if dma_z is None or t_reg_z is None:
        logger.warning(
            "Bloque 'trend' devuelve None para %s: normalizacion interna insuficiente.",
            ticker,
        )
        return None

    raw_score = 0.4 * dma_z + 0.6 * t_reg_z
    universe_raw = _universe_raw_scores_trend(universe_closes)
    if universe_raw.empty:
        logger.warning(
            "Bloque 'trend' devuelve None para %s: distribucion compuesta vacia.",
            ticker,
        )
        return None

    norm = robust_sigmoid_normalize(raw_score, universe_raw, k=0.75)
    if norm is None:
        logger.warning(
            "Bloque 'trend' devuelve None para %s: normalizacion robusta insuficiente.",
            ticker,
        )
        return None
    z = float(norm["z_score"])
    score = float(norm["score_0_10"])
    signal = assign_signal(score, labels=("bajista", "neutral", "alcista"))

    pos = "por encima" if dma > 0 else "por debajo"
    summary = (
        f"{ticker} cotiza {pos} de su MA200 (DMA={dma * 100:.1f}%, z_DMA={dma_z:.2f}). "
        f"Regresion {WIN_REG} sesiones: beta_diario={beta:.6f}, R2={r2:.3f}, "
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
