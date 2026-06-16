"""
block1_momentum.py
Bloque 1: Price Momentum — peso 35 % en el Technical Score.

Mide la persistencia del rendimiento reciente ajustada por volatilidad,
incorporando ademas la fortaleza relativa dentro del sector GICS (RS sector).

Variables:
    p_skip = Close[t-22]   (precio hace ~1 mes, excluye efecto reversal)
    M12_1     = (p_skip / Close[t-274]) - 1      retorno bruto 12-1
    M6_1      = (p_skip / Close[t-148]) - 1      retorno bruto 6-1
    sigma_12m = std(daily_returns[-274:-22])      volatilidad ventana 12m (excl. skip)
    sigma_6m  = std(daily_returns[-148:-22])      volatilidad ventana 6m  (excl. skip)

    M12_1_adj    = M12_1 / sigma_12m
    M6_1_adj     = M6_1  / sigma_6m
    RS_sector_pct = percentil del retorno 12-1 de la empresa dentro de su sector GICS
                    en el universo de referencia (escala 0-1)

    Momentum_Raw = 0.50 * M12_1_adj + 0.25 * M6_1_adj + 0.25 * RS_sector_pct

Normalizacion:
    z-score robusto (mediana/MAD) frente al universo -> sigmoide -> score 0-10

    La normalizacion se aplica UNA SOLA VEZ sobre Momentum_Raw combinado.
    RS_sector_pct entra como senyal cruda (0-1) antes de normalizar; no se
    normaliza por separado.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.technical.engine.utils.data_loader import load_sector_map_from_fundamental
from app.services.technical.engine.utils.normalization import (
    assign_signal,
    robust_sigmoid_normalize,
)

logger = logging.getLogger(__name__)

SKIP_SESSIONS = 21  # ultimo mes a excluir (skip month)
LOOKBACK_12M = 252  # ventana de formacion 12 meses
LOOKBACK_6M = 126  # ventana de formacion 6 meses
MIN_SESSIONS = LOOKBACK_12M + SKIP_SESSIONS + 7  # = 280, con margen
_UNIVERSE_RAW_CACHE: dict[object, pd.Series] = {}


def _universe_cache_key(universe_closes: pd.DataFrame) -> object:
    return universe_closes.attrs.get("universe_name", id(universe_closes))


# ---------------------------------------------------------------------------
# Calculo para un unico ticker
# ---------------------------------------------------------------------------


def _momentum_components(closes: pd.Series) -> tuple[float, float, float, float, float, float]:
    """
    Calcula (M12_1, M6_1, sigma_12m, sigma_6m, M12_1_adj, M6_1_adj).

    Aplica skip month: el numerador es el precio de hace ~21 sesiones (no de ayer)
    y los denominadores son los precios de hace 252+21 y 126+21 sesiones.
    La volatilidad se calcula sobre la ventana de formacion (excluye el skip).

    Raises:
        ValueError: si la serie tiene menos de MIN_SESSIONS sesiones validas.
    """
    closes = closes.dropna()
    if len(closes) < MIN_SESSIONS:
        raise ValueError(f"Solo {len(closes)} sesiones disponibles; se necesitan {MIN_SESSIONS}.")

    daily_returns = closes.pct_change().dropna()

    p_skip = float(closes.iloc[-SKIP_SESSIONS - 1])  # iloc[-22]
    p_12m = float(closes.iloc[-LOOKBACK_12M - SKIP_SESSIONS - 1])  # iloc[-274]
    p_6m = float(closes.iloc[-LOOKBACK_6M - SKIP_SESSIONS - 1])  # iloc[-148]

    m12_1 = (p_skip / p_12m) - 1.0
    m6_1 = (p_skip / p_6m) - 1.0

    sigma_12m = float(
        daily_returns.iloc[-LOOKBACK_12M - SKIP_SESSIONS : -SKIP_SESSIONS].std(ddof=0)
    )
    sigma_6m = float(daily_returns.iloc[-LOOKBACK_6M - SKIP_SESSIONS : -SKIP_SESSIONS].std(ddof=0))

    if sigma_12m == 0 or np.isnan(sigma_12m):
        raise ValueError("Volatilidad 12m es cero o NaN.")
    if sigma_6m == 0 or np.isnan(sigma_6m):
        raise ValueError("Volatilidad 6m es cero o NaN.")

    return m12_1, m6_1, sigma_12m, sigma_6m, m12_1 / sigma_12m, m6_1 / sigma_6m


# ---------------------------------------------------------------------------
# Percentil sectorial (componente RS integrado)
# ---------------------------------------------------------------------------


def _sector_pcts_for_ticker(
    ticker: str,
    universe_closes: pd.DataFrame,
    m12_1: float,
    m6_1: float,
) -> tuple[float, float]:
    """
    Percentil del retorno 12-1 y 6-1 de la empresa dentro de su sector GICS.

    Devuelve (pct_12_1, pct_6_1) en escala [0, 1].
    Si no hay datos de sector o hay menos de 2 peers, devuelve (0.5, 0.5).
    """
    sector_map = load_sector_map_from_fundamental()
    sector = sector_map.get(ticker, "")
    if not sector:
        return 0.5, 0.5

    valid = universe_closes.count() >= MIN_SESSIONS
    valid_closes = universe_closes.loc[:, valid]

    peers = [t for t in valid_closes.columns if sector_map.get(t, "") == sector]
    if len(peers) < 2:
        return 0.5, 0.5

    peer_closes = valid_closes[peers]
    p_skip = peer_closes.iloc[-SKIP_SESSIONS - 1]
    p_12m = peer_closes.iloc[-LOOKBACK_12M - SKIP_SESSIONS - 1]
    p_6m = peer_closes.iloc[-LOOKBACK_6M - SKIP_SESSIONS - 1]

    ret12 = ((p_skip / p_12m) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    ret6 = ((p_skip / p_6m) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()

    pct12 = float((ret12 < m12_1).sum()) / float(len(ret12)) if not ret12.empty else 0.5
    pct6 = float((ret6 < m6_1).sum()) / float(len(ret6)) if not ret6.empty else 0.5

    return pct12, pct6


def _universe_sector_pcts(ret12: pd.Series) -> pd.Series:
    """
    Percentil sectorial del retorno 12-1 para todo el universo (vectorizado por sector).

    Para cada ticker calcula que fraccion de empresas del mismo sector GICS tiene
    un retorno 12-1 inferior (escala [0, 1]).
    Tickers sin sector o con menos de 2 peers reciben 0.5 (neutral).
    """
    sector_map = load_sector_map_from_fundamental()
    pct_dict: dict[str, float] = {}

    sector_groups: dict[str, list[str]] = {}
    for ticker in ret12.index:
        sector = sector_map.get(ticker, "")
        sector_groups.setdefault(sector, []).append(ticker)

    for sector, tickers_in_sector in sector_groups.items():
        sector_rets = ret12.reindex(tickers_in_sector).dropna()
        n = len(sector_rets)

        if not sector or n < 2:
            for t in tickers_in_sector:
                pct_dict[t] = 0.5
            continue

        for t in sector_rets.index:
            val = float(sector_rets[t])
            pct_dict[t] = float((sector_rets < val).sum()) / float(n)

        for t in tickers_in_sector:
            if t not in pct_dict:
                pct_dict[t] = 0.5

    return pd.Series(pct_dict, dtype=float)


# ---------------------------------------------------------------------------
# Distribucion del universo (vectorizado)
# ---------------------------------------------------------------------------


def _universe_raw_scores(universe_closes: pd.DataFrame) -> pd.Series:
    """
    Calcula Momentum_Raw_Score para todo el universo de forma vectorizada.

    Formula:  0.50 * M12_1_adj + 0.25 * M6_1_adj + 0.25 * RS_sector_pct
    Aplica el mismo skip month que _momentum_components.
    Ignora silenciosamente los tickers con datos insuficientes (< MIN_SESSIONS).
    """
    cache_key = _universe_cache_key(universe_closes)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    if len(universe_closes) < MIN_SESSIONS:
        return pd.Series(dtype=float)

    valid = universe_closes.count() >= MIN_SESSIONS
    closes = universe_closes.loc[:, valid]
    returns = closes.pct_change()

    p_skip = closes.iloc[-SKIP_SESSIONS - 1]
    p_12m = closes.iloc[-LOOKBACK_12M - SKIP_SESSIONS - 1]
    p_6m = closes.iloc[-LOOKBACK_6M - SKIP_SESSIONS - 1]

    m12 = (p_skip / p_12m) - 1.0
    m6 = (p_skip / p_6m) - 1.0

    sigma_12m = (
        returns.iloc[-LOOKBACK_12M - SKIP_SESSIONS : -SKIP_SESSIONS].std(ddof=0).replace(0, np.nan)
    )
    sigma_6m = (
        returns.iloc[-LOOKBACK_6M - SKIP_SESSIONS : -SKIP_SESSIONS].std(ddof=0).replace(0, np.nan)
    )

    m12_adj = (m12 / sigma_12m).replace([np.inf, -np.inf], np.nan)
    m6_adj = (m6 / sigma_6m).replace([np.inf, -np.inf], np.nan)

    m12_clean = m12.replace([np.inf, -np.inf], np.nan).dropna()
    sector_pcts = _universe_sector_pcts(m12_clean).reindex(m12_adj.index).fillna(0.5)

    raw = (0.50 * m12_adj + 0.25 * m6_adj + 0.25 * sector_pcts).dropna()
    _UNIVERSE_RAW_CACHE[cache_key] = raw
    return raw


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------


def compute_momentum_block(
    ticker: str,
    universe_closes: pd.DataFrame,
) -> dict[str, Any] | None:
    """
    Calcula el bloque de Price Momentum (0-10) para el ticker dado.

    Incorpora la fortaleza relativa sectorial (RS) como tercer componente interno:
        Momentum_Raw = 0.50 * M12_1_adj + 0.25 * M6_1_adj + 0.25 * RS_sector_pct_12_1

    Args:
        ticker:          Simbolo del activo (debe estar en universe_closes).
        universe_closes: DataFrame de cierres diarios del universo de referencia
                         (S&P 500 o MSCI World en USD).

    Returns:
        Dict con claves:
            momentum_12_1, momentum_6_1, vol_12m, vol_6m,
            momentum_12_1_adj, momentum_6_1_adj,
            rs_sector_pct_12_1, rs_sector_pct_6_1, rs_sector_included,
            raw_momentum_score, z_score, normalized_score_0_10, signal, summary.
    """
    if ticker not in universe_closes.columns:
        raise ValueError(f"Ticker '{ticker}' no encontrado en universe_closes.")

    m12_1, m6_1, sigma_12m, sigma_6m, m12_adj, m6_adj = _momentum_components(
        universe_closes[ticker]
    )

    rs_pct_12, rs_pct_6 = _sector_pcts_for_ticker(ticker, universe_closes, m12_1, m6_1)
    raw_score = 0.50 * m12_adj + 0.25 * m6_adj + 0.25 * rs_pct_12

    universe_raw = _universe_raw_scores(universe_closes)
    if universe_raw.empty:
        logger.warning(
            "Bloque 'momentum' devuelve None para %s: distribucion del universo vacia.",
            ticker,
        )
        return None

    norm = robust_sigmoid_normalize(raw_score, universe_raw)
    if norm is None:
        logger.warning(
            "Bloque 'momentum' devuelve None para %s: normalizacion robusta insuficiente.",
            ticker,
        )
        return None
    z = float(norm["z_score"])
    score = float(norm["score_0_10"])
    signal = assign_signal(score, labels=("bajista", "neutral", "alcista"))

    pct = float((universe_raw < raw_score).mean() * 100)

    universe_name = str(universe_closes.attrs.get("universe_name", "")).lower()
    universe_label = "MSCI World" if "msci" in universe_name else "S&P 500"

    if score < 4.0:
        direction = "debil o bajista"
    elif score > 6.5:
        direction = "positivo"
    else:
        direction = "neutral"
    summary = (
        f"{ticker} presenta un momentum {direction}: "
        f"M12-1 ajustado={m12_adj:.2f} (retorno={m12_1 * 100:.1f}%, vol={sigma_12m * 100:.1f}%), "
        f"M6-1 ajustado={m6_adj:.2f} (retorno={m6_1 * 100:.1f}%), "
        f"RS sector pct12-1={rs_pct_12:.2f}. "
        f"Score {score:.2f}/10 — percentil {pct:.0f} del universo {universe_label}."
    )

    return {
        "momentum_12_1": m12_1,
        "momentum_6_1": m6_1,
        "vol_12m": sigma_12m,
        "vol_6m": sigma_6m,
        "momentum_12_1_adj": m12_adj,
        "momentum_6_1_adj": m6_adj,
        "rs_sector_pct_12_1": rs_pct_12,
        "rs_sector_pct_6_1": rs_pct_6,
        "rs_sector_included": True,
        "raw_momentum_score": raw_score,
        "winsorized_score": raw_score,
        "z_score": z,
        "normalized_score_0_10": score,
        "normalization_method": norm["method"],
        "normalization_k": norm["k"],
        "signal": signal,
        "summary": summary,
    }
