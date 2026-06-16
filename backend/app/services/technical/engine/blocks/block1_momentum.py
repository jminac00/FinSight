"""Block 1: Price Momentum — 35% weight in the Technical Score.

Measures the persistence of recent performance adjusted for volatility, plus the relative strength
within the GICS sector (sector RS).

Variables:
    p_skip = Close[t-22]   (price ~1 month ago, excludes the reversal effect)
    M12_1     = (p_skip / Close[t-274]) - 1      raw 12-1 return
    M6_1      = (p_skip / Close[t-148]) - 1      raw 6-1 return
    sigma_12m = std(daily_returns[-274:-22])     12m-window volatility (excludes skip)
    sigma_6m  = std(daily_returns[-148:-22])     6m-window volatility (excludes skip)

    M12_1_adj    = M12_1 / sigma_12m
    M6_1_adj     = M6_1  / sigma_6m
    RS_sector_pct = percentile of the company's 12-1 return within its GICS sector in the reference
                    universe (0-1 scale)

    Momentum_Raw = 0.50 * M12_1_adj + 0.25 * M6_1_adj + 0.25 * RS_sector_pct

Normalization:
    robust z-score (median/MAD) against the universe -> sigmoid -> 0-10 score

    Normalization is applied ONCE on the combined Momentum_Raw. RS_sector_pct enters as a raw signal
    (0-1) before normalizing; it is not normalized separately.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.technical.engine.config import (
    MOMENTUM_LOOKBACK_6M,
    MOMENTUM_LOOKBACK_12M,
    MOMENTUM_MIN_SESSIONS,
    MOMENTUM_SKIP_SESSIONS,
)
from app.services.technical.engine.utils.data_loader import load_sector_map_from_fundamental
from app.services.technical.engine.utils.normalization import (
    assign_signal,
    robust_sigmoid_normalize,
)

logger = logging.getLogger(__name__)

_UNIVERSE_RAW_CACHE: dict[object, pd.Series] = {}


def _universe_cache_key(universe_closes: pd.DataFrame) -> object:
    return universe_closes.attrs.get("universe_name", id(universe_closes))


# ---------------------------------------------------------------------------
# Single-ticker calculation
# ---------------------------------------------------------------------------


def _momentum_components(closes: pd.Series) -> tuple[float, float, float, float, float, float]:
    """Return (M12_1, M6_1, sigma_12m, sigma_6m, M12_1_adj, M6_1_adj).

    Applies the skip month: the numerator is the price ~21 sessions ago (not yesterday) and the
    denominators are the prices 252+21 and 126+21 sessions ago. Volatility is measured over the
    formation window (excludes the skip).

    Raises:
        ValueError: if the series has fewer than MOMENTUM_MIN_SESSIONS valid sessions.
    """
    closes = closes.dropna()
    if len(closes) < MOMENTUM_MIN_SESSIONS:
        raise ValueError(
            f"Only {len(closes)} sessions available; {MOMENTUM_MIN_SESSIONS} are required."
        )

    daily_returns = closes.pct_change().dropna()

    p_skip = float(closes.iloc[-MOMENTUM_SKIP_SESSIONS - 1])  # iloc[-22]
    p_12m = float(closes.iloc[-MOMENTUM_LOOKBACK_12M - MOMENTUM_SKIP_SESSIONS - 1])  # iloc[-274]
    p_6m = float(closes.iloc[-MOMENTUM_LOOKBACK_6M - MOMENTUM_SKIP_SESSIONS - 1])  # iloc[-148]

    m12_1 = (p_skip / p_12m) - 1.0
    m6_1 = (p_skip / p_6m) - 1.0

    skip = MOMENTUM_SKIP_SESSIONS
    sigma_12m = float(daily_returns.iloc[-MOMENTUM_LOOKBACK_12M - skip : -skip].std(ddof=0))
    sigma_6m = float(daily_returns.iloc[-MOMENTUM_LOOKBACK_6M - skip : -skip].std(ddof=0))

    if sigma_12m == 0 or np.isnan(sigma_12m):
        raise ValueError("12m volatility is zero or NaN.")
    if sigma_6m == 0 or np.isnan(sigma_6m):
        raise ValueError("6m volatility is zero or NaN.")

    return m12_1, m6_1, sigma_12m, sigma_6m, m12_1 / sigma_12m, m6_1 / sigma_6m


# ---------------------------------------------------------------------------
# Sector percentile (integrated RS component)
# ---------------------------------------------------------------------------


def _sector_pcts_for_ticker(
    ticker: str,
    universe_closes: pd.DataFrame,
    m12_1: float,
    m6_1: float,
) -> tuple[float, float]:
    """Percentile of the company's 12-1 and 6-1 returns within its GICS sector.

    Returns (pct_12_1, pct_6_1) on a [0, 1] scale. If there is no sector data or fewer than 2 peers,
    returns (0.5, 0.5).
    """
    sector_map = load_sector_map_from_fundamental()
    sector = sector_map.get(ticker, "")
    if not sector:
        return 0.5, 0.5

    valid = universe_closes.count() >= MOMENTUM_MIN_SESSIONS
    valid_closes = universe_closes.loc[:, valid]

    peers = [t for t in valid_closes.columns if sector_map.get(t, "") == sector]
    if len(peers) < 2:
        return 0.5, 0.5

    peer_closes = valid_closes[peers]
    p_skip = peer_closes.iloc[-MOMENTUM_SKIP_SESSIONS - 1]
    p_12m = peer_closes.iloc[-MOMENTUM_LOOKBACK_12M - MOMENTUM_SKIP_SESSIONS - 1]
    p_6m = peer_closes.iloc[-MOMENTUM_LOOKBACK_6M - MOMENTUM_SKIP_SESSIONS - 1]

    ret12 = ((p_skip / p_12m) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    ret6 = ((p_skip / p_6m) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()

    pct12 = float((ret12 < m12_1).sum()) / float(len(ret12)) if not ret12.empty else 0.5
    pct6 = float((ret6 < m6_1).sum()) / float(len(ret6)) if not ret6.empty else 0.5

    return pct12, pct6


def _universe_sector_pcts(ret12: pd.Series) -> pd.Series:
    """Sector percentile of the 12-1 return for the whole universe (vectorized per sector).

    For each ticker, computes the fraction of companies in the same GICS sector with a lower 12-1
    return ([0, 1] scale). Tickers without a sector or with fewer than 2 peers get 0.5 (neutral).
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
# Universe distribution (vectorized)
# ---------------------------------------------------------------------------


def _universe_raw_scores(universe_closes: pd.DataFrame) -> pd.Series:
    """Compute Momentum_Raw_Score for the whole universe, vectorized.

    Formula: 0.50 * M12_1_adj + 0.25 * M6_1_adj + 0.25 * RS_sector_pct
    Applies the same skip month as ``_momentum_components`` and silently ignores tickers with
    insufficient data (< MOMENTUM_MIN_SESSIONS).
    """
    cache_key = _universe_cache_key(universe_closes)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    if len(universe_closes) < MOMENTUM_MIN_SESSIONS:
        return pd.Series(dtype=float)

    valid = universe_closes.count() >= MOMENTUM_MIN_SESSIONS
    closes = universe_closes.loc[:, valid]
    returns = closes.pct_change()

    p_skip = closes.iloc[-MOMENTUM_SKIP_SESSIONS - 1]
    p_12m = closes.iloc[-MOMENTUM_LOOKBACK_12M - MOMENTUM_SKIP_SESSIONS - 1]
    p_6m = closes.iloc[-MOMENTUM_LOOKBACK_6M - MOMENTUM_SKIP_SESSIONS - 1]

    m12 = (p_skip / p_12m) - 1.0
    m6 = (p_skip / p_6m) - 1.0

    sigma_12m = (
        returns.iloc[-MOMENTUM_LOOKBACK_12M - MOMENTUM_SKIP_SESSIONS : -MOMENTUM_SKIP_SESSIONS]
        .std(ddof=0)
        .replace(0, np.nan)
    )
    sigma_6m = (
        returns.iloc[-MOMENTUM_LOOKBACK_6M - MOMENTUM_SKIP_SESSIONS : -MOMENTUM_SKIP_SESSIONS]
        .std(ddof=0)
        .replace(0, np.nan)
    )

    m12_adj = (m12 / sigma_12m).replace([np.inf, -np.inf], np.nan)
    m6_adj = (m6 / sigma_6m).replace([np.inf, -np.inf], np.nan)

    m12_clean = m12.replace([np.inf, -np.inf], np.nan).dropna()
    sector_pcts = _universe_sector_pcts(m12_clean).reindex(m12_adj.index).fillna(0.5)

    raw = (0.50 * m12_adj + 0.25 * m6_adj + 0.25 * sector_pcts).dropna()
    _UNIVERSE_RAW_CACHE[cache_key] = raw
    return raw


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_momentum_block(
    ticker: str,
    universe_closes: pd.DataFrame,
) -> dict[str, Any] | None:
    """Compute the Price Momentum block (0-10) for the given ticker.

    Incorporates the sector relative strength (RS) as a third internal component:
        Momentum_Raw = 0.50 * M12_1_adj + 0.25 * M6_1_adj + 0.25 * RS_sector_pct_12_1

    Args:
        ticker:          Asset symbol (must be present in universe_closes).
        universe_closes: DataFrame of daily closes of the reference universe (S&P 500 or MSCI World
                         in USD).

    Returns:
        Dict with keys:
            momentum_12_1, momentum_6_1, vol_12m, vol_6m,
            momentum_12_1_adj, momentum_6_1_adj,
            rs_sector_pct_12_1, rs_sector_pct_6_1, rs_sector_included,
            raw_momentum_score, z_score, normalized_score_0_10, signal, summary.
    """
    if ticker not in universe_closes.columns:
        raise ValueError(f"Ticker '{ticker}' not found in universe_closes.")

    m12_1, m6_1, sigma_12m, sigma_6m, m12_adj, m6_adj = _momentum_components(
        universe_closes[ticker]
    )

    rs_pct_12, rs_pct_6 = _sector_pcts_for_ticker(ticker, universe_closes, m12_1, m6_1)
    raw_score = 0.50 * m12_adj + 0.25 * m6_adj + 0.25 * rs_pct_12

    universe_raw = _universe_raw_scores(universe_closes)
    if universe_raw.empty:
        logger.warning(
            "Block 'momentum' returns None for %s: empty universe distribution.",
            ticker,
        )
        return None

    norm = robust_sigmoid_normalize(raw_score, universe_raw)
    if norm is None:
        logger.warning(
            "Block 'momentum' returns None for %s: insufficient robust normalization.",
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
