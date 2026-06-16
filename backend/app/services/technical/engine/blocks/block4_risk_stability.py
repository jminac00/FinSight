"""Block 4: Risk / Stability — 20% weight in the Technical Score.

Evaluates the robustness of the move over the last 126 sessions:
    S_DD      = -abs(max_drawdown)
    S_Risk    = -downside_volatility
    S_Clarity = R2 of the log-price regression

Each component is normalized separately against the universe:
    Risk Stability Raw = 0.4*z_S_DD + 0.3*z_S_Risk + 0.3*z_S_Clarity

Then the shared normalization is applied to the composite score:
    raw -> robust z-score -> sigmoid -> 0-10 scale
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.services.technical.engine.config import RISK_MIN_NEG_RETURNS, RISK_WINDOW
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


def _risk_stability_components(
    closes: pd.Series,
) -> tuple[float, float, float, float, float, float]:
    """Return (max_dd, downside_vol, r2, s_dd, s_risk, s_clarity)."""
    closes = closes.dropna().iloc[-RISK_WINDOW:]
    if len(closes) < RISK_WINDOW:
        raise ValueError(f"Only {len(closes)} sessions; {RISK_WINDOW} are required.")

    rolling_max = closes.expanding().max()
    drawdowns = (closes - rolling_max) / rolling_max
    max_dd = float(drawdowns.min())
    s_dd = -abs(max_dd)

    daily_rets = closes.pct_change().dropna()
    neg_rets = daily_rets[daily_rets < 0]
    if len(neg_rets) >= RISK_MIN_NEG_RETURNS:
        downside_vol = float(neg_rets.std(ddof=0))
    else:
        downside_vol = float(daily_rets.std(ddof=0))
    s_risk = -downside_vol

    log_p = np.log(closes.values.astype(float))
    t_arr = np.arange(len(log_p), dtype=float)
    _, _, r_value, _, _ = stats.linregress(t_arr, log_p)
    r2 = float(r_value**2)
    s_clarity = r2

    return max_dd, downside_vol, r2, s_dd, s_risk, s_clarity


def _robust_z(value: float, universe_series: pd.Series) -> float | None:
    """Robust z-score of a component against its universe distribution."""
    return compute_robust_zscore(value, universe_series)


def _universe_components_risk(universe_closes: pd.DataFrame) -> pd.DataFrame:
    """Compute S_DD, S_Risk and S_Clarity for the universe."""
    cache_key = _universe_cache_key(universe_closes)
    if cache_key in _UNIVERSE_COMPONENTS_CACHE:
        return _UNIVERSE_COMPONENTS_CACHE[cache_key]

    if len(universe_closes) < RISK_WINDOW:
        return pd.DataFrame(columns=["s_dd", "s_risk", "s_clarity"])

    valid = universe_closes.count() >= RISK_WINDOW
    closes = universe_closes.loc[:, valid].iloc[-RISK_WINDOW:]
    if closes.empty:
        return pd.DataFrame(columns=["s_dd", "s_risk", "s_clarity"])

    rolling_max = closes.expanding().max()
    drawdowns = (closes - rolling_max) / rolling_max
    max_dd_univ = drawdowns.min(axis=0)
    s_dd_univ = -max_dd_univ.abs()

    log_p = np.log(closes.values.astype(float))
    t_arr = np.arange(RISK_WINDOW, dtype=float)
    t_c = t_arr - t_arr.mean()
    t_ss = float((t_c**2).sum())
    y_bar = log_p.mean(axis=0)
    y_c = log_p - y_bar
    y_ss = (y_c**2).sum(axis=0)
    cov_num = t_c @ y_c
    r2_arr = np.clip(cov_num**2 / (t_ss * y_ss + 1e-20), 0.0, 1.0)
    s_clarity_univ = pd.Series(r2_arr, index=closes.columns)

    daily_rets = closes.pct_change()
    s_risk_dict: dict[str, float] = {}
    for col in closes.columns:
        dr = daily_rets[col].dropna()
        neg = dr[dr < 0]
        if len(dr) < 2:
            continue
        dv = float(neg.std(ddof=0)) if len(neg) >= RISK_MIN_NEG_RETURNS else float(dr.std(ddof=0))
        s_risk_dict[col] = -dv
    s_risk_univ = pd.Series(s_risk_dict)

    components = pd.DataFrame(
        {
            "s_dd": s_dd_univ,
            "s_risk": s_risk_univ,
            "s_clarity": s_clarity_univ,
        }
    )
    components = components.replace([np.inf, -np.inf], np.nan).dropna()
    _UNIVERSE_COMPONENTS_CACHE[cache_key] = components
    return components


def _universe_raw_scores_risk(universe_closes: pd.DataFrame) -> pd.Series:
    """Universe Risk Stability Raw after normalizing components separately."""
    cache_key = _universe_cache_key(universe_closes)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    components = _universe_components_risk(universe_closes)
    if components.empty:
        return pd.Series(dtype=float)

    z_dd = components["s_dd"].apply(lambda value: _robust_z(float(value), components["s_dd"]))
    z_risk = components["s_risk"].apply(lambda value: _robust_z(float(value), components["s_risk"]))
    z_clarity = components["s_clarity"].apply(
        lambda value: _robust_z(float(value), components["s_clarity"])
    )
    raw = 0.4 * z_dd + 0.3 * z_risk + 0.3 * z_clarity
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    _UNIVERSE_RAW_CACHE[cache_key] = raw
    return raw


def compute_risk_stability_block(
    ticker: str,
    universe_closes: pd.DataFrame,
) -> dict[str, Any] | None:
    """Compute the Risk/Stability block (0-10)."""
    if ticker not in universe_closes.columns:
        raise ValueError(f"Ticker '{ticker}' not in universe_closes.")

    max_dd, downside_vol, r2, s_dd, s_risk, s_clarity = _risk_stability_components(
        universe_closes[ticker]
    )

    components = _universe_components_risk(universe_closes)
    if components.empty:
        logger.warning(
            "Block 'risk_stability' returns None for %s: empty universe components.",
            ticker,
        )
        return None

    s_dd_z = _robust_z(s_dd, components["s_dd"])
    s_risk_z = _robust_z(s_risk, components["s_risk"])
    s_clarity_z = _robust_z(s_clarity, components["s_clarity"])
    if s_dd_z is None or s_risk_z is None or s_clarity_z is None:
        logger.warning(
            "Block 'risk_stability' returns None for %s: insufficient internal normalization.",
            ticker,
        )
        return None

    raw_score = 0.4 * s_dd_z + 0.3 * s_risk_z + 0.3 * s_clarity_z
    universe_raw = _universe_raw_scores_risk(universe_closes)
    if universe_raw.empty:
        logger.warning(
            "Block 'risk_stability' returns None for %s: empty composite distribution.",
            ticker,
        )
        return None

    norm = robust_sigmoid_normalize(raw_score, universe_raw)
    if norm is None:
        logger.warning(
            "Block 'risk_stability' returns None for %s: insufficient robust normalization.",
            ticker,
        )
        return None
    z = float(norm["z_score"])
    score = float(norm["score_0_10"])
    signal = assign_signal(score, labels=("fragil", "intermedio", "estable"))

    quality = "estable" if score > 6.5 else ("fragil" if score < 4.0 else "intermedio")
    summary = (
        f"{ticker} muestra un perfil de riesgo {quality}. "
        f"Max drawdown 126d={max_dd * 100:.1f}%, "
        f"downside vol={downside_vol * 100:.2f}%/dia, "
        f"R2_trayectoria={r2:.3f}. "
        f"Score estabilidad {score:.2f}/10."
    )

    return {
        "max_drawdown_126d": max_dd,
        "downside_volatility_126d": downside_vol,
        "regression_r2_126d": r2,
        "s_dd": s_dd,
        "s_risk": s_risk,
        "s_clarity": s_clarity,
        "s_dd_z_score": s_dd_z,
        "s_risk_z_score": s_risk_z,
        "s_clarity_z_score": s_clarity_z,
        "risk_stability_raw_score": raw_score,
        "risk_stability_z_score": z,
        "risk_stability_score_0_10": score,
        "normalization_method": norm["method"],
        "normalization_k": norm["k"],
        "signal": signal,
        "summary": summary,
    }
