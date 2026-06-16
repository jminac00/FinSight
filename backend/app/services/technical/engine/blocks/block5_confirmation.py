"""
block5_confirmation.py
Bloque 5: Confirmation, peso 15% en el Technical Score.

Validacion tactica de corto plazo mediante Donchian 20 sesiones y volumen:
    DC_pos = (Close[-1] - low_20) / (high_20 - low_20)
    S_DC   = 2 * DC_pos - 1
    rel_vol = Volume[-1] / mean(Volume[-21:-1])
    S_Vol = S_DC * log(rel_vol)
    Confirmation_Raw = 0.6*S_DC + 0.4*S_Vol

La distribucion del universo usa un proxy basado en precios de cierre porque
universe_closes no contiene volumen. La normalizacion final usa z-score robusto
y sigmoide.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.technical.engine.utils.data_loader import get_price_history
from app.services.technical.engine.utils.normalization import (
    assign_signal,
    robust_sigmoid_normalize,
)

logger = logging.getLogger(__name__)

DONCHIAN_WIN = 20
_UNIVERSE_RAW_CACHE: dict[object, pd.Series] = {}


def _universe_cache_key(
    universe_closes: pd.DataFrame,
    universe_ohlcv: dict[str, pd.DataFrame] | None = None,
) -> object:
    closes_key = universe_closes.attrs.get("universe_name", id(universe_closes))
    if universe_ohlcv is None:
        return (closes_key, None)
    close_df = universe_ohlcv.get("Close")
    ohlcv_key = (
        close_df.attrs.get("universe_name", id(close_df))
        if close_df is not None
        else id(universe_ohlcv)
    )
    return (closes_key, ohlcv_key)


def _confirmation_components(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Devuelve (high_20, low_20, dc_pos, s_dc, vol_t, vol_20, rel_vol, s_vol)."""
    high_20 = float(high.iloc[-DONCHIAN_WIN - 1 : -1].max())
    low_20 = float(low.iloc[-DONCHIAN_WIN - 1 : -1].min())
    price_t = float(close.iloc[-1])
    vol_t = float(volume.iloc[-1])
    vol_20 = float(volume.iloc[-DONCHIAN_WIN - 1 : -1].mean())

    channel_range = high_20 - low_20
    if channel_range == 0:
        dc_pos = 0.5
        s_dc = 0.0
    else:
        dc_pos = (price_t - low_20) / channel_range
        s_dc = 2.0 * dc_pos - 1.0

    rel_vol = (vol_t / vol_20) if vol_20 > 0 else 1.0
    # Solo el volumen SUPERIOR a la media confirma la direccion.
    # log(max(rel_vol, 1.0)) = 0 cuando rel_vol <= 1 (sin penalizacion por volumen bajo),
    # positivo cuando rel_vol > 1 (amplificacion proporcional).
    # La formula anterior log(max(rel_vol, 1e-10)) invertia la senal cuando
    # rel_vol < 1, produciendo scores sistematicamente negativos en dias de
    # volumen bajo (ej. festivos, apertura de mes).
    log_rel_vol = float(np.log(max(rel_vol, 1.0)))
    s_vol = s_dc * log_rel_vol

    return high_20, low_20, dc_pos, s_dc, vol_t, vol_20, rel_vol, s_vol


def _universe_raw_scores_confirmation(
    universe_closes: pd.DataFrame,
    universe_ohlcv: dict[str, pd.DataFrame] | None = None,
) -> pd.Series:
    """
    Distribucion de Confirmation_Raw para el universo.

    Si universe_ohlcv esta disponible (High/Low/Close/Volume), calcula la misma
    formula exacta que se usa para cada ticker individual, incluido el termino de
    volumen con log(max(rel_vol, 1.0)). Esto elimina el sesgo sistematico que
    causaba que el 88%+ del universo puntuara por debajo de 5.

    Si universe_ohlcv no esta disponible, recae en el proxy de cierres (solo s_dc
    sin volumen), que es menos preciso pero evita errores.
    """
    cache_key = _universe_cache_key(universe_closes, universe_ohlcv)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    required = ("High", "Low", "Close", "Volume")
    if (
        universe_ohlcv is not None
        and all(f in universe_ohlcv for f in required)
        and len(universe_ohlcv["Close"]) >= DONCHIAN_WIN + 2
    ):
        high_df = universe_ohlcv["High"]
        low_df = universe_ohlcv["Low"]
        close_df = universe_ohlcv["Close"]
        vol_df = universe_ohlcv["Volume"]

        valid = close_df.count() >= DONCHIAN_WIN + 2
        high_df = high_df.loc[:, valid]
        low_df = low_df.loc[:, valid]
        close_df = close_df.loc[:, valid]
        vol_df = vol_df.reindex(columns=close_df.columns)

        high_20 = high_df.iloc[-DONCHIAN_WIN - 1 : -1].max(axis=0)
        low_20 = low_df.iloc[-DONCHIAN_WIN - 1 : -1].min(axis=0)
        price_t = close_df.iloc[-1]
        vol_t = vol_df.iloc[-1]
        vol_20 = vol_df.iloc[-DONCHIAN_WIN - 1 : -1].mean(axis=0).replace(0, np.nan)

        channel_range = (high_20 - low_20).replace(0, np.nan)
        dc_pos = (price_t - low_20) / channel_range
        s_dc = 2.0 * dc_pos - 1.0

        rel_vol = (vol_t / vol_20).clip(lower=1e-10)
        log_rel_vol = np.log(rel_vol.clip(lower=1.0))  # log(max(rel_vol, 1.0))
        s_vol = s_dc * log_rel_vol
        raw = (0.6 * s_dc + 0.4 * s_vol).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        # Fallback: proxy con cierres (sin volumen). Canal de close es mas estrecho
        # que el OHLCV real, lo que puede producir un sesgo hacia abajo, pero es
        # la unica opcion si no hay datos OHLCV completos.
        if len(universe_closes) < DONCHIAN_WIN + 2:
            return pd.Series(dtype=float)
        valid = universe_closes.count() >= DONCHIAN_WIN + 2
        closes = universe_closes.loc[:, valid]
        if closes.empty:
            return pd.Series(dtype=float)
        high_20 = closes.iloc[-DONCHIAN_WIN - 1 : -1].max(axis=0)
        low_20 = closes.iloc[-DONCHIAN_WIN - 1 : -1].min(axis=0)
        price_t = closes.iloc[-1]
        channel_range = (high_20 - low_20).replace(0, np.nan)
        dc_pos = (price_t - low_20) / channel_range
        s_dc = 2.0 * dc_pos - 1.0
        raw = s_dc.replace([np.inf, -np.inf], np.nan).dropna()

    _UNIVERSE_RAW_CACHE[cache_key] = raw
    return raw


def _ohlcv_from_universe_cache(
    ticker: str,
    universe_ohlcv: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame | None:
    """Construye OHLCV del ticker desde la descarga masiva del universo."""
    if not universe_ohlcv:
        return None
    required = ("High", "Low", "Close", "Volume")
    if any(field not in universe_ohlcv for field in required):
        return None
    if any(ticker not in universe_ohlcv[field].columns for field in required):
        return None

    ohlcv = pd.DataFrame({field: universe_ohlcv[field][ticker] for field in required}).dropna(
        subset=["High", "Low", "Close", "Volume"]
    )
    return None if ohlcv.empty else ohlcv


def compute_confirmation_block(
    ticker: str,
    universe_closes: pd.DataFrame,
    universe_ohlcv: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any] | None:
    """Calcula el bloque de Confirmation (0-10)."""
    ohlcv = _ohlcv_from_universe_cache(ticker, universe_ohlcv)
    if ohlcv is None:
        try:
            ohlcv = get_price_history(ticker, period="3y")
        except Exception as exc:
            raise ValueError(f"No se pudieron obtener datos OHLCV para '{ticker}': {exc}") from exc

    if len(ohlcv) < DONCHIAN_WIN + 2:
        raise ValueError(
            f"{ticker}: solo {len(ohlcv)} sesiones OHLCV; se necesitan al menos {DONCHIAN_WIN + 2}."
        )

    high_20, low_20, dc_pos, s_dc, vol_t, vol_20, rel_vol, s_vol = _confirmation_components(
        ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"]
    )
    raw_score = 0.6 * s_dc + 0.4 * s_vol

    universe_raw = _universe_raw_scores_confirmation(universe_closes, universe_ohlcv)
    if universe_raw.empty:
        logger.warning(
            "Bloque 'confirmation' devuelve None para %s: distribucion del universo vacia.",
            ticker,
        )
        return None

    norm = robust_sigmoid_normalize(raw_score, universe_raw)
    if norm is None:
        logger.warning(
            "Bloque 'confirmation' devuelve None para %s: normalizacion robusta insuficiente.",
            ticker,
        )
        return None
    z = float(norm["z_score"])
    score = float(norm["score_0_10"])
    signal = assign_signal(score, labels=("no confirmada", "neutral", "confirmada"))

    zone = "alta" if dc_pos > 0.7 else ("baja" if dc_pos < 0.3 else "central")
    if abs(s_dc) < 0.15:
        vol_note = "sin ruptura clara de rango"
    elif rel_vol > 1.0 and s_dc > 0:
        vol_note = "ruptura alcista respaldada por volumen"
    elif rel_vol > 1.0 and s_dc < 0:
        vol_note = "ruptura bajista respaldada por volumen"
    else:
        vol_note = "movimiento sin volumen superior a la media"

    summary = (
        f"{ticker} cotiza en zona {zone} del canal Donchian 20d "
        f"(posicion={dc_pos:.2f}, S_DC={s_dc:.3f}). "
        f"Volumen relativo={rel_vol:.2f}x; {vol_note}. "
        f"Score confirmacion {score:.2f}/10."
    )

    return {
        "upper_donchian_20": high_20,
        "lower_donchian_20": low_20,
        "donchian_position": dc_pos,
        "donchian_score": s_dc,
        "current_volume": vol_t,
        "average_volume_20": vol_20,
        "relative_volume_20": rel_vol,
        "volume_confirmation_score": s_vol,
        "confirmation_raw_score": raw_score,
        "confirmation_z_score": z,
        "confirmation_score_0_10": score,
        "normalization_method": norm["method"],
        "normalization_k": norm["k"],
        "signal": signal,
        "summary": summary,
    }
