"""
technical_score.py
Orquestador del Technical Score final (0-10).

Pesos nominales:
    Price Momentum      35%  (incluye RS sector como componente interno)
    Trend               30%
    Risk / Stability    20%
    Confirmation        15%

Si un bloque devuelve None o falla, su peso se redistribuye proporcionalmente
entre los bloques disponibles. La descarga del universo S&P 500 se cachea en
memoria durante la sesion e incluye SPY de forma explicita para el benchmark.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.services.technical.engine.blocks.block1_momentum import compute_momentum_block
from app.services.technical.engine.blocks.block2_trend import compute_trend_block
from app.services.technical.engine.blocks.block4_risk_stability import compute_risk_stability_block
from app.services.technical.engine.blocks.block5_confirmation import compute_confirmation_block
from app.services.technical.engine.utils.data_loader import (
    get_fx_to_usd_series,
    get_last_universe_ohlcv,
    get_price_history,
    get_sp500_tickers,
    get_universe_closes,
)
from app.services.technical.engine.utils.normalization import assign_signal

logger = logging.getLogger(__name__)

FMP_API_KEY: str | None = os.environ.get("FMP_API_KEY")

_WEIGHTS: dict[str, float] = {
    "momentum": 0.35,  # incluye RS sector como componente interno
    "trend": 0.30,
    "risk_stability": 0.20,
    "confirmation": 0.15,
}

_SCORE_KEY: dict[str, str] = {
    "momentum": "normalized_score_0_10",
    "trend": "trend_score_0_10",
    "risk_stability": "risk_stability_score_0_10",
    "confirmation": "confirmation_score_0_10",
}

_universe_cache: dict[str, Any] | None = None
_technical_universe_cache: dict[str, dict[str, Any]] = {}


def _normalize_ticker(ticker: str) -> str:
    """Homogeneiza el ticker: uppercase y puntos sustituidos por guiones."""
    return ticker.strip().upper().replace(".", "-")


def _normalize_global_ticker(ticker: str) -> str:
    """Homogeneiza tickers internacionales preservando sufijos de yfinance."""
    return ticker.strip().upper()


def _canonical_universe(universe: str) -> str:
    name = (universe or "sp500").strip().lower()
    if name in {"sp500", "domestic"}:
        return "sp500"
    if name in {"msci_world", "global"}:
        return "msci_world"
    raise KeyError(f"Universo tecnico no soportado: {universe}")


def _tag_ohlcv_cache(
    universe_ohlcv: dict[str, pd.DataFrame] | None,
    universe_name: str,
) -> dict[str, pd.DataFrame] | None:
    if universe_ohlcv is None:
        return None
    for frame in universe_ohlcv.values():
        try:
            frame.attrs["universe_name"] = universe_name
        except Exception:  # noqa: BLE001
            pass
    return universe_ohlcv


def _get_cached_universe() -> dict[str, Any]:
    """
    Descarga el universo S&P 500 la primera vez y usa cache despues.

    SPY se anade explicitamente a la descarga masiva aunque no sea componente
    del S&P 500 para evitar descargas repetidas en Relative Strength.
    """
    global _universe_cache
    if _universe_cache is None:
        logger.info("Descargando universo S&P 500 (primera llamada)...")
        sp500 = get_sp500_tickers()
        tickers_con_spy = sp500 + ["SPY"] if "SPY" not in sp500 else sp500
        closes = get_universe_closes(tickers_con_spy)
        closes.attrs["universe_name"] = "sp500_closes"
        _universe_cache = {
            "tickers": sp500,
            "closes": closes,
            "ohlcv": _tag_ohlcv_cache(get_last_universe_ohlcv(), "sp500_ohlcv"),
        }
    return _universe_cache


def _get_cached_global_universe(universe: str) -> dict[str, Any]:
    """
    Descarga/construye el universo tecnico global una vez por ejecucion.

    Los cierres llegan ya convertidos a USD desde UniverseManager.
    """
    universe = _canonical_universe(universe)
    cache_key = "msci_world_closes_usd" if universe == "msci_world" else "sp500_closes"
    if cache_key in _technical_universe_cache:
        return _technical_universe_cache[cache_key]

    if universe == "sp500":
        cached = _get_cached_universe()
        _technical_universe_cache[cache_key] = cached
        return cached

    from fundamental.universe_manager import get_universe_manager

    logger.info("Descargando universo tecnico %s en USD (primera llamada)...", universe)
    mgr = get_universe_manager()
    cfg = mgr.get_config(universe)
    closes = mgr.get_universe_closes(universe=universe)
    closes.attrs["universe_name"] = cache_key
    cached = {
        "tickers": cfg.tickers,
        "closes": closes,
        "ohlcv": None,
    }
    _technical_universe_cache[cache_key] = cached
    return cached


def _universe_with_ticker(
    ticker: str,
    universe_closes: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """
    Si el ticker no esta en el universo descargado, intenta anadir su historico
    individual temporalmente para mantener la comparacion contra S&P 500.
    """
    if ticker in universe_closes.columns:
        return universe_closes, None

    try:
        history = get_price_history(ticker, period="3y")
        close = history["Close"].rename(ticker)
        closes = universe_closes.copy()
        closes[ticker] = close.reindex(closes.index).ffill()
        if closes[ticker].dropna().empty:
            return universe_closes, (
                f"Ticker '{ticker}' no pertenece al universo S&P 500 y no se pudo "
                "alinear su historico individual."
            )
        return closes, None
    except Exception as exc:
        return universe_closes, (
            f"Ticker '{ticker}' no pertenece al universo S&P 500 descargado y "
            f"no se pudo descargar su historico individual: {exc}"
        )


def _universe_with_ticker_usd(
    ticker: str,
    universe_closes: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """
    Si el ticker no está en el universo global (cierres ya en USD), lo descarga
    individualmente, aplica conversión FX y lo añade temporalmente.

    Equivalente a _universe_with_ticker para el path sp500, pero con la capa FX
    necesaria en universos multi-divisa.  Se usa cuando el ticker falla en la
    descarga masiva del universo (p. ej. por rate limiting de yfinance).
    """
    if ticker in universe_closes.columns:
        return universe_closes, None

    try:
        history = get_price_history(ticker, period="3y")
        close_local = history["Close"]

        # Moneda: currency map MSCI World → yfinance info → default USD
        ccy = "USD"
        try:
            from fundamental.msci_world import load_currency_map

            ccy = (load_currency_map().get(ticker) or "USD").strip().upper()
        except Exception:  # noqa: BLE001
            pass
        if ccy == "USD":
            try:
                import yfinance as _yf

                ccy = (_yf.Ticker(ticker).info.get("currency") or "USD").strip().upper()
            except Exception:  # noqa: BLE001
                pass

        close_usd = close_local
        if ccy != "USD":
            fx = get_fx_to_usd_series(ccy, close_local.index, period="3y")
            if fx is not None:
                close_usd = close_local * fx
            else:
                logger.warning(
                    "FX no disponible para %s (%s); precio local añadido al universo global.",
                    ticker,
                    ccy,
                )

        closes = universe_closes.copy()
        closes[ticker] = close_usd.rename(ticker).reindex(closes.index).ffill()
        if closes[ticker].dropna().empty:
            return universe_closes, (
                f"Ticker '{ticker}' no pertenece al universo MSCI World y no se pudo "
                "alinear su historico individual."
            )
        return closes, None
    except Exception as exc:
        return universe_closes, (
            f"Ticker '{ticker}' no pertenece al universo MSCI World y "
            f"fallo la descarga individual: {exc}"
        )


def compute_technical_score(ticker: str, universe: str = "sp500") -> dict[str, Any]:
    """
    Calcula el Technical Score completo (0-10) para el ticker indicado.

    Mantiene las claves publicas principales:
    ticker, technical_score, signal, weights, effective_weights,
    individual_scores y blocks. Anade errors para explicar fallos por bloque.
    """
    if _canonical_universe(universe) == "sp500":
        ticker = _normalize_ticker(ticker)
        logger.info("Calculando Technical Score para %s...", ticker)

        universe = _get_cached_universe()
        universe_closes: pd.DataFrame = universe["closes"]
        universe_ohlcv: dict[str, pd.DataFrame] | None = universe.get("ohlcv")
        universe_closes, ticker_error = _universe_with_ticker(ticker, universe_closes)

        blocks: dict[str, dict | None] = {}
        errors: dict[str, str] = {}
        if ticker_error is not None:
            errors["ticker"] = ticker_error

        block_specs = [
            ("momentum", compute_momentum_block, {}),
            ("trend", compute_trend_block, {}),
            ("risk_stability", compute_risk_stability_block, {}),
            ("confirmation", compute_confirmation_block, {"universe_ohlcv": universe_ohlcv}),
        ]

        for name, fn, kwargs in block_specs:
            if ticker_error is not None:
                blocks[name] = None
                errors[name] = ticker_error
                continue

            try:
                result = fn(ticker, universe_closes, **kwargs)
                blocks[name] = result
                if result is None:
                    errors[name] = (
                        "El bloque devolvio None por datos insuficientes o por una "
                        "distribucion de universo no normalizable."
                    )
            except Exception as exc:
                logger.error("Bloque '%s' fallido para %s: %s", name, ticker, exc)
                blocks[name] = None
                errors[name] = str(exc)

        individual_scores: dict[str, float | None] = {
            name: (result.get(_SCORE_KEY[name]) if result is not None else None)
            for name, result in blocks.items()
        }

        available = {k: v for k, v in individual_scores.items() if v is not None}
        total_nominal = sum(_WEIGHTS[k] for k in available)

        data_completeness: float = total_nominal  # fraccion del peso nominal con datos [0, 1]

        if total_nominal == 0:
            technical_score: float | None = None
            signal_global = "sin_datos"
            effective_weights: dict[str, float] = {}
        else:
            weighted_sum = sum(
                (_WEIGHTS[k] / total_nominal) * score for k, score in available.items()
            )
            technical_score = round(float(weighted_sum), 2)
            signal_global = assign_signal(
                technical_score,
                thresholds=(4.0, 6.5),
                labels=("bajista", "neutral", "alcista"),
            )
            effective_weights = {k: round(_WEIGHTS[k] / total_nominal, 4) for k in available}

        # Un score es fiable cuando al menos el 60% del peso nominal tiene datos validos.
        # Con pesos 35/30/20/15, el minimo fiable es momentum+trend (65%) o cualquier
        # combinacion que supere el umbral. Sin momentum ni trend el score no es comparable.
        score_reliable: bool = data_completeness >= 0.60

        return {
            "ticker": ticker,
            "technical_score": technical_score,
            "technical_signal": signal_global,
            "momentum_score": individual_scores.get("momentum"),
            "trend_score": individual_scores.get("trend"),
            "risk_stability_score": individual_scores.get("risk_stability"),
            "confirmation_score": individual_scores.get("confirmation"),
            "summary": _build_technical_summary(
                ticker, technical_score, signal_global, individual_scores
            ),
            "signal": signal_global,
            "weights": _WEIGHTS,
            "effective_weights": effective_weights,
            "individual_scores": individual_scores,
            "data_completeness": round(data_completeness, 4),
            "score_reliable": score_reliable,
            "errors": errors,
            "blocks": blocks,
        }

    return _compute_technical_score_global(ticker, universe)


def _compute_technical_score_global(ticker: str, universe: str) -> dict[str, Any]:
    """Calcula Technical Score contra un universo global ya convertido a USD."""
    ticker = _normalize_global_ticker(ticker)
    universe_name = _canonical_universe(universe)
    logger.info("Calculando Technical Score para %s contra %s...", ticker, universe_name)

    cached_universe = _get_cached_global_universe(universe_name)
    universe_closes: pd.DataFrame = cached_universe["closes"]
    universe_ohlcv: dict[str, pd.DataFrame] | None = cached_universe.get("ohlcv")

    blocks: dict[str, dict | None] = {}
    errors: dict[str, str] = {}
    universe_closes, ticker_error = _universe_with_ticker_usd(ticker, universe_closes)
    if ticker_error is not None:
        errors["ticker"] = ticker_error

    block_specs = [
        ("momentum", compute_momentum_block, {}),
        ("trend", compute_trend_block, {}),
        ("risk_stability", compute_risk_stability_block, {}),
        ("confirmation", compute_confirmation_block, {"universe_ohlcv": universe_ohlcv}),
    ]

    for name, fn, kwargs in block_specs:
        if ticker_error is not None:
            blocks[name] = None
            errors[name] = ticker_error
            continue

        try:
            result = fn(ticker, universe_closes, **kwargs)
            blocks[name] = result
            if result is None:
                errors[name] = (
                    "El bloque devolvio None por datos insuficientes o por una "
                    "distribucion de universo no normalizable."
                )
        except Exception as exc:
            logger.error("Bloque '%s' fallido para %s: %s", name, ticker, exc)
            blocks[name] = None
            errors[name] = str(exc)

    individual_scores: dict[str, float | None] = {
        name: (result.get(_SCORE_KEY[name]) if result is not None else None)
        for name, result in blocks.items()
    }

    available = {k: v for k, v in individual_scores.items() if v is not None}
    total_nominal = sum(_WEIGHTS[k] for k in available)

    data_completeness: float = total_nominal

    if total_nominal == 0:
        technical_score: float | None = None
        signal_global = "sin_datos"
        effective_weights: dict[str, float] = {}
    else:
        weighted_sum = sum((_WEIGHTS[k] / total_nominal) * score for k, score in available.items())
        technical_score = round(float(weighted_sum), 2)
        signal_global = assign_signal(
            technical_score,
            thresholds=(4.0, 6.5),
            labels=("bajista", "neutral", "alcista"),
        )
        effective_weights = {k: round(_WEIGHTS[k] / total_nominal, 4) for k in available}

    score_reliable: bool = data_completeness >= 0.60

    return {
        "ticker": ticker,
        "technical_score": technical_score,
        "technical_signal": signal_global,
        "momentum_score": individual_scores.get("momentum"),
        "trend_score": individual_scores.get("trend"),
        "risk_stability_score": individual_scores.get("risk_stability"),
        "confirmation_score": individual_scores.get("confirmation"),
        "summary": _build_technical_summary(
            ticker, technical_score, signal_global, individual_scores
        ),
        "signal": signal_global,
        "weights": _WEIGHTS,
        "effective_weights": effective_weights,
        "individual_scores": individual_scores,
        "data_completeness": round(data_completeness, 4),
        "score_reliable": score_reliable,
        "errors": errors,
        "blocks": blocks,
        "universe": universe_name,
    }


def _build_technical_summary(
    ticker: str,
    score: float | None,
    signal: str,
    individual_scores: dict[str, float | None],
) -> str:
    if score is None:
        return f"{ticker}: technical score no calculable por datos insuficientes."
    parts = [
        f"Momentum={individual_scores.get('momentum')}",
        f"Trend={individual_scores.get('trend')}",
        f"Risk/Stability={individual_scores.get('risk_stability')}",
        f"Confirmation={individual_scores.get('confirmation')}",
    ]
    return (
        f"{ticker}: Technical Score {score:.2f}/10 ({signal}). "
        "Modelo oficial de 4 bloques: " + ", ".join(parts) + "."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = compute_technical_score(test_ticker)
    print(json.dumps(result, indent=2, default=str))
