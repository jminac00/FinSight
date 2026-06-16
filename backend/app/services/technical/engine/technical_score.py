"""Final Technical Score orchestrator (0-10).

Nominal weights:
    Price Momentum      35%  (includes sector RS as an internal component)
    Trend               30%
    Risk / Stability    20%
    Confirmation        15%

If a block returns None or fails, its weight is redistributed proportionally among the available
blocks.

The reference universe (closes + OHLCV) can be injected by the caller; this is how the service
feeds the frozen on-disk snapshot. When it is not injected the engine falls back to downloading the
S&P 500 universe and caching it in memory for the session.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.services.technical.engine.blocks.block1_momentum import compute_momentum_block
from app.services.technical.engine.blocks.block2_trend import compute_trend_block
from app.services.technical.engine.blocks.block4_risk_stability import compute_risk_stability_block
from app.services.technical.engine.blocks.block5_confirmation import compute_confirmation_block
from app.services.technical.engine.config import (
    BLOCK_WEIGHTS,
    RELIABILITY_THRESHOLD,
    SCORE_KEYS,
    SIGNAL_THRESHOLDS,
)
from app.services.technical.engine.utils.data_loader import (
    get_fx_to_usd_series,
    get_last_universe_ohlcv,
    get_price_history,
    get_sp500_tickers,
    get_universe_closes,
)
from app.services.technical.engine.utils.normalization import assign_signal

logger = logging.getLogger(__name__)

_BLOCK_NONE_REASON = (
    "The block returned None because of insufficient data or a non-normalizable universe "
    "distribution."
)

_universe_cache: dict[str, Any] | None = None
_technical_universe_cache: dict[str, dict[str, Any]] = {}


def _normalize_ticker(ticker: str) -> str:
    """Normalize the ticker: uppercase and dots replaced by dashes."""
    return ticker.strip().upper().replace(".", "-")


def _normalize_global_ticker(ticker: str) -> str:
    """Normalize international tickers, preserving yfinance suffixes."""
    return ticker.strip().upper()


def _canonical_universe(universe: str) -> str:
    name = (universe or "sp500").strip().lower()
    if name in {"sp500", "domestic"}:
        return "sp500"
    if name in {"msci_world", "global"}:
        return "msci_world"
    raise KeyError(f"Unsupported technical universe: {universe}")


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
    """Download the S&P 500 universe on first use and cache it afterwards.

    SPY is added explicitly to the bulk download (even though it is not an S&P 500 constituent) to
    avoid repeated downloads in Relative Strength.
    """
    global _universe_cache
    if _universe_cache is None:
        logger.info("Downloading S&P 500 universe (first call)...")
        sp500 = get_sp500_tickers()
        tickers_with_spy = sp500 + ["SPY"] if "SPY" not in sp500 else sp500
        closes = get_universe_closes(tickers_with_spy)
        closes.attrs["universe_name"] = "sp500_closes"
        _universe_cache = {
            "tickers": sp500,
            "closes": closes,
            "ohlcv": _tag_ohlcv_cache(get_last_universe_ohlcv(), "sp500_ohlcv"),
        }
    return _universe_cache


def _get_cached_global_universe(universe: str) -> dict[str, Any]:
    """Download/build the global technical universe once per execution.

    The closes arrive already converted to USD from the universe manager.

    NOTE: this internal download is rewired to the frozen on-disk snapshot by the technical universe
    manager; it is only reached when the global universe is requested without an injected universe.
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

    logger.info("Downloading %s technical universe in USD (first call)...", universe)
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
    """If the ticker is not in the downloaded universe, try to append its individual history
    temporarily to keep the comparison against the S&P 500.
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
                f"Ticker '{ticker}' is not in the S&P 500 universe and its individual history "
                "could not be aligned."
            )
        return closes, None
    except Exception as exc:
        return universe_closes, (
            f"Ticker '{ticker}' is not in the downloaded S&P 500 universe and its individual "
            f"history could not be downloaded: {exc}"
        )


def _universe_with_ticker_usd(
    ticker: str,
    universe_closes: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """If the ticker is not in the global universe (closes already in USD), download it
    individually, apply the FX conversion and append it temporarily.

    Equivalent to _universe_with_ticker for the sp500 path, but with the FX layer needed in
    multi-currency universes. Used when the ticker is missing from the bulk universe download
    (e.g. yfinance rate limiting).
    """
    if ticker in universe_closes.columns:
        return universe_closes, None

    try:
        history = get_price_history(ticker, period="3y")
        close_local = history["Close"]

        # Currency: MSCI World currency map -> yfinance info -> default USD
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
                    "FX not available for %s (%s); local price appended to the global universe.",
                    ticker,
                    ccy,
                )

        closes = universe_closes.copy()
        closes[ticker] = close_usd.rename(ticker).reindex(closes.index).ffill()
        if closes[ticker].dropna().empty:
            return universe_closes, (
                f"Ticker '{ticker}' is not in the MSCI World universe and its individual history "
                "could not be aligned."
            )
        return closes, None
    except Exception as exc:
        return universe_closes, (
            f"Ticker '{ticker}' is not in the MSCI World universe and its individual download "
            f"failed: {exc}"
        )


def _run_blocks_and_aggregate(
    ticker: str,
    universe_closes: pd.DataFrame,
    universe_ohlcv: dict[str, pd.DataFrame] | None,
    ticker_error: str | None,
) -> dict[str, Any]:
    """Run the four blocks, aggregate their scores and build the result dict.

    Universe-agnostic: shared by the S&P 500 and global paths. Does not add the optional
    ``universe`` key (the global path adds it).
    """
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
                errors[name] = _BLOCK_NONE_REASON
        except Exception as exc:
            logger.error("Block '%s' failed for %s: %s", name, ticker, exc)
            blocks[name] = None
            errors[name] = str(exc)

    individual_scores: dict[str, float | None] = {
        name: (result.get(SCORE_KEYS[name]) if result is not None else None)
        for name, result in blocks.items()
    }

    available = {k: v for k, v in individual_scores.items() if v is not None}
    total_nominal = sum(BLOCK_WEIGHTS[k] for k in available)

    data_completeness: float = total_nominal  # fraction of the nominal weight with data [0, 1]

    if total_nominal == 0:
        technical_score: float | None = None
        signal_global = "sin_datos"
        effective_weights: dict[str, float] = {}
    else:
        weighted_sum = sum(
            (BLOCK_WEIGHTS[k] / total_nominal) * score for k, score in available.items()
        )
        technical_score = round(float(weighted_sum), 2)
        signal_global = assign_signal(
            technical_score,
            thresholds=SIGNAL_THRESHOLDS,
            labels=("bajista", "neutral", "alcista"),
        )
        effective_weights = {k: round(BLOCK_WEIGHTS[k] / total_nominal, 4) for k in available}

    # A score is reliable when at least 60% of the nominal weight has valid data. With weights
    # 35/30/20/15 the reliable minimum is momentum+trend (65%) or any combination above the
    # threshold. Without momentum or trend the score is not comparable.
    score_reliable: bool = data_completeness >= RELIABILITY_THRESHOLD

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
        "weights": BLOCK_WEIGHTS,
        "effective_weights": effective_weights,
        "individual_scores": individual_scores,
        "data_completeness": round(data_completeness, 4),
        "score_reliable": score_reliable,
        "errors": errors,
        "blocks": blocks,
    }


def compute_technical_score(
    ticker: str,
    universe: str = "sp500",
    *,
    universe_closes: pd.DataFrame | None = None,
    universe_ohlcv: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Compute the full Technical Score (0-10) for the given ticker.

    Args:
        ticker: Stock symbol.
        universe: Reference universe: ``sp500``/``domestic`` or ``msci_world``/``global``.
        universe_closes: Optional injected universe closes (tickers in columns). When provided the
            engine skips the internal download and normalizes against it. The caller must set
            ``universe_closes.attrs['universe_name']`` so the per-block caches key correctly.
        universe_ohlcv: Optional injected universe OHLCV (High/Low/Close/Volume), used by the
            confirmation block.

    Returns:
        Result dict with technical_score, signal, weights, effective_weights, individual_scores and
        the per-block detail; the global path also adds a ``universe`` key.
    """
    canonical = _canonical_universe(universe)

    if canonical == "sp500":
        ticker = _normalize_ticker(ticker)
        logger.info("Computing Technical Score for %s...", ticker)
        if universe_closes is None:
            cached = _get_cached_universe()
            universe_closes = cached["closes"]
            universe_ohlcv = cached.get("ohlcv")
        universe_closes, ticker_error = _universe_with_ticker(ticker, universe_closes)
        return _run_blocks_and_aggregate(ticker, universe_closes, universe_ohlcv, ticker_error)

    ticker = _normalize_global_ticker(ticker)
    logger.info("Computing Technical Score for %s against %s...", ticker, canonical)
    if universe_closes is None:
        cached = _get_cached_global_universe(canonical)
        universe_closes = cached["closes"]
        universe_ohlcv = cached.get("ohlcv")
    universe_closes, ticker_error = _universe_with_ticker_usd(ticker, universe_closes)
    result = _run_blocks_and_aggregate(ticker, universe_closes, universe_ohlcv, ticker_error)
    result["universe"] = canonical
    return result


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
