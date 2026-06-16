"""
data_loader.py
Descarga y caché de datos de precios, tickers del S&P 500 e información sectorial.

Todos los precios son ajustados (auto_adjust=True en yfinance).
La descarga del universo completo es la operación más costosa (~30-60 s);
se cachea en memoria durante la sesión desde technical_score.py.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Cachés en memoria de sesión ───────────────────────────────────────────────
_sp500_tickers_cache: list[str] | None = None
_sector_cache: dict[str, str] = {}  # ticker → nombre de sector
_last_universe_ohlcv_cache: dict[str, pd.DataFrame] | None = None


# ---------------------------------------------------------------------------
# Precios
# ---------------------------------------------------------------------------


def get_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    """
    Descarga el histórico OHLCV ajustado para un ticker.

    Returns:
        DataFrame con columnas Open, High, Low, Close, Volume e índice DatetimeIndex.

    Raises:
        ValueError: si el ticker no existe o el histórico queda vacío.
    """
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No se encontraron datos de precios para '{ticker}'.")
    # yfinance con un solo ticker devuelve columnas planas (no MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data = data.dropna(subset=["Close"])
    if data.empty:
        raise ValueError(f"Historico de precios vacio tras eliminar NaN para '{ticker}'.")
    return data


# ---------------------------------------------------------------------------
# Universo S&P 500
# ---------------------------------------------------------------------------


def get_sp500_tickers() -> list[str]:
    """
    Obtiene la lista de tickers del S&P 500 desde Wikipedia.
    Cachea el resultado en memoria para la sesión actual.

    Nota metodologica: este universo actual sirve para analisis actual, pero
    no para backtesting historico sin sesgo de supervivencia. Para backtests se
    necesita la composicion historica del indice.
    """
    global _sp500_tickers_cache
    if _sp500_tickers_cache is not None:
        return _sp500_tickers_cache

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    _sp500_tickers_cache = tickers
    logger.info("Tickers S&P 500 cargados: %d", len(tickers))
    return tickers


def get_universe_closes(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """
    Descarga los cierres ajustados de todos los tickers en una sola llamada.

    - Elimina columnas con más del 20 % de valores nulos.
    - Rellena los nulos restantes con ffill.

    Returns:
        DataFrame con tickers como columnas y fechas como índice.
    """
    if not tickers:
        raise ValueError("Lista de tickers vacia para descargar universo.")

    global _last_universe_ohlcv_cache

    logger.info("Descargando cierres del universo (%d tickers)...", len(tickers))
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("La descarga del universo devolvio un DataFrame vacio.")

    def field_frame(field: str) -> pd.DataFrame | None:
        if isinstance(data.columns, pd.MultiIndex):
            if field not in data.columns.get_level_values(0):
                return None
            return data[field]
        if field in data.columns:
            name = tickers[0] if len(tickers) == 1 else field
            return data[[field]].rename(columns={field: name})
        return None

    # Con múltiples tickers yfinance devuelve MultiIndex (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("La descarga del universo no contiene columnas Close.")
        closes = data["Close"]
    else:
        if "Close" in data.columns:
            name = tickers[0] if len(tickers) == 1 else "Close"
            closes = data[["Close"]].rename(columns={"Close": name})
        else:
            closes = data

    if closes.empty:
        raise ValueError("No hay cierres disponibles para el universo.")

    # Eliminar tickers con cobertura insuficiente
    threshold = 0.20
    valid = closes.isnull().mean() <= threshold
    closes = closes.loc[:, valid]
    closes = closes.ffill()
    closes = closes.dropna(axis=1, how="all")
    if closes.empty:
        raise ValueError("Todos los tickers del universo quedaron sin cierres validos.")

    ohlcv: dict[str, pd.DataFrame] = {"Close": closes}
    for field in ("High", "Low", "Volume"):
        frame = field_frame(field)
        if frame is not None:
            ohlcv[field] = frame.reindex(index=closes.index, columns=closes.columns).ffill()
    _last_universe_ohlcv_cache = ohlcv if {"High", "Low", "Close", "Volume"} <= set(ohlcv) else None

    logger.info(
        "Universo descargado: %d tickers validos, %d sesiones.",
        closes.shape[1],
        closes.shape[0],
    )
    return closes


def get_last_universe_ohlcv() -> dict[str, pd.DataFrame] | None:
    """
    Devuelve el OHLCV de la ultima descarga masiva de universo.

    get_universe_closes() descarga todos los campos OHLCV desde yfinance aunque
    devuelva solo cierres. Guardar High/Low/Volume evita repetir una descarga
    individual por ticker en el bloque Confirmation.
    """
    return _last_universe_ohlcv_cache


# ---------------------------------------------------------------------------
# Conversion de divisa a USD (Fase 3 — solo universos con fx_convert=True)
# ---------------------------------------------------------------------------
#
# Momentum, tendencia y riesgo deben calcularse sobre retornos en una divisa
# comun (USD) para ser comparables entre mercados. La conversion se aplica a los
# NIVELES de precio antes de calcular cualquier indicador: precio_usd = precio_local
# * (divisa->USD). El path del S&P 500 (todo USD) NO pasa por aqui.
_fx_cache: dict[str, pd.Series] = {}

_MIN_FX_VALID_RATIO = 0.80


def _extract_close_series(data: pd.DataFrame) -> pd.Series | None:
    """Extrae una serie Close plana desde la salida de yfinance."""
    if data.empty:
        return None
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.droplevel(1)
    if "Close" not in frame.columns:
        return None
    series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if series.empty:
        return None
    return series


def _min_fx_observations(index: pd.Index | None = None) -> int:
    if index is None or len(index) == 0:
        return 5
    return max(2, min(20, int(len(index) * 0.05)))


def _has_enough_fx_data(series: pd.Series | None, index: pd.Index | None = None) -> bool:
    return series is not None and int(series.dropna().shape[0]) >= _min_fx_observations(index)


def _download_fx_pair(pair: str, period: str = "3y") -> pd.Series | None:
    try:
        data = yf.download(pair, period=period, auto_adjust=True, progress=False)
        return _extract_close_series(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FX: descarga de %s fallida: %s", pair, exc)
        return None


def get_fx_series(currency: str, period: str = "3y") -> pd.Series | None:
    """
    Serie historica FX->USD: cuanto vale 1 unidad de `currency` en USD.

    Intenta primero el par directo {currency}USD=X. Si no existe, falla o tiene
    datos insuficientes, intenta el inverso USD{currency}=X y devuelve 1 / serie.
    Devuelve None para USD o si no hay FX fiable.
    """
    ccy = (currency or "USD").strip().upper()
    if ccy == "USD" or not ccy:
        return None
    cache_key = f"{ccy}:{period}"
    if cache_key in _fx_cache:
        return _fx_cache[cache_key]

    direct_pair = f"{ccy}USD=X"
    direct = _download_fx_pair(direct_pair, period=period)
    if _has_enough_fx_data(direct):
        _fx_cache[cache_key] = direct
        return direct

    logger.warning("FX: datos insuficientes para %s; probando par inverso.", direct_pair)
    inverse_pair = f"USD{ccy}=X"
    inverse = _download_fx_pair(inverse_pair, period=period)
    if _has_enough_fx_data(inverse):
        inverse = inverse[inverse != 0]
        if not inverse.empty:
            series = 1.0 / inverse
            series.name = "Close"
            _fx_cache[cache_key] = series
            logger.info("FX: usando %s invertido para obtener %s->USD.", inverse_pair, ccy)
            return series

    logger.warning(
        "FX: sin serie fiable para %s (directo %s, inverso %s).", ccy, direct_pair, inverse_pair
    )
    return None


def get_fx_to_usd_series(currency: str, index: pd.Index, period: str = "3y") -> pd.Series | None:
    """
    Devuelve la serie historica FX->USD alineada al indice de precios.

    USD devuelve una serie de 1.0. Para otras divisas se usa get_fx_series()
    con fallback a pares inversos, se alinea por fecha y se rellena solo hacia
    delante para respetar la informacion disponible en cada sesion.
    """
    ccy = (currency or "USD").strip().upper()
    if ccy == "USD" or not ccy:
        return pd.Series(1.0, index=index, name="USDUSD=X")

    fx = get_fx_series(ccy, period=period)
    if fx is None:
        logger.warning("FX: no hay serie disponible para %s.", ccy)
        return None

    aligned = fx.reindex(index).ffill()
    valid_ratio = float(aligned.notna().mean()) if len(aligned) else 0.0
    nan_count = int(aligned.isna().sum())
    aligned.attrs["currency"] = ccy
    aligned.attrs["valid_ratio"] = valid_ratio
    aligned.attrs["nan_count"] = nan_count

    if not _has_enough_fx_data(aligned, index):
        logger.warning(
            "FX: serie alineada insuficiente para %s (%d datos validos, %d requeridos).",
            ccy,
            int(aligned.notna().sum()),
            _min_fx_observations(index),
        )
        return None
    if valid_ratio < _MIN_FX_VALID_RATIO:
        logger.warning(
            "FX: cobertura baja para %s tras alineado (%.1f%% valido, %d NaN).",
            ccy,
            valid_ratio * 100,
            nan_count,
        )
    return aligned


def get_returns_usd(ticker: str, currency: str, period: str = "2y") -> pd.Series:
    """
    Retornos diarios de `ticker` expresados en USD.

    Si la divisa es USD, equivale a pct_change() del precio local. En otro caso
    convierte el precio local a USD (alineando el FX por ffill) antes de calcular
    los retornos. Pensada para analisis individual; el batch usa
    convert_closes_to_usd() sobre el DataFrame completo del universo.
    """
    prices_local = get_price_history(ticker, period=period)["Close"]
    if (currency or "USD").strip().upper() == "USD":
        return prices_local.pct_change()
    fx_aligned = get_fx_to_usd_series(currency, prices_local.index, period=period)
    if fx_aligned is None:
        logger.warning("FX no disponible para %s (%s); se usan retornos locales.", ticker, currency)
        return prices_local.pct_change()
    prices_usd = prices_local * fx_aligned
    return prices_usd.pct_change()


def convert_closes_to_usd(
    closes: pd.DataFrame,
    currency_map: dict[str, str],
    period: str = "3y",
) -> pd.DataFrame:
    """
    Convierte a USD las columnas de un DataFrame de cierres del universo.

    Cada columna (ticker) se multiplica por el FX de su divisa (ffill sobre el
    indice de fechas). Las columnas en USD (o sin divisa conocida) se dejan
    intactas. No muta el DataFrame de entrada.

    Args:
        closes:        cierres con tickers en columnas, fechas en el indice.
        currency_map:  {ticker -> divisa de cotizacion}. Lo que no este aqui se
                       asume USD.
    """
    if closes.empty:
        return closes
    converted = closes.copy()
    needed = {(currency_map.get(col, "USD") or "USD").strip().upper() for col in closes.columns}
    fx_by_ccy: dict[str, pd.Series] = {}
    for ccy in needed:
        if ccy == "USD":
            continue
        fx = get_fx_to_usd_series(ccy, closes.index, period=period)
        if fx is not None:
            fx_by_ccy[ccy] = fx

    for col in closes.columns:
        ccy = (currency_map.get(col, "USD") or "USD").strip().upper()
        if ccy == "USD":
            continue
        fx_aligned = fx_by_ccy.get(ccy)
        if fx_aligned is None:
            logger.warning("FX ausente para %s (%s); columna sin convertir.", col, ccy)
            continue
        converted[col] = closes[col] * fx_aligned
    return converted


# ---------------------------------------------------------------------------
# Sector de un ticker
# ---------------------------------------------------------------------------


def get_ticker_sector(ticker: str) -> dict[str, str]:
    """
    Devuelve sector e industry de un ticker usando yfinance .info.
    Cachea en memoria para evitar llamadas repetidas.
    """
    global _sector_cache
    if ticker in _sector_cache:
        return {"sector": _sector_cache[ticker], "industry": ""}

    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or ""
        industry = info.get("industry") or ""
        _sector_cache[ticker] = sector
        return {"sector": sector, "industry": industry}
    except Exception as exc:
        logger.warning("No se pudo obtener sector de '%s': %s", ticker, exc)
        return {"sector": "", "industry": ""}


# Caché en memoria del mapa ticker→sector (se llena la primera vez que se pide).
# Sin esto, load_sector_map_from_fundamental() leería el CSV del disco una vez
# por cada ticker analizado: en un batch de 500 empresas son 500+ lecturas
# innecesarias de un fichero de ~100 KB.
_sector_map_cache: dict[str, str] | None = None


def load_sector_map_from_fundamental() -> dict[str, str]:
    """
    Carga el mapa ticker→sector desde el CSV del universo fundamental
    (app/services/fundamental/engine/data/sp500_universe.csv).

    Evita hacer ~500 llamadas a yfinance para el fallback sectorial del Bloque 3.
    Devuelve un dict vacío si el CSV no existe o no se puede leer.

    El resultado se cachea en memoria la primera vez. Las llamadas posteriores
    (dentro de la misma sesión Python) devuelven el dict cacheado sin releer
    el disco.
    """
    global _sector_map_cache
    if _sector_map_cache is not None:
        return _sector_map_cache

    csv_path = (
        Path(__file__).parent.parent.parent.parent
        / "fundamental"
        / "engine"
        / "data"
        / "sp500_universe.csv"
    )
    if not csv_path.exists():
        logger.debug("CSV de universo fundamental no encontrado en %s.", csv_path)
        _sector_map_cache = {}
        return _sector_map_cache
    try:
        df = pd.read_csv(csv_path, usecols=["ticker", "sector"])
        tickers = df["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
        _sector_map_cache = dict(zip(tickers, df["sector"].fillna(""), strict=False))
        logger.debug("Mapa sector cargado desde CSV: %d tickers.", len(_sector_map_cache))
        return _sector_map_cache
    except Exception as exc:
        logger.warning("No se pudo cargar sector map desde CSV: %s", exc)
        _sector_map_cache = {}
        return _sector_map_cache
