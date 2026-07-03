"""Frozen technical-universe snapshots.

The technical blocks normalize each ticker against a reference universe (S&P 500 or MSCI World).
Downloading that universe (~500-1500 tickers, 3 years) on every request is unviable on the free
tier, so it is frozen offline to disk and refreshed daily by the scheduler. At request time the
service loads the snapshot and only the requested ticker may need an individual download.

Snapshots live in ``engine/data/`` as CSV files, committed as seed data (Render's free tier has
no persistent disk) and refreshed in place by the scheduler. For ``sp500`` the OHLCV fields are
frozen too (the confirmation block uses them); for
``msci_world`` only the USD-converted closes are frozen (the global path runs the confirmation
block against the close-price proxy).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.services.fundamental.engine.universe_manager import (
    resolve_universe as _resolve_index_universe,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_SUPPORTED = ("sp500", "msci_world")
# Close is stored as the closes snapshot; only the extra fields need their own file.
_OHLCV_EXTRA_FIELDS = ("High", "Low", "Volume")


class TechnicalUniverseNotReadyError(Exception):
    """Raised when a required technical-universe snapshot is missing on disk."""


def _closes_path(data_dir: Path, universe: str) -> Path:
    return data_dir / f"technical_{universe}_closes.csv"


def _field_path(data_dir: Path, universe: str, field: str) -> Path:
    return data_dir / f"technical_{universe}_{field.lower()}.csv"


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def save_snapshot(
    universe: str,
    closes: pd.DataFrame,
    ohlcv: dict[str, pd.DataFrame] | None,
    data_dir: Path | None = None,
) -> None:
    """Persist a universe snapshot (closes, plus High/Low/Volume when available) to disk."""
    data_dir = data_dir or _DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    closes.to_csv(_closes_path(data_dir, universe))
    if ohlcv:
        for field in _OHLCV_EXTRA_FIELDS:
            frame = ohlcv.get(field)
            if frame is not None:
                frame.to_csv(_field_path(data_dir, universe, field))
    logger.info(
        "Saved technical %s snapshot (%d tickers, %d sessions) to %s",
        universe,
        closes.shape[1],
        closes.shape[0],
        data_dir,
    )


class TechnicalUniverseManager:
    """Loads frozen technical-universe snapshots and resolves a ticker's reference universe."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self._cache: dict[str, dict] = {}

    def resolve_universe(self, ticker: str, mode: str = "auto") -> str:
        """Decide the reference universe for a ticker (delegates to the index-membership router).

        mode: 'domestic' -> sp500, 'global' -> msci_world, 'auto' -> sp500 if an S&P 500 member,
        otherwise msci_world.
        """
        return _resolve_index_universe(ticker, mode)

    def load(self, universe: str = "sp500") -> dict:
        """Return the snapshot payload {'closes': DataFrame, 'ohlcv': dict | None}.

        Raises:
            TechnicalUniverseNotReadyError: if the snapshot is not present on disk.
        """
        if universe in self._cache:
            return self._cache[universe]

        closes_path = _closes_path(self._data_dir, universe)
        if not closes_path.exists():
            raise TechnicalUniverseNotReadyError(
                f"Technical universe '{universe}' snapshot not found at {closes_path}. "
                "Run scripts/build_technical_universe.py (or wait for the daily refresh)."
            )

        closes = _read_frame(closes_path)
        closes.attrs["universe_name"] = f"technical_{universe}"

        field_frames: dict[str, pd.DataFrame] = {}
        for field in _OHLCV_EXTRA_FIELDS:
            path = _field_path(self._data_dir, universe, field)
            if path.exists():
                field_frames[field] = _read_frame(path)

        ohlcv: dict[str, pd.DataFrame] | None = None
        if len(field_frames) == len(_OHLCV_EXTRA_FIELDS):
            # Share the closes object as "Close" so its cache key matches the other fields.
            ohlcv = {"Close": closes, **field_frames}

        payload = {"closes": closes, "ohlcv": ohlcv}
        self._cache[universe] = payload
        return payload

    def clear_cache(self) -> None:
        """Drop the in-memory snapshot cache (call after a refresh)."""
        self._cache.clear()


# -- Offline build (network-bound; run from the script or the scheduler) -------


def _download_sp500() -> tuple[pd.DataFrame, dict[str, pd.DataFrame] | None]:
    from app.services.technical.engine.utils.data_loader import (
        get_last_universe_ohlcv,
        get_sp500_tickers,
        get_universe_closes,
    )

    sp500 = get_sp500_tickers()
    tickers = sp500 + ["SPY"] if "SPY" not in sp500 else sp500
    closes = get_universe_closes(tickers)
    return closes, get_last_universe_ohlcv()


def _download_msci_world() -> tuple[pd.DataFrame, None]:
    from app.services.fundamental.engine.msci_world import MSCI_UNIVERSE_CSV, load_currency_map
    from app.services.technical.engine.utils.data_loader import (
        convert_closes_to_usd,
        get_universe_closes,
    )

    tickers = pd.read_csv(MSCI_UNIVERSE_CSV)["ticker"].astype(str).tolist()
    closes = get_universe_closes(tickers)
    closes_usd = convert_closes_to_usd(closes, load_currency_map())
    return closes_usd, None


def build_snapshot(universe: str = "sp500", data_dir: Path | None = None) -> pd.DataFrame:
    """Download the reference universe and freeze it to disk. Network-bound; run offline/scheduled.

    Returns the closes DataFrame and invalidates the default manager's cache.
    """
    if universe not in _SUPPORTED:
        raise KeyError(f"Unsupported technical universe: {universe}")

    logger.info("Building technical %s universe snapshot...", universe)
    if universe == "sp500":
        closes, ohlcv = _download_sp500()
    else:
        closes, ohlcv = _download_msci_world()

    save_snapshot(universe, closes, ohlcv, data_dir)
    _default_manager.clear_cache()
    return closes


_default_manager = TechnicalUniverseManager()


def get_technical_universe_manager() -> TechnicalUniverseManager:
    """Return the shared technical universe manager (session cache)."""
    return _default_manager
