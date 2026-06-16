"""Reference-universe SELECTION layer for the multi-universe extension.

The normalization universe is no longer fixed to the S&P 500 but a parameter. This layer does
NOT change the normalization math: the fundamental sub-blocks still receive a `universe_df` and
apply their pipeline. What it adds:

  - `UniverseConfig`: describes a universe (tickers, precomputed stats, ratio profile and whether
    the technical module should convert currency to USD).
  - `UniverseManager`: loads and caches those configs, precomputing the statistics (median and
    MAD per indicator) once per universe.

Supported universes:
  - "sp500": wraps the validated universe (load_universe()), ratio_profile="sp500_validated",
    fx_convert=False.
  - "msci_world": global reference, ratio_profile="global_robust", fx_convert=True.

With universe="sp500" the returned `universe_df` is the same DataFrame produced by
load_universe(), so the S&P 500 scores do not change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.fundamental.engine.ratio_profiles import get_profile
from app.services.fundamental.engine.universe import load_universe

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"

# Indicators for which stats are precomputed (union of universe columns used by the 4 sub-blocks).
# median/MAD is computed for whichever are present.
_STAT_INDICATORS: list[str] = [
    # valuation
    "ebitda_yield",
    "earnings_yield",
    "fcf_yield",
    "book_yield",
    # quality
    "gp_a",
    "returnOnAssets",
    "returnOnEquity",
    "operatingMargins",
    "fcf_ni",
    "roce",
    # growth
    "revenue_trend_ols",
    "fcf_trend_ols",
    "fcf_cagr_3y",
    "delta_gross_margin",
    "asset_growth",
    "share_dilution",
    # solvency
    "dn_ebitda",
    "interest_coverage",
    "currentRatio",
    "debtToEquity",
    "cfo_debt",
]


@dataclass
class UniverseConfig:
    name: str
    tickers: list[str]
    stats: dict  # {indicator: {"median": float, "mad": float, "n": int}}
    ratio_profile: str  # "sp500_validated" | "global_robust"
    fx_convert: bool
    universe_df: pd.DataFrame = field(repr=False, default=None)  # type: ignore[assignment]


# Static spec of each universe (no data loaded yet).
_UNIVERSE_SPECS: dict[str, dict] = {
    "sp500": {"ratio_profile": "sp500_validated", "fx_convert": False},
    "msci_world": {"ratio_profile": "global_robust", "fx_convert": True},
}


def _robust_stats(series: pd.Series) -> dict | None:
    """Median and MAD (scaled ×1.4826) of a series, ignoring inf/NaN."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    return {
        "median": median,
        "mad": mad,
        "mad_scaled": 1.4826 * mad,
        "n": int(clean.shape[0]),
    }


def compute_stats(universe_df: pd.DataFrame) -> dict:
    """Precompute median/MAD per indicator present in the universe."""
    stats: dict = {}
    for col in _STAT_INDICATORS:
        if col in universe_df.columns:
            s = _robust_stats(universe_df[col])
            if s is not None:
                stats[col] = s
    return stats


class UniverseManager:
    """Loads and caches universe configurations (including the DataFrame and stats)."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or _DATA_DIR
        self._cache: dict[str, UniverseConfig] = {}

    # -- Public API --------------------------------------------------------

    def get_config(self, universe: str = "sp500") -> UniverseConfig:
        """Return the universe config (loading and caching it the first time)."""
        if universe in self._cache:
            return self._cache[universe]

        spec = _UNIVERSE_SPECS.get(universe)
        if spec is None:
            raise KeyError(
                f"Universe '{universe}' not supported. Available: {sorted(_UNIVERSE_SPECS)}"
            )

        universe_df = self._load_universe_df(universe)
        # Stable label so the sub-blocks' internal caches can distinguish universes without
        # depending on the DataFrame's id().
        try:
            universe_df.attrs["universe_name"] = universe
        except Exception:  # noqa: BLE001
            pass

        stats = self._load_or_compute_stats(universe, universe_df)
        config = UniverseConfig(
            name=universe,
            tickers=universe_df["ticker"].astype(str).tolist(),
            stats=stats,
            ratio_profile=spec["ratio_profile"],
            fx_convert=spec["fx_convert"],
            universe_df=universe_df,
        )
        # Sanity check the declared profile.
        get_profile(config.ratio_profile).assert_weights_sum_to_one()
        self._cache[universe] = config
        return config

    def get_universe_df(self, universe: str = "sp500") -> pd.DataFrame:
        return self.get_config(universe).universe_df

    def get_stats(self, universe: str = "sp500") -> dict:
        return self.get_config(universe).stats

    def profile_name(self, universe: str) -> str:
        """Name of the ratio_profile bound to a universe (without loading data)."""
        spec = _UNIVERSE_SPECS.get(universe)
        if spec is None:
            raise KeyError(f"Universe '{universe}' not supported.")
        return spec["ratio_profile"]

    def resolve_universe(self, ticker: str, mode: str = "auto") -> str:
        """Decide the reference universe for a ticker.

        mode:
          - "domestic": force sp500.
          - "global":   force msci_world (cross-market comparison).
          - "auto":     S&P 500 member → sp500; anything else → msci_world.
        """
        if mode == "global":
            return "msci_world"
        if mode == "domestic":
            return "sp500"
        norm = ticker.strip().upper().replace(".", "-")
        sp500 = set(self.get_config("sp500").tickers)
        return "sp500" if norm in sp500 else "msci_world"

    def resolve_portfolio(self, tickers: list[str], mode: str = "auto") -> dict[str, str]:
        """Assign a universe to each ticker of a portfolio.

        Mixed-portfolio rule: in 'auto' mode, if tickers fall into more than one universe, force
        ALL of them to msci_world so the scores are comparable. 'global' always forces msci_world;
        'domestic' always sp500.
        """
        if mode == "global":
            return {t: "msci_world" for t in tickers}
        if mode == "domestic":
            return {t: "sp500" for t in tickers}
        per_ticker = {t: self.resolve_universe(t, "auto") for t in tickers}
        if len(set(per_ticker.values())) > 1:
            return {t: "msci_world" for t in tickers}
        return per_ticker

    def compute_and_cache_stats(self, universe: str, tickers: list[str]) -> None:
        """Recompute and persist the universe stats to disk."""
        universe_df = self._load_universe_df(universe)
        stats = compute_stats(universe_df)
        self._write_stats(universe, stats)
        logger.info(
            "Stats recomputed for '%s': %d indicators (%d tickers in universe).",
            universe,
            len(stats),
            len(tickers) or len(universe_df),
        )

    # -- Internals ---------------------------------------------------------

    def _load_universe_df(self, universe: str) -> pd.DataFrame:
        if universe == "sp500":
            return load_universe()
        if universe == "msci_world":
            from app.services.fundamental.engine.msci_world import MSCI_UNIVERSE_CSV

            if not MSCI_UNIVERSE_CSV.exists():
                raise FileNotFoundError(
                    "The MSCI World universe is not built. Run build_msci_world_universe() "
                    f"(needs data/URTH_holdings.csv) to generate {MSCI_UNIVERSE_CSV.name}."
                )
            df = pd.read_csv(MSCI_UNIVERSE_CSV)
            logger.info(
                "MSCI World universe loaded: %d companies, %d columns", len(df), len(df.columns)
            )
            return df
        raise KeyError(f"Universe '{universe}' has no loader implemented.")

    def _stats_path(self, universe: str) -> Path:
        return self._data_dir / f"{universe}_stats.json"

    def _load_or_compute_stats(self, universe: str, universe_df: pd.DataFrame) -> dict:
        path = self._stats_path(universe)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unreadable stats cache (%s); recomputing: %s", path, exc)
        stats = compute_stats(universe_df)
        self._write_stats(universe, stats)
        return stats

    def _write_stats(self, universe: str, stats: dict) -> None:
        path = self._stats_path(universe)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info("Stats for '%s' saved to %s", universe, path)


# Shared default instance (session cache).
_default_manager = UniverseManager()


def get_universe_manager() -> UniverseManager:
    return _default_manager


def resolve_universe(ticker: str, mode: str = "auto") -> str:
    """Convenience: route a ticker using the default manager."""
    return _default_manager.resolve_universe(ticker, mode)
