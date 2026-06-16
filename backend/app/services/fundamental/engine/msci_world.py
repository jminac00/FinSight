"""
msci_world.py
Builds the MSCI World universe from the public holdings of the iShares ETF (URTH).

Source robustness (project principle 4): the iShares site is now a SPA and the
CSV download link is built client-side, so direct scraping is fragile. This layer
uses, in order of preference:

  1. An official CSV placed manually at `data/URTH_holdings.csv`
     ("Detailed Holdings and Analytics" button on the ETF page). RECOMMENDED.
  2. A best-effort download of the legacy `.ajax` endpoint (may return HTML if the
     portfolio id expires; in that case an actionable error is raised).

The MSCI World universe INCLUDES the S&P 500 companies (with their suffix-less
yfinance ticker), so a US company has a coherent position in both universes.

Pipeline:
    load_urth_holdings()           -> raw holdings DataFrame (equity)
    build_constituents()           -> DataFrame [ticker, name, sector, country, currency]
                                       (tickers normalized to yfinance format, + S&P 500)
    build_msci_world_universe()    -> data/msci_world_universe.csv (+ currency map json)

The resulting CSV has the SAME schema as sp500_universe.csv (it reuses
universe._extract_ticker_data), with extra columns: country, currency.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from app.services.fundamental.engine.universe import _extract_ticker_data, get_sp500_tickers

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
URTH_LOCAL_CSV = _DATA_DIR / "URTH_holdings.csv"
MSCI_UNIVERSE_CSV = _DATA_DIR / "msci_world_universe.csv"
MSCI_CURRENCY_JSON = _DATA_DIR / "msci_world_currency.json"

# Legacy endpoint (best-effort). The portfolio id may expire.
URTH_AJAX_URL = (
    "https://www.ishares.com/us/products/239696/ishares-msci-world-etf/"
    "1467271812596.ajax?fileType=csv&fileName=URTH_holdings&dataType=fund"
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# Exchange (iShares "Exchange") -> yfinance suffix mapping.
# Keys are compared lower-cased and by substring to tolerate variations in the
# exchange name published by iShares.
# ---------------------------------------------------------------------------
_EXCHANGE_SUFFIX: list[tuple[str, str]] = [
    ("nasdaq", ""),
    ("new york stock exchange", ""),
    ("nyse", ""),
    ("cboe bzx", ""),
    ("bats", ""),
    ("xetra", ".DE"),
    ("deutsche boerse", ".DE"),
    ("frankfurt", ".F"),
    ("euronext paris", ".PA"),
    ("euronext amsterdam", ".AS"),
    ("euronext brussels", ".BR"),
    ("euronext lisbon", ".LS"),
    ("euronext dublin", ".IR"),
    ("irish stock exchange", ".IR"),
    ("london stock exchange", ".L"),
    ("six swiss", ".SW"),
    ("swiss exchange", ".SW"),
    ("tokyo", ".T"),
    ("borsa italiana", ".MI"),
    ("bolsa de madrid", ".MC"),
    ("nasdaq omx helsinki", ".HE"),
    ("nasdaq helsinki", ".HE"),
    ("nasdaq omx stockholm", ".ST"),
    ("nasdaq stockholm", ".ST"),
    ("nasdaq omx copenhagen", ".CO"),
    ("nasdaq copenhagen", ".CO"),
    ("oslo", ".OL"),
    ("wiener", ".VI"),
    ("vienna", ".VI"),
    ("australian securities", ".AX"),
    ("asx", ".AX"),
    ("hong kong", ".HK"),
    ("stock exchange of hong kong", ".HK"),
    ("toronto", ".TO"),
    ("tsx", ".TO"),
    ("singapore", ".SI"),
    ("tel aviv", ".TA"),
    ("new zealand", ".NZ"),
]

# Country fallback (iShares "Location") when the exchange is not enough.
_COUNTRY_SUFFIX: dict[str, str] = {
    "united states": "",
    "germany": ".DE",
    "france": ".PA",
    "netherlands": ".AS",
    "belgium": ".BR",
    "portugal": ".LS",
    "ireland": ".IR",
    "united kingdom": ".L",
    "switzerland": ".SW",
    "japan": ".T",
    "italy": ".MI",
    "spain": ".MC",
    "finland": ".HE",
    "sweden": ".ST",
    "denmark": ".CO",
    "norway": ".OL",
    "austria": ".VI",
    "australia": ".AX",
    "hong kong": ".HK",
    "canada": ".TO",
    "singapore": ".SI",
    "israel": ".TA",
    "new zealand": ".NZ",
}

_COUNTRY_CURRENCY: dict[str, str] = {
    "united states": "USD",
    "germany": "EUR",
    "france": "EUR",
    "netherlands": "EUR",
    "belgium": "EUR",
    "portugal": "EUR",
    "ireland": "EUR",
    "united kingdom": "GBP",
    "switzerland": "CHF",
    "japan": "JPY",
    "italy": "EUR",
    "spain": "EUR",
    "finland": "EUR",
    "sweden": "SEK",
    "denmark": "DKK",
    "norway": "NOK",
    "austria": "EUR",
    "australia": "AUD",
    "hong kong": "HKD",
    "canada": "CAD",
    "singapore": "SGD",
    "israel": "ILS",
    "new zealand": "NZD",
}


def _suffix_for(exchange: str, country: str) -> str | None:
    """Return the yfinance suffix or None if the market is not mapped."""
    ex = (exchange or "").strip().lower()
    for needle, suffix in _EXCHANGE_SUFFIX:
        if needle in {"nasdaq", "nyse"}:
            continue
        if needle in ex:
            return suffix
    if ex in {"nasdaq", "nyse"}:
        return ""
    co = (country or "").strip().lower()
    if co in _COUNTRY_SUFFIX:
        return _COUNTRY_SUFFIX[co]
    return None


def normalize_ticker(
    ticker: str,
    exchange: str,
    country: str,
    known_us_tickers: set[str] | None = None,
) -> str | None:
    """
    Convert an iShares ticker to the format yfinance understands.

    General cleanup of the raw symbol (applies to all markets):
      - trailing dots from LSE nomenclature are removed ('BP.' -> 'BP')
      - share-class separators (space or dot) become a hyphen
        ('NOVO B' -> 'NOVO-B', 'BT.A' -> 'BT-A', 'BRK.B' -> 'BRK-B')
      - placeholders without alphanumeric content ('--') are discarded

    Per-market rules:
      - US: no suffix. If `known_us_tickers` is provided (an authoritative list,
        e.g. the S&P 500), share classes written without a separator are resolved:
        'BRKB' -> 'BRK-B' (the hyphenated variant exists) and 'HEIA' -> 'HEI-A'
        (class A/B of a known ticker).
      - Hong Kong: numeric code padded to 4 digits ('388' -> '0388.HK').
      - Rest of international: clean code + the exchange suffix (.DE, .PA, .L, ...).

    Returns None if the market is not mapped or the symbol is a placeholder.
    """
    raw = (ticker or "").strip().upper()
    if not raw or not any(ch.isalnum() for ch in raw):
        return None
    suffix = _suffix_for(exchange, country)
    if suffix is None:
        return None

    base = raw.rstrip(".").replace(" ", "-").replace(".", "-")
    if not base or not any(ch.isalnum() for ch in base):
        return None

    if suffix == "":
        known = known_us_tickers or set()
        if base in known or "-" in base or len(base) < 3:
            return base
        dashed = f"{base[:-1]}-{base[-1]}"
        if dashed in known:
            return dashed  # 'BRKB' -> 'BRK-B'
        if base[-1] in "AB" and base[:-1] in known:
            return dashed  # 'HEIA' -> 'HEI-A'
        return base

    if suffix == ".HK" and base.isdigit():
        base = base.zfill(4)  # '388' -> '0388'
    return f"{base}{suffix}"


def _infer_market_currency(country: str, raw_currency: str | None, fx_rate: str | None) -> str:
    """
    Return the local quotation currency for the technical FX conversion.

    Some current BlackRock "Data Download Excel" files publish the Currency column
    as USD for all positions and the FX Rate as local/USD. In that case we infer
    the currency by country. If a Market Currency exists, it is used directly.
    """
    ccy = (raw_currency or "USD").strip().upper() or "USD"
    if ccy != "USD":
        return ccy

    try:
        fx = float(str(fx_rate or "1").replace(",", ""))
    except ValueError:
        fx = 1.0
    country_key = (country or "").strip().lower()
    if country_key != "united states" and fx != 1.0:
        return _COUNTRY_CURRENCY.get(country_key, ccy)
    return ccy


# ---------------------------------------------------------------------------
# Loading / downloading the ETF holdings
# ---------------------------------------------------------------------------


def _looks_like_csv(text: str) -> bool:
    head = text.lstrip()[:200].lower()
    return not head.startswith("<!doctype") and not head.startswith("<html")


def download_urth_holdings(timeout: int = 30) -> str | None:
    """Best-effort download of the legacy CSV. Returns the CSV text or None on failure."""
    try:
        r = requests.get(URTH_AJAX_URL, headers=_BROWSER_HEADERS, timeout=timeout)
        r.raise_for_status()
        if _looks_like_csv(r.text):
            return r.text
        logger.warning(
            "iShares returned HTML (not CSV): the legacy endpoint's portfolio id "
            "has likely expired. Use the local CSV at %s.",
            URTH_LOCAL_CSV,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("URTH holdings download failed: %s", exc)
    return None


def _parse_holdings_text(text: str) -> pd.DataFrame:
    """
    Parse the iShares holdings CSV, skipping the metadata rows.

    Locates the header row (the one starting with 'Ticker') and reads from there.
    Filters to equity (Asset Class == Equity) when the column exists.
    """
    lines = text.splitlines()
    header_idx = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.lstrip("﻿").replace('"', "").strip().lower().startswith("ticker,")
        ),
        None,
    )
    if header_idx is None:
        raise ValueError("The 'Ticker' header row was not found in the iShares CSV.")

    body = "\n".join(lines[header_idx:])
    # keep_default_na=False: prevents pandas from turning the ticker "NA"
    # (National Bank of Canada) into NaN; empty cells stay NaN.
    df = pd.read_csv(io.StringIO(body), keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]

    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.strip().str.lower() == "equity"]
    return df.reset_index(drop=True)


def load_urth_holdings() -> pd.DataFrame:
    """
    Load the URTH ETF holdings. Preference: local CSV; otherwise download.

    Raises:
        FileNotFoundError: if there is no local CSV nor a valid download.
    """
    if URTH_LOCAL_CSV.exists():
        logger.info("Loading URTH holdings from local file: %s", URTH_LOCAL_CSV)
        return _parse_holdings_text(URTH_LOCAL_CSV.read_text(encoding="utf-8", errors="replace"))

    logger.info("No local CSV; trying a best-effort iShares download...")
    text = download_urth_holdings()
    if text is None:
        raise FileNotFoundError(
            "Could not obtain the MSCI World holdings.\n"
            f"Download the official 'Detailed Holdings and Analytics' CSV of the URTH ETF "
            f"and save it to:\n  {URTH_LOCAL_CSV}\n"
            "(page: https://www.ishares.com/us/products/239696/ishares-msci-world-etf)"
        )
    return _parse_holdings_text(text)


# ---------------------------------------------------------------------------
# Building the constituents list (normalized + S&P 500)
# ---------------------------------------------------------------------------


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def build_constituents(include_sp500: bool = True) -> pd.DataFrame:
    """
    DataFrame of normalized MSCI World constituents:
        columns = [ticker, name, sector, country, currency]

    - Tickers normalized to yfinance format; the unmappable ones are discarded (logged).
    - Positions with price <= 0 (dead lines, coupons, contra-entries) are discarded.
    - Includes the S&P 500 companies (US, suffix-less, USD), deduplicating.
    """
    holdings = load_urth_holdings()
    c_ticker = _col(holdings, "Ticker")
    c_name = _col(holdings, "Name", "Issuer Name")
    c_sector = _col(holdings, "Sector")
    c_exch = _col(holdings, "Exchange")
    c_country = _col(holdings, "Location", "Country")
    c_ccy = _col(holdings, "Market Currency", "Currency")
    c_fx = _col(holdings, "FX Rate")
    c_price = _col(holdings, "Price")
    if c_ticker is None:
        raise ValueError("The holdings CSV has no 'Ticker' column.")

    # Authoritative US ticker list to resolve share classes without a separator.
    sp500_tickers: list[str] = []
    if include_sp500:
        try:
            sp500_tickers = get_sp500_tickers()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch the S&P 500 list: %s", exc)
    sp500_set = set(sp500_tickers)

    rows: list[dict] = []
    skipped: list[str] = []
    for _, h in holdings.iterrows():
        raw_ticker = str(h[c_ticker]).strip()
        exch = str(h[c_exch]) if c_exch else ""
        country = str(h[c_country]) if c_country else ""
        if c_price is not None:
            try:
                price = float(str(h[c_price]).replace(",", ""))
            except (TypeError, ValueError):
                price = None
            if price is not None and price <= 0.0:
                skipped.append(f"{raw_ticker}@{exch or country} (price<=0)")
                continue
        yf_ticker = normalize_ticker(raw_ticker, exch, country, known_us_tickers=sp500_set)
        if yf_ticker is None:
            skipped.append(f"{raw_ticker}@{exch or country}")
            continue
        rows.append(
            {
                "ticker": yf_ticker,
                "name": str(h[c_name]).strip() if c_name else "",
                "sector": str(h[c_sector]).strip() if c_sector else "",
                "country": country.strip(),
                "currency": _infer_market_currency(
                    country,
                    str(h[c_ccy]).strip().upper() if c_ccy else "USD",
                    str(h[c_fx]).strip() if c_fx else None,
                ),
            }
        )

    df = pd.DataFrame(rows)

    if include_sp500 and sp500_tickers:
        existing = set(df["ticker"]) if not df.empty else set()
        extra = [
            {"ticker": t, "name": "", "sector": "", "country": "United States", "currency": "USD"}
            for t in sp500_tickers
            if t not in existing
        ]
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    if skipped:
        logger.warning(
            "%d positions discarded for an unmapped market (first ones: %s).",
            len(skipped),
            skipped[:15],
        )
    logger.info("MSCI World constituents: %d normalized tickers.", len(df))
    return df


# ---------------------------------------------------------------------------
# Building the fundamental universe (CSV with the sp500_universe.csv schema)
# ---------------------------------------------------------------------------


def build_msci_world_universe(
    limit: int | None = None,
    delay: float = 0.6,
    constituents: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Download the fundamentals of each constituent (reusing
    universe._extract_ticker_data) and save data/msci_world_universe.csv.

    Robustness (principle 4): tickers whose data cannot be downloaded are left OUT
    of the stats computation with a log; the batch never aborts.

    Args:
        limit: process only the first N tickers (debug / testing).
        delay: seconds between tickers (rate-limiting).
        constituents: precomputed list; if None, build_constituents() is called.

    Returns:
        Universe DataFrame (also written to CSV). Includes country and currency
        columns in addition to the standard fundamental schema.
    """
    if constituents is None:
        constituents = build_constituents(include_sp500=True)
    if limit is not None:
        constituents = constituents.head(limit)

    meta_by_ticker = {
        r["ticker"]: {"country": r["country"], "currency": r["currency"]}
        for _, r in constituents.iterrows()
    }
    tickers = constituents["ticker"].tolist()
    total = len(tickers)
    logger.info("Building MSCI World universe: %d tickers (delay=%.2fs).", total, delay)

    rows: list[dict] = []
    failed: list[str] = []
    for i, ticker in enumerate(tickers, start=1):
        row = _extract_ticker_data(ticker)
        meta = meta_by_ticker.get(ticker, {})
        row["country"] = meta.get("country", "")
        row["currency"] = meta.get("currency", "USD")
        # An "empty" ticker (only the ticker key) counts as a download failure.
        if len(row) <= 3 or row.get("marketCap") is None:
            failed.append(ticker)
        rows.append(row)
        if i % 50 == 0:
            logger.info("MSCI World progress: %d / %d (%.0f%%)", i, total, 100 * i / total)
        time.sleep(delay)

    df = pd.DataFrame(rows)
    MSCI_UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MSCI_UNIVERSE_CSV, index=False)

    currency_map = {str(r.get("ticker")): str(r.get("currency", "USD")) for _, r in df.iterrows()}
    MSCI_CURRENCY_JSON.write_text(json.dumps(currency_map, indent=2), encoding="utf-8")

    logger.info(
        "MSCI World universe saved: %d rows (%d without market data) at %s",
        len(df),
        len(failed),
        MSCI_UNIVERSE_CSV,
    )
    return df


def load_currency_map() -> dict[str, str]:
    """Map {ticker -> quotation currency} of the MSCI World universe (for FX)."""
    if MSCI_CURRENCY_JSON.exists():
        try:
            return json.loads(MSCI_CURRENCY_JSON.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("currency map unreadable: %s", exc)
    return {}
