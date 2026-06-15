"""Download and cache the S&P 500 universe fundamental data required for the cross-sectional
normalization of the scores.

Typical usage:
    from app.services.fundamental.engine.universe import load_universe
    df = load_universe(path)                      # load CSV if it exists, build it otherwise
    df = build_universe(path)                     # force a full rebuild
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from app.services.fundamental.engine.data_fetcher import compute_growth_trend

logger = logging.getLogger(__name__)

# Default cache location inside the engine package; overridable via the caller (settings).
DEFAULT_UNIVERSE_PATH = Path(__file__).parent / "data" / "sp500_universe.csv"

# Fields extracted directly from the yfinance .info dict
_INFO_FIELDS: list[str] = [
    "sector",
    "industry",
    "marketCap",
    "enterpriseValue",
    "ebitda",
    "netIncomeToCommon",
    "freeCashflow",
    "bookValue",  # book value per share
    "sharesOutstanding",
    "totalDebt",
    "totalCash",
    "totalAssets",
    "grossProfits",
    "operatingCashflow",
    "returnOnAssets",  # ROA already computed by yfinance
    "returnOnEquity",  # ROE already computed by yfinance
    "operatingMargins",
    "grossMargins",
    "currentRatio",
    "debtToEquity",  # D/E × 100 in yfinance
    "totalRevenue",
    "revenueGrowth",  # YoY growth (proxy to normalize growth)
    "earningsGrowth",
    "interestExpense",  # not always available in .info; None if missing
]

# ---------------------------------------------------------------------------
# GICS sector fallback (tickers where yfinance returns sector=NaN)
# ---------------------------------------------------------------------------

_SECTOR_FALLBACK: dict[str, str] = {
    # Source: official GICS S&P 500. Extend if new tickers appear with NaN.
    "FISV": "Financial Services",  # Fiserv — payment processing
    "BRK-B": "Financial Services",  # Berkshire Hathaway
    "BF-B": "Consumer Defensive",  # Brown-Forman
    "CEG": "Utilities",  # Constellation Energy
    "KVUE": "Consumer Defensive",  # Kenvue
    "SOLV": "Healthcare",  # Solventum
    "SW": "Basic Materials",  # Smurfit WestRock
    "VLTO": "Industrials",  # Veralto
    "AMTM": "Industrials",  # Amentum
}


def _apply_sector_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the 'sector' field with hardcoded GICS values for tickers where yfinance returns
    NaN systematically. Modifies the DataFrame in-place.
    """
    if "sector" not in df.columns or "ticker" not in df.columns:
        return df

    missing_before = df["sector"].isna().sum()
    if missing_before == 0:
        return df

    for ticker, sector in _SECTOR_FALLBACK.items():
        mask = (df["ticker"] == ticker) & df["sector"].isna()
        if mask.any():
            df.loc[mask, "sector"] = sector
            logger.info("Sector fallback applied: %s → %s", ticker, sector)

    missing_after = df["sector"].isna().sum()
    if missing_after > 0:
        still_missing = df[df["sector"].isna()]["ticker"].tolist()
        logger.warning(
            "%d tickers without sector after fallback: %s",
            missing_after,
            still_missing[:20],
        )
    return df


# ---------------------------------------------------------------------------
# Ticker download
# ---------------------------------------------------------------------------


def get_sp500_tickers() -> list[str]:
    """Fetch the list of S&P 500 tickers from Wikipedia."""
    import io

    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia returns 403 without a browser User-Agent
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})
    df = tables[0]
    # BRK.B and BF.B use a hyphen in yfinance
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info("S&P 500 tickers fetched: %d", len(tickers))
    return tickers


# ---------------------------------------------------------------------------
# Per-ticker extraction
# ---------------------------------------------------------------------------


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Safe division; returns None if any operand is invalid."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _normalize_debt_to_equity(value: float | None) -> float | None:
    """Normalize the D/E ratio returned by yfinance, which is not consistent across tickers.
    - value > 20  → assume it came ×100 (e.g. 176 → 1.76)
    - 0 ≤ value ≤ 20 → already the direct ratio
    - value < 0   → negative equity; returned as-is
    - None        → None
    """
    if value is None:
        return None
    if value > 20:
        return value / 100.0
    return value  # covers the [0, 20] range and negatives


def _compute_growth_metrics_from_history(
    ticker_obj: yf.Ticker,
    inc: pd.DataFrame | None = None,
    cf: pd.DataFrame | None = None,
    bs: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Compute the five growth variables that require financial history:
        revenue_cagr_3y, fcf_cagr_3y, delta_gross_margin, asset_growth, share_dilution.

    Accepts pre-downloaded statements (inc, cf, bs) to avoid re-downloading when they were
    already fetched in _extract_ticker_data. If None is passed, it downloads them.

    Never raises: on error it returns a dict with all fields set to None.
    """
    out: dict[str, float | None] = {
        "revenue_cagr_3y": None,
        "fcf_cagr_3y": None,
        "revenue_trend_ols": None,
        "fcf_trend_ols": None,
        "delta_gross_margin": None,
        "asset_growth": None,
        "share_dilution": None,
    }
    try:
        if inc is None:
            inc = ticker_obj.income_stmt
        if cf is None:
            cf = ticker_obj.cashflow
        if bs is None:
            bs = ticker_obj.balance_sheet

        # ---- Revenue CAGR 3y ----
        if inc is not None and not inc.empty and "Total Revenue" in inc.index:
            rev = inc.loc["Total Revenue"].dropna()
            if len(rev) >= 4:
                r_t, r_t_3 = float(rev.iloc[0]), float(rev.iloc[3])
                if r_t > 0 and r_t_3 > 0:
                    out["revenue_cagr_3y"] = (r_t / r_t_3) ** (1 / 3) - 1.0

            # Revenue OLS trend (oldest→newest)
            if len(rev) >= 3:
                rev_list = [float(rev.iloc[i]) for i in range(len(rev) - 1, -1, -1)]
                out["revenue_trend_ols"] = compute_growth_trend(rev_list)

        # ---- FCF CAGR 3y (CFO + Capex; capex comes negative in yfinance) ----
        if cf is not None and not cf.empty:
            cfo_label = next(
                (
                    k
                    for k in (
                        "Operating Cash Flow",
                        "Cash Flow From Continuing Operating Activities",
                        "Total Cash From Operating Activities",
                    )
                    if k in cf.index
                ),
                None,
            )
            capex_label = next(
                (k for k in ("Capital Expenditure", "Capital Expenditures") if k in cf.index),
                None,
            )
            if cfo_label and capex_label:
                cfo_s = cf.loc[cfo_label].dropna()
                capex_s = cf.loc[capex_label].dropna()
                n = min(len(cfo_s), len(capex_s))
                if n >= 4:
                    fcf_t = float(cfo_s.iloc[0]) + float(capex_s.iloc[0])
                    fcf_t_3 = float(cfo_s.iloc[3]) + float(capex_s.iloc[3])
                    if fcf_t > 0 and fcf_t_3 > 0:
                        out["fcf_cagr_3y"] = (fcf_t / fcf_t_3) ** (1 / 3) - 1.0

                # FCF OLS trend (oldest→newest)
                if n >= 3:
                    fcf_list = [
                        float(cfo_s.iloc[i]) + float(capex_s.iloc[i]) for i in range(n - 1, -1, -1)
                    ]
                    out["fcf_trend_ols"] = compute_growth_trend(fcf_list)

        # ---- Delta Gross Margin (3 years, in percentage points as a decimal) ----
        if inc is not None and not inc.empty:
            gp_label = "Gross Profit" if "Gross Profit" in inc.index else None
            rev_label = "Total Revenue" if "Total Revenue" in inc.index else None
            if gp_label and rev_label:
                gp = inc.loc[gp_label]
                rev = inc.loc[rev_label]
                if len(gp) >= 4 and len(rev) >= 4:
                    rev_t, rev_t_3 = float(rev.iloc[0]), float(rev.iloc[3])
                    if rev_t > 0 and rev_t_3 > 0:
                        gm_t = float(gp.iloc[0]) / rev_t
                        gm_t_3 = float(gp.iloc[3]) / rev_t_3
                        out["delta_gross_margin"] = gm_t - gm_t_3

        # ---- Asset Growth (annual change, t vs t-1) ----
        if bs is not None and not bs.empty and "Total Assets" in bs.index:
            ta = bs.loc["Total Assets"].dropna()
            if len(ta) >= 2:
                ta_t, ta_t_1 = float(ta.iloc[0]), float(ta.iloc[1])
                if ta_t_1 > 0:
                    out["asset_growth"] = (ta_t - ta_t_1) / ta_t_1

        # ---- Share Dilution (annual change of shares outstanding) ----
        if bs is not None and not bs.empty:
            sh_label = next(
                (
                    k
                    for k in (
                        "Share Issued",
                        "Ordinary Shares Number",
                        "Common Stock Shares Outstanding",
                        "Basic Average Shares",
                        "Diluted Average Shares",
                    )
                    if k in bs.index
                ),
                None,
            )
            if sh_label:
                sh = bs.loc[sh_label].dropna()
                if len(sh) >= 2:
                    sh_t, sh_t_1 = float(sh.iloc[0]), float(sh.iloc[1])
                    if sh_t_1 > 0:
                        out["share_dilution"] = (sh_t - sh_t_1) / sh_t_1

    except Exception as exc:
        logger.debug("Growth metrics failed: %s", exc)

    return out


def _extract_ticker_data(ticker_str: str) -> dict:
    """Download the yfinance .info dict for a ticker and compute the derived ratios required for
    the normalization of the four sub-blocks.

    The financial statements (income_stmt, cashflow, balance_sheet) are downloaded once and
    reused both for the quality/solvency ratios and for the growth metrics, since several fields
    are no longer in .info (yfinance >= 0.2.x):
        - totalAssets      → balance_sheet["Total Assets"]
        - operatingIncome  → income_stmt["EBIT"] / ["Operating Income"]
        - interestExpense  → income_stmt["Interest Expense"] (except financials)

    Never raises: on any error it returns the partial dict.
    """
    row: dict = {"ticker": ticker_str}

    try:
        ticker_obj = yf.Ticker(ticker_str)
        info: dict = ticker_obj.info

        # Raw fields from the info dict
        for field in _INFO_FIELDS:
            row[field] = info.get(field)

        # Sector fallback for tickers where yfinance returns None/NaN
        if not row.get("sector"):
            fallback = _SECTOR_FALLBACK.get(ticker_str)
            if fallback:
                row["sector"] = fallback
                logger.info("Sector fallback (build): %s → %s", ticker_str, fallback)

        # ---- Pre-download financial statements (once per ticker) ----
        try:
            inc_stmt = ticker_obj.income_stmt
        except Exception:
            inc_stmt = None
        try:
            cf_stmt = ticker_obj.cashflow
        except Exception:
            cf_stmt = None
        try:
            bs_stmt = ticker_obj.balance_sheet
        except Exception:
            bs_stmt = None

        # ---- totalAssets: .info no longer includes it → balance_sheet ----
        if not row.get("totalAssets") and bs_stmt is not None and not bs_stmt.empty:
            if "Total Assets" in bs_stmt.index:
                ta_vals = bs_stmt.loc["Total Assets"].dropna()
                if not ta_vals.empty:
                    row["totalAssets"] = float(ta_vals.iloc[0])

        # ---- Base quantities ----
        mc: float = row["marketCap"] or 0.0
        ev: float = row["enterpriseValue"] or (
            mc + (row["totalDebt"] or 0.0) - (row["totalCash"] or 0.0)
        )
        ebitda: float = row["ebitda"] or 0.0
        ni: float = row["netIncomeToCommon"] or 0.0
        fcf: float = row["freeCashflow"] or 0.0
        bv_ps: float = row["bookValue"] or 0.0
        shares: float = row["sharesOutstanding"] or 0.0
        ta: float = row["totalAssets"] or 0.0
        gp: float = row["grossProfits"] or 0.0
        td: float = row["totalDebt"] or 0.0
        cash: float = row["totalCash"] or 0.0
        cfo: float = row["operatingCashflow"] or 0.0

        # book_value_total: primary source = Stockholders Equity from the balance sheet.
        # Avoids the dual-class share problem (BRK-B, BF-B) where yfinance may return bookValue
        # as the class-A value × number of class-B shares, producing an artificially huge
        # book_yield (e.g. 685% for BRK-B).
        equity_from_bs: float | None = None
        if bs_stmt is not None and not bs_stmt.empty:
            for _eq_label in (
                "Stockholders Equity",
                "Total Stockholders Equity",
                "Common Stock Equity",
            ):
                if _eq_label in bs_stmt.index:
                    _eq_vals = bs_stmt.loc[_eq_label].dropna()
                    if not _eq_vals.empty:
                        equity_from_bs = float(_eq_vals.iloc[0])
                        break

        if equity_from_bs is not None:
            bv_total: float = equity_from_bs
        elif shares > 0 and bv_ps > 0:
            bv_total = bv_ps * shares
        else:
            bv_total = 0.0

        # ---- Base derived fields ----
        row["enterprise_value"] = ev if ev > 0 else None
        row["book_value_total"] = bv_total if bv_total > 0 else None
        row["net_debt"] = td - cash

        # ---- Valuation yields (floored to 0; negative → 0) ----
        row["ebitda_yield"] = max(_safe_div(ebitda, ev) or 0.0, 0.0) if ev > 0 else None
        row["earnings_yield"] = max(_safe_div(ni, mc) or 0.0, 0.0) if mc > 0 else None
        # The universe FCF always comes from info.get("freeCashflow") (source "yfinance_info").
        # For maximum consistency in the FCF yield z-score, the analyzed company should use the
        # same source; see the "fcf_ttm_source" key in data_fetcher.py.
        row["fcf_yield"] = max(_safe_div(fcf, mc) or 0.0, 0.0) if mc > 0 else None
        row["book_yield"] = (
            max(_safe_div(bv_total, mc) or 0.0, 0.0) if mc > 0 and bv_total > 0 else None
        )

        # ---- Quality ratios ----
        row["gp_a"] = _safe_div(gp, ta) if ta > 0 else None  # GP/Assets (Novy-Marx)
        row["fcf_ni"] = _safe_div(fcf, ni) if ni > 0 else None  # FCF/NI; None if NI<0

        # ---- Growth columns (reuses already downloaded statements) ----
        growth_metrics = _compute_growth_metrics_from_history(
            ticker_obj,
            inc=inc_stmt,
            cf=cf_stmt,
            bs=bs_stmt,
        )
        row["revenue_cagr_3y"] = growth_metrics["revenue_cagr_3y"]
        row["fcf_cagr_3y"] = growth_metrics["fcf_cagr_3y"]
        row["revenue_trend_ols"] = growth_metrics["revenue_trend_ols"]
        row["fcf_trend_ols"] = growth_metrics["fcf_trend_ols"]
        row["delta_gross_margin"] = growth_metrics["delta_gross_margin"]
        row["asset_growth"] = growth_metrics["asset_growth"]
        row["share_dilution"] = growth_metrics["share_dilution"]

        # ---- Solvency ratios ----
        net_debt: float = td - cash
        row["dn_ebitda"] = _safe_div(net_debt, ebitda) if ebitda > 0 else None

        # Interest Coverage = EBIT / interest expense.
        # Hierarchy: info["operatingIncome"] → income_stmt["EBIT"] → margin × revenue.
        # interestExpense: info → income_stmt (except financial sector).
        # EBITDA is NOT used as fallback because it overstates coverage in capital-intensive
        # sectors and breaks universe comparability.
        op_income: float | None = info.get("operatingIncome") or info.get("ebit")
        if op_income is None and inc_stmt is not None and not inc_stmt.empty:
            for _label in ("EBIT", "Operating Income", "Total Operating Income As Reported"):
                if _label in inc_stmt.index:
                    _vals = inc_stmt.loc[_label].dropna()
                    if not _vals.empty:
                        _v = float(_vals.iloc[0])
                        if not pd.isna(_v):
                            op_income = _v
                            break
        if op_income is None:
            op_margin: float = row.get("operatingMargins") or 0.0
            rev_ic: float = row.get("totalRevenue") or 0.0
            if op_margin and rev_ic:
                op_income = op_margin * rev_ic

        # interestExpense: .info no longer includes it reliably.
        # Fallback: income_stmt, but only for non-financial companies (for banks, "Interest
        # Expense" is payment on deposits, not corporate debt cost).
        int_exp_raw: float | None = row.get("interestExpense")
        if int_exp_raw is None and inc_stmt is not None and not inc_stmt.empty:
            sector_str: str = row.get("sector") or ""
            if sector_str not in ("Financials", "Financial Services"):
                if "Interest Expense" in inc_stmt.index:
                    _ie_vals = inc_stmt.loc["Interest Expense"].dropna()
                    if not _ie_vals.empty:
                        _ie_v = float(_ie_vals.iloc[0])
                        if not pd.isna(_ie_v):
                            int_exp_raw = _ie_v
        int_exp: float = abs(int_exp_raw or 0.0)

        row["interest_coverage"] = (
            _safe_div(op_income, int_exp) if (op_income is not None and int_exp > 0) else None
        )

        row["cfo_debt"] = _safe_div(cfo, td) if td > 0 else None

        row["debtToEquity"] = _normalize_debt_to_equity(row["debtToEquity"])

    except Exception as exc:  # noqa: BLE001
        logger.warning("Error downloading %s: %s", ticker_str, exc)

    return row


# ---------------------------------------------------------------------------
# Universe build and load
# ---------------------------------------------------------------------------


def build_universe(path: Path | None = None, delay: float = 0.6) -> pd.DataFrame:
    """Download fundamental data for all S&P 500 constituents and save the result to CSV.

    Args:
        path:  Output CSV path; defaults to DEFAULT_UNIVERSE_PATH.
        delay: Seconds to wait between requests (avoids rate-limiting).

    Returns:
        DataFrame with one row per ticker and all computed metrics.

    Estimated time: ~6-8 minutes for the 503 tickers with delay=0.6.
    """
    output_path = path or DEFAULT_UNIVERSE_PATH
    tickers = get_sp500_tickers()
    total = len(tickers)
    logger.info("Starting download of %d tickers (~%.0f min)...", total, total * delay / 60)

    rows: list[dict] = []
    for i, ticker in enumerate(tickers, start=1):
        rows.append(_extract_ticker_data(ticker))
        if i % 50 == 0:
            logger.info("Progress: %d / %d (%.0f%%)", i, total, 100 * i / total)
        time.sleep(delay)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Universe saved to %s (%d rows, %d columns)", output_path, len(df), len(df.columns))
    return df


def load_universe(path: Path | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Load the S&P 500 universe from the cached CSV.
    If the CSV does not exist (or force_refresh=True) it calls build_universe().

    Args:
        path:          CSV path; defaults to DEFAULT_UNIVERSE_PATH.
        force_refresh: If True, ignore the cache and re-download everything.

    Returns:
        DataFrame with one row per ticker and all fundamental metrics.
    """
    cache_path = path or DEFAULT_UNIVERSE_PATH
    if not force_refresh and cache_path.exists():
        logger.info("Loading universe from cache: %s", cache_path)
        df = pd.read_csv(cache_path)
        df = _apply_sector_fallback(df)
        logger.info("Universe loaded: %d companies, %d columns", len(df), len(df.columns))
        return df

    return build_universe(cache_path)
