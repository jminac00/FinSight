"""Fetches and pre-processes all fundamental data for a single company.

The main function get_company_data(ticker) returns a single dict with every variable the four
sub-blocks need (valuation, quality, growth, solvency). It never raises: if a value is not
available it returns None for that key.

Sources:
  - yfinance .info          → pre-computed TTM values and market data
  - yfinance .income_stmt   → annual history (up to 4 years) for CAGR
  - yfinance .balance_sheet → assets, debt, equity, shares
  - yfinance .cash_flow     → FCF, CFO, D&A
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_row(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """Return the first row found in df among the given names."""
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def _get_equity(balance_sheet: pd.DataFrame) -> float | None:
    """Extract stockholders' equity with defensive yfinance fallbacks."""
    row = _get_row(
        balance_sheet,
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Total Stockholders Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
    )
    val = _val(row)
    return val if val is not None and val != 0 else None


def _get_gross_profit(financials: pd.DataFrame) -> pd.Series | None:
    """Extract gross profit with a Revenue - COGS fallback."""
    gp = _get_row(financials, "Gross Profit", "Gross Income")
    if gp is not None and gp.notna().sum() >= 2:
        return gp

    rev = _get_row(financials, "Total Revenue", "Net Revenue", "Operating Revenue")
    cogs = _get_row(financials, "Cost Of Revenue", "Cost of Goods Sold", "Cost Of Goods Sold")
    if rev is not None and cogs is not None:
        derived = rev - cogs
        if derived.notna().sum() >= 2:
            return derived
    return None


def _get_interest_expense_debt_only(financials: pd.DataFrame) -> float | None:
    """For banks, avoid the generic Interest Expense, which includes deposits.
    Returns None if there is no reliable debt interest expense.
    """
    for name in ("Interest Expense Non Operating",):
        row = _get_row(financials, name)
        val = _val(row)
        if val is not None and val > 0:
            return val
    return None


def _val(series: pd.Series | None, pos: int = 0) -> float | None:
    """Extract the scalar value at position `pos` of a Series (ignoring NaN).
    yfinance orders columns from most recent to oldest, so pos=0 → latest year,
    pos=3 → 3 years ago.
    """
    if series is None:
        return None
    try:
        clean = series.dropna()
        if pos >= len(clean):
            return None
        v = float(clean.iloc[pos])
        return None if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return None


def _safe_div(a: float | None, b: float | None) -> float | None:
    """Safe division; returns None if any operand is invalid or b == 0."""
    if a is None or b is None or b == 0:
        return None
    result = a / b
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def _is_us_exchange(exchange: str | None, market: str | None) -> bool:
    """Defensive heuristic to detect a US-market listing."""
    text = f"{exchange or ''} {market or ''}".upper()
    return any(token in text for token in ("NYSE", "NASDAQ", "AMEX", "BATS", "US MARKET"))


def _has_non_us_suffix(ticker: str) -> bool:
    """yfinance uses suffixes like .T/.AS/.DE for non-US local listings."""
    return "." in ticker.strip()


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
    return value


def compute_growth_trend(values: list[float | None], min_obs: int = 3) -> float | None:
    """Growth trend via OLS: fit X_t = alpha + beta*t and return beta / mean(|X|).

    t=0 for the oldest point, t=n-1 for the most recent.
    Normalizing by mean(|X|) makes the metric scale-invariant (comparable across companies of
    different size) and allows direct comparison with CAGR.

    Advantage over CAGR: it uses all available points and works with series that contain
    negative values (e.g. historically negative FCF).
    """
    if not values:
        return None
    clean = [(i, float(v)) for i, v in enumerate(values) if v is not None and np.isfinite(float(v))]
    if len(clean) < min_obs:
        return None
    ts = np.array([x[0] for x in clean], dtype=float)
    xs = np.array([x[1] for x in clean], dtype=float)
    mean_abs = float(np.mean(np.abs(xs)))
    if mean_abs == 0.0:
        return None
    t_mean = ts.mean()
    x_mean = xs.mean()
    ss_tt = float(np.sum((ts - t_mean) ** 2))
    if ss_tt == 0.0:
        return None
    beta = float(np.sum((ts - t_mean) * (xs - x_mean)) / ss_tt)
    result = beta / mean_abs
    return None if (np.isnan(result) or np.isinf(result)) else result


def _cagr(end: float | None, start: float | None, years: int) -> float | None:
    """Compound annual growth rate.
    Returns None if start ≤ 0 or end ≤ 0 (the n-th root of a negative ratio produces complex
    numbers in modern numpy; it also has no clear economic interpretation).
    The rule 'negative FCF base → 0' is applied in growth.py, not here.
    """
    if end is None or start is None or years <= 0:
        return None
    if start <= 0 or end <= 0:
        return None
    try:
        ratio = end / start
        result = float(ratio ** (1.0 / years)) - 1.0
        return None if (np.isnan(result) or np.isinf(result)) else result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def get_company_data(ticker_str: str, apply_c6_fallbacks: bool = True) -> dict[str, Any]:
    """Download and pre-process all fundamental data for a company.

    Args:
        ticker_str: Stock symbol (e.g. "AAPL").
        apply_c6_fallbacks: Enable the extra international/data-quality fallbacks (currency-aware
            interest coverage, ROE/D&E no-data flags, ...). Disabled for the sp500_validated
            profile so the validated S&P 500 scores stay stable.

    Returns:
        Dict with identity, market, TTM income statement, balance sheet, TTM cash flow, computed
        TTM ratios (including roce), valuation yields and historical growth metrics. Missing
        values are None; data quality issues are recorded in the "data_flags" list.
    """
    result: dict[str, Any] = {"ticker": ticker_str, "data_flags": []}

    try:
        t = yf.Ticker(ticker_str)
        info: dict = t.info

        # Download financial statements; degrade to empty DF on failure
        try:
            income = t.income_stmt  # cols: dates desc, rows: line items
        except Exception:
            income = pd.DataFrame()
        try:
            balance = t.balance_sheet
        except Exception:
            balance = pd.DataFrame()
        try:
            cashflow = t.cash_flow
        except Exception:
            cashflow = pd.DataFrame()

        # ----------------------------------------------------------------
        # IDENTITY
        # ----------------------------------------------------------------
        result["name"] = info.get("longName") or info.get("shortName")
        result["sector"] = info.get("sector")
        result["industry"] = info.get("industry")
        result["country"] = info.get("country")
        result["exchange"] = info.get("exchange")
        result["market"] = info.get("market")
        result["quote_type"] = info.get("quoteType")

        price_currency = (info.get("currency") or "").upper() or None
        financial_currency = (info.get("financialCurrency") or price_currency or "").upper() or None
        result["price_currency"] = price_currency
        result["market_cap_currency"] = price_currency
        result["ticker_currency"] = price_currency
        result["financial_currency"] = financial_currency

        country = str(result.get("country") or "")
        is_foreign_company = bool(country and country.lower() not in {"united states", "usa", "us"})
        is_us_listing = _is_us_exchange(result.get("exchange"), result.get("market"))
        result["is_foreign_company"] = is_foreign_company
        result["is_adr"] = bool(
            info.get("isAdr")
            or ("ADR" in str(result.get("name") or "").upper())
            or (is_foreign_company and is_us_listing and not _has_non_us_suffix(ticker_str))
        )
        result["valuation_currency_mismatch"] = bool(
            price_currency and financial_currency and price_currency != financial_currency
        )

        # ----------------------------------------------------------------
        # MARKET DATA
        # ----------------------------------------------------------------
        mc: float | None = info.get("marketCap")
        result["market_cap"] = mc

        td_info: float = info.get("totalDebt") or 0.0
        cash_info: float = info.get("totalCash") or 0.0
        ev_info: float | None = info.get("enterpriseValue")
        ev = ev_info if ev_info else ((mc or 0.0) + td_info - cash_info if mc else None)
        result["enterprise_value"] = ev if ev and ev > 0 else None

        # ----------------------------------------------------------------
        # TTM INCOME STATEMENT
        # ----------------------------------------------------------------
        rev_row = _get_row(income, "Total Revenue")
        gp_row = _get_gross_profit(income)
        oi_row = _get_row(income, "Operating Income", "EBIT")
        ni_row = _get_row(
            income,
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Including Noncontrolling Interests",
        )
        int_row = _get_row(income, "Interest Expense", "Interest Expense Non Operating")
        da_row = _get_row(
            cashflow,
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
            "Depreciation",
        )

        revenue_info = info.get("totalRevenue")
        revenue_stmt = _val(rev_row)
        revenue_ttm = revenue_info if revenue_info is not None else revenue_stmt
        revenue_source = (
            "yfinance_info_ttm"
            if revenue_info is not None
            else ("annual_statement_latest" if revenue_stmt is not None else "missing")
        )

        gross_profit_info = info.get("grossProfits")
        gross_profit_stmt = _val(gp_row)
        gross_profit_ttm = gross_profit_info if gross_profit_info is not None else gross_profit_stmt
        gross_profit_source = (
            "yfinance_info_ttm"
            if gross_profit_info is not None
            else ("annual_statement_latest" if gross_profit_stmt is not None else "missing")
        )

        operating_income_ttm = _val(oi_row)
        operating_income_source = (
            "annual_statement_latest" if operating_income_ttm is not None else "missing"
        )

        net_income_info = info.get("netIncomeToCommon")
        net_income_stmt = _val(ni_row)
        net_income_ttm = net_income_info if net_income_info is not None else net_income_stmt
        net_income_source = (
            "yfinance_info_ttm"
            if net_income_info is not None
            else ("annual_statement_latest" if net_income_stmt is not None else "missing")
        )

        # interest_expense: yfinance returns it negative (it is an expense)
        interest_expense_ttm = _val(int_row)
        if interest_expense_ttm is not None:
            interest_expense_ttm = abs(interest_expense_ttm)

        # EBITDA: prefer the info value (already TTM), otherwise compute manually
        ebitda_info = info.get("ebitda")
        if ebitda_info:
            ebitda_ttm: float | None = float(ebitda_info)
        else:
            da_ttm = abs(_val(da_row) or 0.0)
            ebitda_ttm = (
                (operating_income_ttm + da_ttm) if operating_income_ttm is not None else None
            )

        result["revenue_ttm"] = revenue_ttm
        result["revenue_source"] = revenue_source
        result["gross_profit_ttm"] = gross_profit_ttm
        result["gross_profit_source"] = gross_profit_source
        result["operating_income_ttm"] = operating_income_ttm
        result["operating_income_source"] = operating_income_source
        result["net_income_ttm"] = net_income_ttm
        result["net_income_source"] = net_income_source
        result["ebitda_ttm"] = ebitda_ttm
        result["interest_expense_ttm"] = interest_expense_ttm

        # ----------------------------------------------------------------
        # BALANCE SHEET (latest point)
        # ----------------------------------------------------------------
        ta_row = _get_row(balance, "Total Assets")
        ca_row = _get_row(balance, "Current Assets")
        cl_row = _get_row(balance, "Current Liabilities")
        td_row = _get_row(balance, "Total Debt", "Long Term Debt And Capital Lease Obligation")
        cash_row = _get_row(
            balance, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"
        )

        total_assets = _val(ta_row) or info.get("totalAssets")
        current_assets = _val(ca_row)
        current_liabilities = _val(cl_row)
        equity = _get_equity(balance)
        total_debt = _val(td_row) or info.get("totalDebt") or 0.0
        cash = _val(cash_row) or info.get("totalCash") or 0.0

        # Shares: historical shares are read from income stmt (more reliable)
        shares_row = _get_row(
            income, "Basic Average Shares", "Diluted Average Shares", "Basic Shares Outstanding"
        )
        shares_current = (
            _val(shares_row)
            or info.get("sharesOutstanding")
            or info.get("impliedSharesOutstanding")
        )

        # Primary: totalStockholderEquity from the balance sheet (avoids issues with dual-class
        # shares such as BRK-B, where yfinance may return bookValue as the class-A per-share value
        # instead of the class-B per-share value).
        book_value = equity
        if book_value is None:
            bv_ps = info.get("bookValue")
            if bv_ps and shares_current:
                book_value = bv_ps * shares_current

        result["total_assets"] = total_assets
        result["current_assets"] = current_assets
        result["current_liabilities"] = current_liabilities
        result["equity"] = equity
        result["total_debt"] = total_debt
        result["cash"] = cash
        result["shares_outstanding"] = shares_current
        result["book_value_total"] = book_value
        result["net_debt"] = total_debt - cash

        # ----------------------------------------------------------------
        # TTM CASH FLOW
        # ----------------------------------------------------------------
        cfo_row = _get_row(
            cashflow, "Operating Cash Flow", "Net Cash Provided By Operating Activities"
        )
        fcf_row = _get_row(cashflow, "Free Cash Flow")
        capex_row = _get_row(
            cashflow, "Capital Expenditure", "Purchase Of Property Plant And Equipment"
        )

        cfo_ttm = _val(cfo_row) or info.get("operatingCashflow")

        # METHODOLOGICAL NOTE: the S&P 500 universe (universe.py) always computes fcf_yield using
        # info.get("freeCashflow") (source "yfinance_info"). If the analyzed company derives its FCF
        # from the cash flow statement or as CFO-Capex, the FCF yield z-score may be slightly biased
        # because the numerator was computed differently than the universe's denominator. The
        # "fcf_ttm_source" key allows identifying and auditing these cases.
        fcf_info = info.get("freeCashflow")
        if fcf_info is not None:
            fcf_ttm: float | None = fcf_info
            fcf_ttm_source = "yfinance_info_ttm"
        elif fcf_row is not None:
            fcf_ttm = _val(fcf_row)
            fcf_ttm_source = "cashflow_statement_latest"
        elif cfo_ttm is not None and capex_row is not None:
            capex = abs(_val(capex_row) or 0.0)
            fcf_ttm = cfo_ttm - capex
            fcf_ttm_source = "cfo_minus_capex_latest"
        else:
            fcf_ttm = None
            fcf_ttm_source = "missing"

        result["cfo_ttm"] = cfo_ttm
        result["fcf_ttm"] = fcf_ttm
        result["fcf_ttm_source"] = fcf_ttm_source

        # ----------------------------------------------------------------
        # TTM RATIOS
        # ----------------------------------------------------------------
        ev_v = result["enterprise_value"]
        mc_v = result["market_cap"]

        result["operating_margin"] = _safe_div(operating_income_ttm, revenue_ttm) or info.get(
            "operatingMargins"
        )
        result["gross_margin"] = _safe_div(gross_profit_ttm, revenue_ttm) or info.get(
            "grossMargins"
        )
        result["roa"] = _safe_div(net_income_ttm, total_assets) or info.get("returnOnAssets")
        result["roe"] = _safe_div(net_income_ttm, equity) or info.get("returnOnEquity")
        if apply_c6_fallbacks and result["roe"] is None and equity is None:
            result["data_flags"].append("roe_no_data")
        result["gp_a"] = _safe_div(gross_profit_ttm, total_assets)
        result["fcf_ni"] = (
            _safe_div(fcf_ttm, net_income_ttm)
            if (net_income_ttm is not None and net_income_ttm > 0)
            else None
        )
        result["current_ratio"] = _safe_div(current_assets, current_liabilities) or info.get(
            "currentRatio"
        )

        de_raw = info.get("debtToEquity")
        result["debt_to_equity"] = (
            _normalize_debt_to_equity(de_raw)
            if de_raw is not None
            else _safe_div(total_debt, equity)
        )
        if apply_c6_fallbacks and result["debt_to_equity"] is None and equity is None:
            result["data_flags"].append("de_no_data")

        result["dn_ebitda"] = (
            _safe_div(result["net_debt"], ebitda_ttm)
            if (ebitda_ttm is not None and ebitda_ttm > 0)
            else None
        )

        # Interest Coverage = EBIT / interest expense.
        # EBITDA is not used as a fallback, to keep comparability with universe.py.
        ic_numerator = operating_income_ttm
        interest_coverage_source = "operating_income" if ic_numerator is not None else "missing"
        if ic_numerator is None:
            op_margin_for_ic = result.get("operating_margin")
            if op_margin_for_ic is not None and revenue_ttm is not None:
                ic_numerator = op_margin_for_ic * revenue_ttm
                interest_coverage_source = "operating_margin_x_revenue"
        if apply_c6_fallbacks and result.get("sector") in ("Financials", "Financial Services"):
            debt_only_interest = _get_interest_expense_debt_only(income)
            if debt_only_interest is None:
                interest_expense_for_coverage = None
                result["data_flags"].append("coverage_bank_no_data")
            else:
                interest_expense_for_coverage = debt_only_interest
                interest_coverage_source = "interest_expense_non_operating"
        else:
            interest_expense_for_coverage = interest_expense_ttm

        result["interest_coverage"] = (
            _safe_div(ic_numerator, interest_expense_for_coverage)
            if (interest_expense_for_coverage is not None and interest_expense_for_coverage > 0)
            else None
        )
        if result["interest_coverage"] is None:
            interest_coverage_source = "missing"
        result["interest_coverage_source"] = interest_coverage_source
        result["cfo_debt"] = _safe_div(cfo_ttm, total_debt) if total_debt > 0 else None

        # ROCE = EBIT / Capital Employed (= Total Assets − Current Liabilities).
        # Used only by the global_robust profile (international comparability);
        # the sp500_validated profile does not consume it.
        capital_employed = (
            total_assets - current_liabilities
            if (total_assets is not None and current_liabilities is not None)
            else None
        )
        result["roce"] = (
            _safe_div(operating_income_ttm, capital_employed)
            if (capital_employed is not None and capital_employed > 0)
            else None
        )

        # Valuation yields (floored to 0)
        result["ebitda_yield"] = max(_safe_div(ebitda_ttm, ev_v) or 0.0, 0.0) if ev_v else None
        result["earnings_yield"] = (
            max(_safe_div(net_income_ttm, mc_v) or 0.0, 0.0) if mc_v else None
        )
        result["fcf_yield"] = max(_safe_div(fcf_ttm, mc_v) or 0.0, 0.0) if mc_v else None
        result["book_yield"] = (
            max(_safe_div(book_value, mc_v) or 0.0, 0.0) if (mc_v and book_value) else None
        )

        # ----------------------------------------------------------------
        # HISTORICAL GROWTH
        # yfinance financial statements have up to 4 columns ordered from
        # most recent (pos=0) to oldest (pos=3).
        # ----------------------------------------------------------------

        # Revenue CAGR 3y → try pos=3, then pos=2 as fallback.
        # "revenue_cagr_years" records how many real years the CAGR covers.
        r0, r3, r2 = _val(rev_row, 0), _val(rev_row, 3), _val(rev_row, 2)
        if r0 is not None and r3 is not None:
            result["revenue_cagr_3y"] = _cagr(r0, r3, 3)
            result["revenue_cagr_years"] = 3
        elif r0 is not None and r2 is not None:
            result["revenue_cagr_3y"] = _cagr(r0, r2, 2)
            result["revenue_cagr_years"] = 2
        else:
            result["revenue_cagr_3y"] = info.get("revenueGrowth")  # YoY proxy
            result["revenue_cagr_years"] = 1

        # FCF CAGR 3y (negative base → None; growth.py treats it as 0)
        f0 = _val(fcf_row, 0) if fcf_row is not None else None
        f3 = _val(fcf_row, 3) if fcf_row is not None else None
        f2 = _val(fcf_row, 2) if fcf_row is not None else None
        if f0 is not None and f3 is not None:
            result["fcf_cagr_3y"] = _cagr(f0, f3, 3)
            result["fcf_cagr_years"] = 3
        elif f0 is not None and f2 is not None:
            result["fcf_cagr_3y"] = _cagr(f0, f2, 2)
            result["fcf_cagr_years"] = 2
        else:
            result["fcf_cagr_3y"] = None
            result["fcf_cagr_years"] = None

        # Revenue OLS trend (oldest→newest): pos=3 is the oldest available
        rev_series = [_val(rev_row, i) for i in range(3, -1, -1)]  # [t-3, t-2, t-1, t]
        result["revenue_trend_ols"] = compute_growth_trend(rev_series)

        # FCF OLS trend
        fcf_series = [_val(fcf_row, i) for i in range(3, -1, -1)] if fcf_row is not None else []
        result["fcf_trend_ols"] = compute_growth_trend(fcf_series)

        # Gross margin delta (current - 3 years ago)
        gm0 = _safe_div(_val(gp_row, 0), _val(rev_row, 0))
        gm3 = _safe_div(_val(gp_row, 3), _val(rev_row, 3))
        gm2 = _safe_div(_val(gp_row, 2), _val(rev_row, 2))
        if gm0 is not None and gm3 is not None:
            result["delta_gross_margin"] = gm0 - gm3
        elif gm0 is not None and gm2 is not None:
            result["delta_gross_margin"] = gm0 - gm2
        else:
            result["delta_gross_margin"] = None
            if apply_c6_fallbacks and gp_row is None:
                result["data_flags"].append("gross_profit_no_data")

        # Asset Growth 1y (total assets: current year vs previous year)
        ta0, ta1 = _val(ta_row, 0), _val(ta_row, 1)
        result["asset_growth"] = (_safe_div(ta0, ta1) - 1.0) if (ta0 and ta1) else None

        # Share Dilution 3y (current shares vs 3 years ago)
        s0 = _val(shares_row, 0) or shares_current
        s3, s2 = _val(shares_row, 3), _val(shares_row, 2)
        if s0 and s3:
            result["share_dilution"] = _safe_div(s0, s3) - 1.0
        elif s0 and s2:
            result["share_dilution"] = _safe_div(s0, s2) - 1.0
        else:
            result["share_dilution"] = None

    except Exception as exc:
        logger.error("Unexpected error processing %s: %s", ticker_str, exc)

    return result
