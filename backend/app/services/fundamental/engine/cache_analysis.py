"""Lightweight adapter to score a company from a cached universe row, without per-ticker
yfinance downloads. Used by the offline regression and batch paths.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.fundamental.engine.fmp_fetcher import augment_with_fmp
from app.services.fundamental.engine.fundamental_score import combine_fundamental_scores
from app.services.fundamental.engine.growth import compute_growth_score
from app.services.fundamental.engine.quality import compute_quality_score
from app.services.fundamental.engine.ratio_profiles import get_profile
from app.services.fundamental.engine.solvency import compute_solvency_score
from app.services.fundamental.engine.valuation import compute_valuation_score


def _none_if_nan(value: Any) -> Any:
    try:
        return None if pd.isna(value) else value
    except TypeError:
        return value


def _company_data_from_universe_row(row: pd.Series) -> dict[str, Any]:
    get = lambda key: _none_if_nan(row.get(key))  # noqa: E731
    total_debt = get("totalDebt") or 0.0
    total_cash = get("totalCash") or 0.0

    flags: list[str] = []
    sector = get("sector")
    equity = get("book_value_total")
    if get("returnOnEquity") is None and equity is None:
        flags.append("roe_no_data")
    if get("debtToEquity") is None and equity is None:
        flags.append("de_no_data")
    if get("delta_gross_margin") is None:
        flags.append("gross_profit_no_data")
    if sector in ("Financials", "Financial Services") and get("interest_coverage") is None:
        flags.append("coverage_bank_no_data")

    return {
        "ticker": get("ticker"),
        "sector": sector,
        "industry": get("industry"),
        "country": get("country"),
        "price_currency": get("currency") or "USD",
        "market_cap_currency": get("currency") or "USD",
        "ticker_currency": get("currency") or "USD",
        "financial_currency": get("financialCurrency") or get("currency") or "USD",
        "is_foreign_company": bool(
            get("country") and str(get("country")).lower() not in {"united states", "usa", "us"}
        ),
        "is_adr": False,
        "valuation_currency_mismatch": False,
        "market_cap": get("marketCap"),
        "enterprise_value": get("enterprise_value"),
        "revenue_ttm": get("totalRevenue"),
        "gross_profit_ttm": get("grossProfits"),
        "net_income_ttm": get("netIncomeToCommon"),
        "ebitda_ttm": get("ebitda"),
        "interest_expense_ttm": get("interestExpense"),
        "total_assets": get("totalAssets"),
        "equity": equity,
        "total_debt": total_debt,
        "cash": total_cash,
        "shares_outstanding": get("sharesOutstanding"),
        "book_value_total": equity,
        "net_debt": get("net_debt"),
        "cfo_ttm": get("operatingCashflow"),
        "fcf_ttm": get("freeCashflow"),
        "operating_margin": get("operatingMargins"),
        "gross_margin": get("grossMargins"),
        "roa": get("returnOnAssets"),
        "roe": get("returnOnEquity"),
        "gp_a": get("gp_a"),
        "fcf_ni": get("fcf_ni"),
        "current_ratio": get("currentRatio"),
        "debt_to_equity": get("debtToEquity"),
        "dn_ebitda": get("dn_ebitda"),
        "interest_coverage": get("interest_coverage"),
        "cfo_debt": get("cfo_debt"),
        "ebitda_yield": get("ebitda_yield"),
        "earnings_yield": get("earnings_yield"),
        "fcf_yield": get("fcf_yield"),
        "book_yield": get("book_yield"),
        "revenue_trend_ols": get("revenue_trend_ols"),
        "fcf_trend_ols": get("fcf_trend_ols"),
        "revenue_cagr_3y": get("revenue_cagr_3y"),
        "fcf_cagr_3y": get("fcf_cagr_3y"),
        "delta_gross_margin": get("delta_gross_margin"),
        "asset_growth": get("asset_growth"),
        "share_dilution": get("share_dilution"),
        "roce": get("roce"),
        "data_flags": flags,
    }


def analyze_fundamental_from_cache(
    ticker: str,
    universe_df: pd.DataFrame,
    universe_by_ticker: dict[str, pd.Series],
    profile_name: str = "sp500_validated",
) -> dict[str, Any]:
    key = ticker.upper()
    row = universe_by_ticker.get(key)
    if row is None:
        row = universe_by_ticker.get(key.replace(".", "-"))
    if row is None:
        raise ValueError(f"{ticker} is not in the cached fundamental CSV.")

    company_data = _company_data_from_universe_row(row)
    augment_with_fmp(company_data, company_data.get("sector"))
    profile = get_profile(profile_name)

    val_result = compute_valuation_score(company_data, universe_df, profile=profile)
    qual_result = compute_quality_score(company_data, universe_df, profile=profile)
    grow_result = compute_growth_score(company_data, universe_df, profile=profile)
    solv_result = compute_solvency_score(company_data, universe_df, profile=profile)
    result = combine_fundamental_scores(val_result, qual_result, grow_result, solv_result)
    if solv_result and "solvency_methodology" in solv_result:
        result["solvency_methodology"] = solv_result["solvency_methodology"]
    return result
