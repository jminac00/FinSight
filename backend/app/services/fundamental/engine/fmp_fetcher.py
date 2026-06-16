"""
fmp_fetcher.py
Solvency data fallback from Financial Modeling Prep (FMP).

Active ONLY when:
  - sector == "Financial Services"
  - yfinance returns cfo_ttm == 0/None or interest_expense_ttm == 0/None
  - FMP_API_KEY is configured in .env

Call control:
  - In-memory cache with a 24h TTL.
  - Session counter: warning when exceeding 240 calls (daily limit = 250).
    Each ticker consumes 3 calls (income + balance + cashflow).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Load .env (repo root) if python-dotenv is available
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=_ENV_PATH)
except ImportError:
    # Without dotenv: read .env manually (once at import time)
    try:
        with open(_ENV_PATH) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
    except OSError:
        pass

FMP_API_KEY: str | None = os.getenv("FMP_API_KEY") or None
_FMP_BASE_URL = "https://financialmodelingprep.com/stable"
_FMP_CALLS_PER_TICKER = 3
_FMP_DAILY_LIMIT = 250
_FMP_CALL_WARNING_AT = 240

# Session state (module-level, persists during the batch)
_fmp_call_count: int = 0
_fmp_resolved: int = 0
_fmp_skipped_limit: int = 0

# In-memory cache: {ticker: (timestamp_float, data_dict | None)}
_FMP_CACHE: dict[str, tuple[float, dict | None]] = {}
_CACHE_TTL = 86_400.0  # 24 hours


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _is_missing_or_zero(v: object) -> bool:
    """True if the value is None, NaN, or numerically == 0."""
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return True


def needs_fmp_fallback(company_data: dict, sector: str | None) -> bool:
    """
    True only if the sector is FS AND the three key derived ratios are all missing.

    For large banks yfinance returns operatingCashflow=0 and interest_coverage=None,
    which leaves interest_coverage=None, current_ratio=None and cfo_debt=0/None.
    For non-bank FS companies (V, MA, AXP, FISV) at least one of the three already
    has a real value.
    """
    if sector != "Financial Services":
        return False
    ic = company_data.get("interest_coverage")
    cr = company_data.get("current_ratio")
    cfo = company_data.get("cfo_debt")
    return _is_missing_or_zero(ic) and _is_missing_or_zero(cr) and _is_missing_or_zero(cfo)


def augment_with_fmp(company_data: dict, sector: str | None) -> dict:
    """
    Augment company_data with FMP data if yfinance is insufficient.

    Always sets company_data["solvency_data_source"]:
      "yf"      → yfinance was sufficient, FMP was not called
      "fmp"     → FMP responded with useful data
      "partial" → FMP unavailable or returned no data either
    Returns the same dict (modified in-place).
    """
    global _fmp_resolved

    ticker = str(company_data.get("ticker", "?"))

    if not needs_fmp_fallback(company_data, sector):
        company_data["solvency_data_source"] = "yf"
        return company_data

    if not FMP_API_KEY:
        company_data["solvency_data_source"] = "partial"
        return company_data

    # Check cache
    now = time.time()
    if ticker in _FMP_CACHE:
        ts, cached = _FMP_CACHE[ticker]
        if now - ts < _CACHE_TTL:
            if cached is not None:
                _apply_fmp_data(company_data, cached)
                company_data["solvency_data_source"] = "fmp"
            else:
                company_data["solvency_data_source"] = "partial"
            return company_data

    fmp_data = _get_from_fmp(ticker)
    _FMP_CACHE[ticker] = (now, fmp_data)

    if fmp_data:
        _apply_fmp_data(company_data, fmp_data)
        company_data["solvency_data_source"] = "fmp"
        _fmp_resolved += 1
        logger.info("FMP fallback applied: %s", ticker)
    else:
        company_data["solvency_data_source"] = "partial"

    return company_data


def print_fmp_summary() -> None:
    """Print the FMP usage summary at the end of the batch."""
    print(
        f"\n[FMP] {_fmp_resolved} tickers resolved, "
        f"{_fmp_call_count} calls made, "
        f"{_fmp_skipped_limit} tickers unresolved due to limit or error"
    )


# ---------------------------------------------------------------------------
# FMP call
# ---------------------------------------------------------------------------


def _get_from_fmp(ticker: str) -> dict[str, Any] | None:
    """
    Call the 3 FMP endpoints and extract the solvency fields.
    Returns None if any call fails or the response is empty.
    """
    global _fmp_call_count, _fmp_skipped_limit

    if not FMP_API_KEY:
        return None

    if _fmp_call_count + _FMP_CALLS_PER_TICKER > _FMP_CALL_WARNING_AT:
        logger.warning(
            "FMP: session limit reached (%d/%d); skipping %s.",
            _fmp_call_count,
            _FMP_DAILY_LIMIT,
            ticker,
        )
        _fmp_skipped_limit += 1
        return None

    params = {"symbol": ticker, "apikey": FMP_API_KEY, "limit": 1}

    try:
        r_inc = requests.get(f"{_FMP_BASE_URL}/income-statement", params=params, timeout=10)
        r_inc.raise_for_status()
        inc = r_inc.json()
        _fmp_call_count += 1

        r_bal = requests.get(f"{_FMP_BASE_URL}/balance-sheet-statement", params=params, timeout=10)
        r_bal.raise_for_status()
        bal = r_bal.json()
        _fmp_call_count += 1

        r_cf = requests.get(f"{_FMP_BASE_URL}/cash-flow-statement", params=params, timeout=10)
        r_cf.raise_for_status()
        cf = r_cf.json()
        _fmp_call_count += 1

        if not inc or not bal or not cf:
            logger.warning("FMP: empty response for %s", ticker)
            return None

        # FMP returns interest expense as negative in some periods; use abs()
        int_exp_raw = inc[0].get("interestExpense")
        int_exp = abs(float(int_exp_raw)) if int_exp_raw is not None else None

        return {
            "interest_expense_ttm": int_exp,
            "operating_income_ttm": inc[0].get("operatingIncome"),
            "cfo_ttm": cf[0].get("operatingCashFlow"),
            "total_debt": bal[0].get("totalDebt"),
            "current_assets": bal[0].get("totalCurrentAssets"),
            "current_liabilities": bal[0].get("totalCurrentLiabilities"),
        }

    except Exception as exc:
        logger.warning("FMP error for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Apply FMP data and recompute derived ratios
# ---------------------------------------------------------------------------


def _apply_fmp_data(company_data: dict, fmp_data: dict) -> None:
    """Overwrite company_data fields with FMP values (only if they are valid)."""
    for key, val in fmp_data.items():
        if val is None:
            continue
        try:
            if pd.isna(val) or not np.isfinite(float(val)):
                continue
        except (TypeError, ValueError):
            continue
        company_data[key] = float(val)

    _recalc_solvency_ratios(company_data)


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    r = a / b
    return None if (np.isnan(r) or np.isinf(r)) else r


def _recalc_solvency_ratios(company_data: dict) -> None:
    """Recompute interest_coverage, cfo_debt and current_ratio after the FMP update."""
    int_exp = company_data.get("interest_expense_ttm")
    op_inc = company_data.get("operating_income_ttm")
    cfo = company_data.get("cfo_ttm")
    total_debt = company_data.get("total_debt") or 0.0
    ca = company_data.get("current_assets")
    cl = company_data.get("current_liabilities")

    if int_exp and int_exp > 0:
        ic = _safe_div(op_inc, int_exp)
        if ic is not None:
            company_data["interest_coverage"] = ic

    if total_debt > 0:
        cfo_debt = _safe_div(cfo, total_debt)
        if cfo_debt is not None:
            company_data["cfo_debt"] = cfo_debt

    if ca is not None and cl is not None:
        cr = _safe_div(ca, cl)
        if cr is not None:
            company_data["current_ratio"] = cr
