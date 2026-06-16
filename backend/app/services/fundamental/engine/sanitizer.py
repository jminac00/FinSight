"""Centralizes the ratio-sanitization logic for the four fundamental sub-blocks.

Each sanitize_* function encapsulates the business rules that decide when a ratio must be
neutralized, floored or assigned a value by convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Financial sectors: interest coverage and EBITDA are not comparable with industrial companies
# (banks treat interest as operating income).
_FINANCIAL_SECTORS: frozenset[str] = frozenset(
    {
        "Financial Services",
        "Financials",
    }
)

_REAL_ESTATE_SECTORS: frozenset[str] = frozenset({"Real Estate"})

_MIN_COVERAGE_SAMPLE = 10
_DEBT_THRESHOLD = 1_000_000  # USD 1M — threshold to consider "no significant debt"


def is_financial_sector(sector: str | None) -> bool:
    return bool(sector and sector in _FINANCIAL_SECTORS)


def is_real_estate_sector(sector: str | None) -> bool:
    return bool(sector and sector in _REAL_ESTATE_SECTORS)


# ---------------------------------------------------------------------------
# Per-domain sub-dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValuationRatios:
    ebitda_yield: float
    earnings_yield: float
    fcf_yield: float
    book_yield: float
    ebitda_yield_floored: bool  # was negative, forced to 0
    ev_negative: bool = False  # enterprise_value was negative/null with positive EBITDA
    fcf_yield_capped: bool = False  # fcf_yield > 40%; valuation.py applies universe p95 cap
    fcf_yield_raw: float = 0.0  # original value before cap
    book_yield_outlier: bool = False  # book_yield > 20x (2000%); contribution adjusted to p95
    book_yield_raw: float = 0.0  # original value before cap


@dataclass
class QualityRatios:
    gp_a: float | None
    roa: float | None
    roe: float | None
    operating_margin: float | None
    fcf_ni: float | None
    fcf_ni_zeroed: bool  # net_income < 0 → neutral contribution
    fcf_ni_extreme: bool = False  # |fcf_ni| > 10x → neutral contribution (implausible ratio)


@dataclass
class GrowthRatios:
    revenue_cagr_3y: float | None  # OLS primary; CAGR fallback (Revenue field)
    fcf_trend: float | None  # unified FCF metric: OLS if >=3 points, else CAGR fallback
    fcf_cagr_zeroed: bool  # fcf_trend is None → neutral contribution
    delta_gross_margin: float | None
    asset_growth: float | None
    share_dilution: float | None
    fcf_cagr_base_negative: bool = False  # FCF CAGR >500% or extreme OLS: neutralized
    revenue_cagr_outlier: bool = False  # Revenue CAGR >100% without OLS: outlier
    fcf_cagr_fallback: float | None = None  # CAGR value used if method='cagr_fallback'
    fcf_growth_method: str = "insufficient_data"  # 'ols' | 'cagr_fallback' | 'insufficient_data'


@dataclass
class SolvencyRatios:
    dn_ebitda: float | None
    interest_coverage_used: float | None
    interest_coverage_assigned: bool  # no real debt → p95 is assigned
    interest_coverage_missing: bool  # data not available
    current_ratio: float | None
    debt_to_equity: float | None
    cfo_debt: float | None
    debt_to_equity_z_override: float | None  # -3.0 when equity < 0


@dataclass
class SanitizedRatios:
    valuation: ValuationRatios
    quality: QualityRatios
    growth: GrowthRatios
    solvency: SolvencyRatios
    is_financial: bool
    is_real_estate: bool
    negative_equity: bool


# ---------------------------------------------------------------------------
# Sanitization functions per sub-block
# ---------------------------------------------------------------------------


def sanitize_valuation(data: dict) -> ValuationRatios:
    """Floor negative yields to 0 and flag outliers so valuation.py can cap them with the
    universe's 95th percentile:
      - fcf_yield > 40%: fcf_yield_capped=True; raw kept for the later cap.
      - book_yield > 20x (2000%): book_yield_outlier=True; same.
      - negative enterprise_value with positive EBITDA: ev_negative=True.
    No cap is applied here: valuation.py has access to the universe to use the real p95.
    """
    ey = data.get("ebitda_yield") or 0.0
    ery = data.get("earnings_yield") or 0.0
    fy_raw = float(data.get("fcf_yield") or 0.0)
    by_raw = float(data.get("book_yield") or 0.0)

    ebitda_floored = float(ey) < 0.0

    # Negative EV: data_fetcher/universe stores None when EV <= 0
    ev = data.get("enterprise_value")
    ev_negative = (
        ev is None
        and data.get("ebitda_ttm") is not None
        and float(data.get("ebitda_ttm") or 0.0) > 0
    )

    fcf_yield_capped = fy_raw > 0.40
    book_yield_outlier = by_raw > 20.0

    return ValuationRatios(
        ebitda_yield=max(float(ey), 0.0),
        earnings_yield=max(float(ery), 0.0),
        fcf_yield=max(fy_raw, 0.0),
        book_yield=max(by_raw, 0.0),
        ebitda_yield_floored=ebitda_floored,
        ev_negative=ev_negative,
        fcf_yield_capped=fcf_yield_capped,
        fcf_yield_raw=fy_raw,
        book_yield_outlier=book_yield_outlier,
        book_yield_raw=by_raw,
    )


def sanitize_quality(data: dict) -> QualityRatios:
    """If net_income < 0, FCF/NI is neutralized (z = 0).
    If |FCF/NI| > 10 it is also neutralized: such an extreme ratio reflects a near-zero NI
    (edge case), not a recurring quality signal.
    """
    net_income = data.get("net_income_ttm")
    fcf_ni = data.get("fcf_ni")

    fcf_ni_extreme = bool(fcf_ni is not None and abs(float(fcf_ni)) > 10.0)
    fcf_ni_zeroed = bool(
        (net_income is not None and net_income < 0)
        or (fcf_ni is None and net_income is not None and net_income <= 0)
        or fcf_ni_extreme
    )
    return QualityRatios(
        gp_a=data.get("gp_a"),
        roa=data.get("roa"),
        roe=data.get("roe"),
        operating_margin=data.get("operating_margin"),
        fcf_ni=fcf_ni,
        fcf_ni_zeroed=fcf_ni_zeroed,
        fcf_ni_extreme=fcf_ni_extreme,
    )


def sanitize_growth(data: dict) -> GrowthRatios:
    """FCF: OLS as the primary metric (>= 3 historical points); CAGR as fallback when OLS is not
    computable and the base is positive.
      fcf_growth_method = 'ols'               → OLS available and valid
      fcf_growth_method = 'cagr_fallback'     → OLS absent, CAGR usable
      fcf_growth_method = 'insufficient_data' → neither available → z = 0

    Revenue: same logic (OLS primary, CAGR fallback).
    Outlier guards:
      - Revenue CAGR fallback > 100%: revenue_cagr_outlier=True → z = 0
      - FCF CAGR fallback > 500%: fcf_cagr_base_negative=True → z = 0
      - Any OLS with |value| > 5.0: neutralized defensively
    """
    # -- Revenue --------------------------------------------------------------
    rev_trend = data.get("revenue_trend_ols")
    revenue_cagr_outlier = False
    if rev_trend is None:
        rev_cagr = data.get("revenue_cagr_3y")
        if rev_cagr is not None and abs(float(rev_cagr)) > 1.0:
            revenue_cagr_outlier = True
            rev_cagr = None
        rev_trend = rev_cagr
    if rev_trend is not None and abs(float(rev_trend)) > 5.0:
        revenue_cagr_outlier = True
        rev_trend = None

    # -- FCF ------------------------------------------------------------------
    ols_val = data.get("fcf_trend_ols")
    cagr_val = data.get("fcf_cagr_3y")
    fcf_cagr_base_negative = False
    fcf_cagr_fallback: float | None = None
    fcf_growth_method: str = "insufficient_data"
    fcf_trend_val: float | None = None

    if ols_val is not None:
        ols_f = float(ols_val)
        if abs(ols_f) > 5.0:
            # Extreme OLS: aberrant-data signal, neutralize
            fcf_cagr_base_negative = True
        else:
            fcf_trend_val = ols_f
            fcf_growth_method = "ols"
    else:
        # OLS unavailable (< 3 valid historical points)
        if cagr_val is not None:
            cagr_f = float(cagr_val)
            if abs(cagr_f) > 5.0:
                fcf_cagr_base_negative = True
            else:
                fcf_trend_val = cagr_f
                fcf_cagr_fallback = cagr_f
                fcf_growth_method = "cagr_fallback"
        # else: both None → fcf_growth_method = 'insufficient_data' (already set)

    return GrowthRatios(
        revenue_cagr_3y=rev_trend,
        fcf_trend=fcf_trend_val,
        fcf_cagr_zeroed=fcf_trend_val is None,
        delta_gross_margin=data.get("delta_gross_margin"),
        asset_growth=data.get("asset_growth"),
        share_dilution=data.get("share_dilution"),
        fcf_cagr_base_negative=fcf_cagr_base_negative,
        revenue_cagr_outlier=revenue_cagr_outlier,
        fcf_cagr_fallback=fcf_cagr_fallback,
        fcf_growth_method=fcf_growth_method,
    )


def sanitize_solvency(
    data: dict,
    dist_interest_coverage: pd.Series | None = None,
) -> SolvencyRatios:
    """Three rules for interest coverage:
      - interest_coverage available → used as-is.
      - interest_expense = 0 and total_debt < threshold → company with no real debt;
        the 95th percentile of the universe distribution is assigned (or a 50x fallback).
      - interest_expense None or = 0 with relevant debt → data not available.

    Negative equity → debt_to_equity_z_override = -3.0 (forced strongly negative signal).
    """
    int_cov = data.get("interest_coverage")
    int_exp = data.get("interest_expense_ttm")
    total_debt = data.get("total_debt")
    equity = data.get("equity")

    ic_used, ic_assigned, ic_missing = _resolve_interest_coverage(
        int_cov, int_exp, total_debt, dist_interest_coverage
    )

    negative_equity = equity is not None and float(equity) < 0.0
    de_override: float | None = -3.0 if negative_equity else None

    return SolvencyRatios(
        dn_ebitda=data.get("dn_ebitda"),
        interest_coverage_used=ic_used,
        interest_coverage_assigned=ic_assigned,
        interest_coverage_missing=ic_missing,
        current_ratio=data.get("current_ratio"),
        debt_to_equity=data.get("debt_to_equity"),
        cfo_debt=data.get("cfo_debt"),
        debt_to_equity_z_override=de_override,
    )


def sanitize_all(
    data: dict,
    dist_interest_coverage: pd.Series | None = None,
) -> SanitizedRatios:
    """Apply the four sanitizations at once."""
    sector = data.get("sector")
    equity = data.get("equity")
    return SanitizedRatios(
        valuation=sanitize_valuation(data),
        quality=sanitize_quality(data),
        growth=sanitize_growth(data),
        solvency=sanitize_solvency(data, dist_interest_coverage),
        is_financial=is_financial_sector(sector),
        is_real_estate=is_real_estate_sector(sector),
        negative_equity=equity is not None and float(equity) < 0.0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assign_debt_free_coverage(dist: pd.Series | None) -> float:
    if dist is None:
        return 50.0
    clean = dist.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) >= _MIN_COVERAGE_SAMPLE:
        return float(clean.quantile(0.95))
    return 50.0


def _resolve_interest_coverage(
    int_cov: float | None,
    int_exp: float | None,
    total_debt: float | None,
    dist_ic: pd.Series | None,
) -> tuple[float | None, bool, bool]:
    """Return (coverage_used, assigned_debt_free, missing)."""
    if int_cov is not None:
        return int_cov, False, False
    if int_exp is None:
        return None, False, True
    if int_exp == 0:
        if total_debt is None or float(total_debt) <= _DEBT_THRESHOLD:
            return _assign_debt_free_coverage(dist_ic), True, False
        return None, False, True
    return None, False, True
