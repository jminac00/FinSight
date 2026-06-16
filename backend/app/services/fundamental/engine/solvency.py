"""
solvency.py
Solvency sub-block, 20% weight in the fundamental score.

Interest Coverage is treated conservatively:
    - interest_expense_ttm is None does not imply the absence of debt.
    - interest_expense_ttm == 0 is only interpreted as no debt if total_debt
      is null or very low.

Methodology per GICS sector:
    - Financial Services: excludes DN/EBITDA and D/E (structural debt not
      comparable with ordinary corporate debt); redistributes onto IC (40%),
      CR (35%), CFO/Debt (25%); composite z-score computed only against sector peers.
    - Real Estate / Utilities: 5 original ratios with standard weights; the
      composite raw score is normalized with 80% sector z-score + 20% universe
      z-score, to reduce the bias from comparing against companies without
      structural debt.
    - Other sectors: standard methodology (universe z-score).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.fundamental.engine.ratio_profiles import SP500_VALIDATED, RatioProfile
from app.services.fundamental.engine.sanitizer import sanitize_solvency

logger = logging.getLogger(__name__)

# Direction of each ratio when normalizing: inverted = a higher value is worse.
_INVERTED_RATIOS = frozenset({"dn_ebitda", "debt_to_equity"})


def _universe_key(universe_df: pd.DataFrame) -> object:
    return universe_df.attrs.get("universe_name", id(universe_df))


_WEIGHTS: dict[str, float] = {
    "dn_ebitda": 0.30,
    "interest_coverage": 0.25,
    "current_ratio": 0.20,
    "debt_to_equity": 0.15,
    "cfo_debt": 0.10,
}

# Adjusted weights for Financial Services (excludes DN/EBITDA and D/E)
_WEIGHTS_FINANCIAL: dict[str, float] = {
    "interest_coverage": 0.40,
    "current_ratio": 0.35,
    "cfo_debt": 0.25,
}

_UNIVERSE_COL: dict[str, str] = {
    "dn_ebitda": "dn_ebitda",
    "interest_coverage": "interest_coverage",
    "current_ratio": "currentRatio",
    "debt_to_equity": "debtToEquity",
    "cfo_debt": "cfo_debt",
}

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
MIN_SECTOR_SIZE = 10

_FINANCIAL_SECTOR = "Financial Services"
_REIT_SECTORS = frozenset({"Real Estate"})
_UTILITY_SECTORS = frozenset({"Utilities"})

_UNIVERSE_RAW_CACHE: dict[int, pd.Series] = {}
_SECTOR_RAW_CACHE: dict[tuple, pd.Series] = {}


# ---------------------------------------------------------------------------
# Normalization helpers (unchanged from the previous fix)
# ---------------------------------------------------------------------------


def _winsorize(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return series
    return series.clip(
        lower=float(clean.quantile(WINSOR_LOWER)),
        upper=float(clean.quantile(WINSOR_UPPER)),
    )


def _z_score(value: float, distribution: pd.Series) -> float | None:
    clean = distribution.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None
    std = float(clean.std(ddof=0))
    if std == 0.0:
        return None
    return float((value - float(clean.mean())) / std)


def _z_to_score(z: float, k: float = 1.2) -> float:
    x = float(np.clip(k * z, -60.0, 60.0))
    return float(10.0 / (1.0 + np.exp(-x)))


def _signal(score: float) -> str:
    if score > 6.5:
        return "solvencia solida"
    if score < 4.0:
        return "debil"
    return "moderada"


def _get_distribution(universe_df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in universe_df.columns:
        logger.debug("Column '%s' not found in universe_df.", col)
        return None
    return universe_df[col]


def _normalize_direct(value: float | None, distribution: pd.Series | None) -> float | None:
    if value is None or distribution is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    wins = _winsorize(distribution)
    clean = wins.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    # No clip: clipping to the winsorized range collapses outliers to the same boundary → clusters.
    return _z_score(float(value), wins)


def _normalize_inverted(value: float | None, distribution: pd.Series | None) -> float | None:
    z = _normalize_direct(value, distribution)
    return -z if z is not None else None


def _is_valid_z(z: object) -> bool:
    """True if z is a finite float (not None, not NaN, not inf)."""
    if z is None:
        return False
    try:
        return bool(pd.notna(z) and np.isfinite(float(z)))
    except (TypeError, ValueError):
        return False


def _combine_z_scores(z_map: dict[str, float | None]) -> float | None:
    total_weight = 0.0
    weighted_sum = 0.0
    for var, z_val in z_map.items():
        if _is_valid_z(z_val):
            weighted_sum += _WEIGHTS[var] * float(z_val)
            total_weight += _WEIGHTS[var]
    return weighted_sum / total_weight if total_weight > 0.0 else None


def _combine_z_scores_financial(z_map: dict[str, float | None]) -> float | None:
    """Combine z-scores with Financial Services weights (no DN/EBITDA or D/E)."""
    total_weight = 0.0
    weighted_sum = 0.0
    for var, z_val in z_map.items():
        if _is_valid_z(z_val):
            weighted_sum += _WEIGHTS_FINANCIAL[var] * float(z_val)
            total_weight += _WEIGHTS_FINANCIAL[var]
    return weighted_sum / total_weight if total_weight > 0.0 else None


# ---------------------------------------------------------------------------
# Sector classification
# ---------------------------------------------------------------------------


def _get_solvency_methodology(sector: str | None) -> str:
    """Return the solvency methodology for the given GICS sector."""
    if not sector or (isinstance(sector, float) and pd.isna(sector)):
        return "standard"
    if sector in (_FINANCIAL_SECTOR, "Financials"):
        return "financial_services_adjusted"
    if sector in _REIT_SECTORS:
        return "reit_adjusted"
    if sector in _UTILITY_SECTORS:
        return "utilities_adjusted"
    return "standard"


# ---------------------------------------------------------------------------
# Raw score computation
# ---------------------------------------------------------------------------


def _solvency_raw_from_values(
    dn_ebitda: float | None,
    int_cov: float | None,
    curr_ratio: float | None,
    de: float | None,
    cfo_debt: float | None,
    equity: float | None,
    universe_df: pd.DataFrame,
    neutral_flags: set[str] | None = None,
) -> tuple[float | None, dict[str, float | None]]:
    neutral_flags = neutral_flags or set()
    dist_dn = _get_distribution(universe_df, _UNIVERSE_COL["dn_ebitda"])
    dist_ic = _get_distribution(universe_df, _UNIVERSE_COL["interest_coverage"])
    dist_cr = _get_distribution(universe_df, _UNIVERSE_COL["current_ratio"])
    dist_de = _get_distribution(universe_df, _UNIVERSE_COL["debt_to_equity"])
    dist_cfo = _get_distribution(universe_df, _UNIVERSE_COL["cfo_debt"])

    z_dn = _normalize_inverted(dn_ebitda, dist_dn)
    z_ic = 0.0 if "coverage_bank_no_data" in neutral_flags else _normalize_direct(int_cov, dist_ic)
    z_cr = _normalize_direct(curr_ratio, dist_cr)
    if "de_no_data" in neutral_flags:
        z_de = 0.0
    else:
        z_de = -3.0 if (equity is not None and equity < 0) else _normalize_inverted(de, dist_de)
    z_cfo = _normalize_direct(cfo_debt, dist_cfo)

    z_map = {
        "dn_ebitda": z_dn,
        "interest_coverage": z_ic,
        "current_ratio": z_cr,
        "debt_to_equity": z_de,
        "cfo_debt": z_cfo,
    }
    return _combine_z_scores(z_map), {
        "z_dn_ebitda": z_dn,
        "z_interest_coverage": z_ic,
        "z_current_ratio": z_cr,
        "z_debt_to_equity": z_de,
        "z_cfo_debt": z_cfo,
    }


def _solvency_raw_financial(
    int_cov: float | None,
    curr_ratio: float | None,
    cfo_debt: float | None,
    sector_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    neutral_flags: set[str] | None = None,
) -> tuple[float | None, dict[str, float | None]]:
    """
    Raw score for Financial Services: only IC, CR and CFO/Debt.

    DN/EBITDA and D/E are excluded because the structural debt of banks and
    financials (deposits, debt issuance for intermediation) is not comparable
    with ordinary corporate debt.

    Reference distribution: the sector if it has >= MIN_SECTOR_SIZE valid points
    for that metric; the full universe as a fallback (avoids z-scores with a
    single observation, as happens with interest_coverage in the Financial
    Services sector).

    cfo_debt <= 0 is treated as None: for large banks yfinance returns
    operatingCashflow=0 systematically (a data artifact, not a real value).
    """
    neutral_flags = neutral_flags or set()

    # cfo_debt = exactly 0.0 is a yfinance artifact for large banks
    cfo_clean: float | None = (
        cfo_debt if (cfo_debt is not None and not pd.isna(cfo_debt) and cfo_debt > 0) else None
    )

    def _pick_dist(col_key: str) -> pd.Series | None:
        col = _UNIVERSE_COL[col_key]
        sector_dist = _get_distribution(sector_df, col)
        if sector_dist is not None and int(sector_dist.dropna().count()) >= MIN_SECTOR_SIZE:
            return sector_dist
        return _get_distribution(universe_df, col)

    dist_ic = _pick_dist("interest_coverage")
    dist_cr = _pick_dist("current_ratio")
    dist_cfo = _pick_dist("cfo_debt")

    z_ic = 0.0 if "coverage_bank_no_data" in neutral_flags else _normalize_direct(int_cov, dist_ic)
    z_cr = _normalize_direct(curr_ratio, dist_cr)
    z_cfo = _normalize_direct(cfo_clean, dist_cfo)

    z_map = {
        "interest_coverage": z_ic,
        "current_ratio": z_cr,
        "cfo_debt": z_cfo,
    }
    return _combine_z_scores_financial(z_map), {
        "z_dn_ebitda": None,  # excluded for Financial Services
        "z_interest_coverage": z_ic,
        "z_current_ratio": z_cr,
        "z_debt_to_equity": None,  # excluded for Financial Services
        "z_cfo_debt": z_cfo,
    }


# ---------------------------------------------------------------------------
# Universe distributions
# ---------------------------------------------------------------------------


def _universe_raw_scores(universe_df: pd.DataFrame) -> pd.Series:
    cache_key = id(universe_df)
    if cache_key in _UNIVERSE_RAW_CACHE:
        return _UNIVERSE_RAW_CACHE[cache_key]

    if universe_df.empty:
        return pd.Series(dtype=float)

    def row_score(row: pd.Series) -> float | None:
        raw, _ = _solvency_raw_from_values(
            row.get("dn_ebitda"),
            row.get("interest_coverage"),
            row.get("currentRatio"),
            row.get("debtToEquity"),
            row.get("cfo_debt"),
            row.get("equity"),
            universe_df,
        )
        return raw

    raw_scores = universe_df.apply(row_score, axis=1)
    raw_scores = raw_scores.replace([np.inf, -np.inf], np.nan).dropna()
    _UNIVERSE_RAW_CACHE[cache_key] = raw_scores
    return raw_scores


def _universe_raw_scores_sector(
    universe_df: pd.DataFrame,
    sector: str,
    financial: bool = False,
) -> pd.Series:
    """
    Distribution of raw scores for companies in the same sector.

    If financial=True it uses _solvency_raw_financial (IC, CR, CFO/Debt with FS
    weights) comparing against sector peers. If False it uses the standard 5-ratio
    logic comparing against the full universe (only the composite composition changes).
    """
    cache_key = (id(universe_df), sector, financial)
    if cache_key in _SECTOR_RAW_CACHE:
        return _SECTOR_RAW_CACHE[cache_key]

    if "sector" not in universe_df.columns:
        return pd.Series(dtype=float)

    mask = universe_df["sector"] == sector
    sector_df = universe_df[mask]

    if len(sector_df) < MIN_SECTOR_SIZE:
        logger.info(
            "Sector '%s' with only %d companies; sector distribution not available.",
            sector,
            len(sector_df),
        )
        return pd.Series(dtype=float)

    if financial:

        def row_score(row: pd.Series) -> float | None:
            raw, _ = _solvency_raw_financial(
                row.get("interest_coverage"),
                row.get("currentRatio"),
                row.get("cfo_debt"),
                sector_df,
                universe_df,
            )
            return raw
    else:

        def row_score(row: pd.Series) -> float | None:
            raw, _ = _solvency_raw_from_values(
                row.get("dn_ebitda"),
                row.get("interest_coverage"),
                row.get("currentRatio"),
                row.get("debtToEquity"),
                row.get("cfo_debt"),
                row.get("equity"),
                universe_df,
            )
            return raw

    raw_scores = sector_df.apply(row_score, axis=1)
    raw_scores = raw_scores.replace([np.inf, -np.inf], np.nan).dropna()
    _SECTOR_RAW_CACHE[cache_key] = raw_scores
    return raw_scores


def _normalize_composite(raw_score: float | None, universe_raw: pd.Series) -> float | None:
    if raw_score is None or universe_raw.empty:
        return None
    wins = _winsorize(universe_raw)
    clean = wins.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    # No clip on raw_score: clipping collapses outliers to the same P1/P99 percentile,
    # producing artificial clusters. A more extreme z-score is the correct behavior for
    # a genuinely atypical company.
    return _z_score(raw_score, wins)


# ---------------------------------------------------------------------------
# Generic per-profile path (profiles != sp500_validated, e.g. global_robust)
# ---------------------------------------------------------------------------
#
# To preserve the S&P 500 EXACTLY, the sp500_validated profile keeps using the
# original body (dispatched at the start of compute_solvency_score). Other
# profiles use this flat standard normalization with the profile's weights/columns
# (e.g. global_robust drops D/E), without the sector methodologies designed for
# the S&P 500.
_GENERIC_RAW_CACHE: dict[tuple, pd.Series] = {}


def _combine_z_scores_generic(
    z_map: dict[str, float | None], weights: dict[str, float]
) -> float | None:
    total_weight = 0.0
    weighted_sum = 0.0
    for var, z_val in z_map.items():
        if _is_valid_z(z_val):
            weighted_sum += weights[var] * float(z_val)
            total_weight += weights[var]
    return weighted_sum / total_weight if total_weight > 0.0 else None


def _solvency_raw_generic(
    values: dict[str, float | None],
    equity: float | None,
    universe_df: pd.DataFrame,
    weights: dict[str, float],
    columns: dict[str, str],
    neutral_flags: set[str] | None = None,
) -> tuple[float | None, dict[str, float | None]]:
    neutral_flags = neutral_flags or set()
    z_map: dict[str, float | None] = {}
    detail: dict[str, float | None] = {}  # only the active profile's variables
    for var in weights:
        dist = _get_distribution(universe_df, columns[var])
        if var == "interest_coverage" and "coverage_bank_no_data" in neutral_flags:
            z: float | None = 0.0
        elif var == "debt_to_equity" and "de_no_data" in neutral_flags:
            z = 0.0
        elif var == "debt_to_equity" and equity is not None and equity < 0:
            z: float | None = -3.0
        elif var in _INVERTED_RATIOS:
            z = _normalize_inverted(values.get(var), dist)
        else:
            z = _normalize_direct(values.get(var), dist)
        z_map[var] = z
        detail[f"z_{var}"] = z
    return _combine_z_scores_generic(z_map, weights), detail


def _universe_raw_scores_generic(
    universe_df: pd.DataFrame, weights: dict[str, float], columns: dict[str, str]
) -> pd.Series:
    cache_key = (_universe_key(universe_df), tuple(weights.items()))
    if cache_key in _GENERIC_RAW_CACHE:
        return _GENERIC_RAW_CACHE[cache_key]
    if universe_df.empty:
        return pd.Series(dtype=float)

    def row_score(row: pd.Series) -> float | None:
        values = {var: row.get(columns[var]) for var in weights}
        raw, _ = _solvency_raw_generic(values, row.get("equity"), universe_df, weights, columns)
        return raw

    raw_scores = universe_df.apply(row_score, axis=1)
    raw_scores = raw_scores.replace([np.inf, -np.inf], np.nan).dropna()
    _GENERIC_RAW_CACHE[cache_key] = raw_scores
    return raw_scores


def _compute_solvency_generic(
    company_data: dict[str, Any], universe_df: pd.DataFrame, profile: RatioProfile
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sub_bloque": "solvencia",
        "ticker": company_data.get("ticker"),
        "sector": company_data.get("sector"),
        "data_flags": (
            list(company_data.get("data_flags") or []) if profile.name != "sp500_validated" else []
        ),
    }
    weights = profile.solvency_weights
    columns = profile.solvency_columns

    ic_col = columns.get("interest_coverage", "interest_coverage")
    dist_ic = _get_distribution(universe_df, ic_col)
    ratios = sanitize_solvency(company_data, dist_ic)
    equity = company_data.get("equity")
    neutral_flags = set(company_data.get("data_flags") or [])

    base = {
        "dn_ebitda": ratios.dn_ebitda,
        "interest_coverage": ratios.interest_coverage_used,
        "current_ratio": ratios.current_ratio,
        "debt_to_equity": ratios.debt_to_equity,
        "cfo_debt": ratios.cfo_debt,
    }
    values = {var: base.get(var) for var in weights}

    raw_score, z_detail = _solvency_raw_generic(
        values, equity, universe_df, weights, columns, neutral_flags=neutral_flags
    )
    solvency_z_score = _normalize_composite(
        raw_score, _universe_raw_scores_generic(universe_df, weights, columns)
    )

    result.update(z_detail)
    result.update(
        {
            "dn_ebitda": ratios.dn_ebitda,
            "interest_coverage": company_data.get("interest_coverage"),
            "interest_coverage_used": ratios.interest_coverage_used,
            "interest_coverage_assigned": ratios.interest_coverage_assigned,
            "interest_coverage_missing": ratios.interest_coverage_missing,
            "current_ratio": ratios.current_ratio,
            "debt_to_equity": ratios.debt_to_equity,
            "cfo_debt": ratios.cfo_debt,
            "total_debt": company_data.get("total_debt"),
        }
    )
    result["solvency_methodology"] = f"standard_{profile.name}"
    result["de_negative_equity"] = equity is not None and float(equity) < 0.0
    result["z_combined"] = raw_score
    result["solvency_raw_score"] = raw_score
    result["solvency_z_score"] = solvency_z_score

    if solvency_z_score is None:
        score: float | None = None
        signal = "sin_datos"
    else:
        score = _z_to_score(solvency_z_score)
        signal = _signal(score)
    result["score"] = score
    result["signal"] = signal
    result["solvency_summary"] = _build_summary(result)
    return result


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_solvency_score(
    company_data: dict[str, Any],
    universe_df: pd.DataFrame,
    profile: RatioProfile = SP500_VALIDATED,
) -> dict[str, Any]:
    """Compute the solvency score (0-10) with a sector-adjusted methodology."""
    # Non-validated profiles (e.g. global_robust) use the flat standard
    # normalization with their weights. The S&P 500 keeps its original body intact.
    if profile.name != "sp500_validated":
        return _compute_solvency_generic(company_data, universe_df, profile)

    result: dict[str, Any] = {
        "sub_bloque": "solvencia",
        "ticker": company_data.get("ticker"),
        "sector": company_data.get("sector"),
        "data_flags": [],
    }

    dist_ic = _get_distribution(universe_df, _UNIVERSE_COL["interest_coverage"])
    ratios = sanitize_solvency(company_data, dist_ic)
    neutral_flags: set[str] = set()

    dn_ebitda = ratios.dn_ebitda
    int_cov = company_data.get("interest_coverage")
    int_cov_used = ratios.interest_coverage_used
    ic_assigned = ratios.interest_coverage_assigned
    ic_missing = ratios.interest_coverage_missing
    curr_ratio = ratios.current_ratio
    de = ratios.debt_to_equity
    cfo_debt = ratios.cfo_debt
    equity = company_data.get("equity")

    result.update(
        {
            "dn_ebitda": dn_ebitda,
            "interest_coverage": int_cov,
            "interest_coverage_used": int_cov_used,
            "interest_coverage_assigned": ic_assigned,
            "interest_coverage_missing": ic_missing,
            "current_ratio": curr_ratio,
            "debt_to_equity": de,
            "cfo_debt": cfo_debt,
            "total_debt": company_data.get("total_debt"),
        }
    )

    # --- Determine the methodology by sector ---
    sector = company_data.get("sector")
    methodology = _get_solvency_methodology(sector)
    result["solvency_methodology"] = methodology

    # --- Compute the raw score and individual z-scores per methodology ---
    if methodology == "financial_services_adjusted":
        # Financial Services: only IC, CR, CFO/Debt vs sector peers
        if "sector" in universe_df.columns:
            sector_df = universe_df[universe_df["sector"] == _FINANCIAL_SECTOR]
        else:
            sector_df = pd.DataFrame()

        if len(sector_df) >= MIN_SECTOR_SIZE:
            raw_score, z_detail = _solvency_raw_financial(
                int_cov_used,
                curr_ratio,
                cfo_debt,
                sector_df,
                universe_df,
                neutral_flags=neutral_flags,
            )
            sector_raw = _universe_raw_scores_sector(universe_df, _FINANCIAL_SECTOR, financial=True)
            solvency_z_score = _normalize_composite(raw_score, sector_raw)
        else:
            # Fallback when there are not enough peers
            logger.warning(
                "Financial Services with fewer than %d peers; using standard methodology.",
                MIN_SECTOR_SIZE,
            )
            raw_score, z_detail = _solvency_raw_from_values(
                dn_ebitda,
                int_cov_used,
                curr_ratio,
                de,
                cfo_debt,
                equity,
                universe_df,
                neutral_flags=neutral_flags,
            )
            solvency_z_score = _normalize_composite(raw_score, _universe_raw_scores(universe_df))
            result["solvency_methodology"] = "standard"

    elif methodology in ("reit_adjusted", "utilities_adjusted"):
        # Real Estate / Utilities: 5 standard ratios, but blend 80% sector + 20% universe
        raw_score, z_detail = _solvency_raw_from_values(
            dn_ebitda,
            int_cov_used,
            curr_ratio,
            de,
            cfo_debt,
            equity,
            universe_df,
            neutral_flags=neutral_flags,
        )
        z_univ = _normalize_composite(raw_score, _universe_raw_scores(universe_df))
        sector_raw = _universe_raw_scores_sector(universe_df, sector, financial=False)
        z_sect = _normalize_composite(raw_score, sector_raw) if not sector_raw.empty else None

        if z_univ is not None and z_sect is not None:
            solvency_z_score = 0.20 * z_univ + 0.80 * z_sect
        elif z_univ is not None:
            solvency_z_score = z_univ
        else:
            solvency_z_score = z_sect

    else:
        # Standard: original logic
        raw_score, z_detail = _solvency_raw_from_values(
            dn_ebitda,
            int_cov_used,
            curr_ratio,
            de,
            cfo_debt,
            equity,
            universe_df,
            neutral_flags=neutral_flags,
        )
        solvency_z_score = _normalize_composite(raw_score, _universe_raw_scores(universe_df))

    result.update(z_detail)
    result["de_negative_equity"] = ratios.debt_to_equity_z_override is not None
    result["z_combined"] = raw_score
    result["solvency_raw_score"] = raw_score
    result["solvency_z_score"] = solvency_z_score

    # Propagate the data source to the methodology suffix for Financial Services
    data_source = company_data.get("solvency_data_source")
    if data_source and result.get("solvency_methodology") == "financial_services_adjusted":
        if data_source == "fmp":
            result["solvency_methodology"] = "financial_services_adjusted_fmp"
        elif data_source == "partial":
            result["solvency_methodology"] = "financial_services_partial"
        else:
            result["solvency_methodology"] = "financial_services_adjusted_yf"

    # Fallback for financial_services_partial: CR-only → sector imputation
    if (
        solvency_z_score is None
        and result.get("solvency_methodology") == "financial_services_partial"
    ):
        cr_val = ratios.current_ratio
        if cr_val is not None and not pd.isna(cr_val) and float(cr_val) > 0:
            dist_cr = _get_distribution(universe_df, _UNIVERSE_COL["current_ratio"])
            z_cr_only = _normalize_direct(float(cr_val), dist_cr)
            if z_cr_only is not None:
                solvency_z_score = z_cr_only
                result["z_current_ratio"] = z_cr_only
                result["solvency_methodology"] = "financial_services_minimal"

        if solvency_z_score is None:
            logger.warning(
                "Solvency imputed to 5.0 for %s: no ratio available.",
                company_data.get("ticker", "?"),
            )
            result["solvency_methodology"] = "financial_services_imputed"

    if (
        solvency_z_score is None
        and result.get("solvency_methodology") == "financial_services_imputed"
    ):
        score: float | None = 5.0
        signal = _signal(score)
    elif solvency_z_score is None:
        score = None
        signal = "sin_datos"
    else:
        score = _z_to_score(solvency_z_score)
        signal = _signal(score)

    result["score"] = score
    result["signal"] = signal
    result["solvency_summary"] = _build_summary(result)
    return result


# ---------------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------------


def _build_summary(r: dict[str, Any]) -> str:
    ticker = r.get("ticker", "?")
    score = r.get("score")
    signal = r.get("signal", "?")
    methodology = r.get("solvency_methodology", "standard")
    score_str = f"{score:.2f}/10" if score is not None else "N/A"

    def fmt(v: float | None, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}x" if v is not None else "N/A"

    def z_fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    if r.get("interest_coverage_assigned"):
        ic_note = " [sin deuda real -> p95]"
    elif r.get("interest_coverage_missing"):
        ic_note = " [dato no disponible]"
    else:
        ic_note = ""
    de_note = " [equity<0 -> z=-3]" if r.get("de_negative_equity") else ""

    if methodology == "financial_services_adjusted":
        components = (
            f"CobInt={fmt(r.get('interest_coverage_used'))}{ic_note} "
            f"(z={z_fmt(r.get('z_interest_coverage'))}, x0.40), "
            f"CurrRatio={fmt(r.get('current_ratio'))} "
            f"(z={z_fmt(r.get('z_current_ratio'))}, x0.35), "
            f"CFO/Deuda={fmt(r.get('cfo_debt'))} (z={z_fmt(r.get('z_cfo_debt'))}, x0.25) "
            f"[DN/EBITDA y D/E excluidos — metodologia Financial Services]"
        )
    else:
        method_note = ""
        if methodology in ("reit_adjusted", "utilities_adjusted"):
            method_note = " [blend 80% sector + 20% universo]"
        components = (
            f"DN/EBITDA={fmt(r.get('dn_ebitda'))} inv (z={z_fmt(r.get('z_dn_ebitda'))}, x0.30), "
            f"CobInt={fmt(r.get('interest_coverage_used'))}{ic_note} "
            f"(z={z_fmt(r.get('z_interest_coverage'))}, x0.25), "
            f"CurrRatio={fmt(r.get('current_ratio'))} "
            f"(z={z_fmt(r.get('z_current_ratio'))}, x0.20), "
            f"D/E={fmt(r.get('debt_to_equity'))}{de_note} inv "
            f"(z={z_fmt(r.get('z_debt_to_equity'))}, x0.15), "
            f"CFO/Deuda={fmt(r.get('cfo_debt'))} (z={z_fmt(r.get('z_cfo_debt'))}, x0.10)"
            + method_note
        )

    label = {
        "solvencia solida": "presenta una posicion financiera solida",
        "moderada": "muestra una solvencia moderada",
        "debil": "registra senales de fragilidad financiera",
        "sin_datos": "solvencia no evaluable por datos insuficientes",
    }.get(signal, signal)

    return (
        f"{ticker} {label} - score de solvencia {score_str} ({signal}). Componentes: {components}."
    )


def _run_internal_checks() -> None:
    universe = pd.DataFrame(
        {
            "dn_ebitda": [0.0, 1.0, 2.0, 4.0, 6.0],
            "interest_coverage": [2.0, 5.0, 10.0, 20.0, 40.0],
            "currentRatio": [0.8, 1.0, 1.5, 2.0, 3.0],
            "debtToEquity": [0.0, 0.5, 1.0, 2.0, 4.0],
            "cfo_debt": [0.1, 0.2, 0.4, 0.8, 1.2],
        }
    )
    missing = compute_solvency_score(
        {"ticker": "MISS", "interest_coverage": None, "interest_expense_ttm": None},
        universe,
    )
    assert missing["interest_coverage_assigned"] is False
    assert missing["interest_coverage_missing"] is True
    assert missing["z_interest_coverage"] is None

    debt_free = compute_solvency_score(
        {
            "ticker": "FREE",
            "interest_coverage": None,
            "interest_expense_ttm": 0,
            "total_debt": 0,
        },
        universe,
    )
    assert debt_free["interest_coverage_assigned"] is True
    assert debt_free["interest_coverage_used"] is not None
