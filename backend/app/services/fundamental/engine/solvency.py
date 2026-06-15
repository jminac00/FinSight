"""Solvency sub-block, 20% weight in the fundamental score.

Interest Coverage is handled conservatively:
    - interest_expense_ttm is None does not imply absence of debt.
    - interest_expense_ttm == 0 is only interpreted as no debt when total_debt is null
      or very low.

Methodology per GICS sector:
    - Financial Services: excludes DN/EBITDA and D/E (structural debt not comparable with
      ordinary corporate debt); redistributes over IC (40%), CR (35%), CFO/Debt (25%);
      composite z-score computed only against sector peers.
    - Real Estate / Utilities: 5 original ratios with standard weights; the composite raw score
      is normalized with 80% sector z-score + 20% universe z-score, to reduce the bias caused by
      comparing against companies with no structural debt.
    - Other sectors: standard methodology (universe z-score).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.fundamental.engine.sanitizer import sanitize_solvency

logger = logging.getLogger(__name__)

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
# Normalization helpers
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
    """Combine z-scores with Financial Services weights (no DN/EBITDA, no D/E)."""
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
    if sector == _FINANCIAL_SECTOR:
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
) -> tuple[float | None, dict[str, float | None]]:
    dist_dn = _get_distribution(universe_df, _UNIVERSE_COL["dn_ebitda"])
    dist_ic = _get_distribution(universe_df, _UNIVERSE_COL["interest_coverage"])
    dist_cr = _get_distribution(universe_df, _UNIVERSE_COL["current_ratio"])
    dist_de = _get_distribution(universe_df, _UNIVERSE_COL["debt_to_equity"])
    dist_cfo = _get_distribution(universe_df, _UNIVERSE_COL["cfo_debt"])

    z_dn = _normalize_inverted(dn_ebitda, dist_dn)
    z_ic = _normalize_direct(int_cov, dist_ic)
    z_cr = _normalize_direct(curr_ratio, dist_cr)
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
) -> tuple[float | None, dict[str, float | None]]:
    """Raw score for Financial Services: only IC, CR and CFO/Debt.

    DN/EBITDA and D/E are excluded because the structural debt of banks and financials
    (deposits, debt issuance for intermediation) is not comparable with ordinary corporate debt.

    Reference distribution: sector if it has >= MIN_SECTOR_SIZE valid points for that metric;
    full universe as fallback (avoids z-scores with a single observation, as happens with
    interest_coverage in the Financial Services sector).

    cfo_debt <= 0 is treated as None: for large banks yfinance returns operatingCashflow=0
    systematically (a data artifact, not a real value).
    """
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

    z_ic = _normalize_direct(int_cov, dist_ic)
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
    """Distribution of raw scores for companies in the same sector.

    If financial=True it uses _solvency_raw_financial (IC, CR, CFO/Debt with FS weights)
    comparing against sector peers. If False it uses the standard 5-ratio logic comparing
    against the full universe (only the composite composition changes).
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
            "Sector '%s' has only %d companies; sector distribution not available.",
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
    # No clip on raw_score: clipping collapses outliers to the same P1/P99 percentile producing
    # artificial clusters. A more extreme z-score is the correct behavior for a genuinely atypical
    # company.
    return _z_score(raw_score, wins)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_solvency_score(
    company_data: dict[str, Any],
    universe_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute the solvency score (0-10) with sector-adjusted methodology."""
    result: dict[str, Any] = {
        "sub_bloque": "solvencia",
        "ticker": company_data.get("ticker"),
        "sector": company_data.get("sector"),
    }

    dist_ic = _get_distribution(universe_df, _UNIVERSE_COL["interest_coverage"])
    ratios = sanitize_solvency(company_data, dist_ic)

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

    # --- Determine methodology per sector ---
    sector = company_data.get("sector")
    methodology = _get_solvency_methodology(sector)
    result["solvency_methodology"] = methodology

    # --- Compute raw score and individual z-scores per methodology ---
    if methodology == "financial_services_adjusted":
        # Financial Services: only IC, CR, CFO/Debt vs sector peers
        if "sector" in universe_df.columns:
            sector_df = universe_df[universe_df["sector"] == _FINANCIAL_SECTOR]
        else:
            sector_df = pd.DataFrame()

        if len(sector_df) >= MIN_SECTOR_SIZE:
            raw_score, z_detail = _solvency_raw_financial(
                int_cov_used, curr_ratio, cfo_debt, sector_df, universe_df
            )
            sector_raw = _universe_raw_scores_sector(universe_df, _FINANCIAL_SECTOR, financial=True)
            solvency_z_score = _normalize_composite(raw_score, sector_raw)
        else:
            # Fallback if there are not enough peers
            logger.warning(
                "Financial Services with fewer than %d peers; using standard methodology.",
                MIN_SECTOR_SIZE,
            )
            raw_score, z_detail = _solvency_raw_from_values(
                dn_ebitda, int_cov_used, curr_ratio, de, cfo_debt, equity, universe_df
            )
            solvency_z_score = _normalize_composite(raw_score, _universe_raw_scores(universe_df))
            result["solvency_methodology"] = "standard"

    elif methodology in ("reit_adjusted", "utilities_adjusted"):
        # Real Estate / Utilities: 5 standard ratios, but blend 80% sector + 20% universe
        raw_score, z_detail = _solvency_raw_from_values(
            dn_ebitda, int_cov_used, curr_ratio, de, cfo_debt, equity, universe_df
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
            dn_ebitda, int_cov_used, curr_ratio, de, cfo_debt, equity, universe_df
        )
        solvency_z_score = _normalize_composite(raw_score, _universe_raw_scores(universe_df))

    result.update(z_detail)
    result["de_negative_equity"] = ratios.debt_to_equity_z_override is not None
    result["z_combined"] = raw_score
    result["solvency_raw_score"] = raw_score
    result["solvency_z_score"] = solvency_z_score

    # Propagate data source to the methodology suffix for Financial Services
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
