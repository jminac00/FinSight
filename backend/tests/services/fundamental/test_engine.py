"""Tests for the vendored fundamental engine using a synthetic universe (no network).

These guard against regressions introduced while translating/adapting the collaborator's code:
they verify the core methodology (relative normalization, z-score inversions, weight
redistribution) rather than exact numeric values.
"""

import numpy as np
import pandas as pd

from app.services.fundamental.engine.growth import compute_growth_score
from app.services.fundamental.engine.sanitizer import sanitize_all, sanitize_valuation
from app.services.fundamental.engine.scoring import combine_fundamental_scores
from app.services.fundamental.engine.solvency import compute_solvency_score
from app.services.fundamental.engine.valuation import compute_valuation_score


def _synthetic_universe(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(n)],
            "sector": ["Technology"] * n,
            "ebitda_yield": rng.uniform(0.02, 0.12, n),
            "earnings_yield": rng.uniform(0.02, 0.08, n),
            "fcf_yield": rng.uniform(0.01, 0.07, n),
            "book_yield": rng.uniform(0.05, 0.5, n),
            "gp_a": rng.uniform(0.1, 0.6, n),
            "returnOnAssets": rng.uniform(0.02, 0.2, n),
            "returnOnEquity": rng.uniform(0.05, 0.4, n),
            "operatingMargins": rng.uniform(0.1, 0.4, n),
            "fcf_ni": rng.uniform(0.5, 1.5, n),
            "revenue_trend_ols": rng.uniform(-0.1, 0.3, n),
            "fcf_trend_ols": rng.uniform(-0.1, 0.3, n),
            "delta_gross_margin": rng.uniform(-0.05, 0.05, n),
            "asset_growth": rng.uniform(-0.05, 0.25, n),
            "share_dilution": rng.uniform(-0.05, 0.05, n),
            "dn_ebitda": rng.uniform(0.0, 5.0, n),
            "interest_coverage": rng.uniform(2.0, 40.0, n),
            "currentRatio": rng.uniform(0.8, 3.0, n),
            "debtToEquity": rng.uniform(0.0, 3.0, n),
            "cfo_debt": rng.uniform(0.1, 1.2, n),
            "equity": rng.uniform(1e9, 1e11, n),
        }
    )


def test_sanitize_valuation_floors_negative_yields_and_flags_outliers():
    ratios = sanitize_valuation(
        {"ebitda_yield": -0.01, "earnings_yield": 0.04, "fcf_yield": 0.50, "book_yield": 25.0}
    )
    assert ratios.ebitda_yield == 0.0
    assert ratios.ebitda_yield_floored is True
    assert ratios.fcf_yield_capped is True
    assert ratios.book_yield_outlier is True


def test_sanitize_all_negative_equity_and_zeroed_fcf_ni():
    sanitized = sanitize_all(
        {
            "sector": "Technology",
            "net_income_ttm": -1e9,
            "fcf_ni": None,
            "equity": -5e9,
            "interest_expense_ttm": 0,
            "total_debt": 0,
        }
    )
    assert sanitized.quality.fcf_ni_zeroed is True
    assert sanitized.negative_equity is True
    assert sanitized.solvency.debt_to_equity_z_override == -3.0
    assert sanitized.solvency.interest_coverage_assigned is True


def test_valuation_is_monotonic_in_yields():
    """A company with higher yields than its peers must score at least as high as a cheaper one."""
    universe = _synthetic_universe()
    base = {"ticker": "X", "sector": "Other"}  # sector absent from universe → S&P-only z-score

    cheap = {
        **base,
        "ebitda_yield": 0.02,
        "earnings_yield": 0.02,
        "fcf_yield": 0.01,
        "book_yield": 0.05,
    }
    rich = {
        **base,
        "ebitda_yield": 0.12,
        "earnings_yield": 0.08,
        "fcf_yield": 0.07,
        "book_yield": 0.5,
    }

    score_cheap = compute_valuation_score(cheap, universe)["score"]
    score_rich = compute_valuation_score(rich, universe)["score"]
    assert score_rich > score_cheap


def test_valuation_median_company_scores_near_five():
    """A company whose raw score equals the universe mean gets ~5.0 (sigmoid at z=0)."""
    universe = _synthetic_universe()
    means = {
        "ebitda_yield": float(universe["ebitda_yield"].mean()),
        "earnings_yield": float(universe["earnings_yield"].mean()),
        "fcf_yield": float(universe["fcf_yield"].mean()),
        "book_yield": float(universe["book_yield"].mean()),
    }
    company = {"ticker": "MID", "sector": "Other", **means}
    score = compute_valuation_score(company, universe)["score"]
    assert 4.0 < score < 6.0


def test_growth_inverts_asset_growth_sign():
    """Lower asset growth (conservative) must yield a higher inverted z than aggressive growth."""
    universe = _synthetic_universe()
    low = compute_growth_score(
        {"ticker": "LOW", "sector": "Other", "asset_growth": -0.05, "share_dilution": 0.0},
        universe,
    )
    high = compute_growth_score(
        {"ticker": "HIGH", "sector": "Other", "asset_growth": 0.25, "share_dilution": 0.0},
        universe,
    )
    assert low["z_asset_growth_inv"] > high["z_asset_growth_inv"]


def test_solvency_interest_coverage_rules():
    universe = _synthetic_universe()
    missing = compute_solvency_score(
        {
            "ticker": "MISS",
            "sector": "Other",
            "interest_coverage": None,
            "interest_expense_ttm": None,
        },
        universe,
    )
    assert missing["interest_coverage_missing"] is True
    assert missing["z_interest_coverage"] is None

    debt_free = compute_solvency_score(
        {
            "ticker": "FREE",
            "sector": "Other",
            "interest_coverage": None,
            "interest_expense_ttm": 0,
            "total_debt": 0,
        },
        universe,
    )
    assert debt_free["interest_coverage_assigned"] is True
    assert debt_free["interest_coverage_used"] is not None


def test_combine_redistributes_weights_when_a_block_is_missing():
    val = {"ticker": "Z", "sector": "Other", "score": 6.0, "signal": "neutral"}
    qual = {"score": 6.0, "signal": "calidad media"}
    grow = {"score": 6.0, "signal": "moderado"}
    solv = {"score": None, "signal": "sin_datos"}  # missing block

    result = combine_fundamental_scores(val, qual, grow, solv)

    assert result["weights_used"]["solvencia"] == 0.0
    # Remaining weights must sum to 1 and the score stay at 6.0
    assert abs(sum(result["weights_used"].values()) - 1.0) < 1e-9
    assert result["score_final"] == 6.0
