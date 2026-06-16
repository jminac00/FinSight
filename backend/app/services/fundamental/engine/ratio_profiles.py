"""Declarative registry of fundamental ratio profiles.

A *ratio profile* defines which indicators enter each sub-block and with what weight. It is the
piece that `UniverseConfig.ratio_profile` points to.

- `sp500_validated`: mirrors EXACTLY the live constants of the scoring modules (the test
  `test_ratio_profiles.py` checks this), so the modules can be wired to this registry without
  changing a single S&P 500 score.
- `global_robust`: applies the three international-comparability adjustments (no Book Yield,
  ROE→ROCE, no D/E) for the MSCI World universe.

IMPORTANT: the `sp500_validated` weights are the ones in the REAL, validated code
(EBITDA 0.40 / Earnings 0.25 / FCF 0.25 / Book 0.10). The validated code is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RatioProfile:
    """Per-sub-block weights and universe columns for a given profile."""

    name: str
    valuation_weights: dict[str, float]
    quality_weights: dict[str, float]
    quality_columns: dict[str, str]
    growth_weights: dict[str, float]
    solvency_weights: dict[str, float]
    solvency_columns: dict[str, str]

    def assert_weights_sum_to_one(self, tol: float = 1e-9) -> None:
        for block, weights in (
            ("valuation", self.valuation_weights),
            ("quality", self.quality_weights),
            ("growth", self.growth_weights),
            ("solvency", self.solvency_weights),
        ):
            total = sum(weights.values())
            if abs(total - 1.0) > tol:
                raise ValueError(
                    f"Profile '{self.name}': '{block}' weights sum to {total}, not 1.0"
                )


# ---------------------------------------------------------------------------
# sp500_validated — mirrors EXACTLY the live constants of the modules.
# Any change here must stay in sync with valuation/quality/growth/solvency.py
# (verified by tests/test_ratio_profiles.py).
# ---------------------------------------------------------------------------

SP500_VALIDATED = RatioProfile(
    name="sp500_validated",
    valuation_weights={
        "ebitda_yield": 0.40,
        "earnings_yield": 0.25,
        "fcf_yield": 0.25,
        "book_yield": 0.10,
    },
    quality_weights={
        "gp_a": 0.30,
        "roa": 0.25,
        "roe": 0.20,
        "operating_margin": 0.15,
        "fcf_ni": 0.10,
    },
    quality_columns={
        "gp_a": "gp_a",
        "roa": "returnOnAssets",
        "roe": "returnOnEquity",
        "operating_margin": "operatingMargins",
        "fcf_ni": "fcf_ni",
    },
    growth_weights={
        "revenue_trend_ols": 0.30,
        "fcf_trend_ols": 0.25,
        "delta_gross_margin": 0.20,
        "asset_growth_inv": 0.15,
        "share_dilution_inv": 0.10,
    },
    solvency_weights={
        "dn_ebitda": 0.30,
        "interest_coverage": 0.25,
        "current_ratio": 0.20,
        "debt_to_equity": 0.15,
        "cfo_debt": 0.10,
    },
    solvency_columns={
        "dn_ebitda": "dn_ebitda",
        "interest_coverage": "interest_coverage",
        "current_ratio": "currentRatio",
        "debt_to_equity": "debtToEquity",
        "cfo_debt": "cfo_debt",
    },
)


# ---------------------------------------------------------------------------
# global_robust — international comparability.
#   Valuation: drops Book Yield (IAS 16) → redistributes.
#   Quality:   ROE → ROCE (= EBIT / Capital Employed).
#   Solvency:  drops D/E (IAS 16 + IFRS pensions) → weight to DN/EBITDA.
#   Growth:    unchanged.
# ---------------------------------------------------------------------------

GLOBAL_ROBUST = RatioProfile(
    name="global_robust",
    valuation_weights={
        "ebitda_yield": 0.40,
        "earnings_yield": 0.35,
        "fcf_yield": 0.25,
    },
    quality_weights={
        "gp_a": 0.25,
        "roce": 0.25,
        "roa": 0.20,
        "operating_margin": 0.20,
        "fcf_ni": 0.10,
    },
    quality_columns={
        "gp_a": "gp_a",
        "roce": "roce",
        "roa": "returnOnAssets",
        "operating_margin": "operatingMargins",
        "fcf_ni": "fcf_ni",
    },
    growth_weights=dict(SP500_VALIDATED.growth_weights),
    solvency_weights={
        "dn_ebitda": 0.45,
        "interest_coverage": 0.25,
        "current_ratio": 0.20,
        "cfo_debt": 0.10,
    },
    solvency_columns={
        "dn_ebitda": "dn_ebitda",
        "interest_coverage": "interest_coverage",
        "current_ratio": "currentRatio",
        "cfo_debt": "cfo_debt",
    },
)


PROFILES: dict[str, RatioProfile] = {
    SP500_VALIDATED.name: SP500_VALIDATED,
    GLOBAL_ROBUST.name: GLOBAL_ROBUST,
}

DEFAULT_PROFILE = "sp500_validated"


def get_profile(name: str) -> RatioProfile:
    """Return the ratio profile by name. Raises KeyError if it does not exist."""
    if name not in PROFILES:
        raise KeyError(f"Unknown ratio_profile '{name}'. Available: {sorted(PROFILES)}")
    return PROFILES[name]
