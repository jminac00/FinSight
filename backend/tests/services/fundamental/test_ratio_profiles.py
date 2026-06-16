"""Tests for the ratio-profile registry.

sp500_validated must mirror EXACTLY the live constants of the scoring modules, so wiring the
modules to the registry does not change a single S&P 500 score. Both profiles must have weights
that sum to 1 per sub-block.
"""

import pytest

from app.services.fundamental.engine import growth, quality, solvency, valuation
from app.services.fundamental.engine.ratio_profiles import (
    GLOBAL_ROBUST,
    SP500_VALIDATED,
    get_profile,
)


def test_all_profiles_weights_sum_to_one():
    SP500_VALIDATED.assert_weights_sum_to_one()
    GLOBAL_ROBUST.assert_weights_sum_to_one()


def test_sp500_profile_matches_live_module_constants():
    assert SP500_VALIDATED.valuation_weights == valuation._WEIGHTS
    assert SP500_VALIDATED.quality_weights == quality._WEIGHTS
    assert SP500_VALIDATED.quality_columns == quality._UNIVERSE_COL
    assert SP500_VALIDATED.growth_weights == growth._WEIGHTS
    assert SP500_VALIDATED.solvency_weights == solvency._WEIGHTS
    assert SP500_VALIDATED.solvency_columns == solvency._UNIVERSE_COL


def test_global_robust_drops_book_yield_and_de_uses_roce():
    assert "book_yield" not in GLOBAL_ROBUST.valuation_weights
    assert "roce" in GLOBAL_ROBUST.quality_weights
    assert "roe" not in GLOBAL_ROBUST.quality_weights
    assert "debt_to_equity" not in GLOBAL_ROBUST.solvency_weights


def test_get_profile_known_and_unknown():
    assert get_profile("sp500_validated") is SP500_VALIDATED
    assert get_profile("global_robust") is GLOBAL_ROBUST
    with pytest.raises(KeyError):
        get_profile("does_not_exist")
