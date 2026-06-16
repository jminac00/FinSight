"""Tests for the valuation data-quality rules (currency comparability + plausibility limits)."""

from app.services.fundamental.engine.data_quality import (
    DATA_QUALITY_OK,
    DATA_QUALITY_UNRELIABLE,
    currencies_comparable,
    evaluate_valuation_data_quality,
    is_component_valid,
)

_VAL_WEIGHTS = {
    "ebitda_yield": 0.40,
    "earnings_yield": 0.25,
    "fcf_yield": 0.25,
    "book_yield": 0.10,
}


def test_currencies_comparable_matching_and_mismatch():
    comparable = {
        "price_currency": "USD",
        "market_cap_currency": "USD",
        "financial_currency": "USD",
    }
    mismatch = {
        "price_currency": "USD",
        "market_cap_currency": "USD",
        "financial_currency": "EUR",
    }
    assert currencies_comparable(comparable) is True
    assert currencies_comparable(mismatch) is False
    assert currencies_comparable({}) is False  # missing fields → not comparable


def test_is_component_valid_respects_plausibility_limits():
    assert is_component_valid("ebitda_yield", 0.10) is True
    assert is_component_valid("ebitda_yield", -0.01) is False  # negative
    assert is_component_valid("ebitda_yield", 0.60) is False  # > 0.50 limit
    assert is_component_valid("fcf_yield", 0.50) is False  # > 0.40 limit
    assert is_component_valid("book_yield", 2.50) is True  # within 3.00 limit


def test_currency_mismatch_neutralizes_valuation():
    company = {
        "price_currency": "USD",
        "market_cap_currency": "USD",
        "financial_currency": "EUR",
    }
    values = {"ebitda_yield": 0.10, "earnings_yield": 0.06, "fcf_yield": 0.05, "book_yield": 0.2}
    result = evaluate_valuation_data_quality(company, values, _VAL_WEIGHTS)
    assert result["data_quality"] == DATA_QUALITY_UNRELIABLE
    assert "valoracion" in result["neutralized_blocks"]
    assert result["valid_components"] == []


def test_comparable_currency_with_valid_components_is_ok():
    company = {
        "price_currency": "USD",
        "market_cap_currency": "USD",
        "financial_currency": "USD",
        "valuation_currency_mismatch": False,
    }
    values = {"ebitda_yield": 0.10, "earnings_yield": 0.06, "fcf_yield": 0.05, "book_yield": 0.2}
    result = evaluate_valuation_data_quality(company, values, _VAL_WEIGHTS)
    assert result["data_quality"] == DATA_QUALITY_OK
    assert set(result["valid_components"]) == set(_VAL_WEIGHTS)
    assert result["neutralized_blocks"] == []


def test_negative_enterprise_value_invalidates_ebitda_yield():
    company = {
        "price_currency": "USD",
        "market_cap_currency": "USD",
        "financial_currency": "USD",
        "valuation_currency_mismatch": False,
        "enterprise_value": None,
        "ebitda_ttm": 1e9,
    }
    values = {"ebitda_yield": 0.10, "earnings_yield": 0.06, "fcf_yield": 0.05, "book_yield": 0.2}
    result = evaluate_valuation_data_quality(company, values, _VAL_WEIGHTS)
    assert "ebitda_yield" in result["invalid_components"]
    assert "ebitda_yield" not in result["valid_components"]
