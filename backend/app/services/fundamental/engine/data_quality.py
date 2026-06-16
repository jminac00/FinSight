"""Centralized fundamental data-quality rules.

This layer decides whether the valuation ratios are comparable and which ones may enter the
score. It does not convert currencies: if price/market cap and the financial statements are not
in a comparable currency, valuation is neutralized conservatively until a traceable fundamental
FX conversion exists.

Product-facing warning strings are kept in Spanish on purpose (they surface to the user).
"""

from __future__ import annotations

from typing import Any

import numpy as np

DATA_QUALITY_OK = "OK"
DATA_QUALITY_REVIEW = "Revisar"
DATA_QUALITY_UNRELIABLE = "No fiable"

VALUATION_DISPLAY_NAMES: dict[str, str] = {
    "ebitda_yield": "EBITDA Yield",
    "earnings_yield": "Earnings Yield",
    "fcf_yield": "FCF Yield",
    "book_yield": "Book Yield",
}

VALUATION_PLAUSIBILITY_LIMITS: dict[str, float] = {
    "ebitda_yield": 0.50,
    "earnings_yield": 0.50,
    "fcf_yield": 0.40,
    "book_yield": 3.00,
}


def safe_float(value: Any) -> float | None:
    """Convert to a finite float or return None."""
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def currencies_comparable(data: dict[str, Any]) -> bool:
    """True if the market currency and the financial currency are comparable."""
    price_ccy = (data.get("price_currency") or data.get("ticker_currency") or "").upper()
    market_cap_ccy = (data.get("market_cap_currency") or price_ccy or "").upper()
    financial_ccy = (data.get("financial_currency") or "").upper()
    if not price_ccy or not market_cap_ccy or not financial_ccy:
        return False
    return price_ccy == market_cap_ccy == financial_ccy


def currency_warning(data: dict[str, Any]) -> str:
    market_ccy = data.get("market_cap_currency") or data.get("price_currency") or "N/A"
    financial_ccy = data.get("financial_currency") or "N/A"
    return (
        "Moneda no comparable: market cap/precio en "
        f"{market_ccy} y estados financieros en {financial_ccy}."
    )


def is_component_valid(name: str, value: Any) -> bool:
    """Validate a valuation component by availability, sign and plausibility."""
    value_f = safe_float(value)
    if value_f is None or value_f < 0.0:
        return False
    limit = VALUATION_PLAUSIBILITY_LIMITS.get(name)
    return limit is None or value_f <= limit


def evaluate_valuation_data_quality(
    company_data: dict[str, Any],
    component_values: dict[str, Any],
    weights: dict[str, float],
    min_valid_components: int = 2,
) -> dict[str, Any]:
    """Evaluate valuation data quality and decide which components are valid.

    Returns:
        {
            "data_quality": "OK" | "Revisar" | "No fiable",
            "warnings": [...],
            "valid_components": [...],
            "invalid_components": [...],
            "effective_values": {...},
            "neutralized_blocks": [...]
        }
    """
    warnings: list[str] = []
    valid_components: list[str] = []
    invalid_components: list[str] = []
    effective_values: dict[str, float | None] = {}
    neutralized_blocks: list[str] = []

    comparable = bool(
        company_data.get("valuation_currency_mismatch") is False
    ) and currencies_comparable(company_data)
    if not comparable:
        warnings.append(currency_warning(company_data))
    if company_data.get("is_adr"):
        warnings.append(
            "Ticker ADR o cotizacion extranjera en mercado US: "
            "ratios fundamentales requieren control de moneda."
        )

    for name in weights:
        value = component_values.get(name)
        if not comparable:
            invalid_components.append(name)
            effective_values[name] = None
            continue

        if is_component_valid(name, value):
            valid_components.append(name)
            effective_values[name] = safe_float(value)
            continue

        invalid_components.append(name)
        effective_values[name] = None
        raw_value = safe_float(component_values.get(name))
        if raw_value is None:
            warnings.append(f"{VALUATION_DISPLAY_NAMES.get(name, name)} no disponible o invalido.")
        else:
            warnings.append(
                f"{VALUATION_DISPLAY_NAMES.get(name, name)} sospechoso o invalido "
                f"(raw={raw_value * 100:.2f}%)."
            )

    ev = company_data.get("enterprise_value")
    ebitda = safe_float(company_data.get("ebitda_ttm"))
    ev_negative = ev is None and ebitda is not None and ebitda > 0.0
    if ev_negative and "ebitda_yield" in weights:
        if "ebitda_yield" in valid_components:
            valid_components.remove("ebitda_yield")
        if "ebitda_yield" not in invalid_components:
            invalid_components.append("ebitda_yield")
        effective_values["ebitda_yield"] = None
        warnings.append("Enterprise Value no positivo: EBITDA Yield no fiable.")

    if len(valid_components) < min_valid_components:
        neutralized_blocks.append("valoracion")
        warnings.append(
            "Menos de dos componentes de valoracion validos; sub-bloque neutralizado en 5.00."
        )
        quality = DATA_QUALITY_UNRELIABLE
    elif warnings:
        quality = DATA_QUALITY_REVIEW
    else:
        quality = DATA_QUALITY_OK

    return {
        "data_quality": quality,
        "warnings": list(dict.fromkeys(warnings)),
        "valid_components": valid_components,
        "invalid_components": invalid_components,
        "effective_values": effective_values,
        "neutralized_blocks": neutralized_blocks,
    }
