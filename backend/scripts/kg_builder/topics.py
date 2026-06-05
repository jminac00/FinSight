"""Controlled vocabulary for news topic and event classification."""

TOPICS: list[str] = [
    "earnings_results",
    "mergers_acquisitions",
    "regulatory_legal",
    "analyst_ratings",
    "dividends_buybacks",
    "product_launch",
    "executive_changes",
    "macroeconomic",
    "commodities_energy",
    "geopolitical",
    "debt_financing",
    "market_outlook",
    "sector_rotation",
    "ipo_listing",
    "technology_innovation",
]

EVENT_TYPES: list[str] = [
    "earnings_report",
    "merger_acquisition",
    "regulatory_action",
    "analyst_rating",
    "dividend",
    "product_launch",
    "executive_change",
    "economic_data",
    "geopolitical",
    "other",
]
