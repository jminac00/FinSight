"""Fundamental scoring engine vendored from the Finance collaborator's project.

Computes a 0–10 fundamental score for a single ticker, normalized against a reference universe
(S&P 500 or MSCI World) and its GICS sector. Comments, docstrings and log messages are in English
per the project conventions; product-facing strings (qualitative signals and the natural-language
summaries consumed by the LLM) are kept in Spanish on purpose.

Main exports:
  - analyze_ticker(ticker, universe_df=None, universe="sp500")  → full pipeline
  - load_universe(force_refresh=False)                          → S&P 500 universe
  - get_company_data(ticker)                                    → one company's data
  - compute_valuation_score / compute_quality_score / compute_growth_score / compute_solvency_score
  - combine_fundamental_scores(val, qual, grow, solv)
"""

from app.services.fundamental.engine.data_fetcher import get_company_data
from app.services.fundamental.engine.fundamental_score import (
    analyze_ticker,
    combine_fundamental_scores,
)
from app.services.fundamental.engine.growth import compute_growth_score
from app.services.fundamental.engine.quality import compute_quality_score
from app.services.fundamental.engine.solvency import compute_solvency_score
from app.services.fundamental.engine.universe import (
    build_universe,
    get_sp500_tickers,
    load_universe,
)
from app.services.fundamental.engine.valuation import compute_valuation_score

__all__ = [
    "analyze_ticker",
    "build_universe",
    "combine_fundamental_scores",
    "compute_growth_score",
    "compute_quality_score",
    "compute_solvency_score",
    "compute_valuation_score",
    "get_company_data",
    "get_sp500_tickers",
    "load_universe",
]
