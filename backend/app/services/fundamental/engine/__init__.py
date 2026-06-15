"""Fundamental scoring engine vendored from the Finance collaborator's project.

Computes a 0–10 fundamental score for a single ticker, normalized against the S&P 500
universe (valuation, quality, growth, solvency). Comments, docstrings and log messages are
in English per the project conventions; product-facing strings (qualitative signals and the
natural-language summaries consumed by the LLM) are kept in Spanish on purpose.
"""
