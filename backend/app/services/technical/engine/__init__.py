"""
technical_analysis engine
Bloque de análisis técnico (universos S&P 500 y MSCI World en USD).

Export principal:
    compute_technical_score(ticker) → dict con technical_score (0-10),
    signal, pesos efectivos y resultado completo de cada sub-bloque.
"""

from app.services.technical.engine.technical_score import compute_technical_score

__all__ = ["compute_technical_score"]
