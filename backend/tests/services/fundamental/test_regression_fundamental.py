"""Deterministic offline regression of the fundamental block.

Recomputes the 4 sub-scores + score_final for every company in the cached S&P 500 universe
(engine/data/sp500_universe.csv) without touching the network, and compares them against a
versioned baseline. This guarantees that vendoring/translating the engine does NOT change a
single validated S&P 500 score. FMP is disabled so the result is reproducible.
"""

from pathlib import Path

import pandas as pd

import app.services.fundamental.engine.fmp_fetcher as fmp_fetcher

# Disable FMP → zero network calls, deterministic result.
fmp_fetcher.FMP_API_KEY = None

from app.services.fundamental.engine.cache_analysis import (  # noqa: E402
    analyze_fundamental_from_cache,
)
from app.services.fundamental.engine.universe import load_universe  # noqa: E402

_BASELINE = Path(__file__).parent / "baseline_fund.csv"
_SCORE_COLS = [
    "score_valoracion",
    "score_calidad",
    "score_crecimiento",
    "score_solvencia",
    "score_final",
]
_TOL = 1e-6


def _compute_scores() -> pd.DataFrame:
    universe_df = load_universe()
    universe_by_ticker = {
        str(row["ticker"]).upper().replace(".", "-"): row for _, row in universe_df.iterrows()
    }
    rows: list[dict] = []
    for ticker in universe_df["ticker"].tolist():
        key = str(ticker).upper().replace(".", "-")
        r = analyze_fundamental_from_cache(key, universe_df, universe_by_ticker)
        rows.append(
            {
                "ticker": key,
                "score_valoracion": r.get("score_valoracion"),
                "score_calidad": r.get("score_calidad"),
                "score_crecimiento": r.get("score_crecimiento"),
                "score_solvencia": r.get("score_solvencia"),
                "score_final": r.get("score_final"),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def _rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _SCORE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def test_fundamental_scores_match_baseline():
    baseline = _rounded(pd.read_csv(_BASELINE).set_index("ticker"))
    current = _rounded(_compute_scores())

    assert set(baseline.index) == set(current.index), "Universe ticker set changed vs baseline"

    diffs: list[str] = []
    for tkr in baseline.index:
        for col in _SCORE_COLS:
            b = baseline.loc[tkr, col]
            c = current.loc[tkr, col]
            if pd.isna(b) and pd.isna(c):
                continue
            if pd.isna(b) != pd.isna(c) or abs(float(b) - float(c)) > _TOL:
                diffs.append(f"{tkr}/{col}: baseline={b} -> current={c}")

    assert not diffs, f"{len(diffs)} fundamental scores differ from baseline: {diffs[:20]}"
