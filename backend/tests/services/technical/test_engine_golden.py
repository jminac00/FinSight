"""Golden (characterization) test for the vendored technical engine.

It pins the engine's current numeric output against a fixed, network-free synthetic universe so the
subsequent SOLID/DRY/KISS refactor can be proven behavior-preserving: the committed baseline must
keep matching after the refactor.

Regenerate the baseline (only when the engine output is intentionally changed) with:

    REGEN_GOLDEN=1 uv run pytest tests/services/technical/test_engine_golden.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from app.services.technical.engine import technical_score
from app.services.technical.engine.blocks import (
    block1_momentum,
    block2_trend,
    block4_risk_stability,
    block5_confirmation,
)
from app.services.technical.engine.technical_score import compute_technical_score

_BASELINE_PATH = Path(__file__).parent / "golden" / "technical_score_baseline.json"
_BASKET = ["T05", "T15", "T25"]

# Cross-platform float tolerance: a behavior-preserving refactor reproduces these numbers, while a
# real regression shifts them far above this threshold.
_REL_TOL = 1e-6
_ABS_TOL = 1e-9


def _reset_engine_caches() -> None:
    """Clear the engine's module-level universe caches so each run is independent."""
    technical_score._universe_cache = None
    technical_score._technical_universe_cache = {}
    block1_momentum._UNIVERSE_RAW_CACHE.clear()
    block2_trend._UNIVERSE_COMPONENTS_CACHE.clear()
    block2_trend._UNIVERSE_RAW_CACHE.clear()
    block4_risk_stability._UNIVERSE_COMPONENTS_CACHE.clear()
    block4_risk_stability._UNIVERSE_RAW_CACHE.clear()
    block5_confirmation._UNIVERSE_RAW_CACHE.clear()


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a result into {dotted_path: leaf}, skipping free-text ``summary`` fields."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "summary":
                continue
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            flat.update(_flatten(value, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


def _compute_basket(synthetic_universe: dict, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    """Run the engine for the basket against the injected synthetic universe and flatten results."""
    _reset_engine_caches()
    # Hermetic: synthetic tickers are absent from the fundamental CSV anyway, so an empty sector map
    # reproduces the real behavior (RS sector percentile defaults to 0.5).
    monkeypatch.setattr(block1_momentum, "load_sector_map_from_fundamental", lambda: {})

    closes = synthetic_universe["closes"]
    ohlcv = synthetic_universe["ohlcv"]
    return {
        ticker: _flatten(
            compute_technical_score(
                ticker, universe="sp500", universe_closes=closes, universe_ohlcv=ohlcv
            )
        )
        for ticker in _BASKET
    }


def _assert_close(ticker: str, path: str, expected: Any, actual: Any) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert expected == actual, f"{ticker}.{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        assert math.isclose(actual, expected, rel_tol=_REL_TOL, abs_tol=_ABS_TOL), (
            f"{ticker}.{path}: {actual!r} != {expected!r}"
        )
    else:
        assert expected == actual, f"{ticker}.{path}: {actual!r} != {expected!r}"


def test_engine_matches_golden_baseline(
    synthetic_universe: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    computed = _compute_basket(synthetic_universe, monkeypatch)

    if os.environ.get("REGEN_GOLDEN") or not _BASELINE_PATH.exists():
        _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE_PATH.write_text(json.dumps(computed, indent=2, sort_keys=True), encoding="utf-8")
        pytest.skip(f"Golden baseline (re)generated at {_BASELINE_PATH}")

    baseline = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

    assert set(computed) == set(baseline), "Basket tickers diverged from the baseline"
    for ticker in _BASKET:
        expected = baseline[ticker]
        actual = computed[ticker]
        assert set(actual) == set(expected), (
            f"{ticker}: output keys diverged from the baseline "
            f"(added={set(actual) - set(expected)}, removed={set(expected) - set(actual)})"
        )
        for path, exp_value in expected.items():
            _assert_close(ticker, path, exp_value, actual[path])
