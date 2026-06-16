"""Central configuration for the technical analysis engine.

All windows, weights, thresholds and the sigmoid steepness live here so the blocks and the
orchestrator share a single source of truth instead of redefining magic numbers locally.
"""

from __future__ import annotations

# --- Momentum block (block 1) ------------------------------------------------
# Skip the most recent month to avoid the short-term reversal effect.
MOMENTUM_SKIP_SESSIONS = 21
MOMENTUM_LOOKBACK_12M = 252
MOMENTUM_LOOKBACK_6M = 126
# Minimum history required, with a small margin over the 12-month formation window plus skip.
MOMENTUM_MIN_SESSIONS = MOMENTUM_LOOKBACK_12M + MOMENTUM_SKIP_SESSIONS + 7  # 280

# --- Trend block (block 2) ---------------------------------------------------
TREND_MA_SESSIONS = 200
TREND_REGRESSION_WINDOW = 126
# Trend uses a gentler sigmoid than the other blocks.
TREND_SIGMOID_K = 0.75

# --- Risk / stability block (block 4) ----------------------------------------
RISK_WINDOW = 126
RISK_MIN_NEG_RETURNS = 10

# --- Confirmation block (block 5) --------------------------------------------
DONCHIAN_WINDOW = 20

# --- Normalization (shared) --------------------------------------------------
ROBUST_SIGMOID_K = 0.9
MIN_OBSERVATIONS = 10

# --- Aggregation -------------------------------------------------------------
# Nominal weights; if a block is missing its weight is redistributed proportionally.
BLOCK_WEIGHTS: dict[str, float] = {
    "momentum": 0.35,  # includes the sector relative-strength sub-component
    "trend": 0.30,
    "risk_stability": 0.20,
    "confirmation": 0.15,
}

# Per-block key holding the final 0-10 score in each block's result dict.
SCORE_KEYS: dict[str, str] = {
    "momentum": "normalized_score_0_10",
    "trend": "trend_score_0_10",
    "risk_stability": "risk_stability_score_0_10",
    "confirmation": "confirmation_score_0_10",
}

# Score thresholds mapping a 0-10 score to (bearish, neutral, bullish).
SIGNAL_THRESHOLDS: tuple[float, float] = (4.0, 6.5)

# A score is reliable when at least this fraction of the nominal weight has valid data.
RELIABILITY_THRESHOLD = 0.60
