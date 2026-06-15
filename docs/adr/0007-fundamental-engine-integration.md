# ADR-0007 — Fundamental engine integration via vendoring and a precomputed universe

- **Status**: Accepted
- **Date**: June 2026
- **Source**: Internal architecture design

## Context

The Finance collaborator delivered a standalone Python project that ranks the
S&P 500 by combining fundamental and technical analysis on a 0–10 scale. FinSight
needs the fundamental module (Module 3) as a per-ticker, on-demand service behind
`GET /api/v1/fundamental/{ticker}`.

The collaborator's scoring is **relative to the S&P 500 universe**: each metric is
winsorized (p1–p99), converted to a robust median/MAD z-score against the
cross-sectional distribution, and mapped to 0–10 with a sigmoid (the median
company scores 5.0 by construction). Two consequences follow:

1. A single ticker **cannot be scored in isolation** — the normalization needs the
   reference statistics of the whole universe.
2. The collaborator's code is **batch-oriented** (it scores all ~500 companies in
   one run) and produces deterministic text summaries; it never calls an LLM.

This conflicts with FinSight's on-demand model on Render's free tier (no GPU,
ephemeral filesystem, sleeps after inactivity), and with the project conventions
(code in English; product output in Spanish; LLM accessed only through
`LLMService`, see [ADR-0001](0001-openai-as-llm-and-embeddings-provider.md)).

## Decision

1. **Vendor and adapt, do not reimplement.** The scoring engine is copied into
   `app/services/fundamental/engine/` (data fetching, the four sub-blocks,
   sanitizer, scoring, universe builder). Comments, docstrings and log messages are
   translated to English; the validated formulas are preserved verbatim. The
   qualitative signals and narrative summaries stay in Spanish on purpose — they are
   product content and the input the LLM rewrites for the user.

2. **Precompute the universe + refresh daily.** The cross-sectional reference is an
   offline-built CSV snapshot (`build_fundamental_universe.py`). At request time the
   service loads the cached snapshot and normalizes only the requested ticker
   against it. A scheduler job (`daily_universe_refresh`, 22:30 CET, staggered
   behind the model update) rebuilds it; on a fresh production deploy a background
   warm-up build runs at startup and the endpoint returns **503** until the snapshot
   exists.

3. **LLM owns the narrative.** The engine's deterministic `fundamental_summary`
   becomes the user prompt for `LLMService.complete()`, which renders the analysis
   in Spanish for non-experts. The numeric `score_final` and a curated metrics dict
   populate `FundamentalResult`.

4. **Reuse existing infrastructure.** Ticker existence is checked with the shared
   fail-open `TickerValidator` (Finnhub); results are cached per ticker with
   `cachetools.TTLCache` (`CACHE_TTL_FUNDAMENTAL`); the engine runs in a worker
   thread (`asyncio.to_thread`) so it never blocks the event loop.

Data sources stay on yfinance (Yahoo Finance), already a dependency and permitted
for this module; the offline builder adds only `requests` + `lxml` (S&P 500 list
from Wikipedia).

## Consequences

- The relative 0–10 methodology is preserved exactly, including the value–momentum
  and sector-specific treatments (Financial Services, REITs, Utilities).
- Scores are normalized against the S&P 500. Tickers outside the index are still
  scored against that reference distribution; this is a documented limitation.
- The snapshot is at most one day old, consistent with the quarterly/annual nature
  of fundamental data and the existing daily cadence.
- The **technical** module (Module 4) is out of scope here; it carries a heavier
  universe dependency (live OHLCV for ~500 tickers per session) and will be planned
  separately, reusing this precompute-and-refresh pattern.
