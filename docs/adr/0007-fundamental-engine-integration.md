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

## Amendment (June 2026) — Universal support (MSCI World)

The collaborator delivered a follow-up project extending coverage from the S&P 500
to **any listed stock**. The delta is multi-universe; it re-vendors the whole engine
(the eight Phase-1 modules are rewritten) and adds six new ones. The decisions above
still hold; the following refine them.

1. **Two reference universes.** Besides the S&P 500, the engine builds an **MSCI World**
   universe (~1,300 companies) from the public iShares URTH ETF holdings
   (`engine/msci_world.py`). `UniverseManager.resolve_universe(ticker, mode)` selects the
   reference per request:
   - `auto`: S&P 500 if the ticker is in the index, otherwise MSCI World;
   - `domestic`: force S&P 500; `global`: force MSCI World.
   The endpoint exposes `?mode=auto|domestic|global`. With `mode` resolving to `sp500`
   the returned DataFrame is the same one Phase-1 produced, so **S&P 500 scores do not
   change** (guarded by a golden-master regression, below).

2. **Ratio profiles.** `engine/ratio_profiles.py` declares `sp500_validated` (mirrors the
   live module constants exactly) and `global_robust` for international comparability:
   drops Book Yield (IAS 16), uses ROCE instead of ROE, and drops D/E. Each sub-block now
   takes a `profile=`; the S&P 500 path is unchanged.

3. **Data-quality gating.** `engine/data_quality.py` checks currency comparability
   (price/market-cap vs. financial-statement currency) and plausibility limits, exposing
   `valuation_data_quality` (OK / Revisar / No fiable). When the currencies are not
   comparable the engine **neutralizes valuation** conservatively (no FX conversion is
   performed). `engine/ticker_routing.py` recommends an ADR↔local listing (e.g. NVO →
   NOVO-B.CO) when it improves comparability; this is informative only.

4. **Frozen seed data + weekly rebuild.** The reference universes are **committed** as
   frozen CSVs in `engine/data/` (`sp500_universe.csv`, `msci_world_universe.csv`,
   `msci_world_currency.json`, `URTH_holdings.csv`), so a fresh deploy works without a
   warm-up. The Phase-1 daily refresh and startup warm-up are removed; a weekly job
   (`weekly_fundamental_refresh`, Sunday ~04:00 CET) rebuilds both universes and is
   resilient (a failed rebuild keeps the committed seed). Only the runtime `*_stats.json`
   is git-ignored. Fundamental data is quarterly/annual, so weekly is sufficient; the
   daily refresh is reserved for the technical module (Phase 2).

5. **Caching and existence.** Results are cached per **(universe, ticker)** since scores
   differ by reference. The Finnhub `TickerValidator` gate is **dropped**: Finnhub covers
   North America only and would reject valid international symbols. Existence is now
   determined by the engine — no usable data → `score_final is None` →
   `FundamentalAnalysisError` → **404**. The ticker regex is relaxed to accept exchange
   suffixes and class separators (`^[A-Z0-9][A-Z0-9.\-]{0,14}$`, e.g. `ASML.AS`,
   `NOVO-B.CO`, `7203.T`, `BRK-B`), superseding the US-only 2–5 alphanumeric rule of
   CLAUDE.md §3.8 for this module.

6. **Golden-master regression.** `tests/services/fundamental/test_regression_fundamental.py`
   recomputes the 503 S&P 500 scores offline from the committed seed (no network, FMP
   disabled) and asserts equality with `baseline_fund.csv` to 1e-6. This proves the
   re-vendoring and the comment-only translation introduce no score drift, and guards
   against future code changes.

### Amended consequences

- Any listed stock can be analyzed, normalized against the appropriate universe; the
  Phase-1 "scored against the S&P 500 distribution" limitation no longer applies in
  `auto`/`global` mode.
- `scipy` is **not** required on the fundamental path (numpy/pandas/yfinance/requests/
  dotenv only); it is a technical-module dependency, out of scope here.
- Connecting the live service to the consolidated report (`_mock_fundamental` in
  `report.py`) and the technical module remain out of scope.
