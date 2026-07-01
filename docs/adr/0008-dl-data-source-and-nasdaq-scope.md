# ADR-0008 — yfinance as the DL price data source and Nasdaq-only coverage scope

- **Status**: Accepted
- **Date**: July 2026
- **Source**: Design decision (issue #54); implementation consolidated in PR #51

## Context

The deep learning module (Module 2) requires two data feeds:

- **Training**: a long historical OHLC series (years of daily data) to fit a GRU
  per ticker.
- **Inference**: the most recent `lookback = 24` daily candles to build the input
  window at prediction time.

During the research phase the design referenced Finnhub as the DL price source,
and an offline CSV dataset (`data/nasdaq_prices/`) was considered as the training
corpus. Two problems arose:

1. **Finnhub free tier has no historical OHLC endpoint.** The free plan provides
   real-time quotes and company news but does not expose candlestick history;
   obtaining it would require a paid subscription.

2. **The CSV corpus was a static snapshot**, tied to a fixed date and a curated
   Nasdaq universe. Maintaining and updating it would have required a separate
   pipeline, and the snapshot would become stale between training runs.

The technical analysis module (Module 4), implemented in parallel, had already
adopted **yfinance** via the shared `app.core.market_data.get_price_history()`
abstraction for the same reason: free, no API key required, covers all globally
listed equities, and returns split- and dividend-adjusted OHLC data. Reusing that
module for the DL data feed was the obvious path to avoid duplicating price-fetch
logic across modules.

A separate question is **coverage scope**: although yfinance covers any exchange,
the Hyperparameter Optimization study (Optuna, `study_name = "gru_h10"`) was run
exclusively on AAPL, NVDA, and PEP — three US large-cap stocks listed on the
Nasdaq. The frozen recipe (`gru_frozen_config.json`) is therefore validated on
that domain only.

## Decision

### 1. yfinance via `app.core.market_data` as the sole DL price source

`app.core.market_data.get_price_history(ticker, period="max")` (yfinance under
the hood) replaces both the Finnhub OHLC endpoint and the static CSV corpus for
all DL data needs:

- **Training** (`scripts/train_models.py`): `period="max"` to obtain the full
  adjusted history from IPO.
- **Inference** (`DLService.predict`): the most recent candles, sufficient for
  a `lookback = 24` window.

The static CSV files under `data/` remain in the repository (gitignored) as
research artefacts from the exploratory phase but are no longer part of the
production pipeline.

### 2. Nasdaq-only coverage scope (HPO boundary)

The system restricts the set of tickers for which pre-trained models are offered
to **Nasdaq-listed US equities**. This is not a technical constraint imposed by
yfinance or by the GRU architecture — both are market-agnostic. It is a
deliberate **scope boundary matching the validated domain of the HPO**:

- The frozen recipe was derived from AAPL, NVDA, and PEP. Applying it to markets
  with materially different dynamics (e.g. European mid-caps, emerging-market
  equities, illiquid OTC stocks) would introduce transfer-learning assumptions
  that have not been validated.
- Staying within the Nasdaq universe keeps the product's quality claims honest
  and avoids overpromising on out-of-sample generalization.

When a user requests a ticker for which no model exists, the API returns 404
("model not available") and the report explains that coverage is limited to
Nasdaq-listed stocks. Expanding coverage to additional markets is deferred to
future work and would require a new HPO study on a broader universe.

## Consequences

- `app.core.market_data` is the single price-fetch interface for both the
  technical and the DL modules; no Finnhub OHLC dependency is introduced.
- The CSV corpus (`data/`) is retained for research reproducibility but is not
  referenced by any production code path.
- The Nasdaq restriction is surfaced to users via the API (404 with explanation)
  and documented in the SRS (RF-27) and this ADR.
- Future HPO studies on a broader universe would produce a new frozen config and
  could supersede this scope restriction without changing the architecture.
- See [ADR-0006](0006-gru-architecture-no-vmd.md) for the GRU architecture and
  recipe decisions.
