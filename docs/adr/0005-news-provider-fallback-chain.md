# ADR-0005 — News provider fallback chain for the sentiment module

- **Status**: Accepted
- **Date**: June 2026
- **Source**: Internal architecture design

## Context

The sentiment module needs recent news per ticker. Two viable providers exist,
each with a complementary weakness:

| Provider | Strengths | Weaknesses |
|----------|-----------|------------|
| Finnhub `company_news` | Per-ticker endpoint, no publication delay, 60 req/min free | Only covers North American companies |
| NewsAPI.org `/v2/everything` | Universal keyword search across general media | Articles delayed 24 h, 100 req/day free |

Neither provider alone serves the product goal (analysis of any stock with the
freshest possible news). The coverage gap is **data-dependent**: whether Finnhub
can answer depends on the ticker of each request, not on deployment configuration.

A second, related problem: the ticker is only validated syntactically
(`^[A-Z0-9]{2,5}$`). A nonexistent ticker that reaches NewsAPI's keyword search
can match random articles containing that string, and the LLM then produces a
plausible-looking sentiment analysis of a company that does not exist.

## Decision

1. **Adapter** — introduce a `NewsProvider` abstract interface
   (`app/services/sentiment/news/base.py`) mirroring the `LLMService` design
   ([ADR-0001](0001-openai-as-llm-and-embeddings-provider.md)), with
   `FinnhubNewsProvider` and `NewsAPIProvider` implementations and a shared
   provider-agnostic `NewsArticle` model.

2. **Chain of Responsibility** — a `NewsProviderChain` tries providers in order
   and returns the first non-empty result. An empty result (no coverage) or a
   provider failure falls through to the next link. If no link yields articles,
   a quota error encountered along the way is propagated (HTTP 503); otherwise
   the empty result surfaces as "no recent news" (HTTP 404). The
   `NEWS_PROVIDER` environment variable (default `finnhub`) selects the head of
   the chain; the other provider always backs it.

   [ADR-0004](0004-design-patterns-for-analysis-orchestration.md) explicitly
   **rejected** Chain of Responsibility for LLM fallback. That rejection stands:
   LLM provider choice is a configuration-level decision, resolved determinis-
   tically by an environment variable, and a silent runtime switch would change
   output quality unpredictably. News coverage is different in kind — it varies
   per request with the ticker, cannot be resolved by configuration, and the
   articles themselves are commodity inputs whose source does not change the
   analysis contract. Runtime fallback is therefore justified here and not there.

3. **Ticker existence validation** — before fetching news, the ticker is checked
   against Finnhub `symbol_lookup` (exact symbol match over its fuzzy results),
   cached in memory per ticker. Unknown tickers fail fast with HTTP 404. The
   check is **fail-open**: if the lookup itself fails, analysis proceeds —
   validation protects quality, not availability.

## Consequences

- Any stock with news coverage in either provider can be analysed; North
  American tickers get fresh (non-delayed) news, the rest degrade gracefully to
  NewsAPI's 24 h-delayed results instead of failing.
- NewsAPI's 100 req/day quota is consumed only for tickers Finnhub cannot serve
  (plus the sentiment TTL cache in front), making exhaustion far less likely.
- A shared `get_finnhub_client()` singleton is introduced in `app/core/finnhub.py`;
  the deep-learning (EOD prices for retraining), technical and fundamental
  modules are expected to reuse it later.
- Hallucinated analyses for nonexistent tickers are structurally prevented at
  the cost of one extra (cached) Finnhub call per previously unseen ticker.
- Adding a third news source is a new `NewsProvider` implementation plus one
  line in the factory; module logic does not change.
