# ADR-0004 — Design patterns for analysis orchestration

- **Status**: Accepted
- **Date**: June 2026
- **Source**: Internal architecture design

## Context

The consolidated report is the core use case of FinSight: a single request for a
ticker triggers four independent sub-analyses (sentiment, deep learning, fundamental
and technical) that run in parallel and are then synthesized by the LLM into one
answer in Spanish. The naïve implementation calls each module service directly from
the report layer, with four heterogeneous signatures and four different return
types, hardcoded in sequence.

That approach concentrates several forces in one place:

- The four modules are invoked, awaited, error-handled and logged individually, so
  any change to the orchestration (timeout policy, degradation, adding a fifth
  module) must be repeated four times.
- The system must degrade gracefully: NewsAPI is capped at 100 requests/day, a
  ticker may lack a trained `.pt` model (RF-27), and an external data API may be
  temporarily unavailable. The report must still be produced from the modules that
  did succeed.
- External price/fundamentals data has more than one viable provider (Finnhub,
  Yahoo Finance), and their availability and quotas vary, so the source is a point
  of expected change.

The project already applies the **Adapter** pattern successfully for LLM providers
(`LLMService` with OpenAI / Ollama, see [ADR-0001](0001-openai-as-llm-and-embeddings-provider.md))
and the **Factory + Singleton** pattern in `factory.py`. This ADR records the
patterns chosen for the analysis orchestration itself.

## Decision

Adopt three Gang-of-Four patterns, scoped narrowly to the report flow:

### 1. Strategy — common interface for the four analysis modules

Each module implements one abstract contract:

```python
class AnalysisModule(ABC):
    name: str
    @abstractmethod
    async def analyze(self, ticker: str) -> ModuleResult: ...
```

The orchestrator holds the modules as a list and runs them uniformly with
`asyncio.gather(*[m.analyze(ticker) for m in modules], return_exceptions=True)`.

- **Pros**: enables the parallel `asyncio.gather` execution prescribed for the
  report; graceful degradation comes for free (a failed module yields an exception
  captured in its `ModuleResult` instead of aborting the whole report); a new module
  is a new class with zero changes to the orchestrator; the module list is trivial
  to mock in tests.
- **Cons**: requires normalizing the four outputs into a shared `ModuleResult`
  envelope (`status`, `data`, `error`). This is acceptable because the four modules
  already share the same lifecycle shape (fetch data → process → optional LLM call →
  result).

### 2. Facade — `ReportService` as the single orchestration entry point

The `report` endpoint calls `ReportService.generate(ticker)` and knows nothing about
the four modules or the synthesis step. The facade choreographs: parallel module
execution → collection of results → LLM synthesis in Spanish → legal disclaimer.

- **Pros**: keeps the endpoint trivial and the orchestration logic in one place;
  decouples the API and frontend from the internal module structure.
- **Cons**: risk of the facade growing into a god object. Mitigated by the rule that
  the facade *orchestrates* but never *computes* — all analysis logic stays inside
  the modules.

### 3. Adapter — external data-source providers

Introduce an abstract provider for market data (`PriceDataProvider`, and analogous
fundamentals access) with concrete adapters for Finnhub and Yahoo Finance, mirroring
the existing `LLMService` design.

- **Pros**: switching provider or failing over between them on quota exhaustion does
  not touch module logic; consistent with a pattern the codebase already uses, so no
  new concept for the team.
- **Cons**: if a given datum ends up sourced from a single provider, the abstraction
  is speculative. Justified here because the project already plans to alternate
  between Finnhub and Yahoo Finance for price data, so the volatility is real rather
  than hypothetical.

## Consequences

- The report module is built around an `AnalysisModule` interface and a
  `ReportService` facade; the four module services implement the strategy contract.
- Graceful degradation is a structural property of the orchestrator, not a per-module
  `try/except` duplicated four times. Partial reports are produced when a module
  fails, satisfying the resilience constraints (NewsAPI quota, missing `.pt`, source
  outages).
- Market-data access goes through adapters, decoupling the modules from Finnhub /
  Yahoo Finance specifics.
- Two patterns are explicitly **not** adopted, to avoid speculative complexity:
  - **Template Method** for the shared module lifecycle is deferred; it will only be
    extracted if real duplication of the lifecycle appears across modules (reactive
    refactor, not upfront design).
  - **Chain of Responsibility** for LLM fallback is rejected: provider selection is
    done explicitly via the `LLM_PROVIDER` environment variable (see ADR-0001), not
    through runtime chaining.
- Caching and rate limiting are implemented with function-level decorators / FastAPI
  dependencies rather than the GoF Decorator pattern, since the language idiom
  already covers that need.
- This is an internal architecture decision and does not modify the SRS.
