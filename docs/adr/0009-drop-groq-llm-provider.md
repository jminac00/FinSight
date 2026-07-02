# ADR-0009 — Drop Groq as LLM provider; Ollama and OpenAI only

- **Status**: Accepted
- **Date**: July 2026
- **Source**: Design decision (issue #58)

## Context

[ADR-0001](0001-openai-as-llm-and-embeddings-provider.md) established OpenAI as the
active LLM and embeddings provider in production, with Groq (free tier, Llama 3.1 70B)
kept registered as a free fallback provider and Ollama (local, Llama 3.1 8B) as the
development provider. All three were reachable through the centralized `LLMService`
adapter (`app/llm/`), selected via the `LLM_PROVIDER` environment variable.

In practice, the Groq fallback path was never exercised: OpenAI has remained the
active provider since ADR-0001, and Ollama already covers the offline/local-development
case. Keeping a third provider registered adds a dependency (`groq` SDK), a config
surface (`GROQ_API_KEY`, `GROQ_MODEL`) and a code path (`GroqProvider`, factory
dispatch) with no active use, which goes against keeping the provider set minimal.

Separately, the sentiment module's knowledge graph (Neo4j vector index) was built
using OpenAI's `text-embedding-3-large` embeddings ([ADR-0001](0001-openai-as-llm-and-embeddings-provider.md)).
Embeddings are generated through a dedicated module (`app/llm/embeddings.py`) that is
**not** routed through the `LLM_PROVIDER` switch — it always calls OpenAI, regardless
of which provider is selected for chat completion. This means an OpenAI account with
available credit is a hard production dependency even when `LLM_PROVIDER=ollama` is
selected for chat completions; there is no local/free substitute for the embedder
without rebuilding the graph. This constraint was not previously reflected in
`validate_production_keys()` (`app/core/config.py`), which only required
`OPENAI_API_KEY` when `LLM_PROVIDER=openai`.

## Decision

Drop Groq entirely as an LLM provider option:

- Remove `GroqProvider`, its dispatch branch in `factory.py`, and the `groq_api_key` /
  `groq_model` settings and `GROQ_API_KEY` / `GROQ_MODEL` environment variables.
- Remove the `groq` package dependency.
- `LLM_PROVIDER` now accepts only `ollama` (local, multiple models selectable via
  `OLLAMA_MODEL`) or `openai` (paid, active in production).

Make the OpenAI embedder dependency explicit and enforced:

- `validate_production_keys()` requires `OPENAI_API_KEY` in production **unconditionally**,
  not only when `LLM_PROVIDER=openai`, since the embedder always needs it regardless of
  the chat completion provider.

The `LLMService` Adapter pattern is unchanged: swapping between Ollama and OpenAI for
chat completion still requires no changes to any module's business logic.

## Consequences

- The free-tier Groq fallback described in ADR-0001 no longer exists; OpenAI and
  Ollama are the only two chat completion providers. ADR-0001's status is updated to
  point to this ADR for that part of its decision.
- An OpenAI account with available credit is now an explicit, always-required
  production dependency — even when `LLM_PROVIDER=ollama` is used for chat completion —
  because the knowledge graph embeddings are OpenAI-only. The cost is low (embeddings
  only, no chat completion spend), but the account/funding requirement is unconditional.
- `CLAUDE.md` and `README.md` are updated to remove Groq references and to document
  the mandatory `OPENAI_API_KEY` requirement.
