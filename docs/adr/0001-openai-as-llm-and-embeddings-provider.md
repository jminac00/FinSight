# ADR-0001 — OpenAI as LLM and embeddings provider in production

- **Status**: Accepted
- **Date**: June 2026
- **Source**: SRS v1.1 (requirements document revision)

## Context

The project was initially designed under a strict budget constraint of **0 EUR
operating cost**: every service (hosting, data APIs, database and LLM inference) had
to operate within its free tier. For production LLM inference the plan was Groq API
(Llama 3.1 70B, free tier), with Ollama (Llama 3.1 8B) for local development.

However, the sentiment module (GraphRAG) needs a high-quality embeddings provider to
build and query the Neo4j vector index, and the text-generating modules (sentiment,
fundamental, report synthesis) require structured extraction and consistent answers
in Spanish. Free tiers impose quota and quality limits that compromise the
reliability of the system for the demo and the evaluation of the final degree
project.

## Decision

Adopt **OpenAI as the active LLM and embeddings provider in production**:

- Chat completion and structured extraction: `gpt-5.4-mini`.
- Embeddings: `text-embedding-3-large` (3,072 dimensions, vector index in Neo4j).

Groq becomes the **free fallback provider** (`LLM_PROVIDER=groq`) and Ollama remains
the **local development provider** (`LLM_PROVIDER=ollama`). All modules access the
LLM through the centralized `LLMService` (Adapter pattern), so switching providers
does not require modifying any module's logic.

## Consequences

- **The 0 EUR operating cost constraint is broken**: the system assumes a reduced,
  variable cost driven by OpenAI API consumption (pay per use). All other services
  remain on free tiers.
- OpenAI consumption is bounded through aggressive result caching (per-module TTL),
  usage limited to the offline graph-building batch process and on-demand analysis,
  and no continuous streaming.
- An OpenAI account with available credit is required; it becomes a critical
  production dependency together with its `OPENAI_API_KEY` variable.
- Groq remains available as a free degradation path if OpenAI cost or availability
  demands it, with no code changes thanks to the Adapter pattern.
- The SRS is updated accordingly: budget constraint (§2.4), assumptions and
  dependencies table (§2.5) and product perspective (§2.1).
