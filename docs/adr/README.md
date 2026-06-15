# Architecture Decision Records (ADR)

Record of the architecture and design decisions of the FinSight project.
Each ADR documents one relevant decision: its context, the decision taken and its consequences.

The requirements document (SRS, IEEE 830) is **not part of the repository**: it is a
private document under constant change. The design decisions introduced by its
revisions are recorded here as ADRs, which constitute the public, versioned record
of the project's evolution.

## Index

| ADR | Title | Source | Status |
|-----|-------|--------|--------|
| [0001](0001-openai-as-llm-and-embeddings-provider.md) | OpenAI as LLM and embeddings provider in production | SRS v1.1 | Accepted |
| [0002](0002-cookie-consent-management.md) | Cookie and granular consent management | SRS v1.2 | Accepted |
| [0003](0003-consolidate-deployment-on-render.md) | Consolidate full deployment on Render | SRS v1.3 | Accepted |
| [0004](0004-design-patterns-for-analysis-orchestration.md) | Design patterns for analysis orchestration | Internal architecture design | Accepted |
| [0005](0005-news-provider-fallback-chain.md) | News provider fallback chain for the sentiment module | Internal architecture design | Accepted |
| [0006](0006-gru-architecture-no-vmd.md) | GRU without VMD as the deep learning production model | Internal architecture design | Accepted |

## Format

ADRs follow Michael Nygard's format:

- **Status**: Proposed / Accepted / Superseded by ADR-XXXX.
- **Context**: situation and forces that motivate the decision.
- **Decision**: what was decided and why.
- **Consequences**: positive and negative effects, and derived work.

Files are named `NNNN-short-title.md` with sequential numbering.
