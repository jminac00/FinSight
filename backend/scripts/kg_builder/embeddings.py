"""Batch embedding generation — delegates to the shared runtime embedder in app.llm."""

from app.llm.embeddings import embed_texts

__all__ = ["embed_texts"]
