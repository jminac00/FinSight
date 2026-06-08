import logging
from functools import lru_cache

from app.llm.base import LLMService
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


@lru_cache
def get_llm_service() -> LLMService:
    """Return the singleton LLM provider configured via LLM_PROVIDER env var."""
    from app.core.config import get_settings

    settings = get_settings()
    provider = settings.llm_provider

    if provider == "groq":
        logger.info("LLM provider: Groq (%s)", settings.groq_model)
        return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)

    if provider == "ollama":
        logger.info(
            "LLM provider: Ollama (%s @ %s)", settings.ollama_model, settings.ollama_base_url
        )
        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.ollama_model)

    if provider == "openai":
        logger.info("LLM provider: OpenAI (%s)", settings.openai_model)
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Must be 'groq', 'ollama', or 'openai'.")
