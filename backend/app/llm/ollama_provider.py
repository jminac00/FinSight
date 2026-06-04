import logging

import httpx

from app.llm.base import LLMService

logger = logging.getLogger(__name__)


class OllamaProvider(LLMService):
    """LLM provider backed by a local Ollama instance (development)."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama /api/chat and return the response text."""
        logger.debug("Ollama request — model=%s", self._model)
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return data["message"]["content"]
