import logging

from groq import AsyncGroq

from app.llm.base import LLMService

logger = logging.getLogger(__name__)


class GroqProvider(LLMService):
    """LLM provider backed by the Groq API (production)."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call Groq chat completions and return the response text."""
        logger.debug("Groq request — model=%s", self._model)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
