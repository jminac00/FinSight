import logging

from openai import AsyncOpenAI

from app.llm.base import LLMService

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMService):
    """LLM provider backed by the OpenAI API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, max_retries=6)
        self._model = model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
