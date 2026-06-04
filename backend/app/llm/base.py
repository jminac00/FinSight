from abc import ABC, abstractmethod


class LLMService(ABC):
    """Abstract interface for all LLM providers.

    All modules must call the LLM exclusively through this interface so that
    swapping providers (Groq ↔ Ollama) requires no changes in business logic.
    """

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the assistant's response.

        Args:
            system_prompt: Instructions that define the assistant's behaviour.
            user_prompt: The user-facing content to analyse.

        Returns:
            The model's text response.
        """
