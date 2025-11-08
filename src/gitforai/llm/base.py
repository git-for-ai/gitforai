"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str, **kwargs: Any) -> None:
        """Initialize the LLM provider.

        Args:
            api_key: API key for the provider
            model: Model name to use
            **kwargs: Additional provider-specific parameters
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional provider-specific parameters

        Returns:
            The generated text completion

        Raises:
            Exception: If API call fails
        """
        pass

    @abstractmethod
    async def complete_batch(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> List[str]:
        """Generate completions for multiple prompts.

        Args:
            prompts: List of prompts to process
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            List of generated completions

        Raises:
            Exception: If API calls fail
        """
        pass

    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """Generate an embedding vector for text.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding vector

        Raises:
            Exception: If API call fails
        """
        pass

    @abstractmethod
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors

        Raises:
            Exception: If API calls fail
        """
        pass

    @abstractmethod
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        """Estimate the cost of an API call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pass
