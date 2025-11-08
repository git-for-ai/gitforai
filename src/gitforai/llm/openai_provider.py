"""OpenAI LLM provider implementation."""

import asyncio
from typing import Any, List, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from gitforai.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for completions and embeddings."""

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        embedding_model: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model for completions
            embedding_model: Model for embeddings
            **kwargs: Additional parameters

        Raises:
            ImportError: If openai package not installed
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for OpenAI provider. "
                "Install with: pip install openai"
            )

        super().__init__(api_key, model, **kwargs)
        self.embedding_model = embedding_model
        self.client = AsyncOpenAI(api_key=api_key)
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """Generate a completion from OpenAI.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional OpenAI parameters

        Returns:
            Generated text

        Raises:
            Exception: If API call fails
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Track usage
            if hasattr(response, "usage"):
                self.total_tokens["input"] += response.usage.prompt_tokens
                self.total_tokens["output"] += response.usage.completion_tokens
                self.total_cost += self.estimate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise Exception(f"OpenAI API error: {e}") from e

    async def complete_batch(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.3,
        batch_size: int = 10,
        **kwargs: Any,
    ) -> List[str]:
        """Generate completions for multiple prompts with batching.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            batch_size: Number of concurrent requests
            **kwargs: Additional parameters

        Returns:
            List of completions
        """
        results = []

        # Process in batches to avoid rate limits
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            tasks = [
                self.complete(prompt, max_tokens, temperature, **kwargs)
                for prompt in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle errors in batch
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(f"Error: {result}")
                else:
                    results.append(result)

        return results

    async def generate_embedding(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding vector

        Raises:
            Exception: If API call fails
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                **kwargs,
            )

            # Track usage
            if hasattr(response, "usage"):
                self.total_tokens["input"] += response.usage.prompt_tokens
                self.total_cost += self.estimate_cost(
                    response.usage.prompt_tokens,
                    0,
                    model=self.embedding_model,
                )

            return response.data[0].embedding

        except Exception as e:
            raise Exception(f"OpenAI embedding error: {e}") from e

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Texts per batch (OpenAI limit is ~2048)
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                    **kwargs,
                )

                # Track usage
                if hasattr(response, "usage"):
                    self.total_tokens["input"] += response.usage.prompt_tokens
                    self.total_cost += self.estimate_cost(
                        response.usage.prompt_tokens,
                        0,
                        model=self.embedding_model,
                    )

                # Extract embeddings in order
                batch_embeddings = [data.embedding for data in response.data]
                results.extend(batch_embeddings)

            except Exception as e:
                # Return empty embeddings for failed batch
                results.extend([[] for _ in batch])
                print(f"Batch embedding error: {e}")

        return results

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int = 0,
        model: Optional[str] = None,
    ) -> float:
        """Estimate API call cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model to use for pricing (defaults to self.model)

        Returns:
            Estimated cost in USD
        """
        model_name = model or self.model

        if model_name not in self.PRICING:
            # Default to GPT-4 pricing if model unknown
            pricing = self.PRICING["gpt-4"]
        else:
            pricing = self.PRICING[model_name]

        cost = (input_tokens / 1_000_000) * pricing["input"]
        cost += (output_tokens / 1_000_000) * pricing["output"]

        return cost

    def get_usage_stats(self) -> dict:
        """Get current usage statistics.

        Returns:
            Dictionary with token counts and costs
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "model": self.model,
            "embedding_model": self.embedding_model,
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}
