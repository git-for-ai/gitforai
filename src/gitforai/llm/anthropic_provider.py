"""Anthropic (Claude) LLM provider implementation.

Mirrors the structure of ``openai_provider.py``. Only ``complete`` /
``complete_batch`` are useful — Anthropic doesn't expose embeddings, so the
embedding methods raise ``NotImplementedError`` and callers should pair this
provider with a separate embedding provider (e.g. ``LocalProvider``).
"""

from __future__ import annotations

import asyncio
from typing import Any, List

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    ANTHROPIC_AVAILABLE = False

from gitforai.llm.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider for completions only."""

    # Pricing per 1M tokens (USD). Order of these keys is the resolution order
    # used when ``estimate_cost`` matches a partial model name.
    PRICING = {
        "claude-opus-4": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "claude-haiku-4": {"input": 1.00, "output": 5.00},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ) -> None:
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install with: pip install anthropic"
            )
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.total_tokens["input"] += getattr(usage, "input_tokens", 0)
                self.total_tokens["output"] += getattr(usage, "output_tokens", 0)
                self.total_cost += self.estimate_cost(
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                )
            # Concatenate all text blocks from the first message
            parts = []
            for block in response.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}") from e

    async def complete_batch(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> List[str]:
        return await asyncio.gather(
            *(self.complete(p, max_tokens, temperature, **kwargs) for p in prompts)
        )

    async def generate_embedding(self, text: str, **kwargs: Any) -> List[float]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. Pair AnthropicProvider "
            "with LocalProvider (sentence-transformers) for embeddings."
        )

    async def generate_embeddings_batch(
        self, texts: List[str], **kwargs: Any
    ) -> List[List[float]]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. Pair AnthropicProvider "
            "with LocalProvider (sentence-transformers) for embeddings."
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        # Match by longest pricing key prefix that the model name starts with
        rates = None
        for key in sorted(self.PRICING, key=len, reverse=True):
            if self.model.startswith(key):
                rates = self.PRICING[key]
                break
        if rates is None:
            return 0.0
        return (input_tokens / 1_000_000) * rates["input"] + (
            output_tokens / 1_000_000
        ) * rates["output"]
