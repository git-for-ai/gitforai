"""LLM integration for semantic analysis."""

from gitforai.llm.base import BaseLLMProvider
from gitforai.llm.openai_provider import OpenAIProvider
from gitforai.llm.embeddings import EmbeddingService

# Local provider is optional (requires sentence-transformers)
try:
    from gitforai.llm.local_provider import LocalProvider
    LOCAL_PROVIDER_AVAILABLE = True
except ImportError:
    LOCAL_PROVIDER_AVAILABLE = False
    LocalProvider = None  # type: ignore

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "LocalProvider",
    "EmbeddingService",
    "LOCAL_PROVIDER_AVAILABLE",
]
