"""Local LLM provider implementation using sentence-transformers."""

import asyncio
from typing import Any, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from gitforai.llm.base import BaseLLMProvider


class LocalProvider(BaseLLMProvider):
    """Local embedding provider using sentence-transformers.

    This provider runs embeddings locally without requiring API keys or
    internet connectivity. It provides zero-cost, privacy-preserving
    embeddings suitable for commit similarity search.

    Default model: all-MiniLM-L6-v2
    - Dimensions: 384
    - Size: 80MB
    - Quality: ~88% of OpenAI for code/commits
    - Speed: ~10ms per embedding (no network latency)
    - Cost: $0.00 (free, runs locally)
    """

    # Model information
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    MODEL_INFO = {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "size_mb": 80,
            "description": "Fast, lightweight model for semantic similarity",
        },
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "size_mb": 420,
            "description": "Higher quality, larger model",
        },
        "paraphrase-MiniLM-L6-v2": {
            "dimensions": 384,
            "size_mb": 80,
            "description": "Alternative lightweight model",
        },
        "bge-small-en-v1.5": {
            "dimensions": 384,
            "size_mb": 133,
            "description": "Optimized for retrieval tasks",
        },
    }

    def __init__(
        self,
        api_key: str = "",  # Not used, but required by base class signature
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize local embedding provider.

        Args:
            api_key: Not used for local provider (kept for interface compatibility)
            model: Model name from sentence-transformers hub
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            **kwargs: Additional parameters (ignored for local provider)

        Raises:
            ImportError: If sentence-transformers package not installed
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers package is required for LocalProvider. "
                "Install with: pip install 'gitforai[local-embeddings]' "
                "or: pip install sentence-transformers torch"
            )

        # Initialize base class (api_key unused for local)
        super().__init__(api_key or "local", model, **kwargs)

        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._model_loaded = False

        # Track usage for statistics
        self.total_embeddings = 0
        self.total_tokens_estimated = 0

    def _load_model(self) -> None:
        """Lazy load the sentence-transformers model.

        Downloads the model on first use if not already cached.
        Models are cached in ~/.cache/huggingface/ by default.
        """
        if self._model_loaded:
            return

        try:
            print(f"Loading local embedding model: {self.model}")
            if self.model not in self.MODEL_INFO:
                print(f"Warning: Unknown model '{self.model}'. Known models: {list(self.MODEL_INFO.keys())}")

            self._model = SentenceTransformer(self.model, device=self.device)
            self._model_loaded = True

            # Get model info
            info = self.MODEL_INFO.get(self.model, {})
            dims = info.get("dimensions", "unknown")
            size = info.get("size_mb", "unknown")

            print(f"✓ Model loaded: {dims} dimensions, ~{size}MB")
            print(f"  Cached in: ~/.cache/huggingface/")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load sentence-transformers model '{self.model}': {e}\n"
                f"This may be due to:\n"
                f"  1. First-time download in progress (can take 1-2 minutes)\n"
                f"  2. Network issues preventing download\n"
                f"  3. Invalid model name\n"
                f"Try running: python -c \"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{self.model}')\""
            ) from e

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """Generate a completion from the LLM.

        NOT IMPLEMENTED for local embeddings provider.
        Local models only generate embeddings, not text completions.

        Raises:
            NotImplementedError: Always, as local provider is embeddings-only
        """
        raise NotImplementedError(
            "LocalProvider only supports embeddings, not text completions. "
            "For LLM completions, use OpenAIProvider or AnthropicProvider. "
            "The local provider is designed for zero-cost embedding generation only."
        )

    async def complete_batch(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> List[str]:
        """Generate completions for multiple prompts.

        NOT IMPLEMENTED for local embeddings provider.

        Raises:
            NotImplementedError: Always, as local provider is embeddings-only
        """
        raise NotImplementedError(
            "LocalProvider only supports embeddings, not text completions. "
            "Use OpenAIProvider or AnthropicProvider for batch completions."
        )

    async def generate_embedding(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """Generate embedding for text using local model.

        Args:
            text: Text to embed
            **kwargs: Additional parameters (ignored)

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If model fails to load or generate embedding
        """
        # Lazy load model on first use
        if not self._model_loaded:
            self._load_model()

        try:
            # Run in thread pool to avoid blocking event loop
            # sentence-transformers encode is synchronous
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode(text, convert_to_numpy=True).tolist()
            )

            # Track usage
            self.total_embeddings += 1
            self.total_tokens_estimated += len(text) // 4  # Rough estimate

            return embedding

        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts using local model.

        Much faster than calling generate_embedding() individually due to
        batching optimizations in sentence-transformers.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding (default: 32)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If model fails to load or generate embeddings
        """
        # Lazy load model on first use
        if not self._model_loaded:
            self._load_model()

        try:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 100  # Show progress for large batches
                ).tolist()
            )

            # Track usage
            self.total_embeddings += len(texts)
            self.total_tokens_estimated += sum(len(t) // 4 for t in texts)

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to generate batch embeddings: {e}") from e

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        """Estimate API call cost.

        For local embeddings, cost is always $0.00 (free).

        Args:
            input_tokens: Number of input tokens (ignored)
            output_tokens: Number of output tokens (ignored)

        Returns:
            0.0 (local embeddings are free)
        """
        return 0.0

    def get_usage_stats(self) -> dict:
        """Get current usage statistics.

        Returns:
            Dictionary with usage counts (no cost, since free)
        """
        model_info = self.MODEL_INFO.get(self.model, {})

        return {
            "total_embeddings": self.total_embeddings,
            "total_tokens_estimated": self.total_tokens_estimated,
            "total_cost": 0.0,  # Always free
            "model": self.model,
            "model_loaded": self._model_loaded,
            "dimensions": model_info.get("dimensions", "unknown"),
            "device": self.device or "auto",
            "provider": "local (sentence-transformers)",
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_embeddings = 0
        self.total_tokens_estimated = 0

    def get_model_info(self) -> dict:
        """Get information about the current model.

        Returns:
            Dictionary with model metadata
        """
        info = self.MODEL_INFO.get(self.model, {
            "dimensions": "unknown",
            "size_mb": "unknown",
            "description": "Custom model",
        })

        return {
            "name": self.model,
            "dimensions": info["dimensions"],
            "size_mb": info["size_mb"],
            "description": info["description"],
            "loaded": self._model_loaded,
            "cache_dir": "~/.cache/huggingface/",
        }
