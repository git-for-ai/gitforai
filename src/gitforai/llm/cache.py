"""Caching layer for LLM responses."""

import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional


class LLMCache:
    """File-based cache for LLM responses and embeddings."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Separate directories for different cache types
        self.completions_dir = self.cache_dir / "completions"
        self.embeddings_dir = self.cache_dir / "embeddings"

        self.completions_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)

        # Stats
        self.hits = 0
        self.misses = 0

    def _generate_key(self, text: str, prefix: str = "") -> str:
        """Generate cache key from text.

        Args:
            text: Text to generate key for
            prefix: Optional prefix for the key

        Returns:
            Cache key
        """
        hash_obj = hashlib.sha256(text.encode())
        if prefix:
            hash_obj.update(prefix.encode())
        return hash_obj.hexdigest()

    def get_completion(self, prompt: str, model: str) -> Optional[str]:
        """Get cached completion.

        Args:
            prompt: The prompt
            model: Model name

        Returns:
            Cached completion or None
        """
        key = self._generate_key(prompt, model)
        cache_file = self.completions_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    self.hits += 1
                    return data.get("completion")
            except Exception:
                pass

        self.misses += 1
        return None

    def set_completion(self, prompt: str, model: str, completion: str) -> None:
        """Cache a completion.

        Args:
            prompt: The prompt
            model: Model name
            completion: The completion to cache
        """
        key = self._generate_key(prompt, model)
        cache_file = self.completions_dir / f"{key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "prompt": prompt[:200],  # Store preview
                        "model": model,
                        "completion": completion,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Failed to cache completion: {e}")

    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding.

        Args:
            text: The text
            model: Embedding model name

        Returns:
            Cached embedding or None
        """
        key = self._generate_key(text, model)
        cache_file = self.embeddings_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    self.hits += 1
                    return data.get("embedding")
            except Exception:
                pass

        self.misses += 1
        return None

    def set_embedding(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache an embedding.

        Args:
            text: The text
            model: Embedding model name
            embedding: The embedding vector to cache
        """
        key = self._generate_key(text, model)
        cache_file = self.embeddings_dir / f"{key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "text": text[:200],  # Store preview
                        "model": model,
                        "embedding": embedding,
                    },
                    f,
                )
        except Exception as e:
            print(f"Failed to cache embedding: {e}")

    def get_batch_embeddings(
        self, texts: List[str], model: str
    ) -> tuple[List[Optional[List[float]]], List[int]]:
        """Get cached embeddings for multiple texts.

        Args:
            texts: List of texts
            model: Embedding model name

        Returns:
            Tuple of (embeddings list with None for misses, indices of misses)
        """
        embeddings: List[Optional[List[float]]] = []
        miss_indices: List[int] = []

        for i, text in enumerate(texts):
            embedding = self.get_embedding(text, model)
            embeddings.append(embedding)
            if embedding is None:
                miss_indices.append(i)

        return embeddings, miss_indices

    def set_batch_embeddings(
        self, texts: List[str], model: str, embeddings: List[List[float]]
    ) -> None:
        """Cache multiple embeddings.

        Args:
            texts: List of texts
            model: Embedding model name
            embeddings: List of embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, model, embedding)

    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.completions_dir.glob("*.json"):
            cache_file.unlink()

        for cache_file in self.embeddings_dir.glob("*.json"):
            cache_file.unlink()

        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0

        completion_count = len(list(self.completions_dir.glob("*.json")))
        embedding_count = len(list(self.embeddings_dir.glob("*.json")))

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cached_completions": completion_count,
            "cached_embeddings": embedding_count,
        }
