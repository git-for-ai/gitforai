"""Tests for LLM cache functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from gitforai.llm.cache import LLMCache


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(temp_cache_dir):
    """Create LLMCache instance."""
    return LLMCache(temp_cache_dir)


def test_cache_initialization(temp_cache_dir):
    """Test cache directory structure is created."""
    cache = LLMCache(temp_cache_dir)

    assert cache.cache_dir.exists()
    assert cache.completions_dir.exists()
    assert cache.embeddings_dir.exists()
    assert cache.hits == 0
    assert cache.misses == 0


def test_completion_caching(cache):
    """Test completion caching and retrieval."""
    prompt = "What is the meaning of life?"
    model = "gpt-4"
    completion = "42"

    # First get should be a miss
    result = cache.get_completion(prompt, model)
    assert result is None
    assert cache.misses == 1
    assert cache.hits == 0

    # Set the completion
    cache.set_completion(prompt, model, completion)

    # Second get should be a hit
    result = cache.get_completion(prompt, model)
    assert result == completion
    assert cache.hits == 1
    assert cache.misses == 1


def test_completion_different_models(cache):
    """Test that different models have separate cache entries."""
    prompt = "What is the meaning of life?"
    model1 = "gpt-4"
    model2 = "gpt-3.5-turbo"
    completion1 = "42"
    completion2 = "I don't know"

    cache.set_completion(prompt, model1, completion1)
    cache.set_completion(prompt, model2, completion2)

    result1 = cache.get_completion(prompt, model1)
    result2 = cache.get_completion(prompt, model2)

    assert result1 == completion1
    assert result2 == completion2


def test_embedding_caching(cache):
    """Test embedding caching and retrieval."""
    text = "This is a test sentence"
    model = "text-embedding-3-small"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    # First get should be a miss
    result = cache.get_embedding(text, model)
    assert result is None
    assert cache.misses == 1

    # Set the embedding
    cache.set_embedding(text, model, embedding)

    # Second get should be a hit
    result = cache.get_embedding(text, model)
    assert result == embedding
    assert cache.hits == 1


def test_batch_embedding_caching(cache):
    """Test batch embedding operations."""
    texts = ["text 1", "text 2", "text 3"]
    model = "text-embedding-3-small"
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    # Initially all should be misses
    cached, miss_indices = cache.get_batch_embeddings(texts, model)
    assert len(cached) == 3
    assert all(c is None for c in cached)
    assert miss_indices == [0, 1, 2]

    # Cache first and third
    cache.set_embedding(texts[0], model, embeddings[0])
    cache.set_embedding(texts[2], model, embeddings[2])

    # Now only middle one should be a miss
    cached, miss_indices = cache.get_batch_embeddings(texts, model)
    assert cached[0] == embeddings[0]
    assert cached[1] is None
    assert cached[2] == embeddings[2]
    assert miss_indices == [1]


def test_cache_clear(cache, temp_cache_dir):
    """Test clearing the cache."""
    # Add some cached data
    cache.set_completion("prompt1", "model1", "completion1")
    cache.set_completion("prompt2", "model1", "completion2")
    cache.set_embedding("text1", "model2", [0.1, 0.2])
    cache.set_embedding("text2", "model2", [0.3, 0.4])

    # Verify files exist
    completion_files = list(cache.completions_dir.glob("*.json"))
    embedding_files = list(cache.embeddings_dir.glob("*.json"))
    assert len(completion_files) == 2
    assert len(embedding_files) == 2

    # Clear cache
    cache.clear()

    # Verify files are deleted
    completion_files = list(cache.completions_dir.glob("*.json"))
    embedding_files = list(cache.embeddings_dir.glob("*.json"))
    assert len(completion_files) == 0
    assert len(embedding_files) == 0

    # Verify stats reset
    assert cache.hits == 0
    assert cache.misses == 0


def test_cache_stats(cache):
    """Test cache statistics calculation."""
    # Add some data and track hits/misses
    cache.get_completion("prompt1", "model1")  # miss
    cache.set_completion("prompt1", "model1", "completion1")
    cache.get_completion("prompt1", "model1")  # hit
    cache.get_completion("prompt1", "model1")  # hit

    cache.get_embedding("text1", "model2")  # miss
    cache.set_embedding("text1", "model2", [0.1, 0.2])
    cache.get_embedding("text1", "model2")  # hit

    stats = cache.get_stats()

    assert stats["hits"] == 3
    assert stats["misses"] == 2
    assert stats["hit_rate"] == "60.0%"
    assert stats["cached_completions"] == 1
    assert stats["cached_embeddings"] == 1


def test_cache_key_generation(cache):
    """Test that cache keys are generated consistently."""
    prompt1 = "What is Python?"
    prompt2 = "What is Python?"  # Same text
    prompt3 = "What is JavaScript?"  # Different text
    model = "gpt-4"

    cache.set_completion(prompt1, model, "A programming language")

    # Same prompt should retrieve cached value
    result = cache.get_completion(prompt2, model)
    assert result == "A programming language"

    # Different prompt should not
    result = cache.get_completion(prompt3, model)
    assert result is None


def test_cache_file_format(cache, temp_cache_dir):
    """Test that cache files are properly formatted JSON."""
    prompt = "Test prompt"
    model = "gpt-4"
    completion = "Test completion"

    cache.set_completion(prompt, model, completion)

    # Find the cache file
    cache_files = list(cache.completions_dir.glob("*.json"))
    assert len(cache_files) == 1

    # Read and verify JSON structure
    with open(cache_files[0], "r") as f:
        data = json.load(f)

    assert "prompt" in data
    assert "model" in data
    assert "completion" in data
    assert data["model"] == model
    assert data["completion"] == completion
    assert len(data["prompt"]) <= 200  # Preview should be truncated


def test_cache_handles_long_text(cache):
    """Test caching with very long text."""
    long_prompt = "A" * 10000  # Very long prompt
    model = "gpt-4"
    completion = "Short completion"

    cache.set_completion(long_prompt, model, completion)
    result = cache.get_completion(long_prompt, model)

    assert result == completion


def test_cache_error_handling(cache):
    """Test that cache errors don't crash the application."""
    # This should not raise an exception even if there's an issue
    result = cache.get_completion("test", "model")
    assert result is None

    # Setting invalid data should be handled gracefully
    # (In production, this would just print an error)
    cache.set_completion("test", "model", "valid")


def test_empty_stats(cache):
    """Test stats with no activity."""
    stats = cache.get_stats()

    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == "0.0%"
    assert stats["cached_completions"] == 0
    assert stats["cached_embeddings"] == 0
