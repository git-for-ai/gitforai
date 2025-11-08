"""Tests for LocalProvider (sentence-transformers)."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock sentence_transformers at sys.modules level before importing
mock_st_module = MagicMock()
mock_st_class = MagicMock()
mock_st_module.SentenceTransformer = mock_st_class
sys.modules['sentence_transformers'] = mock_st_module

from gitforai.llm.local_provider import LocalProvider


@pytest.fixture
def mock_sentence_transformer():
    """Create mock SentenceTransformer."""
    model_instance = MagicMock()
    mock_st_class.return_value = model_instance

    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        yield model_instance


@pytest.fixture
def provider(mock_sentence_transformer):
    """Create LocalProvider instance with mocked model."""
    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        return LocalProvider(model="all-MiniLM-L6-v2")


def test_initialization_without_dependencies():
    """Test provider initialization fails without sentence-transformers."""
    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", False):
        with pytest.raises(ImportError) as exc_info:
            LocalProvider()

        assert "sentence-transformers package is required" in str(exc_info.value)


def test_initialization_with_dependencies():
    """Test provider initialization succeeds with dependencies."""
    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        provider = LocalProvider(model="all-MiniLM-L6-v2")

        assert provider.model == "all-MiniLM-L6-v2"
        assert provider.api_key == "local"
        assert provider.device is None
        assert provider._model_loaded is False
        assert provider.total_embeddings == 0


def test_initialization_with_custom_device():
    """Test provider initialization with custom device."""
    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        provider = LocalProvider(model="all-MiniLM-L6-v2", device="cuda")

        assert provider.device == "cuda"


@pytest.mark.asyncio
async def test_complete_not_implemented(provider):
    """Test that complete() raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        await provider.complete("Test prompt")

    assert "only supports embeddings, not text completions" in str(exc_info.value)


@pytest.mark.asyncio
async def test_complete_batch_not_implemented(provider):
    """Test that complete_batch() raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        await provider.complete_batch(["Prompt 1", "Prompt 2"])

    assert "only supports embeddings, not text completions" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_embedding(provider, mock_sentence_transformer):
    """Test single embedding generation."""
    # Mock the encode method to return a numpy-like array
    mock_array = MagicMock()
    mock_array.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_sentence_transformer.encode.return_value = mock_array

    result = await provider.generate_embedding("Test commit message")

    assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert provider.total_embeddings == 1
    assert provider.total_tokens_estimated > 0

    # Verify encode was called with correct parameters
    mock_sentence_transformer.encode.assert_called_once()
    call_args = mock_sentence_transformer.encode.call_args
    assert call_args[0][0] == "Test commit message"
    assert call_args[1]["convert_to_numpy"] is True


@pytest.mark.asyncio
async def test_generate_embedding_lazy_loading():
    """Test that model is loaded lazily on first use."""
    mock_model = MagicMock()
    mock_array = MagicMock()
    mock_array.tolist.return_value = [0.1, 0.2]
    mock_model.encode.return_value = mock_array
    mock_st_class.return_value = mock_model
    mock_st_class.reset_mock()

    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        provider = LocalProvider(model="all-MiniLM-L6-v2")

        # Model should not be loaded yet
        assert provider._model_loaded is False
        mock_st_class.assert_not_called()

        # Generate embedding should trigger loading
        await provider.generate_embedding("Test")

        # Now model should be loaded
        assert provider._model_loaded is True
        mock_st_class.assert_called_once_with("all-MiniLM-L6-v2", device=None)


@pytest.mark.asyncio
async def test_generate_embeddings_batch(provider, mock_sentence_transformer):
    """Test batch embedding generation."""
    # Mock the encode method for batch
    mock_array = MagicMock()
    mock_array.tolist.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    mock_sentence_transformer.encode.return_value = mock_array

    texts = ["Commit 1", "Commit 2", "Commit 3"]
    results = await provider.generate_embeddings_batch(texts)

    assert len(results) == 3
    assert results[0] == [0.1, 0.2, 0.3]
    assert results[1] == [0.4, 0.5, 0.6]
    assert results[2] == [0.7, 0.8, 0.9]
    assert provider.total_embeddings == 3

    # Verify batch encoding was called correctly
    mock_sentence_transformer.encode.assert_called_once()
    call_args = mock_sentence_transformer.encode.call_args
    assert call_args[0][0] == texts
    assert call_args[1]["batch_size"] == 32
    assert call_args[1]["convert_to_numpy"] is True
    assert call_args[1]["show_progress_bar"] is False  # texts < 100


@pytest.mark.asyncio
async def test_generate_embeddings_batch_large(provider, mock_sentence_transformer):
    """Test batch embedding with large number of texts shows progress."""
    # Mock the encode method
    mock_array = MagicMock()
    mock_array.tolist.return_value = [[0.1, 0.2]] * 150
    mock_sentence_transformer.encode.return_value = mock_array

    texts = [f"Commit {i}" for i in range(150)]
    results = await provider.generate_embeddings_batch(texts)

    assert len(results) == 150

    # Verify progress bar is shown for large batches
    call_args = mock_sentence_transformer.encode.call_args
    assert call_args[1]["show_progress_bar"] is True  # texts > 100


@pytest.mark.asyncio
async def test_generate_embeddings_batch_custom_batch_size(provider, mock_sentence_transformer):
    """Test batch embedding with custom batch size."""
    mock_array = MagicMock()
    mock_array.tolist.return_value = [[0.1, 0.2]] * 3
    mock_sentence_transformer.encode.return_value = mock_array

    texts = ["Commit 1", "Commit 2", "Commit 3"]
    await provider.generate_embeddings_batch(texts, batch_size=16)

    # Verify custom batch size was used
    call_args = mock_sentence_transformer.encode.call_args
    assert call_args[1]["batch_size"] == 16


@pytest.mark.asyncio
async def test_generate_embedding_error_handling(provider, mock_sentence_transformer):
    """Test error handling in embedding generation."""
    mock_sentence_transformer.encode.side_effect = Exception("Model error")

    with pytest.raises(RuntimeError) as exc_info:
        await provider.generate_embedding("Test")

    assert "Failed to generate embedding" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_embeddings_batch_error_handling(provider, mock_sentence_transformer):
    """Test error handling in batch embedding generation."""
    mock_sentence_transformer.encode.side_effect = Exception("Model error")

    with pytest.raises(RuntimeError) as exc_info:
        await provider.generate_embeddings_batch(["Text 1", "Text 2"])

    assert "Failed to generate batch embeddings" in str(exc_info.value)


def test_estimate_cost_always_zero(provider):
    """Test that cost estimation always returns $0.00."""
    # Test with various token counts
    assert provider.estimate_cost(0, 0) == 0.0
    assert provider.estimate_cost(1000, 0) == 0.0
    assert provider.estimate_cost(1000, 500) == 0.0
    assert provider.estimate_cost(1_000_000, 1_000_000) == 0.0


def test_get_usage_stats(provider):
    """Test usage statistics retrieval."""
    stats = provider.get_usage_stats()

    assert stats["total_embeddings"] == 0
    assert stats["total_tokens_estimated"] == 0
    assert stats["total_cost"] == 0.0
    assert stats["model"] == "all-MiniLM-L6-v2"
    assert stats["model_loaded"] is False
    assert stats["dimensions"] == 384
    assert stats["device"] == "auto"
    assert stats["provider"] == "local (sentence-transformers)"


@pytest.mark.asyncio
async def test_get_usage_stats_after_embeddings(provider, mock_sentence_transformer):
    """Test usage statistics are updated after embedding generation."""
    mock_array = MagicMock()
    mock_array.tolist.return_value = [0.1, 0.2, 0.3]
    mock_sentence_transformer.encode.return_value = mock_array

    # Generate some embeddings
    await provider.generate_embedding("Test 1")
    await provider.generate_embedding("Test 2")
    await provider.generate_embeddings_batch(["Test 3", "Test 4", "Test 5"])

    stats = provider.get_usage_stats()

    assert stats["total_embeddings"] == 5  # 2 single + 3 batch
    assert stats["total_tokens_estimated"] > 0
    assert stats["total_cost"] == 0.0  # Always free
    assert stats["model_loaded"] is True


def test_reset_usage_stats(provider):
    """Test resetting usage statistics."""
    # Manually set some stats
    provider.total_embeddings = 100
    provider.total_tokens_estimated = 5000

    provider.reset_usage_stats()

    assert provider.total_embeddings == 0
    assert provider.total_tokens_estimated == 0


def test_get_model_info(provider):
    """Test model information retrieval."""
    info = provider.get_model_info()

    assert info["name"] == "all-MiniLM-L6-v2"
    assert info["dimensions"] == 384
    assert info["size_mb"] == 80
    assert "lightweight" in info["description"].lower()
    assert info["loaded"] is False
    assert "huggingface" in info["cache_dir"].lower()


def test_get_model_info_unknown_model():
    """Test model information for unknown model."""
    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        provider = LocalProvider(model="custom-model")

        info = provider.get_model_info()

        assert info["name"] == "custom-model"
        assert info["dimensions"] == "unknown"
        assert info["size_mb"] == "unknown"
        assert info["description"] == "Custom model"


def test_model_info_data():
    """Test that MODEL_INFO contains expected models."""
    assert "all-MiniLM-L6-v2" in LocalProvider.MODEL_INFO
    assert "all-mpnet-base-v2" in LocalProvider.MODEL_INFO
    assert "paraphrase-MiniLM-L6-v2" in LocalProvider.MODEL_INFO
    assert "bge-small-en-v1.5" in LocalProvider.MODEL_INFO

    # Verify structure
    model_info = LocalProvider.MODEL_INFO["all-MiniLM-L6-v2"]
    assert "dimensions" in model_info
    assert "size_mb" in model_info
    assert "description" in model_info
    assert model_info["dimensions"] == 384
    assert model_info["size_mb"] == 80


def test_default_model():
    """Test default model is correct."""
    assert LocalProvider.DEFAULT_MODEL == "all-MiniLM-L6-v2"


@pytest.mark.asyncio
async def test_load_model_only_once():
    """Test that model is only loaded once even with multiple calls."""
    mock_model = MagicMock()
    mock_array = MagicMock()
    mock_array.tolist.return_value = [0.1, 0.2]
    mock_model.encode.return_value = mock_array
    mock_st_class.return_value = mock_model
    mock_st_class.reset_mock()

    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        provider = LocalProvider(model="all-MiniLM-L6-v2")

        # Generate multiple embeddings
        await provider.generate_embedding("Test 1")
        await provider.generate_embedding("Test 2")
        await provider.generate_embeddings_batch(["Test 3", "Test 4"])

        # Model should only be instantiated once
        mock_st_class.assert_called_once()


def test_api_key_not_required():
    """Test that api_key parameter is not actually used."""
    with patch("gitforai.llm.local_provider.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        # Should work with empty string
        provider1 = LocalProvider(api_key="")
        assert provider1.api_key == "local"

        # Should work with any value (unused)
        provider2 = LocalProvider(api_key="anything")
        assert provider2.api_key == "anything"

        # Should work with no api_key specified
        provider3 = LocalProvider()
        assert provider3.api_key == "local"
