"""Tests for OpenAI provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gitforai.llm.openai_provider import OpenAIProvider


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    # Ensure openai is imported
    with patch("openai.AsyncOpenAI") as mock:
        client = MagicMock()
        mock.return_value = client
        # Also need to ensure OPENAI_AVAILABLE is True
        with patch("gitforai.llm.openai_provider.OPENAI_AVAILABLE", True):
            yield client


@pytest.fixture
def provider(mock_openai_client):
    """Create OpenAIProvider instance with mocked client."""
    with patch("openai.AsyncOpenAI"):
        with patch("gitforai.llm.openai_provider.OPENAI_AVAILABLE", True):
            return OpenAIProvider(
                api_key="test-key",
                model="gpt-4-turbo-preview",
                embedding_model="text-embedding-3-small",
            )


@pytest.mark.asyncio
async def test_complete_success(provider, mock_openai_client):
    """Test successful completion."""
    # Mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Call complete
    result = await provider.complete("Test prompt")

    assert result == "This is a test response"
    assert provider.total_tokens["input"] == 10
    assert provider.total_tokens["output"] == 20
    assert provider.total_cost > 0


@pytest.mark.asyncio
async def test_complete_with_parameters(provider, mock_openai_client):
    """Test completion with custom parameters."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 10

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    await provider.complete(
        "Test prompt",
        max_tokens=100,
        temperature=0.7,
    )

    # Verify parameters were passed
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_complete_error_handling(provider, mock_openai_client):
    """Test error handling in completion."""
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API Error")
    )

    with pytest.raises(Exception) as exc_info:
        await provider.complete("Test prompt")

    assert "OpenAI API error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_complete_batch(provider, mock_openai_client):
    """Test batch completion."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 10

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    results = await provider.complete_batch(prompts)

    assert len(results) == 3
    assert all(r == "Response" for r in results)
    assert mock_openai_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_complete_batch_with_errors(provider, mock_openai_client):
    """Test batch completion handles individual errors."""
    # First call succeeds, second fails, third succeeds
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Success"
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 10

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[
            mock_response,
            Exception("API Error"),
            mock_response,
        ]
    )

    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    results = await provider.complete_batch(prompts)

    assert len(results) == 3
    assert results[0] == "Success"
    assert "Error" in results[1]
    assert results[2] == "Success"


@pytest.mark.asyncio
async def test_generate_embedding(provider, mock_openai_client):
    """Test embedding generation."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_response.usage.prompt_tokens = 8

    mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

    result = await provider.generate_embedding("Test text")

    assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert provider.total_tokens["input"] == 8


@pytest.mark.asyncio
async def test_generate_embeddings_batch(provider, mock_openai_client):
    """Test batch embedding generation."""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
        MagicMock(embedding=[0.5, 0.6]),
    ]
    mock_response.usage.prompt_tokens = 15

    mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

    texts = ["Text 1", "Text 2", "Text 3"]
    results = await provider.generate_embeddings_batch(texts)

    assert len(results) == 3
    assert results[0] == [0.1, 0.2]
    assert results[1] == [0.3, 0.4]
    assert results[2] == [0.5, 0.6]


@pytest.mark.asyncio
async def test_generate_embeddings_batch_error(provider, mock_openai_client):
    """Test batch embedding handles errors gracefully."""
    mock_openai_client.embeddings.create = AsyncMock(
        side_effect=Exception("API Error")
    )

    texts = ["Text 1", "Text 2"]
    results = await provider.generate_embeddings_batch(texts)

    # Should return empty embeddings for failed batch
    assert len(results) == 2
    assert results[0] == []
    assert results[1] == []


def test_estimate_cost(provider):
    """Test cost estimation."""
    # Test GPT-4 pricing
    cost = provider.estimate_cost(1000, 500)
    expected = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
    assert cost == pytest.approx(expected)


def test_estimate_cost_embedding(provider):
    """Test cost estimation for embeddings."""
    cost = provider.estimate_cost(
        1000,
        0,
        model="text-embedding-3-small"
    )
    expected = (1000 / 1_000_000) * 0.02
    assert cost == pytest.approx(expected)


def test_estimate_cost_unknown_model(provider):
    """Test cost estimation defaults to GPT-4 for unknown models."""
    cost = provider.estimate_cost(1000, 500, model="unknown-model")
    # Should use GPT-4 pricing as default
    expected = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert cost == pytest.approx(expected)


def test_usage_stats(provider):
    """Test usage statistics tracking."""
    stats = provider.get_usage_stats()

    assert stats["total_tokens"]["input"] == 0
    assert stats["total_tokens"]["output"] == 0
    assert stats["total_cost"] == 0.0
    assert stats["model"] == "gpt-4-turbo-preview"
    assert stats["embedding_model"] == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_usage_stats_after_calls(provider, mock_openai_client):
    """Test usage statistics are updated after API calls."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    await provider.complete("Test")
    await provider.complete("Test again")

    stats = provider.get_usage_stats()

    assert stats["total_tokens"]["input"] == 200
    assert stats["total_tokens"]["output"] == 100
    assert stats["total_cost"] > 0


def test_reset_usage_stats(provider):
    """Test resetting usage statistics."""
    # Manually set some stats
    provider.total_tokens["input"] = 1000
    provider.total_tokens["output"] = 500
    provider.total_cost = 0.05

    provider.reset_usage_stats()

    assert provider.total_tokens["input"] == 0
    assert provider.total_tokens["output"] == 0
    assert provider.total_cost == 0.0


def test_pricing_data(provider):
    """Test pricing data is available."""
    assert "gpt-4-turbo-preview" in provider.PRICING
    assert "gpt-3.5-turbo" in provider.PRICING
    assert "text-embedding-3-small" in provider.PRICING

    # Verify structure
    gpt4_pricing = provider.PRICING["gpt-4-turbo-preview"]
    assert "input" in gpt4_pricing
    assert "output" in gpt4_pricing


def test_initialization():
    """Test provider initialization."""
    with patch("openai.AsyncOpenAI"):
        with patch("gitforai.llm.openai_provider.OPENAI_AVAILABLE", True):
            provider = OpenAIProvider(
                api_key="test-key",
                model="gpt-3.5-turbo",
                embedding_model="text-embedding-ada-002",
            )

            assert provider.model == "gpt-3.5-turbo"
            assert provider.embedding_model == "text-embedding-ada-002"
            assert provider.api_key == "test-key"
