"""Tests for embedding service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from gitforai.llm.embeddings import EmbeddingService
from gitforai.models import CodeChange, CommitMetadata, FileDiff


@pytest.fixture
def mock_provider():
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    provider.generate_embeddings_batch = AsyncMock(
        return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )
    return provider


@pytest.fixture
def embedding_service(mock_provider):
    """Create EmbeddingService instance."""
    return EmbeddingService(mock_provider)


@pytest.fixture
def sample_commit():
    """Create sample commit metadata."""
    return CommitMetadata(
        hash="abc123",
        author_name="Test Author",
        author_email="test@example.com",
        timestamp=datetime.now(timezone.utc),
        message="Add new feature",
        files_changed=["src/feature.py", "tests/test_feature.py"],
        intent="feature",
        topics=["feature", "testing"],
        llm_summary="Adds a new feature with tests",
    )


@pytest.fixture
def sample_diff():
    """Create sample file diff."""
    return FileDiff(
        file_path="src/feature.py",
        change_type="modified",
        additions=10,
        deletions=5,
        diff_text="+def new_feature():\n+    pass",
        is_binary=False,
        context_before="# Old code",
        context_after="# New code",
    )


@pytest.fixture
def sample_code_change():
    """Create sample code change."""
    return CodeChange(
        file_path="src/feature.py",
        change_type="modified",
        summary="Added new feature function",
        reasoning="To support new use case",
        diff_snippet="+def new_feature():\n+    pass",
        context="Feature implementation",
        affected_functions=["new_feature"],
    )


@pytest.mark.asyncio
async def test_embed_commit(embedding_service, mock_provider, sample_commit):
    """Test embedding generation for commit."""
    result = await embedding_service.embed_commit(sample_commit)

    assert result == [0.1, 0.2, 0.3]
    mock_provider.generate_embedding.assert_called_once()

    # Verify text construction includes key fields
    call_args = mock_provider.generate_embedding.call_args[0][0]
    assert "Add new feature" in call_args
    assert "Adds a new feature with tests" in call_args
    assert "feature" in call_args
    assert "testing" in call_args


@pytest.mark.asyncio
async def test_embed_commit_minimal(embedding_service, mock_provider):
    """Test embedding commit with minimal data."""
    commit = CommitMetadata(
        hash="abc123",
        author_name="Test Author",
        author_email="test@example.com",
        timestamp=datetime.now(timezone.utc),
        message="Fix bug",
        files_changed=[],
    )

    result = await embedding_service.embed_commit(commit)

    assert result == [0.1, 0.2, 0.3]
    call_args = mock_provider.generate_embedding.call_args[0][0]
    assert "Fix bug" in call_args


@pytest.mark.asyncio
async def test_embed_commits_batch(embedding_service, mock_provider):
    """Test batch embedding for commits."""
    commits = [
        CommitMetadata(
            hash=f"hash{i}",
            author_name="Author",
            author_email="author@example.com",
            timestamp=datetime.now(timezone.utc),
            message=f"Commit {i}",
            files_changed=[],
        )
        for i in range(3)
    ]

    results = await embedding_service.embed_commits_batch(commits)

    assert len(results) == 3
    assert results == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    mock_provider.generate_embeddings_batch.assert_called_once()


@pytest.mark.asyncio
async def test_embed_diff(embedding_service, mock_provider, sample_diff):
    """Test embedding generation for diff."""
    result = await embedding_service.embed_diff(sample_diff)

    assert result == [0.1, 0.2, 0.3]
    mock_provider.generate_embedding.assert_called_once()

    call_args = mock_provider.generate_embedding.call_args[0][0]
    assert "src/feature.py" in call_args
    assert "modified" in call_args
    assert "+def new_feature()" in call_args


@pytest.mark.asyncio
async def test_embed_diff_binary(embedding_service, mock_provider):
    """Test embedding for binary diff."""
    diff = FileDiff(
        file_path="image.png",
        change_type="added",
        additions=0,
        deletions=0,
        diff_text=None,
        is_binary=True,
    )

    result = await embedding_service.embed_diff(diff)

    assert result == [0.1, 0.2, 0.3]

    call_args = mock_provider.generate_embedding.call_args[0][0]
    assert "image.png" in call_args
    assert "added" in call_args
    # Binary file should not include diff text
    assert "Diff:" not in call_args


@pytest.mark.asyncio
async def test_embed_diff_truncation(embedding_service, mock_provider):
    """Test that long diffs are truncated."""
    long_diff_text = "+" + "x" * 2000  # Very long diff

    diff = FileDiff(
        file_path="file.py",
        change_type="modified",
        additions=100,
        deletions=50,
        diff_text=long_diff_text,
        is_binary=False,
    )

    await embedding_service.embed_diff(diff)

    call_args = mock_provider.generate_embedding.call_args[0][0]
    # Should be truncated to 1000 chars in diff section
    assert len(call_args) < 2000


@pytest.mark.asyncio
async def test_embed_code_change(embedding_service, mock_provider, sample_code_change):
    """Test embedding generation for code change."""
    result = await embedding_service.embed_code_change(sample_code_change)

    assert result == [0.1, 0.2, 0.3]
    mock_provider.generate_embedding.assert_called_once()

    call_args = mock_provider.generate_embedding.call_args[0][0]
    assert "src/feature.py" in call_args
    assert "Added new feature function" in call_args
    assert "To support new use case" in call_args
    assert "new_feature" in call_args


@pytest.mark.asyncio
async def test_embed_code_changes_batch(embedding_service, mock_provider):
    """Test batch embedding for code changes."""
    changes = [
        CodeChange(
            file_path=f"file{i}.py",
            change_type="modified",
            summary=f"Change {i}",
            diff_snippet="code",
            affected_functions=[],
        )
        for i in range(3)
    ]

    results = await embedding_service.embed_code_changes_batch(changes)

    assert len(results) == 3
    mock_provider.generate_embeddings_batch.assert_called_once()


@pytest.mark.asyncio
async def test_embed_text(embedding_service, mock_provider):
    """Test embedding arbitrary text."""
    text = "This is a search query"

    result = await embedding_service.embed_text(text)

    assert result == [0.1, 0.2, 0.3]
    mock_provider.generate_embedding.assert_called_once_with(text)


@pytest.mark.asyncio
async def test_embed_texts_batch(embedding_service, mock_provider):
    """Test batch embedding for texts."""
    texts = ["Query 1", "Query 2", "Query 3"]

    results = await embedding_service.embed_texts_batch(texts)

    assert len(results) == 3
    mock_provider.generate_embeddings_batch.assert_called_once_with(texts)


def test_generate_cache_key():
    """Test cache key generation."""
    text = "Test text for caching"

    key1 = EmbeddingService.generate_cache_key(text)
    key2 = EmbeddingService.generate_cache_key(text)

    # Same text should produce same key
    assert key1 == key2
    assert len(key1) == 64  # SHA256 hex digest length

    # Different text should produce different key
    key3 = EmbeddingService.generate_cache_key("Different text")
    assert key1 != key3


def test_truncate_text():
    """Test text truncation."""
    short_text = "Short text"
    result = EmbeddingService.truncate_text(short_text, max_tokens=100)
    assert result == short_text

    long_text = "x" * 40000  # 40k chars, ~10k tokens
    result = EmbeddingService.truncate_text(long_text, max_tokens=8000)
    assert len(result) <= 8000 * 4 + 3  # +3 for "..."
    assert result.endswith("...")


def test_truncate_text_default():
    """Test text truncation with default token limit."""
    very_long_text = "x" * 40000

    result = EmbeddingService.truncate_text(very_long_text)

    # Default is 8000 tokens = 32000 chars
    assert len(result) <= 32003  # +3 for "..."


@pytest.mark.asyncio
async def test_embed_commit_with_many_files(embedding_service, mock_provider):
    """Test that commits with many files truncate file list."""
    commit = CommitMetadata(
        hash="abc123",
        author_name="Test Author",
        author_email="test@example.com",
        timestamp=datetime.now(timezone.utc),
        message="Large refactor",
        files_changed=[f"file{i}.py" for i in range(20)],  # Many files
    )

    await embedding_service.embed_commit(commit)

    call_args = mock_provider.generate_embedding.call_args[0][0]
    # Should only include first 10 files
    assert "file0.py" in call_args
    assert "file9.py" in call_args
    # File 10+ might not be included (truncated at 10)
