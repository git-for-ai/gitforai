"""Tests for semantic processor."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gitforai.llm.processor import SemanticProcessor
from gitforai.models import CommitMetadata, FileDiff


@pytest.fixture
def mock_provider():
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.model = "gpt-4-turbo-preview"
    provider.complete = AsyncMock(return_value="Test response")
    provider.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    provider.generate_embeddings_batch = AsyncMock(
        return_value=[[0.1, 0.2], [0.3, 0.4]]
    )
    provider.get_usage_stats = MagicMock(
        return_value={
            "total_tokens": {"input": 100, "output": 50},
            "total_cost": 0.01,
        }
    )
    return provider


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def processor_with_cache(mock_provider, temp_cache_dir):
    """Create processor with caching enabled."""
    return SemanticProcessor(mock_provider, cache_dir=temp_cache_dir, use_cache=True)


@pytest.fixture
def processor_no_cache(mock_provider):
    """Create processor without caching."""
    return SemanticProcessor(mock_provider, use_cache=False)


@pytest.fixture
def sample_commit():
    """Create sample commit."""
    return CommitMetadata(
        hash="abc123",
        author_name="Test Author",
        author_email="test@example.com",
        timestamp=datetime.now(timezone.utc),
        message="Add new feature",
        files_changed=["src/feature.py"],
    )


@pytest.fixture
def sample_diff():
    """Create sample diff."""
    return FileDiff(
        file_path="src/feature.py",
        change_type="modified",
        additions=10,
        deletions=5,
        diff_text="+def new_feature():\n+    pass",
        is_binary=False,
    )


@pytest.mark.asyncio
async def test_enrich_commit_basic(processor_no_cache, mock_provider, sample_commit):
    """Test basic commit enrichment."""
    # Mock different responses for different prompts
    mock_provider.complete.side_effect = [
        "feature",  # intent
        "authentication, security",  # topics
        "Adds user authentication",  # summary
    ]

    result = await processor_no_cache.enrich_commit(sample_commit, include_embedding=False)

    assert result.intent == "feature"
    assert result.topics == ["authentication", "security"]
    assert result.llm_summary == "Adds user authentication"
    assert result.embedding is None
    assert mock_provider.complete.call_count == 3


@pytest.mark.asyncio
async def test_enrich_commit_with_embedding(processor_no_cache, mock_provider, sample_commit):
    """Test commit enrichment with embedding."""
    mock_provider.complete.side_effect = ["feature", "api", "Summary"]

    result = await processor_no_cache.enrich_commit(sample_commit, include_embedding=True)

    assert result.embedding == [0.1, 0.2, 0.3]
    mock_provider.generate_embedding.assert_called_once()


@pytest.mark.asyncio
async def test_enrich_commit_caching(processor_with_cache, mock_provider, sample_commit):
    """Test that enrichment results are cached."""
    mock_provider.complete.side_effect = [
        "feature", "api", "Summary",
        "feature", "api", "Summary",  # Second call should not reach here
    ]

    # First enrichment
    result1 = await processor_with_cache.enrich_commit(sample_commit, include_embedding=False)

    # Second enrichment of same commit
    result2 = await processor_with_cache.enrich_commit(sample_commit, include_embedding=False)

    assert result1.intent == result2.intent
    assert result1.topics == result2.topics
    assert result1.llm_summary == result2.llm_summary

    # Should have used cache, so only 3 API calls (not 6)
    assert mock_provider.complete.call_count == 3


@pytest.mark.asyncio
async def test_enrich_commits_batch(processor_no_cache, mock_provider):
    """Test batch commit enrichment."""
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

    # Mock responses for all commits
    mock_provider.complete.side_effect = [
        "feature", "api", "Summary 1",
        "bug_fix", "auth", "Summary 2",
        "refactor", "code", "Summary 3",
    ]

    results = await processor_no_cache.enrich_commits_batch(
        commits,
        batch_size=3,
        include_embeddings=False,
    )

    assert len(results) == 3
    assert results[0].intent == "feature"
    assert results[1].intent == "bug_fix"
    assert results[2].intent == "refactor"


@pytest.mark.asyncio
async def test_enrich_commits_batch_with_errors(processor_no_cache, mock_provider):
    """Test batch enrichment handles errors gracefully."""
    commits = [
        CommitMetadata(
            hash=f"hash{i}",
            author_name="Author",
            author_email="author@example.com",
            timestamp=datetime.now(timezone.utc),
            message=f"Commit {i}",
            files_changed=[],
        )
        for i in range(2)
    ]

    # First commit succeeds, second fails
    async def mock_complete_with_error(prompt, **kwargs):
        if mock_provider.complete.call_count <= 3:
            return "Success"
        raise Exception("API Error")

    mock_provider.complete = AsyncMock(side_effect=mock_complete_with_error)

    results = await processor_no_cache.enrich_commits_batch(commits, batch_size=2)

    # Should return 2 results, even if one failed
    assert len(results) == 2


@pytest.mark.asyncio
async def test_analyze_diff(processor_no_cache, mock_provider, sample_diff):
    """Test diff analysis."""
    mock_provider.complete.side_effect = [
        "Added new feature function",
        "To implement new functionality",
    ]

    result = await processor_no_cache.analyze_diff(
        sample_diff,
        commit_message="Add feature",
    )

    assert result.file_path == "src/feature.py"
    assert result.change_type == "modified"
    assert result.summary == "Added new feature function"
    assert result.reasoning == "To implement new functionality"


@pytest.mark.asyncio
async def test_analyze_diff_binary(processor_no_cache, mock_provider):
    """Test analysis of binary diff."""
    binary_diff = FileDiff(
        file_path="image.png",
        change_type="added",
        additions=0,
        deletions=0,
        diff_text=None,
        is_binary=True,
    )

    result = await processor_no_cache.analyze_diff(binary_diff)

    assert result.summary == "Binary file added"
    assert mock_provider.complete.call_count == 0  # Should not call API for binary


@pytest.mark.asyncio
async def test_analyze_diffs_batch(processor_no_cache, mock_provider):
    """Test batch diff analysis."""
    diffs = [
        FileDiff(
            file_path=f"file{i}.py",
            change_type="modified",
            additions=5,
            deletions=2,
            diff_text="+code",
            is_binary=False,
        )
        for i in range(3)
    ]

    mock_provider.complete.side_effect = [
        "Summary 1", "Reasoning 1",
        "Summary 2", "Reasoning 2",
        "Summary 3", "Reasoning 3",
    ]

    results = await processor_no_cache.analyze_diffs_batch(
        diffs,
        commit_message="Update files",
        batch_size=3,
    )

    assert len(results) == 3
    assert all(r.summary for r in results)


@pytest.mark.asyncio
async def test_generate_embeddings_for_commits(processor_no_cache, mock_provider):
    """Test generating embeddings for commits."""
    commits = [
        CommitMetadata(
            hash="hash1",
            author_name="Author",
            author_email="author@example.com",
            timestamp=datetime.now(timezone.utc),
            message="Commit 1",
            files_changed=[],
            intent="feature",
            llm_summary="Summary 1",
        ),
        CommitMetadata(
            hash="hash2",
            author_name="Author",
            author_email="author@example.com",
            timestamp=datetime.now(timezone.utc),
            message="Commit 2",
            files_changed=[],
            intent="bug_fix",
            llm_summary="Summary 2",
        ),
    ]

    mock_provider.generate_embeddings_batch = AsyncMock(
        return_value=[[0.1, 0.2], [0.3, 0.4]]
    )

    results = await processor_no_cache.generate_embeddings_for_commits(commits)

    assert len(results) == 2
    assert results[0].embedding == [0.1, 0.2]
    assert results[1].embedding == [0.3, 0.4]


@pytest.mark.asyncio
async def test_generate_embeddings_cached(processor_with_cache, mock_provider):
    """Test embedding generation uses cache."""
    commit = CommitMetadata(
        hash="hash1",
        author_name="Author",
        author_email="author@example.com",
        timestamp=datetime.now(timezone.utc),
        message="Commit",
        files_changed=[],
    )

    # First call
    await processor_with_cache.generate_embeddings_for_commits([commit])

    # Second call should use cache
    await processor_with_cache.generate_embeddings_for_commits([commit])

    # Should only call generate_embeddings_batch once
    assert mock_provider.generate_embeddings_batch.call_count == 1


def test_get_stats(processor_with_cache):
    """Test getting processor statistics."""
    stats = processor_with_cache.get_stats()

    assert "cache" in stats
    assert "provider" in stats

    cache_stats = stats["cache"]
    assert "hits" in cache_stats
    assert "misses" in cache_stats


def test_get_stats_no_cache(processor_no_cache):
    """Test getting stats without cache."""
    stats = processor_no_cache.get_stats()

    assert "cache" not in stats
    assert "provider" in stats


def test_clear_cache(processor_with_cache, temp_cache_dir):
    """Test clearing the cache."""
    # Add some data to cache
    processor_with_cache.cache.set_completion("test", "model", "result")

    # Verify file exists
    files = list(temp_cache_dir.rglob("*.json"))
    assert len(files) > 0

    # Clear cache
    processor_with_cache.clear_cache()

    # Verify files deleted
    files = list(temp_cache_dir.rglob("*.json"))
    assert len(files) == 0


def test_processor_initialization_default_cache():
    """Test processor initializes with default cache directory."""
    with patch("gitforai.llm.processor.LLMCache"):
        provider = MagicMock()
        processor = SemanticProcessor(provider, use_cache=True)

        assert processor.use_cache is True
        assert processor.cache is not None


def test_processor_initialization_no_cache():
    """Test processor initializes without cache."""
    provider = MagicMock()
    processor = SemanticProcessor(provider, use_cache=False)

    assert processor.use_cache is False
    assert processor.cache is None


@pytest.mark.asyncio
async def test_topic_extraction_single_topic(processor_no_cache, mock_provider, sample_commit):
    """Test topic extraction with single topic."""
    mock_provider.complete.side_effect = [
        "feature",
        "authentication",  # Single topic
        "Summary",
    ]

    result = await processor_no_cache.enrich_commit(sample_commit, include_embedding=False)

    assert result.topics == ["authentication"]


@pytest.mark.asyncio
async def test_topic_extraction_empty(processor_no_cache, mock_provider, sample_commit):
    """Test topic extraction with empty response."""
    mock_provider.complete.side_effect = [
        "feature",
        "",  # Empty topics
        "Summary",
    ]

    result = await processor_no_cache.enrich_commit(sample_commit, include_embedding=False)

    assert result.topics == []
