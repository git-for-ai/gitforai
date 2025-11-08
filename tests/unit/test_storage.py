"""Tests for vector database storage layer."""

from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from gitforai.storage.chroma_client import ChromaClient
from gitforai.storage.config import VectorDBConfig
from gitforai.storage.query import QueryEngine, QueryResult
from gitforai.storage.schema import (
    CodeChangeDocument,
    CommitDocument,
    FileVersionDocument,
)
from gitforai.storage.vector_store import VectorStore


# ============================================================================
# Configuration Tests
# ============================================================================


class TestVectorDBConfig:
    """Tests for VectorDBConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VectorDBConfig()

        assert config.provider == "chroma"
        assert config.persist_dir == Path("./.gitforai/vectordb")
        assert config.collection_prefix == "gitforai"
        assert config.embedding_dimension == 384
        assert config.distance_metric == "cosine"
        assert config.batch_size == 100
        assert config.host is None
        assert config.port is None

    def test_get_collection_name(self):
        """Test collection name generation with prefix."""
        config = VectorDBConfig()

        assert config.get_collection_name("commits") == "gitforai_commits"
        assert config.get_collection_name("file_versions") == "gitforai_file_versions"
        assert config.get_collection_name("code_changes") == "gitforai_code_changes"

    def test_custom_prefix(self):
        """Test custom collection prefix."""
        config = VectorDBConfig(collection_prefix="custom")

        assert config.get_collection_name("commits") == "custom_commits"

    def test_ensure_persist_dir(self, tmp_path):
        """Test persistence directory creation."""
        persist_dir = tmp_path / "vectordb"
        config = VectorDBConfig(persist_dir=persist_dir)

        assert not persist_dir.exists()

        config.ensure_persist_dir()

        assert persist_dir.exists()
        assert persist_dir.is_dir()


# ============================================================================
# Schema Tests
# ============================================================================


class TestCommitDocument:
    """Tests for CommitDocument schema."""

    def test_create_commit_document(self):
        """Test creating a commit document."""
        commit = CommitDocument(
            id="abc123",
            embedding=[0.1, 0.2, 0.3],
            author="John Doe",
            author_email="john@example.com",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            message="Initial commit",
            summary="Added initial project structure",
            intent="feature",
            topics=["setup", "initialization"],
            files_changed=["README.md", "main.py"],
            num_files_changed=2,
            num_lines_added=100,
            num_lines_deleted=0,
            parent_hashes=["parent123"],
            is_merge=False,
            branch="main",
            repo_path="/path/to/repo",
        )

        assert commit.id == "abc123"
        assert commit.author == "John Doe"
        assert commit.message == "Initial commit"
        assert commit.summary == "Added initial project structure"
        assert commit.intent == "feature"
        assert len(commit.topics) == 2
        assert len(commit.files_changed) == 2

    def test_commit_to_chroma_format(self):
        """Test converting commit to ChromaDB format."""
        commit = CommitDocument(
            id="abc123",
            embedding=[0.1, 0.2, 0.3],
            author="John Doe",
            author_email="john@example.com",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            message="Fix bug",
            summary="Fixed authentication bug",
            intent="bug_fix",
            topics=["auth", "security"],
            files_changed=["auth.py"],
            num_files_changed=1,
            num_lines_added=10,
            num_lines_deleted=5,
            parent_hashes=["parent123"],
            is_merge=False,
        )

        chroma_data = commit.to_chroma_format()

        assert chroma_data["ids"] == ["abc123"]
        assert chroma_data["embeddings"] == [[0.1, 0.2, 0.3]]
        assert len(chroma_data["metadatas"]) == 1
        assert len(chroma_data["documents"]) == 1

        metadata = chroma_data["metadatas"][0]
        assert metadata["author"] == "John Doe"
        assert metadata["author_email"] == "john@example.com"
        assert metadata["message"] == "Fix bug"
        assert metadata["summary"] == "Fixed authentication bug"
        assert metadata["intent"] == "bug_fix"
        assert metadata["topics"] == "auth,security"
        assert metadata["files_changed"] == "auth.py"
        assert metadata["num_files_changed"] == 1
        assert metadata["num_lines_added"] == 10
        assert metadata["num_lines_deleted"] == 5
        assert metadata["is_merge"] is False

        document = chroma_data["documents"][0]
        assert "Fixed authentication bug" in document
        assert "Fix bug" in document


class TestFileVersionDocument:
    """Tests for FileVersionDocument schema."""

    def test_create_id(self):
        """Test unique ID creation."""
        id = FileVersionDocument.create_id("src/main.py", "abc123")

        assert id == "src/main.py:abc123"

    def test_file_version_to_chroma_format(self):
        """Test converting file version to ChromaDB format."""
        file_version = FileVersionDocument(
            id="src/main.py:abc123",
            embedding=[0.1, 0.2],
            file_path="src/main.py",
            commit_hash="abc123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            size=1024,
            language="python",
            file_type="py",
            num_lines=50,
            num_functions=5,
            num_classes=2,
            preview="import os\nimport sys\n...",
        )

        chroma_data = file_version.to_chroma_format()

        assert chroma_data["ids"] == ["src/main.py:abc123"]
        assert len(chroma_data["metadatas"]) == 1

        metadata = chroma_data["metadatas"][0]
        assert metadata["file_path"] == "src/main.py"
        assert metadata["commit_hash"] == "abc123"
        assert metadata["size"] == 1024
        assert metadata["language"] == "python"
        assert metadata["file_type"] == "py"
        assert metadata["num_lines"] == 50
        assert metadata["num_functions"] == 5
        assert metadata["num_classes"] == 2

        document = chroma_data["documents"][0]
        assert "src/main.py" in document
        assert "import os" in document


class TestCodeChangeDocument:
    """Tests for CodeChangeDocument schema."""

    def test_create_id(self):
        """Test unique ID creation."""
        id1 = CodeChangeDocument.create_id("abc123", "src/main.py", 0)
        id2 = CodeChangeDocument.create_id("abc123", "src/main.py", 1)

        assert id1.startswith("change_")
        assert id2.startswith("change_")
        assert id1 != id2  # Different indices should produce different IDs

    def test_code_change_to_chroma_format(self):
        """Test converting code change to ChromaDB format."""
        code_change = CodeChangeDocument(
            id="change_123abc",
            embedding=[0.1, 0.2],
            commit_hash="abc123",
            file_path="src/auth.py",
            change_type="modified",
            lines_added=10,
            lines_deleted=5,
            start_line=42,
            end_line=52,
            summary="Fixed password validation",
            context="User authentication module",
            intent="bug_fix",
            diff_preview="- old validation\n+ new validation",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        chroma_data = code_change.to_chroma_format()

        assert chroma_data["ids"] == ["change_123abc"]
        assert len(chroma_data["metadatas"]) == 1

        metadata = chroma_data["metadatas"][0]
        assert metadata["commit_hash"] == "abc123"
        assert metadata["file_path"] == "src/auth.py"
        assert metadata["change_type"] == "modified"
        assert metadata["lines_added"] == 10
        assert metadata["lines_deleted"] == 5
        assert metadata["start_line"] == 42
        assert metadata["end_line"] == 52
        assert metadata["summary"] == "Fixed password validation"
        assert metadata["intent"] == "bug_fix"

        document = chroma_data["documents"][0]
        assert "src/auth.py" in document
        assert "Fixed password validation" in document
        assert "old validation" in document


# ============================================================================
# ChromaClient Tests
# ============================================================================


class TestChromaClient:
    """Tests for ChromaClient."""

    @patch("gitforai.storage.chroma_client.chromadb.PersistentClient")
    def test_create_persistent_client(self, mock_persistent_client, tmp_path):
        """Test creating persistent client."""
        config = VectorDBConfig(persist_dir=tmp_path / "vectordb")
        client = ChromaClient(config)

        # Access client property to trigger lazy initialization
        _ = client.client

        mock_persistent_client.assert_called_once()

    @patch("gitforai.storage.chroma_client.chromadb.HttpClient")
    def test_create_http_client(self, mock_http_client):
        """Test creating HTTP client."""
        config = VectorDBConfig(host="localhost", port=8000)
        client = ChromaClient(config)

        # Access client property to trigger lazy initialization
        _ = client.client

        mock_http_client.assert_called_once_with(host="localhost", port=8000)

    def test_lazy_initialization(self, tmp_path):
        """Test that client is created lazily."""
        config = VectorDBConfig(persist_dir=tmp_path / "vectordb")
        client = ChromaClient(config)

        assert client._client is None

        # Access client property
        _ = client.client

        assert client._client is not None

    @patch("gitforai.storage.chroma_client.chromadb.PersistentClient")
    def test_get_or_create_collection(self, mock_persistent_client, tmp_path):
        """Test getting or creating a collection."""
        mock_client_instance = Mock()
        mock_persistent_client.return_value = mock_client_instance

        config = VectorDBConfig(persist_dir=tmp_path / "vectordb")
        client = ChromaClient(config)

        client.get_or_create_collection("test_collection", {"key": "value"})

        mock_client_instance.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"key": "value"},
        )

    @patch("gitforai.storage.chroma_client.chromadb.PersistentClient")
    def test_context_manager(self, mock_persistent_client, tmp_path):
        """Test context manager support."""
        config = VectorDBConfig(persist_dir=tmp_path / "vectordb")

        with ChromaClient(config) as client:
            assert client is not None

        # close() should have been called
        # (currently a no-op, but structure is in place)


# ============================================================================
# VectorStore Tests
# ============================================================================


class TestVectorStore:
    """Tests for VectorStore."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock ChromaClient."""
        client = Mock(spec=ChromaClient)
        client.client = Mock()
        return client

    @pytest.fixture
    def mock_collection(self):
        """Create a mock collection."""
        collection = Mock()
        collection.add = Mock()
        collection.upsert = Mock()
        collection.delete = Mock()
        collection.count = Mock(return_value=10)
        return collection

    @pytest.fixture
    def vector_store(self, mock_client):
        """Create a VectorStore with mock client."""
        return VectorStore(client=mock_client)

    def test_insert_commits_empty(self, vector_store):
        """Test inserting empty commit list."""
        result = vector_store.insert_commits([])

        assert result == 0

    def test_insert_commits(self, vector_store, mock_client, mock_collection):
        """Test inserting commits."""
        mock_client.get_or_create_collection.return_value = mock_collection

        commits = [
            CommitDocument(
                id="abc123",
                embedding=[0.1, 0.2],
                author="John Doe",
                author_email="john@example.com",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                message="Test commit",
                num_files_changed=1,
                num_lines_added=10,
                num_lines_deleted=0,
            )
        ]

        result = vector_store.insert_commits(commits)

        assert result == 1
        mock_collection.add.assert_called_once()

    def test_insert_commits_batching(self, vector_store, mock_client, mock_collection):
        """Test commit insertion with batching."""
        mock_client.get_or_create_collection.return_value = mock_collection

        # Create 250 commits (should be 3 batches of 100)
        commits = [
            CommitDocument(
                id=f"commit{i}",
                embedding=[0.1, 0.2],
                author="John Doe",
                author_email="john@example.com",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                message=f"Commit {i}",
                num_files_changed=1,
                num_lines_added=10,
                num_lines_deleted=0,
            )
            for i in range(250)
        ]

        result = vector_store.insert_commits(commits)

        assert result == 250
        # Should have called add 3 times (3 batches)
        assert mock_collection.add.call_count == 3

    def test_upsert_commits(self, vector_store, mock_client, mock_collection):
        """Test upserting commits."""
        mock_client.get_or_create_collection.return_value = mock_collection

        commits = [
            CommitDocument(
                id="abc123",
                embedding=[0.1, 0.2],
                author="John Doe",
                author_email="john@example.com",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                message="Test commit",
                num_files_changed=1,
                num_lines_added=10,
                num_lines_deleted=0,
            )
        ]

        result = vector_store.upsert_commits(commits)

        assert result == 1
        mock_collection.upsert.assert_called_once()

    def test_delete_commits(self, vector_store, mock_client):
        """Test deleting commits."""
        mock_collection = Mock()
        mock_client.client.get_collection.return_value = mock_collection

        result = vector_store.delete_commits(["abc123", "def456"])

        assert result == 2
        mock_collection.delete.assert_called_once_with(ids=["abc123", "def456"])

    def test_delete_commits_collection_not_found(self, vector_store, mock_client):
        """Test deleting commits when collection doesn't exist."""
        mock_client.client.get_collection.side_effect = ValueError("Not found")

        result = vector_store.delete_commits(["abc123"])

        assert result == 0

    def test_get_commits_count(self, vector_store, mock_client):
        """Test getting commit count."""
        mock_client.get_collection_count.return_value = 42

        count = vector_store.get_commits_count()

        assert count == 42

    def test_get_stats(self, vector_store, mock_client):
        """Test getting database statistics."""
        mock_client.get_collection_count.return_value = 10

        stats = vector_store.get_stats()

        assert "collections" in stats
        assert "config" in stats
        assert "commits" in stats["collections"]
        assert "file_versions" in stats["collections"]
        assert "code_changes" in stats["collections"]

    def test_context_manager(self, mock_client):
        """Test context manager support."""
        with VectorStore(client=mock_client) as store:
            assert store is not None

        mock_client.close.assert_called_once()


# ============================================================================
# QueryEngine Tests
# ============================================================================


class TestQueryResult:
    """Tests for QueryResult."""

    def test_create_query_result(self):
        """Test creating query result."""
        result = QueryResult(
            documents=["doc1", "doc2"],
            distances=[0.1, 0.2],
            metadatas=[{"key": "value1"}, {"key": "value2"}],
            ids=["id1", "id2"],
        )

        assert len(result) == 2
        assert result.documents == ["doc1", "doc2"]
        assert result.distances == [0.1, 0.2]

    def test_iterate_results(self):
        """Test iterating over results."""
        result = QueryResult(
            documents=["doc1", "doc2"],
            distances=[0.1, 0.2],
            metadatas=[{"key": "value1"}, {"key": "value2"}],
            ids=["id1", "id2"],
        )

        items = list(result)
        assert len(items) == 2
        assert items[0] == ("doc1", 0.1, {"key": "value1"}, "id1")
        assert items[1] == ("doc2", 0.2, {"key": "value2"}, "id2")


class TestQueryEngine:
    """Tests for QueryEngine."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock ChromaClient."""
        client = Mock(spec=ChromaClient)
        client.client = Mock()
        return client

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock EmbeddingService."""
        service = Mock()
        # Make embed_text return a coroutine
        async def mock_embed_text(text):
            return [0.1, 0.2, 0.3]
        service.embed_text = mock_embed_text
        return service

    @pytest.fixture
    def mock_collection(self):
        """Create a mock collection."""
        collection = Mock()
        collection.query = Mock(
            return_value={
                "ids": [["id1", "id2"]],
                "documents": [["doc1", "doc2"]],
                "metadatas": [[{"key": "val1"}, {"key": "val2"}]],
                "distances": [[0.1, 0.2]],
            }
        )
        collection.get = Mock(
            return_value={
                "ids": ["id1", "id2"],
                "documents": ["doc1", "doc2"],
                "metadatas": [{"key": "val1"}, {"key": "val2"}],
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            }
        )
        return collection

    @pytest.fixture
    def query_engine(self, mock_client, mock_embedding_service):
        """Create a QueryEngine with mocks."""
        return QueryEngine(
            client=mock_client,
            embedding_service=mock_embedding_service,
        )

    async def test_search_commits(self, query_engine, mock_client, mock_collection, mock_embedding_service):
        """Test searching commits."""
        mock_client.client.get_collection.return_value = mock_collection

        result = await query_engine.search_commits("bug fix", n_results=5)

        assert len(result) == 2
        assert result.ids == ["id1", "id2"]
        mock_collection.query.assert_called_once()

    async def test_search_commits_with_filters(self, query_engine, mock_client, mock_collection):
        """Test searching commits with filters."""
        mock_client.client.get_collection.return_value = mock_collection

        filters = {"author": "John Doe"}
        result = await query_engine.search_commits("bug fix", n_results=5, filters=filters)

        assert len(result) == 2
        # Verify filters were passed to query
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["where"] == filters

    async def test_search_commits_collection_not_found(self, query_engine, mock_client):
        """Test searching when collection doesn't exist."""
        mock_client.client.get_collection.side_effect = ValueError("Not found")

        result = await query_engine.search_commits("bug fix")

        assert len(result) == 0

    def test_find_similar_commits(self, query_engine, mock_client, mock_collection):
        """Test finding similar commits."""
        # Setup get() to return source commit
        mock_collection.get.return_value = {
            "ids": ["source_id"],
            "embeddings": [[0.5, 0.6, 0.7]],
        }

        # Setup query() to return similar commits including source
        mock_collection.query.return_value = {
            "ids": [["source_id", "similar1", "similar2"]],
            "documents": [["source_doc", "doc1", "doc2"]],
            "metadatas": [[{"k": "v0"}, {"k": "v1"}, {"k": "v2"}]],
            "distances": [[0.0, 0.1, 0.2]],
        }

        mock_client.client.get_collection.return_value = mock_collection

        result = query_engine.find_similar_commits("source_id", n_results=2)

        # Source commit should be filtered out
        assert len(result) == 2
        assert "source_id" not in result.ids

    def test_search_by_intent(self, query_engine, mock_client, mock_collection):
        """Test searching by intent."""
        mock_client.client.get_collection.return_value = mock_collection

        result = query_engine.search_by_intent("bug_fix", n_results=10)

        assert len(result) == 2
        # Verify get() was called with intent filter
        call_kwargs = mock_collection.get.call_args[1]
        assert call_kwargs["where"] == {"intent": "bug_fix"}

    def test_search_by_author_name(self, query_engine, mock_client, mock_collection):
        """Test searching by author name."""
        mock_client.client.get_collection.return_value = mock_collection

        result = query_engine.search_by_author("John Doe", n_results=10)

        assert len(result) == 2
        call_kwargs = mock_collection.get.call_args[1]
        assert call_kwargs["where"] == {"author": "John Doe"}

    def test_search_by_author_email(self, query_engine, mock_client, mock_collection):
        """Test searching by author email."""
        mock_client.client.get_collection.return_value = mock_collection

        result = query_engine.search_by_author("john@example.com", n_results=10)

        assert len(result) == 2
        call_kwargs = mock_collection.get.call_args[1]
        assert call_kwargs["where"] == {"author_email": "john@example.com"}

    def test_search_date_range(self, query_engine, mock_client, mock_collection):
        """Test searching by date range."""
        mock_client.client.get_collection.return_value = mock_collection

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        result = query_engine.search_date_range(start, end, n_results=10)

        assert len(result) == 2
        call_kwargs = mock_collection.get.call_args[1]
        assert "timestamp" in call_kwargs["where"]
        assert "$gte" in call_kwargs["where"]["timestamp"]
        assert "$lte" in call_kwargs["where"]["timestamp"]

    def test_get_file_history(self, query_engine, mock_client, mock_collection):
        """Test getting file history."""
        # Return results with timestamps in different order
        mock_collection.get.return_value = {
            "ids": ["v1", "v2", "v3"],
            "documents": ["old", "middle", "new"],
            "metadatas": [
                {"timestamp": "2024-01-01T00:00:00"},
                {"timestamp": "2024-06-01T00:00:00"},
                {"timestamp": "2024-12-01T00:00:00"},
            ],
            "embeddings": [[0.1], [0.2], [0.3]],
        }

        mock_client.client.get_collection.return_value = mock_collection

        result = query_engine.get_file_history("src/main.py", n_results=50)

        assert len(result) == 3
        # Results should be sorted by timestamp (most recent first)
        assert result.ids[0] == "v3"
        assert result.ids[1] == "v2"
        assert result.ids[2] == "v1"

    def test_context_manager(self, mock_client, mock_embedding_service):
        """Test context manager support."""
        with QueryEngine(client=mock_client, embedding_service=mock_embedding_service) as engine:
            assert engine is not None

        mock_client.close.assert_called_once()
