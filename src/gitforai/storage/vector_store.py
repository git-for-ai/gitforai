"""High-level vector store operations for managing git history."""

from typing import List, Optional

import structlog

from gitforai.storage.chroma_client import ChromaClient
from gitforai.storage.config import VectorDBConfig
from gitforai.storage.schema import (
    CodeChangeDocument,
    CommitDocument,
    FileVersionDocument,
)

logger = structlog.get_logger(__name__)


class VectorStore:
    """High-level interface for vector database operations.

    Provides convenient methods for inserting, updating, and querying
    commits, file versions, and code changes in the vector database.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        client: Optional[ChromaClient] = None,
    ) -> None:
        """Initialize vector store.

        Args:
            config: Vector database configuration. If None, loads from environment.
            client: Optional pre-configured ChromaClient. If None, creates new client.
        """
        self.config = config or VectorDBConfig()
        self.client = client or ChromaClient(self.config)

        # Collection names with prefix
        self.commits_collection_name = self.config.get_collection_name("commits")
        self.file_versions_collection_name = self.config.get_collection_name(
            "file_versions"
        )
        self.code_changes_collection_name = self.config.get_collection_name(
            "code_changes"
        )

    # ============================================================================
    # Commits Collection Operations
    # ============================================================================

    def insert_commits(self, commits: List[CommitDocument]) -> int:
        """Insert commits into vector database.

        Args:
            commits: List of commit documents to insert

        Returns:
            Number of commits successfully inserted
        """
        if not commits:
            return 0

        collection = self.client.get_or_create_collection(
            name=self.commits_collection_name,
            metadata={"schema": "CommitDocument"},
        )

        # Process in batches
        batch_size = self.config.batch_size
        inserted = 0

        for i in range(0, len(commits), batch_size):
            batch = commits[i : i + batch_size]

            # Combine batch into single ChromaDB format
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for commit in batch:
                chroma_data = commit.to_chroma_format()
                ids.extend(chroma_data["ids"])
                embeddings.extend(chroma_data["embeddings"])
                metadatas.extend(chroma_data["metadatas"])
                documents.extend(chroma_data["documents"])

            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                inserted += len(batch)
                logger.info(
                    "inserted_commits",
                    batch_size=len(batch),
                    total=inserted,
                    of=len(commits),
                )
            except Exception as e:
                logger.error(
                    "commit_insertion_failed", error=str(e), batch_size=len(batch)
                )
                # Continue with next batch

        return inserted

    def upsert_commits(self, commits: List[CommitDocument]) -> int:
        """Insert or update commits in vector database.

        Args:
            commits: List of commit documents to upsert

        Returns:
            Number of commits successfully upserted
        """
        if not commits:
            return 0

        collection = self.client.get_or_create_collection(
            name=self.commits_collection_name,
            metadata={"schema": "CommitDocument"},
        )

        batch_size = self.config.batch_size
        upserted = 0

        for i in range(0, len(commits), batch_size):
            batch = commits[i : i + batch_size]

            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for commit in batch:
                chroma_data = commit.to_chroma_format()
                ids.extend(chroma_data["ids"])
                embeddings.extend(chroma_data["embeddings"])
                metadatas.extend(chroma_data["metadatas"])
                documents.extend(chroma_data["documents"])

            try:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                upserted += len(batch)
                logger.info(
                    "upserted_commits",
                    batch_size=len(batch),
                    total=upserted,
                    of=len(commits),
                )
            except Exception as e:
                logger.error(
                    "commit_upsert_failed", error=str(e), batch_size=len(batch)
                )

        return upserted

    def delete_commits(self, commit_hashes: List[str]) -> int:
        """Delete commits from vector database.

        Args:
            commit_hashes: List of commit hashes to delete

        Returns:
            Number of commits deleted
        """
        if not commit_hashes:
            return 0

        try:
            collection = self.client.client.get_collection(
                name=self.commits_collection_name
            )
            collection.delete(ids=commit_hashes)
            logger.info("deleted_commits", count=len(commit_hashes))
            return len(commit_hashes)
        except ValueError:
            # Collection doesn't exist
            logger.warning("commits_collection_not_found")
            return 0
        except Exception as e:
            logger.error("commit_deletion_failed", error=str(e))
            return 0

    def get_commits_count(self) -> int:
        """Get total number of commits in database.

        Returns:
            Number of commits
        """
        return self.client.get_collection_count(self.commits_collection_name)

    # ============================================================================
    # File Versions Collection Operations
    # ============================================================================

    def insert_file_versions(self, file_versions: List[FileVersionDocument]) -> int:
        """Insert file versions into vector database.

        Args:
            file_versions: List of file version documents to insert

        Returns:
            Number of file versions successfully inserted
        """
        if not file_versions:
            return 0

        collection = self.client.get_or_create_collection(
            name=self.file_versions_collection_name,
            metadata={"schema": "FileVersionDocument"},
        )

        batch_size = self.config.batch_size
        inserted = 0

        for i in range(0, len(file_versions), batch_size):
            batch = file_versions[i : i + batch_size]

            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for file_version in batch:
                chroma_data = file_version.to_chroma_format()
                ids.extend(chroma_data["ids"])
                embeddings.extend(chroma_data["embeddings"])
                metadatas.extend(chroma_data["metadatas"])
                documents.extend(chroma_data["documents"])

            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                inserted += len(batch)
                logger.info(
                    "inserted_file_versions",
                    batch_size=len(batch),
                    total=inserted,
                    of=len(file_versions),
                )
            except Exception as e:
                logger.error(
                    "file_version_insertion_failed",
                    error=str(e),
                    batch_size=len(batch),
                )

        return inserted

    def upsert_file_versions(self, file_versions: List[FileVersionDocument]) -> int:
        """Insert or update file versions in vector database.

        Args:
            file_versions: List of file version documents to upsert

        Returns:
            Number of file versions successfully upserted
        """
        if not file_versions:
            return 0

        collection = self.client.get_or_create_collection(
            name=self.file_versions_collection_name,
            metadata={"schema": "FileVersionDocument"},
        )

        batch_size = self.config.batch_size
        upserted = 0

        for i in range(0, len(file_versions), batch_size):
            batch = file_versions[i : i + batch_size]

            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for file_version in batch:
                chroma_data = file_version.to_chroma_format()
                ids.extend(chroma_data["ids"])
                embeddings.extend(chroma_data["embeddings"])
                metadatas.extend(chroma_data["metadatas"])
                documents.extend(chroma_data["documents"])

            try:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                upserted += len(batch)
                logger.info(
                    "upserted_file_versions",
                    batch_size=len(batch),
                    total=upserted,
                    of=len(file_versions),
                )
            except Exception as e:
                logger.error(
                    "file_version_upsert_failed", error=str(e), batch_size=len(batch)
                )

        return upserted

    def get_file_versions_count(self) -> int:
        """Get total number of file versions in database.

        Returns:
            Number of file versions
        """
        return self.client.get_collection_count(self.file_versions_collection_name)

    # ============================================================================
    # Code Changes Collection Operations
    # ============================================================================

    def insert_code_changes(self, code_changes: List[CodeChangeDocument]) -> int:
        """Insert code changes into vector database.

        Args:
            code_changes: List of code change documents to insert

        Returns:
            Number of code changes successfully inserted
        """
        if not code_changes:
            return 0

        collection = self.client.get_or_create_collection(
            name=self.code_changes_collection_name,
            metadata={"schema": "CodeChangeDocument"},
        )

        batch_size = self.config.batch_size
        inserted = 0

        for i in range(0, len(code_changes), batch_size):
            batch = code_changes[i : i + batch_size]

            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for code_change in batch:
                chroma_data = code_change.to_chroma_format()
                ids.extend(chroma_data["ids"])
                embeddings.extend(chroma_data["embeddings"])
                metadatas.extend(chroma_data["metadatas"])
                documents.extend(chroma_data["documents"])

            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                inserted += len(batch)
                logger.info(
                    "inserted_code_changes",
                    batch_size=len(batch),
                    total=inserted,
                    of=len(code_changes),
                )
            except Exception as e:
                logger.error(
                    "code_change_insertion_failed",
                    error=str(e),
                    batch_size=len(batch),
                )

        return inserted

    def upsert_code_changes(self, code_changes: List[CodeChangeDocument]) -> int:
        """Insert or update code changes in vector database.

        Args:
            code_changes: List of code change documents to upsert

        Returns:
            Number of code changes successfully upserted
        """
        if not code_changes:
            return 0

        collection = self.client.get_or_create_collection(
            name=self.code_changes_collection_name,
            metadata={"schema": "CodeChangeDocument"},
        )

        batch_size = self.config.batch_size
        upserted = 0

        for i in range(0, len(code_changes), batch_size):
            batch = code_changes[i : i + batch_size]

            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for code_change in batch:
                chroma_data = code_change.to_chroma_format()
                ids.extend(chroma_data["ids"])
                embeddings.extend(chroma_data["embeddings"])
                metadatas.extend(chroma_data["metadatas"])
                documents.extend(chroma_data["documents"])

            try:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                upserted += len(batch)
                logger.info(
                    "upserted_code_changes",
                    batch_size=len(batch),
                    total=upserted,
                    of=len(code_changes),
                )
            except Exception as e:
                logger.error(
                    "code_change_upsert_failed", error=str(e), batch_size=len(batch)
                )

        return upserted

    def get_code_changes_count(self) -> int:
        """Get total number of code changes in database.

        Returns:
            Number of code changes
        """
        return self.client.get_collection_count(self.code_changes_collection_name)

    # ============================================================================
    # General Operations
    # ============================================================================

    def reset_all(self) -> None:
        """Reset all collections (delete all data).

        WARNING: This will delete all data from all collections.
        Use with caution, typically only in testing.
        """
        logger.warning("resetting_all_collections")
        self.client.reset()

    def get_stats(self) -> dict:
        """Get statistics about the vector database.

        Returns:
            Dictionary with collection counts and metadata
        """
        return {
            "collections": {
                "commits": {
                    "name": self.commits_collection_name,
                    "count": self.get_commits_count(),
                },
                "file_versions": {
                    "name": self.file_versions_collection_name,
                    "count": self.get_file_versions_count(),
                },
                "code_changes": {
                    "name": self.code_changes_collection_name,
                    "count": self.get_code_changes_count(),
                },
            },
            "config": {
                "provider": self.config.provider,
                "persist_dir": str(self.config.persist_dir),
                "embedding_dimension": self.config.embedding_dimension,
                "distance_metric": self.config.distance_metric,
                "batch_size": self.config.batch_size,
            },
        }

    def close(self) -> None:
        """Close the vector store and underlying client."""
        self.client.close()

    def __enter__(self) -> "VectorStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
