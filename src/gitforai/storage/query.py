"""Query interface for semantic search operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from gitforai.llm.embeddings import EmbeddingService
from gitforai.storage.chroma_client import ChromaClient
from gitforai.storage.config import VectorDBConfig
from gitforai.storage.schema import (
    CodeChangeDocument,
    CommitDocument,
    FileVersionDocument,
)

logger = structlog.get_logger(__name__)


class QueryResult:
    """Container for query results with metadata."""

    def __init__(
        self,
        documents: List[Any],
        distances: List[float],
        metadatas: List[dict],
        ids: List[str],
    ) -> None:
        """Initialize query result.

        Args:
            documents: List of document objects
            distances: List of distance scores
            metadatas: List of metadata dicts
            ids: List of document IDs
        """
        self.documents = documents
        self.distances = distances
        self.metadatas = metadatas
        self.ids = ids

    def __len__(self) -> int:
        """Get number of results."""
        return len(self.documents)

    def __iter__(self):
        """Iterate over results."""
        return iter(zip(self.documents, self.distances, self.metadatas, self.ids))


class QueryEngine:
    """High-level interface for semantic search and queries.

    Provides convenient methods for searching commits, file versions,
    and code changes using semantic similarity and metadata filters.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        client: Optional[ChromaClient] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        """Initialize query engine.

        Args:
            config: Vector database configuration
            client: Optional pre-configured ChromaClient
            embedding_service: Optional EmbeddingService for query embeddings.
                              Required for semantic search operations.
        """
        self.config = config or VectorDBConfig()
        self.client = client or ChromaClient(self.config)
        self.embedding_service = embedding_service

        # Collection names
        self.commits_collection_name = self.config.get_collection_name("commits")
        self.file_versions_collection_name = self.config.get_collection_name(
            "file_versions"
        )
        self.code_changes_collection_name = self.config.get_collection_name(
            "code_changes"
        )

    # ============================================================================
    # Commits Collection Queries
    # ============================================================================

    async def search_commits(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Search commits by semantic similarity to query text.

        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters (e.g., {"author": "john"})

        Returns:
            QueryResult with matching commits

        Example:
            >>> engine = QueryEngine()
            >>> results = await engine.search_commits("bug fixes in authentication")
            >>> for commit, distance, metadata, id in results:
            ...     print(f"{id}: {metadata['message']} (score: {distance})")
        """
        if not self.embedding_service:
            raise ValueError("EmbeddingService required for semantic search")

        try:
            collection = self.client.client.get_collection(
                name=self.commits_collection_name
            )
        except ValueError:
            logger.warning("commits_collection_not_found")
            return QueryResult([], [], [], [])

        # Generate embedding for query
        query_embedding = await self.embedding_service.embed_text(query)

        # Build where clause from filters
        where = self._build_where_clause(filters) if filters else None

        # Execute query
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        logger.info(
            "commits_query_executed",
            query=query,
            n_results=len(results["ids"][0]),
            filters=filters,
        )

        return QueryResult(
            documents=results["documents"][0],
            distances=results["distances"][0],
            metadatas=results["metadatas"][0],
            ids=results["ids"][0],
        )

    def find_similar_commits(
        self,
        commit_hash: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Find commits similar to a given commit.

        Args:
            commit_hash: Commit hash to find similar commits for
            n_results: Number of results to return
            filters: Optional metadata filters

        Returns:
            QueryResult with similar commits
        """
        try:
            collection = self.client.client.get_collection(
                name=self.commits_collection_name
            )

            # Get the source commit's embedding
            source = collection.get(ids=[commit_hash], include=["embeddings"])

            if not source["ids"]:
                logger.warning("source_commit_not_found", commit_hash=commit_hash)
                return QueryResult([], [], [], [])

            embedding = source["embeddings"][0]

            # Build where clause
            where = self._build_where_clause(filters) if filters else None

            # Query for similar commits
            results = collection.query(
                query_embeddings=[embedding],
                n_results=n_results + 1,  # +1 to exclude self
                where=where,
            )

            # Remove the source commit from results
            filtered_results = self._filter_self_from_results(
                results, commit_hash
            )

            logger.info(
                "similar_commits_found",
                commit_hash=commit_hash,
                n_results=len(filtered_results["ids"][0]),
            )

            return QueryResult(
                documents=filtered_results["documents"][0],
                distances=filtered_results["distances"][0],
                metadatas=filtered_results["metadatas"][0],
                ids=filtered_results["ids"][0],
            )

        except ValueError:
            logger.warning("commits_collection_not_found")
            return QueryResult([], [], [], [])

    def search_by_intent(
        self,
        intent: str,
        n_results: int = 10,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Search commits by intent category.

        Args:
            intent: Intent type (bug_fix, feature, refactor, docs, test, etc.)
            n_results: Number of results to return
            additional_filters: Optional additional metadata filters

        Returns:
            QueryResult with matching commits
        """
        filters = {"intent": intent}
        if additional_filters:
            filters.update(additional_filters)

        # For intent search, we don't need semantic search, just metadata filtering
        try:
            collection = self.client.client.get_collection(
                name=self.commits_collection_name
            )

            where = self._build_where_clause(filters)

            # Get all matching documents (no semantic ranking)
            results = collection.get(
                where=where,
                limit=n_results,
                include=["metadatas", "documents", "embeddings"],
            )

            logger.info(
                "intent_search_executed",
                intent=intent,
                n_results=len(results["ids"]),
            )

            # Convert to QueryResult format (no distances for metadata-only query)
            distances = [0.0] * len(results["ids"])

            return QueryResult(
                documents=results["documents"],
                distances=distances,
                metadatas=results["metadatas"],
                ids=results["ids"],
            )

        except ValueError:
            logger.warning("commits_collection_not_found")
            return QueryResult([], [], [], [])

    def search_by_author(
        self,
        author: str,
        n_results: int = 10,
        query: Optional[str] = None,
    ) -> QueryResult:
        """Search commits by author.

        Args:
            author: Author name or email
            n_results: Number of results to return
            query: Optional semantic query to rank results

        Returns:
            QueryResult with matching commits
        """
        # Check if searching by email or name
        if "@" in author:
            filters = {"author_email": author}
        else:
            filters = {"author": author}

        if query:
            # Semantic search with author filter
            return self.search_commits(query, n_results, filters)
        else:
            # Metadata-only search
            try:
                collection = self.client.client.get_collection(
                    name=self.commits_collection_name
                )

                where = self._build_where_clause(filters)

                results = collection.get(
                    where=where,
                    limit=n_results,
                    include=["metadatas", "documents", "embeddings"],
                )

                logger.info(
                    "author_search_executed",
                    author=author,
                    n_results=len(results["ids"]),
                )

                distances = [0.0] * len(results["ids"])

                return QueryResult(
                    documents=results["documents"],
                    distances=distances,
                    metadatas=results["metadatas"],
                    ids=results["ids"],
                )

            except ValueError:
                logger.warning("commits_collection_not_found")
                return QueryResult([], [], [], [])

    def search_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        query: Optional[str] = None,
        n_results: int = 10,
    ) -> QueryResult:
        """Search commits within a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            query: Optional semantic query to rank results
            n_results: Number of results to return

        Returns:
            QueryResult with matching commits
        """
        filters = {
            "timestamp": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat(),
            }
        }

        if query:
            return self.search_commits(query, n_results, filters)
        else:
            # Metadata-only search
            try:
                collection = self.client.client.get_collection(
                    name=self.commits_collection_name
                )

                where = self._build_where_clause(filters)

                results = collection.get(
                    where=where,
                    limit=n_results,
                    include=["metadatas", "documents", "embeddings"],
                )

                logger.info(
                    "date_range_search_executed",
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    n_results=len(results["ids"]),
                )

                distances = [0.0] * len(results["ids"])

                return QueryResult(
                    documents=results["documents"],
                    distances=distances,
                    metadatas=results["metadatas"],
                    ids=results["ids"],
                )

            except ValueError:
                logger.warning("commits_collection_not_found")
                return QueryResult([], [], [], [])

    # ============================================================================
    # File Versions Collection Queries
    # ============================================================================

    def search_file_versions(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Search file versions by semantic similarity.

        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters (e.g., {"language": "python"})

        Returns:
            QueryResult with matching file versions
        """
        try:
            collection = self.client.client.get_collection(
                name=self.file_versions_collection_name
            )
        except ValueError:
            logger.warning("file_versions_collection_not_found")
            return QueryResult([], [], [], [])

        query_embedding = self.embedding_service.embed_text(query)
        where = self._build_where_clause(filters) if filters else None

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        logger.info(
            "file_versions_query_executed",
            query=query,
            n_results=len(results["ids"][0]),
        )

        return QueryResult(
            documents=results["documents"][0],
            distances=results["distances"][0],
            metadatas=results["metadatas"][0],
            ids=results["ids"][0],
        )

    def get_file_history(
        self,
        file_path: str,
        n_results: int = 50,
    ) -> QueryResult:
        """Get evolution history of a specific file.

        Args:
            file_path: Path to the file
            n_results: Maximum number of versions to return

        Returns:
            QueryResult with file versions sorted by timestamp
        """
        filters = {"file_path": file_path}

        try:
            collection = self.client.client.get_collection(
                name=self.file_versions_collection_name
            )

            where = self._build_where_clause(filters)

            results = collection.get(
                where=where,
                limit=n_results,
                include=["metadatas", "documents", "embeddings"],
            )

            # Sort by timestamp (most recent first)
            if results["metadatas"]:
                sorted_indices = sorted(
                    range(len(results["metadatas"])),
                    key=lambda i: results["metadatas"][i].get("timestamp", ""),
                    reverse=True,
                )

                results = {
                    "ids": [results["ids"][i] for i in sorted_indices],
                    "documents": [results["documents"][i] for i in sorted_indices],
                    "metadatas": [results["metadatas"][i] for i in sorted_indices],
                }

            logger.info(
                "file_history_retrieved",
                file_path=file_path,
                n_versions=len(results["ids"]),
            )

            distances = [0.0] * len(results["ids"])

            return QueryResult(
                documents=results["documents"],
                distances=distances,
                metadatas=results["metadatas"],
                ids=results["ids"],
            )

        except ValueError:
            logger.warning("file_versions_collection_not_found")
            return QueryResult([], [], [], [])

    # ============================================================================
    # Code Changes Collection Queries
    # ============================================================================

    def search_code_changes(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Search code changes by semantic similarity.

        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters

        Returns:
            QueryResult with matching code changes
        """
        try:
            collection = self.client.client.get_collection(
                name=self.code_changes_collection_name
            )
        except ValueError:
            logger.warning("code_changes_collection_not_found")
            return QueryResult([], [], [], [])

        query_embedding = self.embedding_service.embed_text(query)
        where = self._build_where_clause(filters) if filters else None

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        logger.info(
            "code_changes_query_executed",
            query=query,
            n_results=len(results["ids"][0]),
        )

        return QueryResult(
            documents=results["documents"][0],
            distances=results["distances"][0],
            metadatas=results["metadatas"][0],
            ids=results["ids"][0],
        )

    def get_file_changes(
        self,
        file_path: str,
        n_results: int = 50,
    ) -> QueryResult:
        """Get all changes for a specific file.

        Args:
            file_path: Path to the file
            n_results: Maximum number of changes to return

        Returns:
            QueryResult with code changes sorted by timestamp
        """
        filters = {"file_path": file_path}

        try:
            collection = self.client.client.get_collection(
                name=self.code_changes_collection_name
            )

            where = self._build_where_clause(filters)

            results = collection.get(
                where=where,
                limit=n_results,
                include=["metadatas", "documents", "embeddings"],
            )

            # Sort by timestamp (most recent first)
            if results["metadatas"]:
                sorted_indices = sorted(
                    range(len(results["metadatas"])),
                    key=lambda i: results["metadatas"][i].get("timestamp", ""),
                    reverse=True,
                )

                results = {
                    "ids": [results["ids"][i] for i in sorted_indices],
                    "documents": [results["documents"][i] for i in sorted_indices],
                    "metadatas": [results["metadatas"][i] for i in sorted_indices],
                }

            logger.info(
                "file_changes_retrieved",
                file_path=file_path,
                n_changes=len(results["ids"]),
            )

            distances = [0.0] * len(results["ids"])

            return QueryResult(
                documents=results["documents"],
                distances=distances,
                metadatas=results["metadatas"],
                ids=results["ids"],
            )

        except ValueError:
            logger.warning("code_changes_collection_not_found")
            return QueryResult([], [], [], [])

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters.

        Args:
            filters: Dictionary of filter conditions

        Returns:
            ChromaDB where clause
        """
        # ChromaDB supports direct metadata filtering
        # Handle nested operators like {"timestamp": {"$gte": "...", "$lte": "..."}}
        return filters

    def _filter_self_from_results(
        self, results: dict, exclude_id: str
    ) -> dict:
        """Filter out a specific ID from results.

        Args:
            results: ChromaDB query results
            exclude_id: ID to exclude

        Returns:
            Filtered results
        """
        filtered = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        for i, result_id in enumerate(results["ids"][0]):
            if result_id != exclude_id:
                filtered["ids"][0].append(result_id)
                filtered["documents"][0].append(results["documents"][0][i])
                filtered["metadatas"][0].append(results["metadatas"][0][i])
                filtered["distances"][0].append(results["distances"][0][i])

        return filtered

    def close(self) -> None:
        """Close the query engine and underlying client."""
        self.client.close()

    def __enter__(self) -> "QueryEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
