"""ChromaDB client wrapper for vector database operations."""

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from gitforai.storage.config import VectorDBConfig


class ChromaClient:
    """Wrapper for ChromaDB client with configuration management.

    Provides a clean interface for creating and managing ChromaDB connections
    with support for both persistent local storage and client/server modes.
    """

    def __init__(self, config: Optional[VectorDBConfig] = None) -> None:
        """Initialize ChromaDB client.

        Args:
            config: Vector database configuration. If None, loads from environment.
        """
        self.config = config or VectorDBConfig()
        self._client: Optional[chromadb.ClientAPI] = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Get or create ChromaDB client.

        Lazy initialization - client is created on first access.

        Returns:
            ChromaDB client instance
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> chromadb.ClientAPI:
        """Create ChromaDB client based on configuration.

        Returns:
            ChromaDB client instance

        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.host and self.config.port:
            # Client/server mode
            return chromadb.HttpClient(
                host=self.config.host,
                port=self.config.port,
            )
        else:
            # Persistent local mode
            self.config.ensure_persist_dir()

            settings = ChromaSettings(
                persist_directory=str(self.config.persist_dir),
                anonymized_telemetry=False,  # Disable telemetry for privacy
            )

            return chromadb.PersistentClient(
                path=str(self.config.persist_dir),
                settings=settings,
            )

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[dict] = None,
    ) -> chromadb.Collection:
        """Get existing collection or create new one.

        Args:
            name: Collection name
            metadata: Optional metadata for the collection

        Returns:
            ChromaDB collection
        """
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {},
        )

    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Collection name to delete
        """
        try:
            self.client.delete_collection(name=name)
        except ValueError:
            # Collection doesn't exist, ignore
            pass

    def list_collections(self) -> list[str]:
        """List all collection names.

        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def reset(self) -> None:
        """Reset the client (delete all data).

        WARNING: This will delete all collections and data.
        Use with caution, typically only in testing.
        """
        if self._client is not None:
            self.client.reset()
            self._client = None

    def get_collection_count(self, collection_name: str) -> int:
        """Get count of items in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of items in the collection, or 0 if collection doesn't exist
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()
        except (ValueError, Exception):
            # Collection doesn't exist or other error
            return 0

    def close(self) -> None:
        """Close the client connection.

        For persistent mode, this ensures data is flushed to disk.
        """
        # ChromaDB handles cleanup automatically
        # This method is here for API completeness
        pass

    def __enter__(self) -> "ChromaClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
