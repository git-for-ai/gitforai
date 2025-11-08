"""Vector database configuration."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorDBConfig(BaseSettings):
    """Configuration for vector database (Chroma).

    Settings can be loaded from environment variables or .env file.
    All settings are prefixed with VECTORDB_ (e.g., VECTORDB_PROVIDER).
    """

    model_config = SettingsConfigDict(
        env_prefix="VECTORDB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database provider (currently only "chroma" supported)
    provider: str = Field(
        default="chroma",
        description="Vector database provider (chroma, pinecone, weaviate)",
    )

    # Chroma-specific settings
    persist_dir: Path = Field(
        default=Path("./.gitforai/vectordb"),
        description="Directory for Chroma persistent storage",
    )

    collection_prefix: str = Field(
        default="gitforai",
        description="Prefix for collection names (e.g., gitforai_commits)",
    )

    # Embedding dimension (depends on provider used)
    # Local: 384, OpenAI text-embedding-3-small: 1536
    embedding_dimension: int = Field(
        default=384,
        description="Dimension of embeddings (384 for local, 1536 for OpenAI)",
    )

    # Distance metric for similarity search
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric (cosine, l2, ip)",
    )

    # Batch size for operations
    batch_size: int = Field(
        default=100,
        description="Batch size for insert/query operations",
    )

    # Client settings
    host: Optional[str] = Field(
        default=None,
        description="Chroma server host (None for local persistent mode)",
    )

    port: Optional[int] = Field(
        default=None,
        description="Chroma server port (None for local persistent mode)",
    )

    # Collection names
    commits_collection: str = Field(
        default="commits",
        description="Collection name for commits",
    )

    file_versions_collection: str = Field(
        default="file_versions",
        description="Collection name for file versions",
    )

    code_changes_collection: str = Field(
        default="code_changes",
        description="Collection name for code changes",
    )

    def get_collection_name(self, base_name: str) -> str:
        """Get full collection name with prefix.

        Args:
            base_name: Base collection name (e.g., "commits")

        Returns:
            Full collection name (e.g., "gitforai_commits")
        """
        return f"{self.collection_prefix}_{base_name}"

    def ensure_persist_dir(self) -> None:
        """Ensure persistence directory exists."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
