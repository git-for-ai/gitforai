"""Configuration models."""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RepositoryConfig(BaseModel):
    """Configuration for a Git repository to process."""

    repo_path: Path = Field(..., description="Path to the Git repository")
    included_extensions: List[str] = Field(
        default_factory=lambda: [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".md"],
        description="File extensions to include",
    )
    excluded_paths: List[str] = Field(
        default_factory=lambda: ["node_modules/", ".git/", "__pycache__/", "venv/", "dist/", "build/"],
        description="Paths to exclude",
    )
    max_file_size_bytes: int = Field(
        default=1_000_000,  # 1MB
        description="Maximum file size to process",
    )
    process_binary_files: bool = Field(False, description="Whether to process binary files")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "repo_path": "/path/to/repo",
                "included_extensions": [".py", ".js", ".md"],
                "excluded_paths": ["node_modules/", ".git/"],
                "max_file_size_bytes": 1000000,
                "process_binary_files": False,
            }
        }


class LLMConfig(BaseModel):
    """Configuration for LLM API."""

    provider: str = Field("openai", description="LLM provider: openai or anthropic")
    model: str = Field("gpt-4-turbo-preview", description="Model name")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model name")
    api_key: Optional[str] = Field(None, description="API key")
    max_tokens: int = Field(2000, description="Maximum tokens for completion")
    temperature: float = Field(0.3, description="Temperature for generation")
    batch_size: int = Field(10, description="Batch size for processing")


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""

    provider: str = Field("chroma", description="Vector DB provider: chroma, pinecone, weaviate")
    persist_directory: Optional[Path] = Field(None, description="Directory for persistent storage")
    collection_name: str = Field("git_history", description="Collection name")
    embedding_dimension: int = Field(1536, description="Dimension of embeddings")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"

    # Vector DB Settings
    vectordb_provider: str = "chroma"
    vectordb_persist_dir: str = "./chroma_data"
    vectordb_collection: str = "git_history"

    # Processing Settings
    batch_size: int = 10
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: str = "./.cache"

    # Logging
    log_level: str = "INFO"
