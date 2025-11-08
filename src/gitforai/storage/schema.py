"""Vector database schema definitions for collections."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class CommitDocument(BaseModel):
    """Document schema for commits collection.

    Stores commit metadata with semantic embeddings for similarity search.
    """

    # Primary key
    id: str = Field(..., description="Commit hash (SHA)")

    # Embedding vector
    embedding: List[float] = Field(..., description="Semantic embedding vector")

    # Metadata fields (searchable/filterable)
    author: str = Field(..., description="Commit author name")
    author_email: str = Field(..., description="Commit author email")
    timestamp: datetime = Field(..., description="Commit timestamp")
    message: str = Field(..., description="Original commit message")

    # Enriched semantic fields (from LLM analysis)
    summary: Optional[str] = Field(None, description="LLM-generated commit summary")
    intent: Optional[str] = Field(
        None, description="Commit intent (bug_fix, feature, refactor, docs, test, etc.)"
    )
    topics: List[str] = Field(default_factory=list, description="Extracted topics")

    # Change metadata
    files_changed: List[str] = Field(
        default_factory=list, description="List of changed file paths"
    )
    num_files_changed: int = Field(0, description="Number of files changed")
    num_lines_added: int = Field(0, description="Total lines added")
    num_lines_deleted: int = Field(0, description="Total lines deleted")

    # Diff preview (first 50 lines of changes for better semantic search)
    diff_preview: Optional[str] = Field(None, description="Preview of code changes (first 50 lines of diffs)")

    # Parent relationships
    parent_hashes: List[str] = Field(
        default_factory=list, description="Parent commit hashes"
    )
    is_merge: bool = Field(False, description="Whether this is a merge commit")

    # Repository context
    branch: Optional[str] = Field(None, description="Branch name")
    repo_path: Optional[str] = Field(None, description="Repository path")

    def to_chroma_format(self) -> dict:
        """Convert to ChromaDB format.

        Returns:
            Dict with ids, embeddings, metadatas, documents structure
        """
        # Prepare metadata (must be JSON-serializable)
        metadata = {
            "author": self.author,
            "author_email": self.author_email,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "num_files_changed": self.num_files_changed,
            "num_lines_added": self.num_lines_added,
            "num_lines_deleted": self.num_lines_deleted,
            "is_merge": self.is_merge,
        }

        # Add optional fields
        if self.summary:
            metadata["summary"] = self.summary
        if self.intent:
            metadata["intent"] = self.intent
        if self.topics:
            metadata["topics"] = ",".join(self.topics)
        if self.files_changed:
            metadata["files_changed"] = ",".join(self.files_changed)
        if self.parent_hashes:
            metadata["parent_hashes"] = ",".join(self.parent_hashes)
        if self.branch:
            metadata["branch"] = self.branch
        if self.repo_path:
            metadata["repo_path"] = self.repo_path
        if self.diff_preview:
            # Truncate to avoid ChromaDB metadata size limits
            metadata["diff_preview"] = self.diff_preview[:2000]

        # Document text for full-text search (commit message + summary + diff)
        document = self.message
        if self.summary:
            document = f"{self.summary}\n\n{self.message}"
        if self.diff_preview:
            document = f"{document}\n\nChanges:\n{self.diff_preview[:1000]}"

        return {
            "ids": [self.id],
            "embeddings": [self.embedding],
            "metadatas": [metadata],
            "documents": [document],
        }


class FileVersionDocument(BaseModel):
    """Document schema for file versions collection.

    Stores file snapshots at specific commits with embeddings.
    """

    # Primary key: file_path:commit_hash
    id: str = Field(..., description="Unique ID (file_path:commit_hash)")

    # Embedding vector
    embedding: List[float] = Field(..., description="Semantic embedding of file content")

    # Metadata
    file_path: str = Field(..., description="File path in repository")
    commit_hash: str = Field(..., description="Commit hash where this version exists")
    timestamp: datetime = Field(..., description="Commit timestamp")

    # File metadata
    size: int = Field(..., description="File size in bytes")
    language: Optional[str] = Field(None, description="Programming language")
    file_type: Optional[str] = Field(None, description="File type/extension")

    # Content info
    num_lines: Optional[int] = Field(None, description="Number of lines")
    num_functions: Optional[int] = Field(None, description="Number of functions/methods")
    num_classes: Optional[int] = Field(None, description="Number of classes")

    # Content preview (not embedded, just metadata)
    preview: Optional[str] = Field(None, description="First 500 chars of content")

    # Repository context
    repo_path: Optional[str] = Field(None, description="Repository path")

    @staticmethod
    def create_id(file_path: str, commit_hash: str) -> str:
        """Create unique ID from file path and commit hash.

        Args:
            file_path: File path in repository
            commit_hash: Commit hash

        Returns:
            Unique ID string
        """
        return f"{file_path}:{commit_hash}"

    def to_chroma_format(self) -> dict:
        """Convert to ChromaDB format.

        Returns:
            Dict with ids, embeddings, metadatas, documents structure
        """
        metadata = {
            "file_path": self.file_path,
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp.isoformat(),
            "size": self.size,
        }

        # Add optional fields
        if self.language:
            metadata["language"] = self.language
        if self.file_type:
            metadata["file_type"] = self.file_type
        if self.num_lines is not None:
            metadata["num_lines"] = self.num_lines
        if self.num_functions is not None:
            metadata["num_functions"] = self.num_functions
        if self.num_classes is not None:
            metadata["num_classes"] = self.num_classes
        if self.repo_path:
            metadata["repo_path"] = self.repo_path

        # Document for full-text search
        document = f"File: {self.file_path}"
        if self.preview:
            document = f"{document}\n\n{self.preview}"

        return {
            "ids": [self.id],
            "embeddings": [self.embedding],
            "metadatas": [metadata],
            "documents": [document],
        }


class CodeChangeDocument(BaseModel):
    """Document schema for code changes collection.

    Stores individual code changes (diffs) with embeddings for fine-grained search.
    """

    # Primary key: unique change ID
    id: str = Field(..., description="Unique change ID")

    # Embedding vector
    embedding: List[float] = Field(..., description="Semantic embedding of the change")

    # Metadata
    commit_hash: str = Field(..., description="Parent commit hash")
    file_path: str = Field(..., description="File path")
    change_type: str = Field(
        ..., description="Change type (added, modified, deleted, renamed)"
    )

    # Change details
    lines_added: int = Field(0, description="Lines added in this change")
    lines_deleted: int = Field(0, description="Lines deleted in this change")
    start_line: Optional[int] = Field(None, description="Starting line number")
    end_line: Optional[int] = Field(None, description="Ending line number")

    # Semantic analysis (from LLM)
    summary: Optional[str] = Field(None, description="LLM summary of the change")
    context: Optional[str] = Field(None, description="Context around the change")
    intent: Optional[str] = Field(None, description="Intent of this specific change")

    # Diff content
    diff_preview: Optional[str] = Field(None, description="Preview of the diff")

    # Timestamp
    timestamp: datetime = Field(..., description="Commit timestamp")

    # Repository context
    repo_path: Optional[str] = Field(None, description="Repository path")

    @staticmethod
    def create_id(commit_hash: str, file_path: str, index: int) -> str:
        """Create unique ID for a code change.

        Args:
            commit_hash: Commit hash
            file_path: File path
            index: Index of change within commit

        Returns:
            Unique ID string
        """
        # Use hash of components for shorter IDs
        import hashlib
        components = f"{commit_hash}:{file_path}:{index}"
        hash_part = hashlib.sha256(components.encode()).hexdigest()[:16]
        return f"change_{hash_part}"

    def to_chroma_format(self) -> dict:
        """Convert to ChromaDB format.

        Returns:
            Dict with ids, embeddings, metadatas, documents structure
        """
        metadata = {
            "commit_hash": self.commit_hash,
            "file_path": self.file_path,
            "change_type": self.change_type,
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "timestamp": self.timestamp.isoformat(),
        }

        # Add optional fields
        if self.start_line is not None:
            metadata["start_line"] = self.start_line
        if self.end_line is not None:
            metadata["end_line"] = self.end_line
        if self.summary:
            metadata["summary"] = self.summary
        if self.context:
            metadata["context"] = self.context
        if self.intent:
            metadata["intent"] = self.intent
        if self.repo_path:
            metadata["repo_path"] = self.repo_path

        # Document for full-text search
        document = f"File: {self.file_path} ({self.change_type})"
        if self.summary:
            document = f"{document}\n{self.summary}"
        if self.diff_preview:
            document = f"{document}\n\n{self.diff_preview}"

        return {
            "ids": [self.id],
            "embeddings": [self.embedding],
            "metadatas": [metadata],
            "documents": [document],
        }


# Collection metadata templates
COLLECTION_METADATA = {
    "commits": {
        "description": "Commit metadata with semantic embeddings",
        "schema": "CommitDocument",
        "primary_key": "commit_hash",
    },
    "file_versions": {
        "description": "File snapshots at specific commits",
        "schema": "FileVersionDocument",
        "primary_key": "file_path:commit_hash",
    },
    "code_changes": {
        "description": "Individual code changes with embeddings",
        "schema": "CodeChangeDocument",
        "primary_key": "unique_change_id",
    },
}
