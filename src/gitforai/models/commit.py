"""Data models for Git commit information."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class CommitStats(BaseModel):
    """Statistics for a commit (lines added/deleted)."""

    insertions: int = Field(0, description="Number of lines added")
    deletions: int = Field(0, description="Number of lines deleted")


class CommitMetadata(BaseModel):
    """Represents metadata for a single Git commit."""

    hash: str = Field(..., description="Full commit SHA hash")
    short_hash: str = Field(..., description="Short commit SHA hash (7 chars)")
    author_name: str = Field(..., description="Author name")
    author_email: str = Field(..., description="Author email")
    committer_name: str = Field(..., description="Committer name")
    committer_email: str = Field(..., description="Committer email")
    timestamp: datetime = Field(..., description="Commit timestamp")
    message: str = Field(..., description="Full commit message")
    message_summary: Optional[str] = Field(None, description="First line of commit message")
    parent_hashes: List[str] = Field(default_factory=list, description="Parent commit hashes")
    files_changed: List[str] = Field(default_factory=list, description="List of changed file paths")
    is_merge: bool = Field(False, description="Whether this is a merge commit")

    # Commit statistics
    stats: Optional[CommitStats] = Field(None, description="Statistics about lines added/deleted")

    # Diff preview (first 50 lines of changes for semantic search)
    diff_preview: Optional[str] = Field(None, description="Preview of code changes (first 50 lines of diffs)")

    # Semantic enrichment (populated later by LLM processing)
    intent: Optional[str] = Field(None, description="Commit intent: bug_fix, feature, refactor, docs, test, chore")
    topics: List[str] = Field(default_factory=list, description="Extracted topics from commit")
    llm_summary: Optional[str] = Field(None, description="LLM-generated summary of changes")

    # Vector embedding (populated for semantic search)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of commit message and changes")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "hash": "abc123def456",
                "short_hash": "abc123d",
                "author_name": "John Doe",
                "author_email": "john@example.com",
                "committer_name": "John Doe",
                "committer_email": "john@example.com",
                "timestamp": "2024-01-15T10:30:00Z",
                "message": "Fix authentication bug\n\nResolves issue with token validation",
                "message_summary": "Fix authentication bug",
                "parent_hashes": ["parent123"],
                "files_changed": ["src/auth.py", "tests/test_auth.py"],
                "is_merge": False,
                "intent": "bug_fix",
                "topics": ["authentication", "security"],
                "llm_summary": "Fixed token validation logic in authentication module",
            }
        }


class FileDiff(BaseModel):
    """Represents a diff for a single file in a commit."""

    commit_hash: str = Field(..., description="Commit hash this diff belongs to")
    file_path: str = Field(..., description="Path to the file")
    change_type: str = Field(..., description="Type of change: added, modified, deleted, renamed")
    old_path: Optional[str] = Field(None, description="Old path for renamed files")
    diff_text: str = Field(..., description="Full unified diff text")
    additions: int = Field(0, description="Number of lines added")
    deletions: int = Field(0, description="Number of lines deleted")
    is_binary: bool = Field(False, description="Whether the file is binary")

    # Contextual information
    context_before: Optional[str] = Field(None, description="Code context before changes")
    context_after: Optional[str] = Field(None, description="Code context after changes")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "commit_hash": "abc123def456",
                "file_path": "src/auth.py",
                "change_type": "modified",
                "old_path": None,
                "diff_text": "@@ -10,7 +10,7 @@\n-    if not token:\n+    if not token or token == '':",
                "additions": 1,
                "deletions": 1,
                "is_binary": False,
            }
        }


class FileSnapshot(BaseModel):
    """Represents the complete state of a file at a specific commit."""

    commit_hash: str = Field(..., description="Commit hash")
    file_path: str = Field(..., description="Path to the file")
    content: str = Field(..., description="Full file content")
    size_bytes: int = Field(..., description="File size in bytes")
    timestamp: datetime = Field(..., description="Commit timestamp")
    file_extension: str = Field(..., description="File extension")
    is_binary: bool = Field(False, description="Whether the file is binary")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "commit_hash": "abc123def456",
                "file_path": "src/auth.py",
                "content": "def authenticate(token):\n    if not token:\n        return False\n    return True",
                "size_bytes": 1024,
                "timestamp": "2024-01-15T10:30:00Z",
                "file_extension": ".py",
                "is_binary": False,
            }
        }


class CodeChange(BaseModel):
    """Represents a semantic code change with LLM-generated context."""

    id: str = Field(..., description="Unique identifier for this change")
    commit_hash: str = Field(..., description="Commit hash")
    file_path: str = Field(..., description="Path to the file")
    change_type: str = Field(..., description="Type of change: added, modified, deleted")
    diff_snippet: str = Field(..., description="Relevant diff snippet")
    context: str = Field(..., description="Surrounding code context")

    # LLM-generated semantic information
    summary: Optional[str] = Field(None, description="LLM-generated summary of the change")
    reasoning: Optional[str] = Field(None, description="LLM-inferred reasoning for the change")
    affected_functions: List[str] = Field(default_factory=list, description="Functions affected by this change")

    # Vector embedding (populated later)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of this change")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "id": "abc123def456:src/auth.py:0",
                "commit_hash": "abc123def456",
                "file_path": "src/auth.py",
                "change_type": "modified",
                "diff_snippet": "-    if not token:\n+    if not token or token == '':",
                "context": "def authenticate(token):\n    if not token or token == '':\n        return False",
                "summary": "Added empty string check to token validation",
                "reasoning": "Prevent authentication bypass with empty tokens",
                "affected_functions": ["authenticate"],
            }
        }
