"""Embedding generation service."""

import hashlib
from typing import List, Optional

from gitforai.llm.base import BaseLLMProvider
from gitforai.models import CodeChange, CommitMetadata, FileDiff


class EmbeddingService:
    """Service for generating embeddings for Git data."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialize embedding service.

        Args:
            provider: LLM provider for generating embeddings
        """
        self.provider = provider

    async def embed_commit(self, commit: CommitMetadata) -> List[float]:
        """Generate embedding for a commit.

        Combines commit message, summary, metadata, and diff preview into a single embedding.

        Args:
            commit: Commit metadata

        Returns:
            Embedding vector
        """
        # Construct text representation
        text_parts = [
            f"Commit: {commit.message}",
        ]

        if commit.llm_summary:
            text_parts.append(f"Summary: {commit.llm_summary}")

        if commit.intent:
            text_parts.append(f"Type: {commit.intent}")

        if commit.topics:
            text_parts.append(f"Topics: {', '.join(commit.topics)}")

        if commit.files_changed:
            text_parts.append(f"Files: {', '.join(commit.files_changed[:10])}")

        # Include diff preview for better code-level semantic search
        if commit.diff_preview:
            text_parts.append(f"Changes:\n{commit.diff_preview}")

        text = "\n".join(text_parts)

        return await self.provider.generate_embedding(text)

    async def embed_commits_batch(
        self, commits: List[CommitMetadata]
    ) -> List[List[float]]:
        """Generate embeddings for multiple commits.

        Args:
            commits: List of commit metadata

        Returns:
            List of embedding vectors
        """
        texts = []
        for commit in commits:
            text_parts = [f"Commit: {commit.message}"]

            if commit.llm_summary:
                text_parts.append(f"Summary: {commit.llm_summary}")
            if commit.intent:
                text_parts.append(f"Type: {commit.intent}")
            if commit.topics:
                text_parts.append(f"Topics: {', '.join(commit.topics)}")
            if commit.diff_preview:
                text_parts.append(f"Changes:\n{commit.diff_preview}")

            texts.append("\n".join(text_parts))

        return await self.provider.generate_embeddings_batch(texts)

    async def embed_diff(self, diff: FileDiff) -> List[float]:
        """Generate embedding for a file diff.

        Args:
            diff: File diff

        Returns:
            Embedding vector
        """
        text_parts = [
            f"File: {diff.file_path}",
            f"Change: {diff.change_type}",
        ]

        if diff.diff_text and not diff.is_binary:
            # Truncate long diffs
            diff_preview = diff.diff_text[:1000]
            text_parts.append(f"Diff:\n{diff_preview}")

        if diff.context_before:
            text_parts.append(f"Context before: {diff.context_before[:300]}")

        if diff.context_after:
            text_parts.append(f"Context after: {diff.context_after[:300]}")

        text = "\n".join(text_parts)

        return await self.provider.generate_embedding(text)

    async def embed_code_change(self, change: CodeChange) -> List[float]:
        """Generate embedding for a code change.

        Args:
            change: Code change with semantic information

        Returns:
            Embedding vector
        """
        text_parts = [
            f"File: {change.file_path}",
            f"Change type: {change.change_type}",
        ]

        if change.summary:
            text_parts.append(f"Summary: {change.summary}")

        if change.reasoning:
            text_parts.append(f"Reasoning: {change.reasoning}")

        if change.diff_snippet:
            text_parts.append(f"Code:\n{change.diff_snippet[:800]}")

        if change.context:
            text_parts.append(f"Context: {change.context[:400]}")

        if change.affected_functions:
            text_parts.append(f"Functions: {', '.join(change.affected_functions)}")

        text = "\n".join(text_parts)

        return await self.provider.generate_embedding(text)

    async def embed_code_changes_batch(
        self, changes: List[CodeChange]
    ) -> List[List[float]]:
        """Generate embeddings for multiple code changes.

        Args:
            changes: List of code changes

        Returns:
            List of embedding vectors
        """
        texts = []
        for change in changes:
            text_parts = [
                f"File: {change.file_path}",
                f"Change: {change.change_type}",
            ]

            if change.summary:
                text_parts.append(f"Summary: {change.summary}")
            if change.diff_snippet:
                text_parts.append(f"Code: {change.diff_snippet[:500]}")

            texts.append("\n".join(text_parts))

        return await self.provider.generate_embeddings_batch(texts)

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for arbitrary text.

        Useful for search queries.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return await self.provider.generate_embedding(text)

    async def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return await self.provider.generate_embeddings_batch(texts)

    @staticmethod
    def generate_cache_key(text: str) -> str:
        """Generate a cache key for text.

        Args:
            text: Text to generate key for

        Returns:
            Cache key (SHA256 hash)
        """
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def truncate_text(text: str, max_tokens: int = 8000) -> str:
        """Truncate text to approximate token limit.

        Rough approximation: 1 token ≈ 4 characters

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens

        Returns:
            Truncated text
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        return text[:max_chars] + "..."
