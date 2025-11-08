"""Semantic enrichment processor for Git data."""

import asyncio
from pathlib import Path
from typing import List, Optional

from gitforai.llm.base import BaseLLMProvider
from gitforai.llm.cache import LLMCache
from gitforai.llm.embeddings import EmbeddingService
from gitforai.llm.prompts import PromptTemplates
from gitforai.models import CodeChange, CommitMetadata, FileDiff


class SemanticProcessor:
    """Processes Git data with LLM analysis and embeddings."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize semantic processor.

        Args:
            provider: LLM provider for completions and embeddings
            cache_dir: Directory for caching (defaults to .gitforai/cache)
            use_cache: Whether to use caching
        """
        self.provider = provider
        self.embedding_service = EmbeddingService(provider)
        self.prompts = PromptTemplates()

        # Set up caching
        self.use_cache = use_cache
        if use_cache:
            if cache_dir is None:
                cache_dir = Path.home() / ".gitforai" / "cache"
            self.cache = LLMCache(cache_dir)
        else:
            self.cache = None

    async def enrich_commit(
        self,
        commit: CommitMetadata,
        include_embedding: bool = True,
    ) -> CommitMetadata:
        """Enrich a commit with LLM analysis.

        Adds intent classification, topic extraction, summary, and embedding.

        Args:
            commit: Commit to enrich
            include_embedding: Whether to generate embedding

        Returns:
            Enriched commit with populated LLM fields
        """
        # Generate diff summary if we have files
        diff_summary = ""
        if commit.files_changed:
            diff_summary = f"{len(commit.files_changed)} files changed"

        # Try LLM enrichment, but continue with embeddings even if it fails
        # (LocalProvider only supports embeddings, not text completions)
        try:
            # 1. Classify intent
            intent = await self._get_completion_cached(
                self.prompts.commit_intent_classification(
                    commit.message,
                    diff_summary,
                ),
                "intent_classification",
            )
            commit.intent = intent.strip()

            # 2. Extract topics
            topics_str = await self._get_completion_cached(
                self.prompts.topic_extraction(
                    commit.message,
                    diff_summary,
                ),
                "topic_extraction",
            )
            commit.topics = [t.strip() for t in topics_str.split(",") if t.strip()]

            # 3. Generate summary
            summary = await self._get_completion_cached(
                self.prompts.commit_summary(
                    commit.message,
                    commit.files_changed,
                ),
                "summary",
            )
            commit.llm_summary = summary.strip()
        except Exception as e:
            # LLM enrichment failed (e.g., LocalProvider doesn't support completions)
            # Continue with embedding generation
            pass

        # 4. Generate embedding (always attempt, even if LLM enrichment failed)
        if include_embedding:
            try:
                embedding = await self._get_embedding_cached(commit)
                commit.embedding = embedding
            except Exception as e:
                # Re-raise embedding errors since they're critical for indexing
                raise

        return commit

    async def enrich_commits_batch(
        self,
        commits: List[CommitMetadata],
        batch_size: int = 10,
        include_embeddings: bool = True,
    ) -> List[CommitMetadata]:
        """Enrich multiple commits with batching.

        Args:
            commits: List of commits to enrich
            batch_size: Number of commits to process concurrently
            include_embeddings: Whether to generate embeddings

        Returns:
            List of enriched commits
        """
        enriched = []

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(commits), batch_size):
            batch = commits[i : i + batch_size]
            tasks = [
                self.enrich_commit(commit, include_embeddings) for commit in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle errors
            for commit, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"Error enriching commit {commit.hash}: {result}")
                    enriched.append(commit)  # Add original commit
                else:
                    enriched.append(result)

        return enriched

    async def analyze_diff(
        self,
        diff: FileDiff,
        commit_message: str = "",
    ) -> CodeChange:
        """Analyze a file diff and generate semantic information.

        Args:
            diff: File diff to analyze
            commit_message: Optional commit message for context

        Returns:
            CodeChange with semantic analysis
        """
        # Create base CodeChange
        change = CodeChange(
            file_path=diff.file_path,
            change_type=diff.change_type,
            diff_snippet=diff.diff_text[:1000] if diff.diff_text else "",
            context=diff.context_before[:400] if diff.context_before else "",
            affected_functions=[],  # Would need AST parsing
        )

        # Skip analysis for binary files
        if diff.is_binary:
            change.summary = f"Binary file {diff.change_type}"
            return change

        # Generate explanation
        if diff.diff_text:
            explanation = await self._get_completion_cached(
                self.prompts.diff_explanation(
                    diff.file_path,
                    diff.diff_text,
                    diff.context_before or "",
                    diff.context_after or "",
                ),
                "diff_explanation",
            )
            change.summary = explanation.strip()

        # Generate reasoning if we have commit message
        if commit_message and diff.diff_text:
            reasoning = await self._get_completion_cached(
                self.prompts.code_change_reasoning(
                    commit_message,
                    diff.file_path,
                    diff.diff_text,
                ),
                "reasoning",
            )
            change.reasoning = reasoning.strip()

        return change

    async def analyze_diffs_batch(
        self,
        diffs: List[FileDiff],
        commit_message: str = "",
        batch_size: int = 5,
    ) -> List[CodeChange]:
        """Analyze multiple diffs with batching.

        Args:
            diffs: List of diffs to analyze
            commit_message: Optional commit message for context
            batch_size: Number of diffs to process concurrently

        Returns:
            List of CodeChange objects with analysis
        """
        changes = []

        for i in range(0, len(diffs), batch_size):
            batch = diffs[i : i + batch_size]
            tasks = [self.analyze_diff(diff, commit_message) for diff in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for diff, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"Error analyzing diff {diff.file_path}: {result}")
                    # Add basic change
                    changes.append(
                        CodeChange(
                            file_path=diff.file_path,
                            change_type=diff.change_type,
                            diff_snippet=diff.diff_text[:1000] if diff.diff_text else "",
                            affected_functions=[],
                        )
                    )
                else:
                    changes.append(result)

        return changes

    async def generate_embeddings_for_commits(
        self,
        commits: List[CommitMetadata],
    ) -> List[CommitMetadata]:
        """Generate embeddings for commits that don't have them.

        Args:
            commits: List of commits

        Returns:
            List of commits with embeddings populated
        """
        # Check cache first if enabled
        if self.use_cache and self.cache:
            for commit in commits:
                if commit.embedding is None:
                    cached = self.cache.get_embedding(
                        self._commit_to_text(commit),
                        self.provider.model,
                    )
                    if cached:
                        commit.embedding = cached

        # Find commits still missing embeddings
        missing_indices = [
            i for i, c in enumerate(commits) if c.embedding is None
        ]

        if missing_indices:
            # Generate embeddings for missing commits
            missing_commits = [commits[i] for i in missing_indices]
            embeddings = await self.embedding_service.embed_commits_batch(
                missing_commits
            )

            # Update commits and cache
            for idx, embedding in zip(missing_indices, embeddings):
                commits[idx].embedding = embedding
                if self.use_cache and self.cache:
                    self.cache.set_embedding(
                        self._commit_to_text(commits[idx]),
                        self.provider.model,
                        embedding,
                    )

        return commits

    async def _get_completion_cached(
        self,
        prompt: str,
        prompt_type: str,
    ) -> str:
        """Get completion with caching.

        Args:
            prompt: Prompt to send
            prompt_type: Type of prompt (for cache key)

        Returns:
            Completion text
        """
        # Check cache
        if self.use_cache and self.cache:
            cached = self.cache.get_completion(prompt, self.provider.model)
            if cached:
                return cached

        # Generate completion
        completion = await self.provider.complete(prompt)

        # Cache result
        if self.use_cache and self.cache:
            self.cache.set_completion(prompt, self.provider.model, completion)

        return completion

    async def _get_embedding_cached(
        self,
        commit: CommitMetadata,
    ) -> List[float]:
        """Get embedding with caching.

        Args:
            commit: Commit to embed

        Returns:
            Embedding vector
        """
        text = self._commit_to_text(commit)

        # Check cache
        if self.use_cache and self.cache:
            cached = self.cache.get_embedding(text, self.provider.model)
            if cached:
                return cached

        # Generate embedding
        embedding = await self.embedding_service.embed_commit(commit)

        # Cache result
        if self.use_cache and self.cache:
            self.cache.set_embedding(text, self.provider.model, embedding)

        return embedding

    def _commit_to_text(self, commit: CommitMetadata) -> str:
        """Convert commit to text representation for caching.

        Args:
            commit: Commit to convert

        Returns:
            Text representation
        """
        parts = [f"Commit: {commit.message}"]

        if commit.llm_summary:
            parts.append(f"Summary: {commit.llm_summary}")
        if commit.intent:
            parts.append(f"Type: {commit.intent}")
        if commit.topics:
            parts.append(f"Topics: {', '.join(commit.topics)}")

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get processing statistics.

        Returns:
            Dictionary with cache stats and provider usage
        """
        stats = {}

        if self.use_cache and self.cache:
            stats["cache"] = self.cache.get_stats()

        if hasattr(self.provider, "get_usage_stats"):
            stats["provider"] = self.provider.get_usage_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear the cache."""
        if self.use_cache and self.cache:
            self.cache.clear()
