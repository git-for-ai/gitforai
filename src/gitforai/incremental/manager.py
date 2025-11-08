"""Incremental update manager - orchestrates the update process."""

from pathlib import Path
from typing import Dict, List, Optional

import git
from rich.console import Console

from gitforai.extraction.git_extractor import GitExtractor
from gitforai.incremental.delta import DeltaDetector
from gitforai.incremental.state import BranchState, StateManager
from gitforai.llm.processor import SemanticProcessor
from gitforai.models.commit import CommitMetadata


class IncrementalUpdateResult:
    """Result of an incremental update operation."""

    def __init__(
        self,
        success: bool,
        commits_processed: int = 0,
        commits_skipped: int = 0,
        is_full_reindex: bool = False,
        rebase_detected: bool = False,
        error: Optional[str] = None,
    ):
        self.success = success
        self.commits_processed = commits_processed
        self.commits_skipped = commits_skipped
        self.is_full_reindex = is_full_reindex
        self.rebase_detected = rebase_detected
        self.error = error


class IncrementalUpdateManager:
    """Manages incremental updates to the vector database.

    Coordinates state management, delta detection, and commit processing
    to efficiently update the database with only new commits.
    """

    def __init__(
        self,
        state_manager: StateManager,
        console: Optional[Console] = None,
    ):
        """Initialize the incremental update manager.

        Args:
            state_manager: State manager for tracking last indexed commits
            console: Rich console for output (optional)
        """
        self.state_manager = state_manager
        self.console = console or Console()

    def should_use_incremental(
        self, repo_path: Path, branch: str, embedding_provider: str
    ) -> bool:
        """Check if incremental update should be used.

        Args:
            repo_path: Path to the repository
            branch: Branch name
            embedding_provider: Current embedding provider

        Returns:
            True if incremental update should be used, False if full reindex needed
        """
        branch_state = self.state_manager.get_branch_state(str(repo_path), branch)

        if branch_state is None:
            # No previous state - need full index
            return False

        # Check if embedding provider changed
        if branch_state.embedding_provider != embedding_provider:
            self.console.print(
                f"[yellow]Embedding provider changed from {branch_state.embedding_provider} "
                f"to {embedding_provider}. Full reindex required.[/yellow]"
            )
            return False

        return True

    def get_commits_to_process(
        self,
        repo: git.Repo,
        repo_path: Path,
        branch: str,
        max_commits: Optional[int] = None,
    ) -> tuple[List[git.Commit], bool]:
        """Get the list of commits that need to be processed.

        Args:
            repo: GitPython repository object
            repo_path: Path to the repository
            branch: Branch name
            max_commits: Maximum number of commits to process (optional)

        Returns:
            Tuple of (commits_to_process, is_full_reindex)
        """
        branch_state = self.state_manager.get_branch_state(str(repo_path), branch)

        if branch_state is None:
            # No previous state - do full index
            self.console.print("[yellow]No previous state found. Performing full index.[/yellow]")
            detector = DeltaDetector(repo)
            all_commits = detector.get_all_commits(branch, max_count=max_commits)
            return (all_commits, True)

        # Try incremental update
        detector = DeltaDetector(repo)

        # Check for rebase
        is_rebased, new_base = detector.detect_rebase(branch_state.last_indexed_commit)

        if is_rebased:
            self.console.print(
                "[yellow]Rebase or force push detected. Performing full reindex.[/yellow]"
            )
            all_commits = detector.get_all_commits(branch, max_count=max_commits)
            return (all_commits, True)

        # Get new commits
        try:
            new_commits = detector.find_new_commits(branch_state.last_indexed_commit, branch)

            if not new_commits:
                self.console.print("[green]No new commits to process.[/green]")
                return ([], False)

            self.console.print(
                f"[green]Found {len(new_commits)} new commits since last index.[/green]"
            )

            # Respect max_commits limit if provided
            if max_commits and len(new_commits) > max_commits:
                self.console.print(
                    f"[yellow]Limiting to {max_commits} most recent commits.[/yellow]"
                )
                new_commits = new_commits[-max_commits:]

            return (new_commits, False)

        except ValueError as e:
            # Error getting new commits - fall back to full reindex
            self.console.print(
                f"[yellow]Error getting new commits: {e}. Performing full reindex.[/yellow]"
            )
            all_commits = detector.get_all_commits(branch, max_count=max_commits)
            return (all_commits, True)

    async def process_commits(
        self,
        commits: List[git.Commit],
        extractor: GitExtractor,
        llm_processor: SemanticProcessor,
        include_llm_enrichment: bool = True,
    ) -> List[CommitMetadata]:
        """Process a list of Git commits into CommitMetadata objects.

        Args:
            commits: List of GitPython commit objects
            extractor: GitExtractor instance
            llm_processor: SemanticProcessor instance
            include_llm_enrichment: Whether to perform LLM enrichment

        Returns:
            List of processed CommitMetadata objects
        """
        processed_commits = []

        for commit in commits:
            # Extract commit metadata
            commit_meta = extractor._extract_commit_metadata(commit)

            # Enrich with LLM if requested
            if include_llm_enrichment:
                commit_meta = await llm_processor.enrich_commit(
                    commit_meta, include_embedding=True
                )
            else:
                # Just generate embedding without LLM enrichment
                from gitforai.llm.embeddings import EmbeddingGenerator
                embedding_gen = EmbeddingGenerator(llm_processor.provider)
                commit_meta.embedding = await embedding_gen.embed_commit(commit_meta)

            processed_commits.append(commit_meta)

        return processed_commits

    def update_state(
        self,
        repo_path: Path,
        branch: str,
        last_commit: str,
        commits_processed: int,
        embedding_provider: str,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Update the state after successful processing.

        Args:
            repo_path: Path to the repository
            branch: Branch name
            last_commit: SHA of the last processed commit
            commits_processed: Number of commits processed in this operation
            embedding_provider: Embedding provider used
            embedding_model: Specific model used (optional)
        """
        self.state_manager.update_branch_state(
            repo_path=str(repo_path),
            branch=branch,
            last_commit=last_commit,
            commits_indexed=commits_processed,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )

    def get_status(
        self, repo_path: Path, branch: Optional[str] = None
    ) -> Dict[str, any]:
        """Get the current status of indexing for a repository.

        Args:
            repo_path: Path to the repository
            branch: Branch name (optional, defaults to current branch)

        Returns:
            Dictionary with status information
        """
        try:
            repo = git.Repo(repo_path)
            if branch is None:
                branch = repo.active_branch.name

            detector = DeltaDetector(repo)
            branch_state = self.state_manager.get_branch_state(str(repo_path), branch)

            if branch_state is None:
                # No state - never indexed
                stats = detector.get_stats()
                return {
                    "indexed": False,
                    "current_branch": stats["current_branch"],
                    "current_head": stats["current_head"],
                    "total_commits": stats["total_commits"],
                    "message": "Repository has not been indexed yet",
                }

            # Get stats with last indexed commit
            stats = detector.get_stats(branch_state.last_indexed_commit, branch)

            return {
                "indexed": True,
                "current_branch": stats["current_branch"],
                "current_head": stats["current_head"],
                "last_indexed_commit": branch_state.last_indexed_commit,
                "last_indexed_at": branch_state.last_indexed_at.isoformat(),
                "commits_indexed": branch_state.commits_indexed,
                "total_commits": stats["total_commits"],
                "new_commits": stats.get("new_commits", 0),
                "rebase_detected": stats.get("rebase_detected", False),
                "embedding_provider": branch_state.embedding_provider,
                "embedding_model": branch_state.embedding_model,
                "up_to_date": stats.get("new_commits", 0) == 0,
            }

        except Exception as e:
            return {
                "error": str(e),
                "message": f"Error getting status: {e}",
            }
