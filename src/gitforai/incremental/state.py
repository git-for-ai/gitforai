"""State management for incremental updates.

Tracks the last indexed commit per repository and branch to enable
efficient incremental updates.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


class BranchState(BaseModel):
    """State for a single branch in a repository."""

    last_indexed_commit: str = Field(..., description="SHA of the last indexed commit")
    last_indexed_at: datetime = Field(..., description="Timestamp of last indexing operation")
    commits_indexed: int = Field(0, description="Total number of commits indexed on this branch")
    embedding_provider: str = Field(..., description="Embedding provider used (local, openai, etc.)")
    embedding_model: Optional[str] = Field(None, description="Specific model used for embeddings")


class RepositoryState(BaseModel):
    """State for a single repository."""

    branches: Dict[str, BranchState] = Field(
        default_factory=dict, description="State per branch"
    )


class GitForAIState(BaseModel):
    """Root state object for GitForAI incremental updates."""

    version: str = Field("1.0", description="State file format version")
    repositories: Dict[str, RepositoryState] = Field(
        default_factory=dict, description="State per repository path"
    )


class StateManager:
    """Manages persistent state for incremental updates.

    The state is stored in .gitforai/state.json and tracks the last indexed
    commit for each repository and branch.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        """Initialize the state manager.

        Args:
            state_dir: Directory to store state file. Defaults to ~/.gitforai/
        """
        if state_dir is None:
            state_dir = Path.home() / ".gitforai"

        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "state.json"
        self._state: Optional[GitForAIState] = None

    def _ensure_state_dir(self) -> None:
        """Ensure the state directory exists."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def load_or_create(self) -> GitForAIState:
        """Load existing state or create a new one.

        Returns:
            GitForAIState object
        """
        if self._state is not None:
            return self._state

        self._ensure_state_dir()

        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self._state = GitForAIState(**data)
            except (json.JSONDecodeError, ValueError) as e:
                # If state file is corrupted, create a new one
                print(f"Warning: Could not load state file ({e}), creating new state")
                self._state = GitForAIState()
        else:
            self._state = GitForAIState()

        return self._state

    def save(self) -> None:
        """Save the current state to disk using atomic write.

        Uses a temporary file and rename to ensure atomicity.
        """
        if self._state is None:
            return

        self._ensure_state_dir()

        # Write to temporary file first
        fd, temp_path = tempfile.mkstemp(
            dir=self.state_dir, prefix=".state_", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._state.model_dump(), f, indent=2, default=str)

            # Atomic rename
            os.replace(temp_path, self.state_file)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def get_branch_state(
        self, repo_path: str, branch: str
    ) -> Optional[BranchState]:
        """Get the state for a specific repository and branch.

        Args:
            repo_path: Absolute path to repository
            branch: Branch name

        Returns:
            BranchState if it exists, None otherwise
        """
        state = self.load_or_create()
        repo_state = state.repositories.get(repo_path)
        if repo_state is None:
            return None
        return repo_state.branches.get(branch)

    def update_branch_state(
        self,
        repo_path: str,
        branch: str,
        last_commit: str,
        commits_indexed: int,
        embedding_provider: str,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Update or create the state for a specific repository and branch.

        Args:
            repo_path: Absolute path to repository
            branch: Branch name
            last_commit: SHA of the last indexed commit
            commits_indexed: Number of commits indexed in this operation
            embedding_provider: Embedding provider used
            embedding_model: Specific model used (optional)
        """
        state = self.load_or_create()

        # Ensure repository state exists
        if repo_path not in state.repositories:
            state.repositories[repo_path] = RepositoryState()

        repo_state = state.repositories[repo_path]

        # Get existing branch state or create new
        if branch in repo_state.branches:
            branch_state = repo_state.branches[branch]
            branch_state.last_indexed_commit = last_commit
            branch_state.last_indexed_at = datetime.now()
            branch_state.commits_indexed += commits_indexed
            branch_state.embedding_provider = embedding_provider
            if embedding_model:
                branch_state.embedding_model = embedding_model
        else:
            repo_state.branches[branch] = BranchState(
                last_indexed_commit=last_commit,
                last_indexed_at=datetime.now(),
                commits_indexed=commits_indexed,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )

        self.save()

    def delete_branch_state(self, repo_path: str, branch: str) -> bool:
        """Delete the state for a specific branch.

        Args:
            repo_path: Absolute path to repository
            branch: Branch name

        Returns:
            True if state was deleted, False if it didn't exist
        """
        state = self.load_or_create()
        repo_state = state.repositories.get(repo_path)
        if repo_state is None or branch not in repo_state.branches:
            return False

        del repo_state.branches[branch]

        # Clean up empty repository state
        if not repo_state.branches:
            del state.repositories[repo_path]

        self.save()
        return True

    def delete_repository_state(self, repo_path: str) -> bool:
        """Delete all state for a repository.

        Args:
            repo_path: Absolute path to repository

        Returns:
            True if state was deleted, False if it didn't exist
        """
        state = self.load_or_create()
        if repo_path not in state.repositories:
            return False

        del state.repositories[repo_path]
        self.save()
        return True

    def list_repositories(self) -> list[str]:
        """Get a list of all repositories with tracked state.

        Returns:
            List of repository paths
        """
        state = self.load_or_create()
        return list(state.repositories.keys())

    def list_branches(self, repo_path: str) -> list[str]:
        """Get a list of all branches tracked for a repository.

        Args:
            repo_path: Absolute path to repository

        Returns:
            List of branch names
        """
        state = self.load_or_create()
        repo_state = state.repositories.get(repo_path)
        if repo_state is None:
            return []
        return list(repo_state.branches.keys())
