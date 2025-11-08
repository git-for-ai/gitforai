"""Tests for incremental update state management."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from gitforai.incremental.state import (
    BranchState,
    GitForAIState,
    RepositoryState,
    StateManager,
)


class TestStateModels:
    """Test Pydantic state models."""

    def test_branch_state_creation(self):
        """Test creating a BranchState."""
        state = BranchState(
            last_indexed_commit="abc123",
            last_indexed_at=datetime.now(),
            commits_indexed=50,
            embedding_provider="local",
            embedding_model="all-MiniLM-L6-v2",
        )
        assert state.last_indexed_commit == "abc123"
        assert state.commits_indexed == 50
        assert state.embedding_provider == "local"

    def test_repository_state_creation(self):
        """Test creating a RepositoryState."""
        state = RepositoryState()
        assert state.branches == {}

        state.branches["main"] = BranchState(
            last_indexed_commit="abc123",
            last_indexed_at=datetime.now(),
            commits_indexed=50,
            embedding_provider="local",
        )
        assert "main" in state.branches

    def test_gitforai_state_creation(self):
        """Test creating a GitForAIState."""
        state = GitForAIState()
        assert state.version == "1.0"
        assert state.repositories == {}

    def test_state_serialization(self):
        """Test serializing state to dict."""
        state = GitForAIState()
        state.repositories["/path/to/repo"] = RepositoryState()
        state.repositories["/path/to/repo"].branches["main"] = BranchState(
            last_indexed_commit="abc123",
            last_indexed_at=datetime.now(),
            commits_indexed=50,
            embedding_provider="local",
        )

        data = state.model_dump()
        assert "version" in data
        assert "repositories" in data
        assert "/path/to/repo" in data["repositories"]


class TestStateManager:
    """Test StateManager functionality."""

    def test_init_creates_default_state_dir(self):
        """Test that StateManager creates default state directory."""
        manager = StateManager()
        assert manager.state_dir == Path.home() / ".gitforai"
        assert manager.state_file == manager.state_dir / "state.json"

    def test_init_with_custom_dir(self):
        """Test StateManager with custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            assert manager.state_dir == Path(tmpdir)

    def test_load_or_create_new_state(self):
        """Test loading or creating a new state when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            state = manager.load_or_create()

            assert isinstance(state, GitForAIState)
            assert state.version == "1.0"
            assert state.repositories == {}

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            # Load/create initial state
            state = manager.load_or_create()
            state.repositories["/test/repo"] = RepositoryState()
            state.repositories["/test/repo"].branches["main"] = BranchState(
                last_indexed_commit="abc123",
                last_indexed_at=datetime.now(),
                commits_indexed=10,
                embedding_provider="local",
            )

            # Save state
            manager.save()
            assert manager.state_file.exists()

            # Create new manager and load
            manager2 = StateManager(Path(tmpdir))
            state2 = manager2.load_or_create()

            assert "/test/repo" in state2.repositories
            assert "main" in state2.repositories["/test/repo"].branches
            assert state2.repositories["/test/repo"].branches["main"].last_indexed_commit == "abc123"

    def test_atomic_save(self):
        """Test that save is atomic using temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            state = manager.load_or_create()

            # Save state
            manager.save()

            # Verify no temporary files left behind
            temp_files = list(Path(tmpdir).glob(".state_*.json.tmp"))
            assert len(temp_files) == 0

    def test_get_branch_state_nonexistent(self):
        """Test getting state for non-existent repo/branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.load_or_create()

            state = manager.get_branch_state("/nonexistent/repo", "main")
            assert state is None

    def test_update_branch_state_new(self):
        """Test updating state for a new repository and branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.load_or_create()

            manager.update_branch_state(
                repo_path="/test/repo",
                branch="main",
                last_commit="abc123",
                commits_indexed=10,
                embedding_provider="local",
                embedding_model="all-MiniLM-L6-v2",
            )

            state = manager.get_branch_state("/test/repo", "main")
            assert state is not None
            assert state.last_indexed_commit == "abc123"
            assert state.commits_indexed == 10
            assert state.embedding_provider == "local"
            assert state.embedding_model == "all-MiniLM-L6-v2"

    def test_update_branch_state_existing(self):
        """Test updating state for an existing branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.load_or_create()

            # Create initial state
            manager.update_branch_state(
                repo_path="/test/repo",
                branch="main",
                last_commit="abc123",
                commits_indexed=10,
                embedding_provider="local",
            )

            # Update with new commit
            manager.update_branch_state(
                repo_path="/test/repo",
                branch="main",
                last_commit="def456",
                commits_indexed=5,
                embedding_provider="local",
            )

            state = manager.get_branch_state("/test/repo", "main")
            assert state.last_indexed_commit == "def456"
            assert state.commits_indexed == 15  # Cumulative

    def test_delete_branch_state(self):
        """Test deleting a branch state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.load_or_create()

            # Create state
            manager.update_branch_state(
                repo_path="/test/repo",
                branch="main",
                last_commit="abc123",
                commits_indexed=10,
                embedding_provider="local",
            )

            # Delete branch state
            result = manager.delete_branch_state("/test/repo", "main")
            assert result is True

            # Verify deleted
            state = manager.get_branch_state("/test/repo", "main")
            assert state is None

            # Try deleting again
            result = manager.delete_branch_state("/test/repo", "main")
            assert result is False

    def test_delete_repository_state(self):
        """Test deleting all state for a repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.load_or_create()

            # Create state for multiple branches
            manager.update_branch_state(
                repo_path="/test/repo",
                branch="main",
                last_commit="abc123",
                commits_indexed=10,
                embedding_provider="local",
            )
            manager.update_branch_state(
                repo_path="/test/repo",
                branch="develop",
                last_commit="def456",
                commits_indexed=5,
                embedding_provider="local",
            )

            # Delete repository state
            result = manager.delete_repository_state("/test/repo")
            assert result is True

            # Verify all branches deleted
            assert manager.get_branch_state("/test/repo", "main") is None
            assert manager.get_branch_state("/test/repo", "develop") is None

    def test_list_repositories(self):
        """Test listing all repositories with state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.load_or_create()

            # Add multiple repositories
            manager.update_branch_state(
                repo_path="/repo1",
                branch="main",
                last_commit="abc123",
                commits_indexed=10,
                embedding_provider="local",
            )
            manager.update_branch_state(
                repo_path="/repo2",
                branch="main",
                last_commit="def456",
                commits_indexed=5,
                embedding_provider="local",
            )

            repos = manager.list_repositories()
            assert len(repos) == 2
            assert "/repo1" in repos
            assert "/repo2" in repos

    def test_list_branches(self):
        """Test listing all branches for a repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.load_or_create()

            # Add multiple branches
            manager.update_branch_state(
                repo_path="/test/repo",
                branch="main",
                last_commit="abc123",
                commits_indexed=10,
                embedding_provider="local",
            )
            manager.update_branch_state(
                repo_path="/test/repo",
                branch="develop",
                last_commit="def456",
                commits_indexed=5,
                embedding_provider="local",
            )

            branches = manager.list_branches("/test/repo")
            assert len(branches) == 2
            assert "main" in branches
            assert "develop" in branches

    def test_corrupted_state_file(self):
        """Test handling of corrupted state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            # Create corrupted state file
            manager.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(manager.state_file, "w") as f:
                f.write("invalid json {{{")

            # Should create new state instead of crashing
            state = manager.load_or_create()
            assert isinstance(state, GitForAIState)
            assert state.repositories == {}
