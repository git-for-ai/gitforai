"""Unit tests for Git extraction module."""

import tempfile
from pathlib import Path

import git
import pytest

from gitforai.extraction import GitExtractor
from gitforai.models import RepositoryConfig


@pytest.fixture
def test_repo():
    """Create a temporary Git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)

        # Configure git
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Create initial commit
        (repo_path / "README.md").write_text("# Test Project\n")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        # Create second commit
        (repo_path / "main.py").write_text("def hello():\n    print('Hello, World!')\n")
        repo.index.add(["main.py"])
        repo.index.commit("Add main.py")

        # Create third commit
        (repo_path / "main.py").write_text("def hello():\n    print('Hello, GitForAI!')\n")
        repo.index.add(["main.py"])
        repo.index.commit("Fix: Update hello message")

        yield repo_path


def test_git_extractor_initialization(test_repo):
    """Test GitExtractor initialization."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    assert extractor.config == config
    assert extractor.repo is not None


def test_git_extractor_invalid_path():
    """Test GitExtractor with invalid repository path."""
    config = RepositoryConfig(repo_path=Path("/nonexistent/path"))

    with pytest.raises(ValueError, match="Repository path does not exist"):
        GitExtractor(config)


def test_extract_all_commits(test_repo):
    """Test extracting all commits from repository."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    commits = list(extractor.extract_all_commits())

    assert len(commits) == 3
    assert commits[0].message == "Fix: Update hello message"
    assert commits[1].message == "Add main.py"
    assert commits[2].message == "Initial commit"


def test_extract_all_commits_with_max_count(test_repo):
    """Test extracting commits with max count limit."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    commits = list(extractor.extract_all_commits(max_count=2))

    assert len(commits) == 2


def test_extract_commit(test_repo):
    """Test extracting a specific commit."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    # Get the latest commit hash
    all_commits = list(extractor.extract_all_commits())
    latest_hash = all_commits[0].hash

    # Extract specific commit
    commit = extractor.extract_commit(latest_hash)

    assert commit.hash == latest_hash
    assert commit.message == "Fix: Update hello message"
    assert commit.author_name == "Test User"
    assert commit.author_email == "test@example.com"


def test_extract_commit_invalid_hash(test_repo):
    """Test extracting a commit with invalid hash."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    with pytest.raises(ValueError, match="Commit not found"):
        extractor.extract_commit("invalid_hash_123")


def test_extract_commit_diffs(test_repo):
    """Test extracting diffs for a commit."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    # Get the latest commit hash
    all_commits = list(extractor.extract_all_commits())
    latest_hash = all_commits[0].hash

    # Extract diffs
    diffs = extractor.extract_commit_diffs(latest_hash)

    assert len(diffs) == 1
    assert diffs[0].file_path == "main.py"
    assert diffs[0].change_type == "modified"
    assert diffs[0].additions >= 1
    assert diffs[0].deletions >= 1


def test_extract_file_snapshot(test_repo):
    """Test extracting a file snapshot."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    # Get the latest commit hash
    all_commits = list(extractor.extract_all_commits())
    latest_hash = all_commits[0].hash

    # Extract file snapshot
    snapshot = extractor.extract_file_snapshot(latest_hash, "main.py")

    assert snapshot is not None
    assert snapshot.file_path == "main.py"
    assert "GitForAI" in snapshot.content
    assert snapshot.file_extension == ".py"
    assert not snapshot.is_binary


def test_extract_file_snapshot_nonexistent(test_repo):
    """Test extracting a snapshot for nonexistent file."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    # Get the latest commit hash
    all_commits = list(extractor.extract_all_commits())
    latest_hash = all_commits[0].hash

    # Try to extract nonexistent file
    snapshot = extractor.extract_file_snapshot(latest_hash, "nonexistent.py")

    assert snapshot is None


def test_extract_all_snapshots(test_repo):
    """Test extracting all file snapshots from a commit."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    # Get the latest commit hash
    all_commits = list(extractor.extract_all_commits())
    latest_hash = all_commits[0].hash

    # Extract all snapshots
    snapshots = extractor.extract_all_snapshots(latest_hash)

    assert len(snapshots) >= 2  # Should have README.md and main.py
    file_paths = [s.file_path for s in snapshots]
    assert "README.md" in file_paths
    assert "main.py" in file_paths


def test_file_filtering(test_repo):
    """Test that files are filtered based on configuration."""
    # Create config that only includes .md files
    config = RepositoryConfig(
        repo_path=test_repo,
        included_extensions=[".md"],
    )
    extractor = GitExtractor(config)

    # Get the latest commit
    all_commits = list(extractor.extract_all_commits())
    latest_hash = all_commits[0].hash

    # Extract snapshots - should only get .md files
    snapshots = extractor.extract_all_snapshots(latest_hash)
    file_extensions = [s.file_extension for s in snapshots]

    assert all(ext == ".md" for ext in file_extensions)


def test_commit_metadata_fields(test_repo):
    """Test that commit metadata contains all expected fields."""
    config = RepositoryConfig(repo_path=test_repo)
    extractor = GitExtractor(config)

    commits = list(extractor.extract_all_commits())
    commit = commits[0]

    # Check all fields are present
    assert commit.hash is not None
    assert commit.short_hash is not None
    assert len(commit.short_hash) == 7
    assert commit.author_name is not None
    assert commit.author_email is not None
    assert commit.committer_name is not None
    assert commit.committer_email is not None
    assert commit.timestamp is not None
    assert commit.message is not None
    assert commit.message_summary is not None
    assert isinstance(commit.parent_hashes, list)
    assert isinstance(commit.files_changed, list)
    assert isinstance(commit.is_merge, bool)
