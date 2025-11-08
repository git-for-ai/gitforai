"""Git repository data extraction."""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import git
from git import Commit, Diff, Repo

from gitforai.models import CommitMetadata, FileDiff, FileSnapshot, RepositoryConfig


class GitExtractor:
    """Extracts data from a Git repository."""

    def __init__(self, config: RepositoryConfig) -> None:
        """Initialize the GitExtractor.

        Args:
            config: Repository configuration

        Raises:
            ValueError: If repository path is invalid
        """
        self.config = config
        if not config.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {config.repo_path}")

        try:
            self.repo = Repo(config.repo_path)
        except git.exc.InvalidGitRepositoryError as e:
            raise ValueError(f"Invalid Git repository: {config.repo_path}") from e

    def extract_all_commits(
        self,
        branch: str = "HEAD",
        max_count: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterator[CommitMetadata]:
        """Extract metadata for all commits in the repository.

        Args:
            branch: Branch name to extract from (default: HEAD)
            max_count: Maximum number of commits to extract
            since: Only commits after this date
            until: Only commits before this date

        Yields:
            CommitMetadata objects
        """
        kwargs = {}
        if max_count:
            kwargs["max_count"] = max_count
        if since:
            kwargs["since"] = since
        if until:
            kwargs["until"] = until

        for commit in self.repo.iter_commits(branch, **kwargs):
            yield self._extract_commit_metadata(commit)

    def extract_commit(self, commit_hash: str) -> CommitMetadata:
        """Extract metadata for a specific commit.

        Args:
            commit_hash: Commit hash (full or short)

        Returns:
            CommitMetadata object

        Raises:
            ValueError: If commit not found
        """
        try:
            commit = self.repo.commit(commit_hash)
            return self._extract_commit_metadata(commit)
        except (git.exc.BadName, ValueError) as e:
            raise ValueError(f"Commit not found: {commit_hash}") from e

    def extract_commit_diffs(self, commit_hash: str) -> List[FileDiff]:
        """Extract diffs for all files changed in a commit.

        Args:
            commit_hash: Commit hash

        Returns:
            List of FileDiff objects
        """
        commit = self.repo.commit(commit_hash)
        diffs = []

        # Handle first commit (no parent)
        if not commit.parents:
            for item in commit.tree.traverse():
                if item.type == "blob":
                    diffs.append(
                        FileDiff(
                            commit_hash=commit.hexsha,
                            file_path=item.path,
                            change_type="added",
                            diff_text="",
                            additions=0,
                            deletions=0,
                            is_binary=False,
                        )
                    )
            return diffs

        # Compare with parent(s)
        parent = commit.parents[0]
        diff_index = parent.diff(commit, create_patch=True)

        for diff_item in diff_index:
            file_diff = self._extract_file_diff(commit.hexsha, diff_item)
            if file_diff and self._should_include_file(file_diff.file_path):
                diffs.append(file_diff)

        return diffs

    def extract_file_snapshot(self, commit_hash: str, file_path: str) -> Optional[FileSnapshot]:
        """Extract the complete content of a file at a specific commit.

        Args:
            commit_hash: Commit hash
            file_path: Path to file in repository

        Returns:
            FileSnapshot object or None if file doesn't exist or is excluded

        Raises:
            ValueError: If commit not found
        """
        if not self._should_include_file(file_path):
            return None

        try:
            commit = self.repo.commit(commit_hash)
            try:
                blob = commit.tree / file_path
            except KeyError:
                # File doesn't exist in this commit
                return None

            if blob.type != "blob":
                return None

            # Check file size
            if blob.size > self.config.max_file_size_bytes:
                return None

            # Get file content
            try:
                content = blob.data_stream.read().decode("utf-8")
                is_binary = False
            except UnicodeDecodeError:
                if self.config.process_binary_files:
                    content = "<binary file>"
                    is_binary = True
                else:
                    return None

            file_extension = Path(file_path).suffix

            return FileSnapshot(
                commit_hash=commit.hexsha,
                file_path=file_path,
                content=content,
                size_bytes=blob.size,
                timestamp=datetime.fromtimestamp(commit.committed_date),
                file_extension=file_extension,
                is_binary=is_binary,
            )

        except (git.exc.BadName, ValueError) as e:
            raise ValueError(f"Commit not found: {commit_hash}") from e

    def extract_all_snapshots(self, commit_hash: str) -> List[FileSnapshot]:
        """Extract snapshots for all files in a commit.

        Args:
            commit_hash: Commit hash

        Returns:
            List of FileSnapshot objects
        """
        snapshots = []
        commit = self.repo.commit(commit_hash)

        for item in commit.tree.traverse():
            if item.type == "blob":
                snapshot = self.extract_file_snapshot(commit_hash, item.path)
                if snapshot:
                    snapshots.append(snapshot)

        return snapshots

    def _extract_commit_metadata(self, commit: Commit) -> CommitMetadata:
        """Extract metadata from a GitPython Commit object.

        Args:
            commit: GitPython Commit object

        Returns:
            CommitMetadata object
        """
        # Get list of changed files
        files_changed = []
        if commit.parents:
            parent = commit.parents[0]
            diff_index = parent.diff(commit)
            files_changed = [
                diff.a_path or diff.b_path
                for diff in diff_index
                if self._should_include_file(diff.a_path or diff.b_path or "")
            ]

        # Extract message summary (first line)
        message_lines = commit.message.strip().split("\n")
        message_summary = message_lines[0] if message_lines else ""

        # Extract commit statistics (lines added/deleted)
        stats = None
        try:
            git_stats = commit.stats.total
            from gitforai.models.commit import CommitStats
            stats = CommitStats(
                insertions=git_stats.get("insertions", 0),
                deletions=git_stats.get("deletions", 0),
            )
        except Exception:
            # Stats not available (e.g., initial commit)
            pass

        # Extract diff preview (first 50 lines of changes)
        diff_preview = None
        if commit.parents:
            try:
                parent = commit.parents[0]
                diff_index = parent.diff(commit, create_patch=True)
                diff_lines = []
                line_count = 0

                for diff in diff_index:
                    # Skip files we don't want to include
                    file_path = diff.a_path or diff.b_path
                    if not file_path or not self._should_include_file(file_path):
                        continue

                    # Get diff text for this file
                    try:
                        diff_text = diff.diff.decode("utf-8") if diff.diff else ""
                        if diff_text:
                            diff_lines.append(f"--- {diff.a_path or 'null'}")
                            diff_lines.append(f"+++ {diff.b_path or 'null'}")
                            line_count += 2

                            for line in diff_text.split("\n"):
                                if line_count >= 50:
                                    break
                                diff_lines.append(line)
                                line_count += 1

                            if line_count >= 50:
                                break
                    except (UnicodeDecodeError, AttributeError):
                        # Skip binary or problematic files
                        continue

                if diff_lines:
                    diff_preview = "\n".join(diff_lines)
                    if line_count >= 50:
                        diff_preview += "\n... (truncated)"
            except Exception:
                # Diff extraction failed - continue without preview
                pass

        return CommitMetadata(
            hash=commit.hexsha,
            short_hash=commit.hexsha[:7],
            author_name=commit.author.name,
            author_email=commit.author.email,
            committer_name=commit.committer.name,
            committer_email=commit.committer.email,
            timestamp=datetime.fromtimestamp(commit.committed_date),
            message=commit.message.strip(),
            message_summary=message_summary,
            parent_hashes=[p.hexsha for p in commit.parents],
            files_changed=files_changed,
            is_merge=len(commit.parents) > 1,
            stats=stats,
            diff_preview=diff_preview,
        )

    def _extract_file_diff(self, commit_hash: str, diff: Diff) -> Optional[FileDiff]:
        """Extract diff information from a GitPython Diff object.

        Args:
            commit_hash: Commit hash
            diff: GitPython Diff object

        Returns:
            FileDiff object or None
        """
        file_path = diff.b_path or diff.a_path
        if not file_path:
            return None

        # Determine change type
        if diff.new_file:
            change_type = "added"
        elif diff.deleted_file:
            change_type = "deleted"
        elif diff.renamed_file:
            change_type = "renamed"
        else:
            change_type = "modified"

        # Get diff text
        try:
            diff_text = diff.diff.decode("utf-8") if diff.diff else ""
            is_binary = False
        except UnicodeDecodeError:
            diff_text = "<binary diff>"
            is_binary = True

        # Count additions and deletions (rough estimate from diff text)
        additions = 0
        deletions = 0
        if not is_binary and diff_text:
            for line in diff_text.split("\n"):
                if line.startswith("+") and not line.startswith("+++"):
                    additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    deletions += 1

        return FileDiff(
            commit_hash=commit_hash,
            file_path=file_path,
            change_type=change_type,
            old_path=diff.a_path if diff.renamed_file else None,
            diff_text=diff_text,
            additions=additions,
            deletions=deletions,
            is_binary=is_binary,
        )

    def _should_include_file(self, file_path: str) -> bool:
        """Check if a file should be included based on configuration.

        Args:
            file_path: File path

        Returns:
            True if file should be included
        """
        # Check excluded paths
        for excluded in self.config.excluded_paths:
            if excluded in file_path:
                return False

        # Check included extensions
        if not self.config.included_extensions:
            return True

        file_extension = Path(file_path).suffix
        return file_extension in self.config.included_extensions
