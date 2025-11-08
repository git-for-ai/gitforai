"""Delta detection for incremental updates.

Detects new commits since the last indexing operation and handles
edge cases like rebases and force pushes.
"""

from typing import Dict, List, Optional, Tuple

import git


class DeltaDetector:
    """Detects changes in a Git repository since last indexing.

    Identifies new commits, detects rebases, and provides statistics
    about what needs to be indexed.
    """

    def __init__(self, repo: git.Repo):
        """Initialize the delta detector.

        Args:
            repo: GitPython repository object
        """
        self.repo = repo

    def find_new_commits(
        self, last_indexed_commit: str, branch: Optional[str] = None
    ) -> List[git.Commit]:
        """Find all commits since the last indexed commit.

        Args:
            last_indexed_commit: SHA of the last indexed commit
            branch: Branch to check (defaults to current branch)

        Returns:
            List of new commits in chronological order (oldest first)

        Raises:
            ValueError: If last_indexed_commit doesn't exist in the repository
        """
        if branch is None:
            branch = self.repo.active_branch.name

        # Verify the last indexed commit exists
        try:
            last_commit_obj = self.repo.commit(last_indexed_commit)
        except (git.BadName, ValueError) as e:
            raise ValueError(
                f"Last indexed commit {last_indexed_commit} not found in repository"
            ) from e

        # Get the current HEAD for the branch
        try:
            head_commit = self.repo.commit(branch)
        except (git.BadName, ValueError) as e:
            raise ValueError(f"Branch {branch} not found in repository") from e

        # If HEAD is the same as last indexed, no new commits
        if head_commit.hexsha == last_indexed_commit:
            return []

        # Get commits between last indexed and HEAD
        # Format: "last_commit..HEAD" means all commits reachable from HEAD
        # but not from last_commit
        try:
            # Use rev_list to get commits in chronological order
            commit_range = f"{last_indexed_commit}..{branch}"
            commits = list(self.repo.iter_commits(commit_range, reverse=True))
            return commits
        except git.GitCommandError as e:
            # This can happen if the last_indexed_commit is not an ancestor of HEAD
            # (i.e., a rebase or force push occurred)
            raise ValueError(
                f"Commit history has changed (possible rebase/force push): {e}"
            ) from e

    def detect_rebase(self, last_indexed_commit: str) -> Tuple[bool, Optional[str]]:
        """Detect if a rebase or force push has occurred.

        A rebase is detected when the last indexed commit is not an ancestor
        of the current HEAD.

        Args:
            last_indexed_commit: SHA of the last indexed commit

        Returns:
            Tuple of (is_rebased, new_base_commit)
            - is_rebased: True if a rebase/force push was detected
            - new_base_commit: SHA of a suitable commit to use as new base, or None
        """
        try:
            last_commit_obj = self.repo.commit(last_indexed_commit)
        except (git.BadName, ValueError):
            # Commit doesn't exist at all - definitely a rebase/force push
            return (True, None)

        try:
            head_commit = self.repo.head.commit

            # Check if last_indexed_commit is an ancestor of HEAD
            # If it is, no rebase occurred
            merge_base = self.repo.merge_base(last_commit_obj, head_commit)

            if merge_base and merge_base[0].hexsha == last_indexed_commit:
                # last_indexed_commit is an ancestor of HEAD - no rebase
                return (False, None)
            else:
                # last_indexed_commit is not an ancestor - rebase detected
                # The merge base is a suitable commit to use as the new starting point
                if merge_base:
                    return (True, merge_base[0].hexsha)
                else:
                    # No common ancestor - complete history rewrite
                    return (True, None)
        except git.GitCommandError:
            # Error checking ancestry - assume rebase
            return (True, None)

    def get_stats(
        self, last_indexed_commit: Optional[str] = None, branch: Optional[str] = None
    ) -> Dict[str, any]:
        """Get statistics about the repository state.

        Args:
            last_indexed_commit: SHA of last indexed commit (optional)
            branch: Branch to check (defaults to current branch)

        Returns:
            Dictionary with statistics:
            - current_branch: Name of current branch
            - current_head: SHA of current HEAD
            - total_commits: Total commits in repository
            - new_commits: Number of new commits (if last_indexed_commit provided)
            - is_rebased: Whether a rebase was detected (if last_indexed_commit provided)
        """
        if branch is None:
            branch = self.repo.active_branch.name

        stats = {
            "current_branch": branch,
            "current_head": self.repo.head.commit.hexsha,
            "total_commits": len(list(self.repo.iter_commits(branch))),
        }

        if last_indexed_commit:
            # Check for rebase
            is_rebased, new_base = self.detect_rebase(last_indexed_commit)
            stats["is_rebased"] = is_rebased

            if is_rebased:
                stats["new_commits"] = None  # Can't count reliably
                stats["rebase_detected"] = True
                stats["suggested_base"] = new_base
            else:
                # Count new commits
                try:
                    new_commits = self.find_new_commits(last_indexed_commit, branch)
                    stats["new_commits"] = len(new_commits)
                    stats["rebase_detected"] = False
                except ValueError:
                    stats["new_commits"] = None
                    stats["rebase_detected"] = True

        return stats

    def get_all_commits(
        self,
        branch: Optional[str] = None,
        max_count: Optional[int] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[git.Commit]:
        """Get all commits on a branch with optional filters.

        Args:
            branch: Branch to get commits from (defaults to current branch)
            max_count: Maximum number of commits to return
            since: Only commits more recent than this date (ISO format)
            until: Only commits older than this date (ISO format)

        Returns:
            List of commits in reverse chronological order (newest first)
        """
        if branch is None:
            branch = self.repo.active_branch.name

        kwargs = {"reverse": True}  # Chronological order (oldest first)
        if max_count:
            kwargs["max_count"] = max_count
        if since:
            kwargs["since"] = since
        if until:
            kwargs["until"] = until

        return list(self.repo.iter_commits(branch, **kwargs))
