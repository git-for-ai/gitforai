"""Prompt templates for LLM analysis."""

from typing import List


class PromptTemplates:
    """Collection of prompt templates for commit analysis."""

    @staticmethod
    def commit_intent_classification(commit_message: str, diff_summary: str = "") -> str:
        """Generate prompt for classifying commit intent.

        Args:
            commit_message: The commit message
            diff_summary: Optional summary of changes

        Returns:
            Formatted prompt
        """
        return f"""Analyze this Git commit and classify its intent into ONE of these categories:
- bug_fix: Fixes a bug or issue
- feature: Adds new functionality
- refactor: Code restructuring without changing behavior
- docs: Documentation changes only
- test: Test additions or modifications
- chore: Build, dependencies, or maintenance tasks
- style: Code style/formatting changes
- perf: Performance improvements

Commit message:
{commit_message}

{f"Changes: {diff_summary}" if diff_summary else ""}

Respond with ONLY the category name (e.g., "bug_fix" or "feature")."""

    @staticmethod
    def commit_summary(
        commit_message: str,
        files_changed: List[str],
        diff_snippet: str = "",
    ) -> str:
        """Generate prompt for summarizing a commit.

        Args:
            commit_message: The commit message
            files_changed: List of changed files
            diff_snippet: Optional code diff snippet

        Returns:
            Formatted prompt
        """
        files_list = "\n".join(f"- {f}" for f in files_changed[:10])
        if len(files_changed) > 10:
            files_list += f"\n... and {len(files_changed) - 10} more files"

        return f"""Provide a concise 1-2 sentence summary of what this commit does and why.
Focus on the purpose and impact, not just what was changed.

Commit message:
{commit_message}

Files changed:
{files_list}

{f"Code changes:\n{diff_snippet[:500]}" if diff_snippet else ""}

Summary:"""

    @staticmethod
    def topic_extraction(commit_message: str, diff_summary: str = "") -> str:
        """Generate prompt for extracting topics from a commit.

        Args:
            commit_message: The commit message
            diff_summary: Optional summary of changes

        Returns:
            Formatted prompt
        """
        return f"""Extract 2-5 key topics or themes from this commit.
Topics should be specific technical concepts, features, or components.

Examples of good topics: "authentication", "database migration", "API endpoint", "error handling"

Commit message:
{commit_message}

{f"Changes: {diff_summary}" if diff_summary else ""}

Respond with a comma-separated list of topics (e.g., "authentication, security, JWT tokens")."""

    @staticmethod
    def diff_explanation(
        file_path: str,
        diff_text: str,
        context_before: str = "",
        context_after: str = "",
    ) -> str:
        """Generate prompt for explaining a code diff.

        Args:
            file_path: Path to the changed file
            diff_text: The diff text
            context_before: Code context before changes
            context_after: Code context after changes

        Returns:
            Formatted prompt
        """
        return f"""Explain what changed in this code diff and why it matters.
Focus on the functional impact and any potential implications.

File: {file_path}

{f"Before:\n{context_before[:300]}\n" if context_before else ""}

Changes:
{diff_text[:800]}

{f"After:\n{context_after[:300]}\n" if context_after else ""}

Provide a brief explanation (2-3 sentences):"""

    @staticmethod
    def code_change_reasoning(
        commit_message: str,
        file_path: str,
        diff_text: str,
    ) -> str:
        """Generate prompt for inferring the reasoning behind a code change.

        Args:
            commit_message: The commit message
            file_path: Path to the changed file
            diff_text: The diff text

        Returns:
            Formatted prompt
        """
        return f"""Based on the commit message and code changes, explain the likely reasoning or motivation for this change.
What problem was being solved? What improvement was being made?

Commit message:
{commit_message}

File: {file_path}

Changes:
{diff_text[:600]}

Reasoning:"""

    @staticmethod
    def batch_commit_analysis(commits: List[dict]) -> str:
        """Generate prompt for analyzing multiple commits together.

        Args:
            commits: List of commit dictionaries with message and metadata

        Returns:
            Formatted prompt
        """
        commits_text = "\n\n".join(
            f"Commit {i+1}:\n{c.get('message', '')}\nFiles: {', '.join(c.get('files', [])[:3])}"
            for i, c in enumerate(commits[:5])
        )

        return f"""Analyze these related commits and identify:
1. Common themes or patterns
2. The overall goal or feature being developed
3. Any significant architectural changes

Commits:
{commits_text}

Analysis:"""

    @staticmethod
    def function_impact_analysis(
        function_name: str,
        old_code: str,
        new_code: str,
    ) -> str:
        """Generate prompt for analyzing impact of function changes.

        Args:
            function_name: Name of the function
            old_code: Original function code
            new_code: Modified function code

        Returns:
            Formatted prompt
        """
        return f"""Analyze the changes to this function and describe:
1. What changed functionally
2. Impact on callers or dependencies
3. Any potential breaking changes

Function: {function_name}

Before:
{old_code[:400]}

After:
{new_code[:400]}

Impact analysis:"""
