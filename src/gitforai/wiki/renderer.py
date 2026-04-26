"""Wiki page renderers.

Two renderers, sharing factual sections:

- ``NaiveRenderer``: deterministic, no LLM. Produces Files / Recent Activity /
  Authors / History / Linked Tasks. Fast, free, useful as a baseline and
  fallback when an LLM provider is unavailable.

- ``LLMRenderer``: subclasses ``NaiveRenderer`` and replaces the page Title
  and Overview with LLM-generated prose. All factual sections stay
  deterministic (no LLM hallucination on data that we can compute exactly).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

from gitforai.integrations.replay import (
    CommitTaskLink,
    FileCluster,
    ReplayResult,
    TaskState,
    TemporalEpic,
)
from gitforai.llm.base import BaseLLMProvider
from gitforai.llm.cache import LLMCache
from gitforai.models.commit import CommitMetadata

from .identity import PageSource, WikiPage, WikiSection
from .prompts import PROMPT_VERSION, parse_topic_response, topic_summary_prompt

logger = logging.getLogger(__name__)


class NaiveRenderer:
    """Renders WikiPage sections from replay data, no LLM involved."""

    def __init__(self, recent_commit_limit: int = 10, top_authors: int = 3):
        self.recent_commit_limit = recent_commit_limit
        self.top_authors = top_authors

    def render(
        self,
        page: WikiPage,
        cluster: FileCluster,
        replay_result: ReplayResult,
        commits_by_hash: Dict[str, CommitMetadata],
    ) -> WikiPage:
        """Populate ``page.sections`` in place and return the page."""
        cluster_files = set(cluster.files)

        # cluster.commits_touching_cluster is unreliable (replay.py never fills it),
        # so derive the commit list ourselves from file membership.
        cluster_commits = [
            c for c in commits_by_hash.values()
            if cluster_files.intersection(c.files_changed or [])
        ]
        cluster_commits.sort(key=lambda c: c.timestamp, reverse=True)
        latest_commit = cluster_commits[0] if cluster_commits else None

        page.sections = [
            self._render_overview(cluster, cluster_commits),
            self._render_files(cluster),
            self._render_recent_activity(cluster_commits),
            self._render_authors(cluster_commits),
            self._render_history(cluster, cluster_commits, replay_result),
            self._render_linked_tasks(cluster_files, replay_result),
        ]
        # Drop empty sections
        page.sections = [s for s in page.sections if s.body.strip()]

        if latest_commit is not None:
            page.last_synced_commit = latest_commit.hash
        page.last_synced_at = datetime.now(timezone.utc)
        return page

    # ---- sections -------------------------------------------------------

    def _render_overview(
        self,
        cluster: FileCluster,
        commits: List[CommitMetadata],
    ) -> WikiSection:
        first = commits[-1].timestamp.date() if commits else None
        last = commits[0].timestamp.date() if commits else None
        lines = [
            f"This page tracks **{len(cluster.files)} file(s)** that frequently change together "
            f"(co-change frequency: {cluster.co_change_frequency:.2f}).",
            "",
            f"- Commits touching these files: **{len(commits)}**",
        ]
        if first and last:
            lines.append(f"- Active range: **{first.isoformat()} → {last.isoformat()}**")
        return WikiSection(
            title="Overview",
            body="\n".join(lines),
            source=PageSource.AUTO,
            source_commits=[c.hash for c in commits[: self.recent_commit_limit]],
        )

    def _render_files(self, cluster: FileCluster) -> WikiSection:
        lines = [f"- `{f}`" for f in sorted(cluster.files)]
        return WikiSection(
            title="Files",
            body="\n".join(lines),
            source=PageSource.AUTO,
        )

    def _render_recent_activity(self, commits: List[CommitMetadata]) -> WikiSection:
        if not commits:
            return WikiSection(title="Recent Activity", body="", source=PageSource.AUTO)
        recent = commits[: self.recent_commit_limit]
        rows = ["| Date | Commit | Author | Subject |", "|---|---|---|---|"]
        for c in recent:
            subject = (c.message_summary or c.message.splitlines()[0] or "").strip()
            subject = subject.replace("|", "\\|")[:80]
            rows.append(
                f"| {c.timestamp.date().isoformat()} | `{c.short_hash}` | {c.author_name} | {subject} |"
            )
        return WikiSection(
            title="Recent Activity",
            body="\n".join(rows),
            source=PageSource.AUTO,
            source_commits=[c.hash for c in recent],
        )

    def _render_authors(self, commits: List[CommitMetadata]) -> WikiSection:
        if not commits:
            return WikiSection(title="Authors", body="", source=PageSource.AUTO)
        counter = Counter(c.author_name for c in commits)
        top = counter.most_common(self.top_authors)
        lines = [f"- **{name}** — {n} commit(s)" for name, n in top]
        return WikiSection(title="Authors", body="\n".join(lines), source=PageSource.AUTO)

    def _render_history(
        self,
        cluster: FileCluster,
        commits: List[CommitMetadata],
        replay_result: ReplayResult,
    ) -> WikiSection:
        if not replay_result.temporal_epics or not commits:
            return WikiSection(title="History", body="", source=PageSource.AUTO)

        commit_hash_set = {c.hash for c in commits}
        relevant: List[TemporalEpic] = []
        for epic in replay_result.temporal_epics:
            if commit_hash_set.intersection(epic.commits):
                relevant.append(epic)

        if not relevant:
            return WikiSection(title="History", body="", source=PageSource.AUTO)

        relevant.sort(key=lambda e: e.start_date, reverse=True)
        lines = []
        for epic in relevant[:10]:
            overlap = len(commit_hash_set.intersection(epic.commits))
            lines.append(
                f"- **{epic.title}** "
                f"({epic.start_date.date()} → {epic.end_date.date()}, "
                f"{overlap}/{epic.commit_count} commits in this cluster)"
            )
        return WikiSection(title="History", body="\n".join(lines), source=PageSource.AUTO)

    def _render_linked_tasks(
        self,
        cluster_files: set,
        replay_result: ReplayResult,
    ) -> WikiSection:
        if not replay_result.commit_task_links or not replay_result.tasks:
            return WikiSection(title="Linked Tasks", body="", source=PageSource.AUTO)

        tasks_by_id: Dict[str, TaskState] = {t.task_id: t for t in replay_result.tasks}

        # Pick tasks whose related commits touched any file in this cluster.
        relevant_task_ids: set = set()
        for link in replay_result.commit_task_links:
            task = tasks_by_id.get(link.task_id)
            if not task:
                continue
            # TaskState has file_locations populated by Layer 4
            if cluster_files.intersection(getattr(task, "file_locations", []) or []):
                relevant_task_ids.add(link.task_id)

        if not relevant_task_ids:
            return WikiSection(title="Linked Tasks", body="", source=PageSource.AUTO)

        lines = []
        for tid in sorted(relevant_task_ids)[:20]:
            task = tasks_by_id[tid]
            title = (getattr(task, "title", "") or "").strip()
            # Suppress synthetic placeholders ("Task TASK-123" etc.) — only show
            # tasks where the replay engine resolved a real human-authored title.
            if not title or title.lower().startswith("task ") and tid.lower() in title.lower():
                continue
            status = getattr(task, "status", "?")
            lines.append(f"- `{tid}` [{status}] {title}")
        if not lines:
            return WikiSection(title="Linked Tasks", body="", source=PageSource.AUTO)
        return WikiSection(title="Linked Tasks", body="\n".join(lines), source=PageSource.AUTO)


# ---------------------------------------------------------------------------
# LLM renderer
# ---------------------------------------------------------------------------


class LLMRenderer(NaiveRenderer):
    """Augments the naive renderer with LLM-generated title + summary prose.

    The factual sections (Files, Recent Activity, Authors, History, Linked
    Tasks) are inherited verbatim from ``NaiveRenderer`` — only the page
    title and the leading prose section come from the LLM. This keeps the
    risk of hallucinated facts to zero on the deterministic parts, and gives
    the LLM a tight, well-scoped job (topic naming + 2-4 sentence intro).

    Caching: the prompt is the cache key (via ``LLMCache``), so unchanged
    clusters cost nothing on a regen. Bump ``PROMPT_VERSION`` in
    ``wiki/prompts.py`` to force re-generation after a prompt edit.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        cache: Optional[LLMCache] = None,
        recent_commit_limit: int = 10,
        top_authors: int = 3,
        commits_for_prompt: int = 12,
        max_tokens: int = 400,
        temperature: float = 0.2,
        precomputed_responses: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            provider: Real LLM provider for completions. May be None if
                ``precomputed_responses`` covers every page being rendered.
            cache: Optional response cache.
            precomputed_responses: Map of ``page_id -> raw completion text``
                (in the ``TITLE:/SUMMARY:`` format). When a page_id is present
                here the renderer skips the provider entirely. Useful for the
                "Claude Code session as the LLM" workflow and as a way to
                replay-from-disk for free.
        """
        super().__init__(recent_commit_limit=recent_commit_limit, top_authors=top_authors)
        self.provider = provider
        self.cache = cache
        self.commits_for_prompt = commits_for_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.precomputed_responses = precomputed_responses or {}

    async def render_async(
        self,
        page: WikiPage,
        cluster: FileCluster,
        replay_result: ReplayResult,
        commits_by_hash: Dict[str, CommitMetadata],
    ) -> WikiPage:
        """Render with LLM prose; fall back to naive on any LLM error."""
        # Always start from the deterministic baseline.
        super().render(page, cluster, replay_result, commits_by_hash)

        # Compute the inputs for the LLM prompt from already-derived facts.
        cluster_files = set(cluster.files)
        cluster_commits = [
            c for c in commits_by_hash.values()
            if cluster_files.intersection(c.files_changed or [])
        ]
        cluster_commits.sort(key=lambda c: c.timestamp, reverse=True)

        commit_subjects = [
            (c.message_summary or c.message.splitlines()[0] or "").strip()
            for c in cluster_commits[: self.commits_for_prompt]
        ]
        commit_subjects = [s for s in commit_subjects if s]

        prompt = topic_summary_prompt(sorted(cluster.files), commit_subjects)

        # Precomputed responses skip the provider entirely.
        if page.page_id in self.precomputed_responses:
            completion = self.precomputed_responses[page.page_id]
        elif self.provider is not None:
            completion = await self._complete_cached(prompt)
        else:
            logger.warning(
                "No LLM provider and no precomputed response for '%s'; keeping naive output.",
                page.page_id,
            )
            return page

        if not completion:
            logger.warning("LLM returned empty for page '%s'; keeping naive title.", page.title)
            return page

        title, summary = parse_topic_response(completion)
        if title:
            page.title = title.strip()
        if summary:
            summary_section = WikiSection(
                title="Summary",
                body=summary,
                source=PageSource.AUTO,
                source_commits=[c.hash for c in cluster_commits[: self.recent_commit_limit]],
            )
            # Prepend the Summary so it appears just under the page title.
            page.sections.insert(0, summary_section)
        return page

    async def _complete_cached(self, prompt: str) -> str:
        model = getattr(self.provider, "model", "unknown")
        cache_key_prompt = f"v{PROMPT_VERSION}\n{prompt}"

        if self.cache is not None:
            cached = self.cache.get_completion(cache_key_prompt, model)
            if cached is not None:
                return cached

        try:
            text = await self.provider.complete(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.warning("LLM completion failed (%s); falling back to naive.", e)
            return ""

        if self.cache is not None and text:
            self.cache.set_completion(cache_key_prompt, model, text)
        return text


# ---------------------------------------------------------------------------
# Provider auto-detection
# ---------------------------------------------------------------------------


def auto_detect_llm_provider(preferred: Optional[str] = None) -> Optional[BaseLLMProvider]:
    """Return a completion-capable LLM provider from env, or None.

    Args:
        preferred: ``"anthropic"``, ``"openai"``, or None for auto.

    Order:
    1. If ``preferred`` is set, only that provider is attempted.
    2. Otherwise: Anthropic (if ANTHROPIC_API_KEY) → OpenAI (if OPENAI_API_KEY).

    Returns ``None`` (and logs at INFO level) if no usable provider is
    available, so callers can fall back to the naive renderer cleanly.
    """
    import os

    def _try_anthropic() -> Optional[BaseLLMProvider]:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        try:
            from gitforai.llm.anthropic_provider import AnthropicProvider
            model = os.environ.get("ANTHROPIC_MODEL") or AnthropicProvider.DEFAULT_MODEL
            return AnthropicProvider(api_key=key, model=model)
        except ImportError as e:
            logger.warning("Anthropic SDK unavailable (%s).", e)
            return None

    def _try_openai() -> Optional[BaseLLMProvider]:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        try:
            from gitforai.llm.openai_provider import OpenAIProvider
            model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
            return OpenAIProvider(api_key=key, model=model)
        except ImportError as e:
            logger.warning("OpenAI SDK unavailable (%s).", e)
            return None

    if preferred == "anthropic":
        return _try_anthropic()
    if preferred == "openai":
        return _try_openai()
    if preferred == "none":
        return None

    return _try_anthropic() or _try_openai()
