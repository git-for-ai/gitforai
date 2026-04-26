"""Drift detection for auto-generated wiki pages.

A page is *drifted* when commits affecting its files have landed since the
page was last synced. This module produces a per-page ``DriftReport`` that an
orchestrator (or a human running ``check_wiki_drift.py``) can use to decide
which pages to regenerate.

Step 3 implements the deterministic, free signal:

- ``new_commits`` = commits with timestamp > ``last_synced_at`` whose
  ``files_changed`` intersects ``page.files``.
- ``drifted_sections`` = sections whose AUTO ``source_commits`` don't already
  cover those new commits.
- ``drift_score`` = ``min(1.0, new_commits / max_drift_commits)``.

Step 4 will layer a semantic drift signal on top, by embedding each AUTO
section and querying ``mcp__gitforai__find_similar_commits`` to detect cases
where the section's *content* has been contradicted by a commit that doesn't
touch the cluster's files (e.g., a behavior change documented elsewhere).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from gitforai.models.commit import CommitMetadata

from .identity import PageSource, WikiPage

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    page_id: str
    page_title: str
    last_synced_commit: Optional[str]
    last_synced_at: Optional[datetime]
    new_commit_hashes: List[str] = field(default_factory=list)
    new_commit_subjects: List[str] = field(default_factory=list)
    drifted_section_titles: List[str] = field(default_factory=list)
    drift_score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    @property
    def is_drifted(self) -> bool:
        return self.drift_score > 0.0

    @property
    def severity(self) -> str:
        if self.drift_score == 0.0:
            return "fresh"
        if self.drift_score < 0.3:
            return "minor"
        if self.drift_score < 0.7:
            return "moderate"
        return "severe"


class DriftDetector:
    """Compute drift reports for one or more WikiPages.

    Uses only the deterministic file-touch signal. Pages that have never been
    synced (``last_synced_commit`` is None) are reported as drifted with the
    highest severity, since they need an initial render.
    """

    def __init__(self, max_drift_commits: int = 5):
        """
        Args:
            max_drift_commits: Number of new-since-sync commits at which
                ``drift_score`` saturates at 1.0. Lower values make pages flip
                to "severe" sooner; higher values are more forgiving.
        """
        self.max_drift_commits = max(1, max_drift_commits)

    def detect(
        self,
        page: WikiPage,
        commits_by_hash: Dict[str, CommitMetadata],
    ) -> DriftReport:
        report = DriftReport(
            page_id=page.page_id,
            page_title=page.title,
            last_synced_commit=page.last_synced_commit,
            last_synced_at=page.last_synced_at,
        )

        if not page.files:
            report.reasons.append(
                "Page has no associated files; cannot compute file-touch drift."
            )
            return report

        if page.last_synced_commit is None:
            # Never rendered. Treat as fully drifted so it gets initial content.
            report.drift_score = 1.0
            report.reasons.append("Page has never been synced.")
            return report

        page_files = set(page.files)
        cutoff_time = _ensure_aware(page.last_synced_at)
        last_synced = page.last_synced_commit

        new_commits: List[CommitMetadata] = []
        for commit in commits_by_hash.values():
            if commit.hash == last_synced:
                continue
            if cutoff_time and _ensure_aware(commit.timestamp) <= cutoff_time:
                continue
            if not page_files.intersection(commit.files_changed or []):
                continue
            new_commits.append(commit)

        new_commits.sort(key=lambda c: c.timestamp, reverse=True)
        new_hashes = {c.hash for c in new_commits}

        report.new_commit_hashes = [c.hash for c in new_commits]
        report.new_commit_subjects = [
            (c.message_summary or c.message.splitlines()[0] or "").strip()
            for c in new_commits
        ]
        report.drift_score = min(1.0, len(new_commits) / self.max_drift_commits)

        # Section-level drift: any AUTO section whose source_commits set is a
        # subset of "stuff before sync" misses anything that's landed since.
        # We mark every AUTO section as drifted when there's *any* new commit,
        # since the deterministic signal can't tell which section should
        # re-mention the new work without reading content.
        if new_commits:
            for section in page.sections:
                if section.source == PageSource.AUTO:
                    report.drifted_section_titles.append(section.title)
            report.reasons.append(
                f"{len(new_commits)} commit(s) touched this page's files since last sync."
            )

        return report

    def detect_all(
        self,
        pages: List[WikiPage],
        commits_by_hash: Dict[str, CommitMetadata],
    ) -> List[DriftReport]:
        return [self.detect(p, commits_by_hash) for p in pages]


def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to timezone-aware (assumes UTC if naive).

    Required because ``synced_at`` in older pages was written with
    ``datetime.utcnow()`` (naive), while git commit timestamps come back
    timezone-aware. Comparing the two raises ``TypeError`` without this.
    """
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)
