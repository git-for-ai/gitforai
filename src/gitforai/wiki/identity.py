"""Wiki page identity model.

A WikiPage is anchored to a FileCluster (Layer 4 of the replay engine), but its
page_id is derived from the *sorted file list* rather than the cluster_id, so
that pages survive small reshufflings of cluster boundaries between runs.
"""

from __future__ import annotations

import hashlib
import os.path
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from gitforai.integrations.replay import FileCluster, ReplayResult


class PageSource(str, Enum):
    """Provenance of a section's content."""

    AUTO = "auto"      # generated from git history; safe to regenerate
    HUMAN = "human"    # human-authored; preserve verbatim across regen
    MIXED = "mixed"    # auto-generated then edited; needs review


@dataclass
class WikiSection:
    title: str
    body: str
    source: PageSource = PageSource.AUTO
    source_commits: List[str] = field(default_factory=list)


@dataclass
class WikiPage:
    page_id: str                       # stable across regens (used as filename)
    title: str                         # human-readable
    cluster_id: str                    # FileCluster this page is anchored to
    files: List[str] = field(default_factory=list)
    sections: List[WikiSection] = field(default_factory=list)
    last_synced_commit: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    drift_score: float = 0.0


# ---------------------------------------------------------------------------
# Page identity helpers
# ---------------------------------------------------------------------------

# GitHub wiki page filenames: keep ASCII letters/digits/dashes only.
_SLUG_BAD = re.compile(r"[^A-Za-z0-9]+")


def _slugify(text: str, max_len: int = 60) -> str:
    """Slugify into a GitHub-wiki-safe filename stem (no extension)."""
    slug = _SLUG_BAD.sub("-", text).strip("-")
    if not slug:
        slug = "Page"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug


def _file_list_fingerprint(files: List[str]) -> str:
    """Deterministic fingerprint of a sorted file list (8 hex chars)."""
    canonical = "\n".join(sorted(files)).encode("utf-8")
    return hashlib.sha1(canonical).hexdigest()[:8]


def _common_path_prefix(files: List[str]) -> str:
    """Longest shared *directory* prefix among files (may be empty)."""
    if not files:
        return ""
    common = os.path.commonpath(files) if len(files) > 1 else os.path.dirname(files[0])
    return common


def derive_page_title(cluster: FileCluster) -> str:
    """Derive a human-ish title from a FileCluster without an LLM.

    Strategy (in order):
    1. Strict common directory prefix (works when all files live in one folder).
    2. Majority-share directory: the deepest folder that contains > 50% of files.
    3. Analyzer's ``cluster.description`` (often a generic blurb like "6 .py files").
    4. ``Cluster <short>`` fallback.

    Why not use ``cluster.description`` first? The analyzer fills it with a
    generic count-based label that makes for poor wiki titles.
    """
    title = _title_from_prefix(_common_path_prefix(cluster.files))
    if title:
        return title

    title = _title_from_majority_prefix(cluster.files)
    if title:
        return title

    if cluster.description:
        return cluster.description

    return f"Cluster {cluster.cluster_id[:8]}"


def _title_from_prefix(prefix: str) -> str:
    """Convert a directory path into a Title Case label, skipping noise dirs."""
    if not prefix:
        return ""
    parts = [p for p in prefix.split(os.sep) if p and p not in {"src", "lib"}]
    if not parts:
        return ""
    tail = parts[-1]
    words = re.split(r"[-_]+", tail)
    return " ".join(w.capitalize() for w in words if w) or tail


def _title_from_majority_prefix(files: List[str], threshold: float = 0.5) -> str:
    """Return a title from the deepest directory that contains > ``threshold`` of files."""
    if not files:
        return ""
    # Count how many files live under each ancestor directory.
    dir_counts: dict = {}
    for f in files:
        d = os.path.dirname(f)
        while d:
            dir_counts[d] = dir_counts.get(d, 0) + 1
            new_d = os.path.dirname(d)
            if new_d == d:
                break
            d = new_d

    n = len(files)
    # Pick the deepest dir whose share exceeds the threshold.
    candidates = [
        (d, count) for d, count in dir_counts.items() if count / n > threshold
    ]
    if not candidates:
        return ""
    # Deepest = longest path
    candidates.sort(key=lambda x: (-x[0].count(os.sep), -x[1]))
    return _title_from_prefix(candidates[0][0])


def derive_page_id(cluster: FileCluster, title: str) -> str:
    """Stable, GitHub-wiki-compatible page identifier (also the filename stem).

    Uses ``<Slug>-<fingerprint>`` so a small reshuffle of cluster files doesn't
    silently rewrite an existing page, while readability stays good.
    """
    slug = _slugify(title)
    fp = _file_list_fingerprint(cluster.files)
    return f"{slug}-{fp}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def derive_pages_from_clusters(
    replay_result: ReplayResult,
    min_files: int = 2,
    max_files: int = 20,
    min_co_change_freq: float = 0.0,
) -> List[WikiPage]:
    """Build empty WikiPage stubs from the file clusters in a ReplayResult.

    Args:
        replay_result: Output of ``GitHistoryReplayEngine.replay_history()``.
        min_files: Skip clusters smaller than this (noise filter).
        max_files: Drop clusters larger than this (typically bulk-commit dumps
            that aren't real topics). Set to 0 to disable.
        min_co_change_freq: Skip clusters with weak co-change signal (0.0..1.0).
            Note: at low commit counts this field is uniformly ~1.0 and provides
            no useful filtering; it only becomes meaningful with rich history.

    Returns:
        One WikiPage stub per surviving cluster, with no sections rendered yet.
        Titles are disambiguated so no two pages share the same title.

    Note: ``cluster.commits_touching_cluster`` is unreliable (the analyzer
    leaves it empty — see ``FileCoChangeAnalyzer`` in replay.py); the renderer
    is responsible for computing the commit list from file membership.
    """
    pages: List[WikiPage] = []
    for cluster in replay_result.file_clusters:
        n_files = len(cluster.files)
        if n_files < min_files:
            continue
        if max_files > 0 and n_files > max_files:
            continue
        if cluster.co_change_frequency < min_co_change_freq:
            continue

        title = derive_page_title(cluster)
        page_id = derive_page_id(cluster, title)
        pages.append(
            WikiPage(
                page_id=page_id,
                title=title,
                cluster_id=cluster.cluster_id,
                files=list(cluster.files),
            )
        )

    _disambiguate_titles(pages)
    return pages


def _disambiguate_titles(pages: List[WikiPage]) -> None:
    """Append a distinguishing suffix to any pages that share a title.

    Strategy:
    1. Try to find a path segment present in this cluster's files but not in
       any colliding cluster's files. Use it as a parenthetical.
    2. Fall back to a file-count suffix.
    """
    by_title: dict = {}
    for p in pages:
        by_title.setdefault(p.title, []).append(p)

    for title, group in by_title.items():
        if len(group) < 2:
            continue

        # Collect path segments per page, then per page find segments unique to it.
        segs_per_page = [_path_segments(p.files) for p in group]
        for i, page in enumerate(group):
            others = set().union(*(segs_per_page[j] for j in range(len(group)) if j != i))
            unique = sorted(segs_per_page[i] - others, key=len, reverse=True)
            differentiator = unique[0] if unique else f"{len(page.files)} files"
            page.title = f"{title} ({differentiator})"


def _path_segments(files: List[str]) -> set:
    """Return the set of all directory + basename segments for a list of files."""
    segs: set = set()
    for f in files:
        parts = [p for p in f.split(os.sep) if p]
        # Use both directory segments and basename-without-extension
        for p in parts[:-1]:
            segs.add(p)
        if parts:
            stem = os.path.splitext(parts[-1])[0]
            if stem:
                segs.add(stem)
    return segs
