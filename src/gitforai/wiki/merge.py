"""Merge a freshly-rendered page with the human-authored content of an
existing on-disk page.

Contract:
- AUTO sections in ``existing`` are *replaced* by their fresh equivalents
  (matched by case-insensitive title) — so the model gets to update its
  own prose and tables.
- HUMAN sections in ``existing`` are preserved verbatim.
- The relative ordering of HUMAN sections within the existing page is kept,
  so a HUMAN section that lived between Summary and Overview stays there.
- AUTO sections that are new in ``fresh`` (no equivalent in ``existing``)
  are appended in fresh order.
- AUTO sections that disappeared (existed before but no longer rendered)
  are dropped — they are auto-generated, so deleting them is safe.
- Page-level metadata (title, last_synced_*) comes from ``fresh``.
"""

from __future__ import annotations

from typing import List

from .identity import PageSource, WikiPage, WikiSection


def merge_human_edits(fresh: WikiPage, existing: WikiPage) -> WikiPage:
    """Return a new WikiPage that combines fresh AUTO content with existing HUMAN content."""
    fresh_by_title = {
        s.title.strip().lower(): s
        for s in fresh.sections
        if s.source != PageSource.HUMAN
    }
    used_titles: set = set()
    merged_sections: List[WikiSection] = []

    # Walk existing sections in order. AUTO -> swap in fresh; HUMAN -> keep.
    for section in existing.sections:
        title_key = section.title.strip().lower()
        if section.source == PageSource.HUMAN:
            merged_sections.append(section)
            continue
        replacement = fresh_by_title.get(title_key)
        if replacement is not None:
            merged_sections.append(replacement)
            used_titles.add(title_key)
        # else: AUTO section that's no longer produced — drop silently.

    # Append any fresh AUTO sections that didn't have an existing counterpart.
    for section in fresh.sections:
        if section.source == PageSource.HUMAN:
            continue
        if section.title.strip().lower() in used_titles:
            continue
        merged_sections.append(section)

    return WikiPage(
        page_id=fresh.page_id,
        title=fresh.title,
        cluster_id=fresh.cluster_id,
        files=fresh.files,
        sections=merged_sections,
        last_synced_commit=fresh.last_synced_commit,
        last_synced_at=fresh.last_synced_at,
        drift_score=0.0,
    )
