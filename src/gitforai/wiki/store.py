"""Read/write WikiPage objects as GitHub-wiki compatible markdown files.

Layout (flat, matching GitHub wiki conventions):

    <root>/
        Home.md            # landing page (auto-generated index)
        _Sidebar.md        # navigation
        <PageId>.md        # one file per WikiPage

Each generated section is wrapped in HTML-comment markers so the renderer can
later distinguish auto-generated content from human edits:

    <!-- gitforai:auto section="Overview" synced="abc123" -->
    ## Overview
    ...
    <!-- /gitforai:auto -->

A page-level meta line at the top stores cluster_id / page_id / sync state.
Anything outside ``gitforai:auto`` markers (and outside the meta block) is
treated as HUMAN content and preserved verbatim on round-trip.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from .identity import PageSource, WikiPage, WikiSection


META_LINE_RE = re.compile(
    r"<!--\s*gitforai:meta\s+(?P<attrs>[^>]+?)\s*-->",
    re.IGNORECASE,
)
AUTO_OPEN_RE = re.compile(
    r"<!--\s*gitforai:auto\s+(?P<attrs>[^>]+?)\s*-->",
    re.IGNORECASE,
)
AUTO_CLOSE_RE = re.compile(r"<!--\s*/gitforai:auto\s*-->", re.IGNORECASE)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


class WikiStore:
    """Filesystem-backed wiki store (one .md file per WikiPage)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- write ----------------------------------------------------------

    def write_page(self, page: WikiPage) -> Path:
        path = self.root / f"{page.page_id}.md"
        path.write_text(self._serialize_page(page), encoding="utf-8")
        return path

    def write_index(self, pages: Iterable[WikiPage]) -> Path:
        """Write Home.md and _Sidebar.md (GitHub wiki conventions)."""
        pages_sorted = sorted(pages, key=lambda p: p.title.lower())
        home = self.root / "Home.md"
        home.write_text(self._render_home(pages_sorted), encoding="utf-8")
        sidebar = self.root / "_Sidebar.md"
        sidebar.write_text(self._render_sidebar(pages_sorted), encoding="utf-8")
        return home

    # ---- read -----------------------------------------------------------

    def read_page(self, page_id: str) -> Optional[WikiPage]:
        path = self.root / f"{page_id}.md"
        if not path.exists():
            return None
        return self._parse_page(path.read_text(encoding="utf-8"), page_id)

    def list_page_ids(self) -> List[str]:
        skip = {"Home", "_Sidebar", "_Footer"}
        return sorted(
            p.stem for p in self.root.glob("*.md") if p.stem not in skip
        )

    # ---- serialization --------------------------------------------------

    def _serialize_page(self, page: WikiPage) -> str:
        lines: List[str] = [f"# {page.title}", ""]
        meta_attrs = {
            "page_id": page.page_id,
            "cluster_id": page.cluster_id,
            "synced": page.last_synced_commit or "",
            "synced_at": page.last_synced_at.isoformat() if page.last_synced_at else "",
        }
        lines.append(self._meta_comment(meta_attrs))
        lines.append("")

        for section in page.sections:
            if section.source == PageSource.HUMAN:
                # Human sections: emit verbatim (no markers)
                lines.append(f"## {section.title}")
                lines.append("")
                lines.append(section.body.rstrip())
                lines.append("")
                continue

            attrs = {
                "section": section.title,
                "synced": page.last_synced_commit or "",
            }
            if section.source_commits:
                attrs["commits"] = ",".join(section.source_commits[:8])
            lines.append(self._auto_open(attrs))
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.body.rstrip())
            lines.append("")
            lines.append("<!-- /gitforai:auto -->")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _parse_page(self, text: str, page_id: str) -> WikiPage:
        # Title: first H1
        title_match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else page_id

        # Meta
        meta_match = META_LINE_RE.search(text)
        meta = _parse_attrs(meta_match.group("attrs")) if meta_match else {}
        last_synced = meta.get("synced") or None
        last_synced_at_str = meta.get("synced_at") or ""
        try:
            last_synced_at = datetime.fromisoformat(last_synced_at_str) if last_synced_at_str else None
        except ValueError:
            last_synced_at = None

        sections = list(_parse_sections(text))

        # Pull file paths back out of the deterministic "Files" section so the
        # page round-trips losslessly. Falls back to an empty list if the page
        # was hand-authored without a Files section.
        files = _extract_files_from_sections(sections)

        return WikiPage(
            page_id=meta.get("page_id", page_id),
            title=title,
            cluster_id=meta.get("cluster_id", ""),
            files=files,
            sections=sections,
            last_synced_commit=last_synced,
            last_synced_at=last_synced_at,
        )

    # ---- index rendering -----------------------------------------------

    def _render_home(self, pages: List[WikiPage]) -> str:
        lines = [
            "# Wiki",
            "",
            "_Auto-generated from git history by GitForAI._",
            "",
            f"_{len(pages)} page(s) tracked._",
            "",
            "## Topics",
            "",
        ]
        for p in pages:
            lines.append(f"- [{p.title}]({p.page_id}) — {len(p.files)} file(s)")
        lines.append("")
        return "\n".join(lines)

    def _render_sidebar(self, pages: List[WikiPage]) -> str:
        lines = ["### Topics", ""]
        for p in pages:
            lines.append(f"- [{p.title}]({p.page_id})")
        lines.append("")
        return "\n".join(lines)

    # ---- comment helpers -----------------------------------------------

    @staticmethod
    def _meta_comment(attrs: dict) -> str:
        return "<!-- gitforai:meta " + _format_attrs(attrs) + " -->"

    @staticmethod
    def _auto_open(attrs: dict) -> str:
        return "<!-- gitforai:auto " + _format_attrs(attrs) + " -->"


# ---------------------------------------------------------------------------
# attribute helpers
# ---------------------------------------------------------------------------


def _format_attrs(attrs: dict) -> str:
    parts = []
    for key, value in attrs.items():
        if value is None:
            continue
        # Sanitize quotes in value
        safe = str(value).replace('"', "'")
        parts.append(f'{key}="{safe}"')
    return " ".join(parts)


def _parse_attrs(blob: str) -> dict:
    return {m.group(1): m.group(2) for m in ATTR_RE.finditer(blob)}


_FILES_LINE_RE = re.compile(r"^\s*[-*]\s+`([^`]+)`\s*$", re.MULTILINE)


def _extract_files_from_sections(sections: Iterable[WikiSection]) -> List[str]:
    """Pull file paths out of a "Files" section's body (``- `path` `` bullets)."""
    for section in sections:
        if section.title.strip().lower() == "files":
            return [m.group(1) for m in _FILES_LINE_RE.finditer(section.body)]
    return []


def _parse_sections(text: str) -> Iterable[WikiSection]:
    """Yield WikiSection objects from a serialized page.

    Auto sections are detected by their ``gitforai:auto`` markers; everything
    else under an ``##`` heading is treated as HUMAN content.
    """
    # Split by H2 headings, but track open/close auto markers across the
    # whole document so we know which H2 sections sit inside a marker pair.
    auto_ranges: List[tuple[int, int, dict]] = []
    for open_match in AUTO_OPEN_RE.finditer(text):
        close_match = AUTO_CLOSE_RE.search(text, open_match.end())
        if not close_match:
            continue
        attrs = _parse_attrs(open_match.group("attrs"))
        auto_ranges.append((open_match.start(), close_match.end(), attrs))

    def _source_for(pos: int) -> tuple[PageSource, dict]:
        for start, end, attrs in auto_ranges:
            if start <= pos < end:
                return PageSource.AUTO, attrs
        return PageSource.HUMAN, {}

    headings = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    for i, h in enumerate(headings):
        title = h.group(1).strip()
        body_start = h.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[body_start:body_end]

        # Strip both auto-marker forms that may sit inside this body. The
        # close marker comes from this section's own AUTO range; the open
        # marker comes from the *next* section's AUTO range when this
        # section is HUMAN (the open marker sits between the human heading
        # and the next H2). Without this, a HUMAN section round-trips with
        # a duplicated opener for the next section.
        body = AUTO_CLOSE_RE.sub("", body)
        body = AUTO_OPEN_RE.sub("", body)
        body = body.strip()

        source, attrs = _source_for(h.start())
        commits = [c for c in attrs.get("commits", "").split(",") if c]
        yield WikiSection(
            title=title,
            body=body,
            source=source,
            source_commits=commits,
        )
