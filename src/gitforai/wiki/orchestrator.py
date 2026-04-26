"""High-level orchestrator that ties together replay, drift, render, and merge.

This is the entry point an MCP tool or scheduled job would call. It folds the
build/check_drift/responses-apply scripts into one coherent operation:

    orch = WikiOrchestrator(repo_path, wiki_dir)
    report = await orch.sync(provider=anthropic_provider)
    # or, for the manual-LLM (Claude Code session) path:
    prompts = orch.dump_drift_prompts()
    # ... fill responses ...
    report = await orch.sync(precomputed_responses=responses)

The orchestrator never deletes existing pages on its own — orphaned pages
(wiki page exists, cluster gone) are flagged in the report so a human can
decide. AUTO sections are replaced; HUMAN sections are preserved.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from gitforai.extraction.git_extractor import GitExtractor
from gitforai.integrations.replay import (
    FileCoChangeAnalyzer,
    GitHistoryReplayEngine,
    ReplayResult,
    TemporalClusterAnalyzer,
)
from gitforai.llm.base import BaseLLMProvider
from gitforai.llm.cache import LLMCache
from gitforai.models.commit import CommitMetadata
from gitforai.models.config import RepositoryConfig

from .drift import DriftDetector, DriftReport
from .identity import WikiPage, derive_pages_from_clusters
from .merge import merge_human_edits
from .renderer import LLMRenderer, NaiveRenderer
from .store import WikiStore

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    pages_total: int = 0           # how many pages should exist (per current clusters)
    pages_new: int = 0             # cluster present, no existing page
    pages_drifted: int = 0         # existed and was drifted
    pages_regenerated: int = 0     # actually rewritten this run
    pages_fresh: int = 0           # existed and was already up to date
    pages_orphaned: int = 0        # existing page whose cluster no longer exists
    drift_reports: List[DriftReport] = field(default_factory=list)
    orphaned_page_ids: List[str] = field(default_factory=list)
    new_page_ids: List[str] = field(default_factory=list)
    regenerated_page_ids: List[str] = field(default_factory=list)


class WikiOrchestrator:
    """Drift-aware wiki sync."""

    def __init__(
        self,
        repo_path: Path,
        wiki_dir: Path,
        *,
        min_files: int = 2,
        max_files: int = 20,
        min_co_change_freq: float = 0.0,
        max_drift_commits: int = 5,
        md_pattern: str = "**/*.md",
    ):
        self.repo_path = Path(repo_path)
        self.wiki_dir = Path(wiki_dir)
        self.min_files = min_files
        self.max_files = max_files
        self.min_co_change_freq = min_co_change_freq
        self.max_drift_commits = max_drift_commits
        self.md_pattern = md_pattern

        self.store = WikiStore(self.wiki_dir)
        self.detector = DriftDetector(max_drift_commits=max_drift_commits)

    # ---------------------------------------------------------------- compute

    def _compute_state(self) -> Tuple[
        ReplayResult,
        Dict[str, CommitMetadata],
        List[WikiPage],
        Dict[str, WikiPage],
    ]:
        """Run replay, derive fresh pages, read existing pages from disk."""
        # 1. Replay
        repo_config = RepositoryConfig(repo_path=self.repo_path)
        extractor = GitExtractor(repo_config)
        engine_config = SimpleNamespace(repo_path=self.repo_path)
        engine = GitHistoryReplayEngine(engine_config, extractor)

        commit_count = sum(1 for _ in extractor.repo.iter_commits())
        if commit_count < 20:
            engine.file_analyzer = FileCoChangeAnalyzer(co_change_threshold=1)
            engine.temporal_analyzer = TemporalClusterAnalyzer(burst_threshold=1)

        result = engine.replay_history(md_file_patterns=[self.md_pattern])

        # 2. commits_by_hash for renderer + drift
        commits = list(extractor.extract_all_commits())
        commits_by_hash = {c.hash: c for c in commits}

        # 3. Fresh pages from current clusters
        fresh_pages = derive_pages_from_clusters(
            result,
            min_files=self.min_files,
            max_files=self.max_files,
            min_co_change_freq=self.min_co_change_freq,
        )

        # 4. Existing pages on disk
        existing: Dict[str, WikiPage] = {}
        for pid in self.store.list_page_ids():
            page = self.store.read_page(pid)
            if page is not None:
                existing[pid] = page

        return result, commits_by_hash, fresh_pages, existing

    def classify(
        self,
        fresh_pages: List[WikiPage],
        existing: Dict[str, WikiPage],
        commits_by_hash: Dict[str, CommitMetadata],
    ) -> Tuple[List[str], Dict[str, DriftReport], List[str]]:
        """Return (new_ids, drift_reports_by_id, orphaned_ids)."""
        fresh_ids = {p.page_id for p in fresh_pages}
        existing_ids = set(existing.keys())

        new_ids = sorted(fresh_ids - existing_ids)
        orphaned_ids = sorted(existing_ids - fresh_ids)

        drift_by_id: Dict[str, DriftReport] = {}
        for pid in fresh_ids & existing_ids:
            drift_by_id[pid] = self.detector.detect(existing[pid], commits_by_hash)

        return new_ids, drift_by_id, orphaned_ids

    # ---------------------------------------------------------------- prompts

    def dump_drift_prompts(self, out_path: Path) -> int:
        """Write prompts only for pages that need regeneration. Returns count."""
        from .prompts import topic_summary_prompt

        result, commits_by_hash, fresh_pages, existing = self._compute_state()
        new_ids, drift_by_id, _ = self.classify(fresh_pages, existing, commits_by_hash)
        regen_ids = set(new_ids) | {pid for pid, r in drift_by_id.items() if r.is_drifted}

        cluster_by_id = {c.cluster_id: c for c in result.file_clusters}
        payload = []
        for page in fresh_pages:
            if page.page_id not in regen_ids:
                continue
            cluster = cluster_by_id[page.cluster_id]
            cluster_files = set(cluster.files)
            cluster_commits = sorted(
                (c for c in commits_by_hash.values()
                 if cluster_files.intersection(c.files_changed or [])),
                key=lambda c: c.timestamp,
                reverse=True,
            )
            commit_subjects = [
                (c.message_summary or c.message.splitlines()[0] or "").strip()
                for c in cluster_commits[:12]
            ]
            commit_subjects = [s for s in commit_subjects if s]
            reason = "new" if page.page_id in new_ids else "drifted"
            payload.append({
                "page_id": page.page_id,
                "reason": reason,
                "current_title": page.title,
                "files": sorted(cluster.files),
                "commit_subjects": commit_subjects,
                "prompt": topic_summary_prompt(sorted(cluster.files), commit_subjects),
            })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return len(payload)

    # ---------------------------------------------------------------- sync

    async def sync(
        self,
        *,
        provider: Optional[BaseLLMProvider] = None,
        precomputed_responses: Optional[Dict[str, str]] = None,
        cache: Optional[LLMCache] = None,
        force_regen_all: bool = False,
    ) -> SyncReport:
        result, commits_by_hash, fresh_pages, existing = self._compute_state()
        new_ids, drift_by_id, orphaned_ids = self.classify(
            fresh_pages, existing, commits_by_hash
        )

        report = SyncReport(
            pages_total=len(fresh_pages),
            pages_new=len(new_ids),
            pages_drifted=sum(1 for r in drift_by_id.values() if r.is_drifted),
            pages_orphaned=len(orphaned_ids),
            pages_fresh=sum(1 for r in drift_by_id.values() if not r.is_drifted),
            drift_reports=list(drift_by_id.values()),
            orphaned_page_ids=orphaned_ids,
            new_page_ids=new_ids,
        )

        regen_ids: set = set(new_ids)
        if force_regen_all:
            regen_ids |= {p.page_id for p in fresh_pages}
        else:
            regen_ids |= {pid for pid, r in drift_by_id.items() if r.is_drifted}

        if not regen_ids:
            logger.info("No pages need regeneration; only the index will be refreshed.")
            self._write_index(fresh_pages, existing, regen_ids={})
            return report

        # Build the renderer once; the orchestrator decides what to render.
        if provider is not None or precomputed_responses:
            renderer = LLMRenderer(
                provider=provider,
                cache=cache,
                precomputed_responses=precomputed_responses or {},
            )
        else:
            renderer = NaiveRenderer()

        cluster_by_id = {c.cluster_id: c for c in result.file_clusters}
        regenerated: List[str] = []

        async def _render_one(page: WikiPage) -> None:
            cluster = cluster_by_id[page.cluster_id]
            if isinstance(renderer, LLMRenderer):
                await renderer.render_async(page, cluster, result, commits_by_hash)
            else:
                renderer.render(page, cluster, result, commits_by_hash)

            existing_page = existing.get(page.page_id)
            final = merge_human_edits(page, existing_page) if existing_page else page
            self.store.write_page(final)
            regenerated.append(page.page_id)
            logger.info(
                "Wrote %s (%d sections, title=%r%s)",
                page.page_id,
                len(final.sections),
                final.title,
                ", merged human edits" if existing_page else "",
            )

        # Render concurrently. LLM providers fan out fine; naive runs are fast.
        await asyncio.gather(*(
            _render_one(p) for p in fresh_pages if p.page_id in regen_ids
        ))

        report.pages_regenerated = len(regenerated)
        report.regenerated_page_ids = regenerated

        # 5. Refresh index. For pages we didn't regen, use existing on-disk title.
        self._write_index(fresh_pages, existing, regen_ids=set(regenerated))
        return report

    def _write_index(
        self,
        fresh_pages: List[WikiPage],
        existing: Dict[str, WikiPage],
        regen_ids: set,
    ) -> None:
        index_pages: List[WikiPage] = []
        for page in fresh_pages:
            if page.page_id in regen_ids:
                index_pages.append(page)
            elif page.page_id in existing:
                # Use the existing on-disk page (its title may be human-edited
                # or LLM-generated from a previous run).
                ex = existing[page.page_id]
                ex.files = page.files
                index_pages.append(ex)
            else:
                index_pages.append(page)
        self.store.write_index(index_pages)
