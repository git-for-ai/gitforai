#!/usr/bin/env python3
"""Build a wiki for a repository from its git history (Step 1, no LLM).

Usage:
    python scripts/build_wiki.py [REPO_PATH] [--out DIR] [--min-files N] [--min-commits N]

Defaults to running on the current repository (gitforai-core) and writing
markdown to ``./wiki/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

# Allow ``python scripts/build_wiki.py`` from a fresh checkout.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gitforai.extraction.git_extractor import GitExtractor
from gitforai.integrations.replay import (
    FileCoChangeAnalyzer,
    GitHistoryReplayEngine,
    TemporalClusterAnalyzer,
)
from gitforai.llm.cache import LLMCache
from gitforai.models.config import RepositoryConfig
from gitforai.wiki import (
    LLMRenderer,
    NaiveRenderer,
    WikiStore,
    auto_detect_llm_provider,
    derive_pages_from_clusters,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "repo_path",
        nargs="?",
        default=str(ROOT),
        help="Path to the git repository (default: this repo)",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "wiki"),
        help="Output directory for wiki markdown (default: ./wiki)",
    )
    p.add_argument("--min-files", type=int, default=2, help="Skip clusters smaller than this")
    p.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Drop clusters larger than this (likely bulk-commit dumps, not real topics). "
        "Set to 0 to disable.",
    )
    p.add_argument(
        "--min-co-change-freq",
        type=float,
        default=0.0,
        help="Skip clusters whose co-change frequency is below this (0.0..1.0)",
    )
    p.add_argument(
        "--co-change-threshold",
        type=int,
        default=None,
        help="Override replay's FileCoChangeAnalyzer threshold (default: 3 for big repos, "
        "auto-lowered to 1 if the repo has <20 commits)",
    )
    p.add_argument(
        "--burst-threshold",
        type=int,
        default=None,
        help="Override replay's TemporalClusterAnalyzer burst threshold (default: 5 commits/day, "
        "auto-lowered to 1 if the repo has <20 commits)",
    )
    p.add_argument(
        "--md-pattern",
        default="**/*.md",
        help="Glob pattern for markdown files passed to replay (default: **/*.md)",
    )
    p.add_argument(
        "--llm-provider",
        choices=["auto", "anthropic", "openai", "none"],
        default="auto",
        help="LLM provider for prose generation. 'auto' detects from API key env vars; "
        "'none' forces the naive renderer.",
    )
    p.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".gitforai" / "cache"),
        help="LLM response cache directory (default: ~/.gitforai/cache)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching",
    )
    p.add_argument(
        "--dump-prompts",
        metavar="FILE",
        default=None,
        help="Write per-page prompts (and prompt context) to FILE as JSON, then exit. "
        "Use this to drive prose generation manually (e.g. with the running Claude Code "
        "session) without an API key.",
    )
    p.add_argument(
        "--responses-file",
        metavar="FILE",
        default=None,
        help="Apply pre-baked completions from FILE (JSON: {page_id: completion_text}) "
        "instead of calling an LLM provider.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("build_wiki")

    repo_path = Path(args.repo_path).resolve()
    out_dir = Path(args.out).resolve()
    log.info("Repo: %s", repo_path)
    log.info("Out:  %s", out_dir)

    # 1. Set up the replay engine.
    repo_config = RepositoryConfig(repo_path=repo_path)
    extractor = GitExtractor(repo_config)
    engine_config = SimpleNamespace(repo_path=repo_path)
    engine = GitHistoryReplayEngine(engine_config, extractor)

    # Auto-tune thresholds for small histories so the demo isn't empty.
    commit_count = sum(1 for _ in extractor.repo.iter_commits())
    log.info("Repo has %d commits.", commit_count)
    co_change = args.co_change_threshold
    burst = args.burst_threshold
    if co_change is None:
        co_change = 1 if commit_count < 20 else 3
    if burst is None:
        burst = 1 if commit_count < 20 else 5
    if co_change != 3 or burst != 5:
        log.info("Tuning analyzers: co_change_threshold=%d, burst_threshold=%d", co_change, burst)
        engine.file_analyzer = FileCoChangeAnalyzer(co_change_threshold=co_change)
        engine.temporal_analyzer = TemporalClusterAnalyzer(burst_threshold=burst)

    # 2. Run replay (Layers 1-4).
    log.info("Replaying git history (this may take a moment)...")
    result = engine.replay_history(md_file_patterns=[args.md_pattern])
    log.info(
        "Replay done: %d commits, %d tasks, %d clusters, %d epics",
        result.commits_analyzed,
        result.tasks_found,
        len(result.file_clusters),
        len(result.temporal_epics),
    )

    if not result.file_clusters:
        log.warning(
            "No file clusters found. Lower co_change_threshold in FileCoChangeAnalyzer "
            "or work on a repo with more co-edit history."
        )
        return 1

    # 3. Build commits_by_hash for the renderer.
    commits = list(extractor.extract_all_commits())
    commits_by_hash = {c.hash: c for c in commits}
    log.info("Indexed %d commits for renderer.", len(commits_by_hash))

    # 4. Derive pages and render.
    pages = derive_pages_from_clusters(
        result,
        min_files=args.min_files,
        max_files=args.max_files,
        min_co_change_freq=args.min_co_change_freq,
    )
    log.info(
        "Derived %d page(s) (filtered from %d clusters; min_files=%d, max_files=%d, min_co_change_freq=%.2f).",
        len(pages),
        len(result.file_clusters),
        args.min_files,
        args.max_files,
        args.min_co_change_freq,
    )
    if not pages:
        log.warning("No pages survived filtering. Try lowering --min-files or --min-co-change-freq.")
        return 1

    cluster_by_id = {c.cluster_id: c for c in result.file_clusters}
    store = WikiStore(out_dir)

    # --- branch A: dump prompts and exit ---
    if args.dump_prompts:
        dump_path = Path(args.dump_prompts)
        _dump_prompts(pages, cluster_by_id, commits_by_hash, dump_path)
        log.info("Wrote %d prompt(s) to %s", len(pages), dump_path)
        print(f"\n📝 Prompts dumped to {dump_path}")
        print("   Fill in responses as JSON of {page_id: completion_text} matching")
        print("   the TITLE:/SUMMARY: format, then re-run with --responses-file.")
        return 0

    # --- branch B: load precomputed responses (manual / Claude-Code-session path) ---
    precomputed: dict = {}
    if args.responses_file:
        precomputed = json.loads(Path(args.responses_file).read_text(encoding="utf-8"))
        log.info("Loaded %d precomputed response(s) from %s", len(precomputed), args.responses_file)

    # --- branch C: pick a real provider ---
    provider = None
    if args.llm_provider != "none" and not precomputed:
        provider = auto_detect_llm_provider(
            None if args.llm_provider == "auto" else args.llm_provider
        )

    if provider is not None or precomputed:
        cache = None if args.no_cache else LLMCache(Path(args.cache_dir))
        renderer = LLMRenderer(provider=provider, cache=cache, precomputed_responses=precomputed)
        log.info(
            "Using LLM renderer: provider=%s, precomputed=%d, cache=%s",
            "none" if provider is None else provider.__class__.__name__,
            len(precomputed),
            "off" if cache is None else args.cache_dir,
        )
        asyncio.run(_render_all_async(renderer, pages, cluster_by_id, result, commits_by_hash, store, log))
        if provider is not None and cache is not None:
            stats = cache.get_stats()
            log.info(
                "LLM cache: %d hit, %d miss (%s); cost=$%.4f",
                stats["hits"],
                stats["misses"],
                stats["hit_rate"],
                getattr(provider, "total_cost", 0.0),
            )
    else:
        if args.llm_provider not in ("none", "auto"):
            log.warning(
                "Requested --llm-provider=%s but no API key/SDK available; falling back to naive.",
                args.llm_provider,
            )
        elif args.llm_provider == "auto":
            log.info(
                "No LLM provider available (set ANTHROPIC_API_KEY/OPENAI_API_KEY, "
                "or use --dump-prompts/--responses-file for manual flow); using naive renderer."
            )
        renderer = NaiveRenderer()
        for page in pages:
            cluster = cluster_by_id[page.cluster_id]
            renderer.render(page, cluster, result, commits_by_hash)
            path = store.write_page(page)
            log.info("Wrote %s (%d sections)", path.name, len(page.sections))

    store.write_index(pages)
    log.info("Wrote Home.md and _Sidebar.md.")

    print(f"\n✅ Wiki built at {out_dir} ({len(pages)} pages)")
    print("   Inspect:")
    for p in sorted(pages, key=lambda p: p.title.lower()):
        print(f"     - {p.title:<40s}  {len(p.files):>3d} files  -> {p.page_id}.md")
    return 0


async def _render_all_async(renderer, pages, cluster_by_id, result, commits_by_hash, store, log):
    """Render all pages concurrently with the LLM renderer."""
    async def _one(page):
        cluster = cluster_by_id[page.cluster_id]
        await renderer.render_async(page, cluster, result, commits_by_hash)
        path = store.write_page(page)
        log.info("Wrote %s (%d sections, title=%r)", path.name, len(page.sections), page.title)

    # Concurrent fan-out — Anthropic/OpenAI handle this fine for ~10s of pages.
    await asyncio.gather(*(_one(p) for p in pages))


def _dump_prompts(pages, cluster_by_id, commits_by_hash, out_path: Path) -> None:
    """Write per-page prompts + context to a JSON file for manual response generation."""
    from gitforai.wiki.prompts import topic_summary_prompt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for page in pages:
        cluster = cluster_by_id[page.cluster_id]
        cluster_files = set(cluster.files)
        cluster_commits = [
            c for c in commits_by_hash.values()
            if cluster_files.intersection(c.files_changed or [])
        ]
        cluster_commits.sort(key=lambda c: c.timestamp, reverse=True)
        commit_subjects = [
            (c.message_summary or c.message.splitlines()[0] or "").strip()
            for c in cluster_commits[:12]
        ]
        commit_subjects = [s for s in commit_subjects if s]

        payload.append({
            "page_id": page.page_id,
            "current_title": page.title,
            "files": sorted(cluster.files),
            "commit_subjects": commit_subjects,
            "prompt": topic_summary_prompt(sorted(cluster.files), commit_subjects),
        })

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
