#!/usr/bin/env python3
"""Drift-aware wiki sync.

Refreshes only pages whose source files have new commits since their
``last_synced_commit``. Preserves any HUMAN-edited sections via the
``gitforai:auto`` markers stored in each page.

Three driving modes (same as build_wiki.py):

    # 1. SDK-driven (autonomous):
    ANTHROPIC_API_KEY=sk-... python scripts/sync_wiki.py REPO --wiki DIR

    # 2. Manual / Claude Code session as LLM:
    python scripts/sync_wiki.py REPO --wiki DIR --dump-prompts /tmp/p.json
    # ... fill /tmp/r.json with {page_id: completion_text} ...
    python scripts/sync_wiki.py REPO --wiki DIR --responses-file /tmp/r.json

    # 3. Naive (no LLM, factual sections only):
    python scripts/sync_wiki.py REPO --wiki DIR --llm-provider none

Use ``--force`` to regenerate every page regardless of drift (e.g. after
upgrading the prompt template).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gitforai.llm.cache import LLMCache
from gitforai.wiki import WikiOrchestrator, auto_detect_llm_provider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo_path", nargs="?", default=str(ROOT))
    p.add_argument("--wiki", required=True, help="Path to the existing wiki directory.")
    p.add_argument("--min-files", type=int, default=2)
    p.add_argument("--max-files", type=int, default=20)
    p.add_argument("--max-drift-commits", type=int, default=5)
    p.add_argument(
        "--md-pattern", default="**/*.md",
        help="Glob for markdown files passed to replay (default: **/*.md)",
    )
    p.add_argument(
        "--llm-provider",
        choices=["auto", "anthropic", "openai", "none"],
        default="auto",
    )
    p.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".gitforai" / "cache"),
    )
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--dump-prompts",
        metavar="FILE",
        help="Write prompts only for pages that need regen, then exit.",
    )
    p.add_argument(
        "--responses-file",
        metavar="FILE",
        help="Apply pre-baked completions ({page_id: text}) instead of an LLM provider.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate every page, ignoring drift state.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("sync_wiki")

    repo_path = Path(args.repo_path).resolve()
    wiki_dir = Path(args.wiki).resolve()
    if not wiki_dir.is_dir():
        print(f"❌ Wiki directory not found: {wiki_dir}", file=sys.stderr)
        return 2

    orch = WikiOrchestrator(
        repo_path=repo_path,
        wiki_dir=wiki_dir,
        min_files=args.min_files,
        max_files=args.max_files,
        max_drift_commits=args.max_drift_commits,
        md_pattern=args.md_pattern,
    )

    # --- branch A: dump prompts only for drifted/new pages ---
    if args.dump_prompts:
        out = Path(args.dump_prompts)
        n = orch.dump_drift_prompts(out)
        log.info("Wrote %d prompt(s) to %s", n, out)
        if n == 0:
            print(f"\n✅ No pages need regeneration. (Wrote empty {out}.)")
        else:
            print(f"\n📝 Dumped {n} prompt(s) to {out}")
            print("   Fill responses as JSON {page_id: completion_text} matching")
            print("   the TITLE:/SUMMARY: format, then re-run with --responses-file.")
        return 0

    # --- branch B: precomputed responses (manual / Claude Code session) ---
    precomputed: dict = {}
    if args.responses_file:
        precomputed = json.loads(Path(args.responses_file).read_text(encoding="utf-8"))
        log.info("Loaded %d precomputed response(s).", len(precomputed))

    # --- branch C: LLM provider ---
    provider = None
    if args.llm_provider != "none" and not precomputed:
        provider = auto_detect_llm_provider(
            None if args.llm_provider == "auto" else args.llm_provider
        )
        if provider is None and args.llm_provider not in ("auto", "none"):
            log.warning(
                "Requested --llm-provider=%s but no key/SDK available; falling back.",
                args.llm_provider,
            )

    cache = None if args.no_cache else LLMCache(Path(args.cache_dir))

    report = asyncio.run(orch.sync(
        provider=provider,
        precomputed_responses=precomputed,
        cache=cache,
        force_regen_all=args.force,
    ))

    # --- summary ---
    print()
    print(f"Wiki: {wiki_dir}")
    print(f"Repo: {repo_path}")
    print()
    print(f"  total clusters/pages : {report.pages_total}")
    print(f"  new                  : {report.pages_new}      {report.new_page_ids or ''}")
    print(f"  drifted              : {report.pages_drifted}")
    print(f"  fresh (no action)    : {report.pages_fresh}")
    print(f"  regenerated this run : {report.pages_regenerated}  {report.regenerated_page_ids or ''}")
    print(f"  orphaned             : {report.pages_orphaned}  {report.orphaned_page_ids or ''}")

    if provider is not None and cache is not None:
        stats = cache.get_stats()
        print(f"\n  LLM cache: {stats['hits']} hit, {stats['misses']} miss ({stats['hit_rate']})")
        print(f"  LLM cost : ${getattr(provider, 'total_cost', 0.0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
