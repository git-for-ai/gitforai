#!/usr/bin/env python3
"""Check whether an existing wiki has drifted relative to current git history.

Usage:
    python scripts/check_wiki_drift.py [REPO_PATH] [--wiki DIR]

Reads each generated page from ``--wiki``, finds commits that have touched
any of the page's files since its ``last_synced_commit``, and prints a
per-page report. Exit code is 0 when no drift, 1 when any page is drifted —
suitable for use as a pre-merge / scheduled check.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow ``python scripts/check_wiki_drift.py`` from a fresh checkout.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gitforai.extraction.git_extractor import GitExtractor
from gitforai.models.config import RepositoryConfig
from gitforai.wiki import DriftDetector, WikiStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo_path", nargs="?", default=str(ROOT))
    p.add_argument("--wiki", default=str(ROOT / "wiki"), help="Path to the wiki directory.")
    p.add_argument(
        "--max-drift-commits",
        type=int,
        default=5,
        help="Number of new commits at which drift_score saturates to 1.0.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo_path = Path(args.repo_path).resolve()
    wiki_dir = Path(args.wiki).resolve()
    if not wiki_dir.is_dir():
        print(f"❌ Wiki directory not found: {wiki_dir}", file=sys.stderr)
        return 2

    store = WikiStore(wiki_dir)
    page_ids = store.list_page_ids()
    if not page_ids:
        print(f"❌ No pages found in {wiki_dir}", file=sys.stderr)
        return 2

    pages = []
    for pid in page_ids:
        page = store.read_page(pid)
        if page is not None:
            pages.append(page)
    if not pages:
        print(f"❌ No parseable pages in {wiki_dir}", file=sys.stderr)
        return 2

    # Pull commits straight from git; we don't need a full replay for drift.
    extractor = GitExtractor(RepositoryConfig(repo_path=repo_path))
    commits = list(extractor.extract_all_commits())
    commits_by_hash = {c.hash: c for c in commits}

    detector = DriftDetector(max_drift_commits=args.max_drift_commits)
    reports = detector.detect_all(pages, commits_by_hash)

    drifted = [r for r in reports if r.is_drifted]

    print(f"\nWiki: {wiki_dir}")
    print(f"Repo: {repo_path}  ({len(commits_by_hash)} commits indexed)\n")

    title_w = max(len(r.page_title) for r in reports) + 2
    print(f"{'Page'.ljust(title_w)}{'Severity':<10} {'New':>4}  {'Score':>6}  Drifted Sections")
    print("-" * (title_w + 38))
    for r in sorted(reports, key=lambda x: -x.drift_score):
        sections = ", ".join(r.drifted_section_titles[:3])
        if len(r.drifted_section_titles) > 3:
            sections += f", +{len(r.drifted_section_titles) - 3} more"
        print(
            f"{r.page_title.ljust(title_w)}"
            f"{r.severity:<10} "
            f"{len(r.new_commit_hashes):>4}  "
            f"{r.drift_score:>6.2f}  "
            f"{sections or '-'}"
        )

    if drifted:
        print(f"\n⚠️  {len(drifted)}/{len(reports)} page(s) drifted.\n")
        for r in drifted:
            print(f"## {r.page_title}  ({r.page_id})")
            if r.last_synced_commit:
                print(f"   Last synced: {r.last_synced_commit[:7]} at {r.last_synced_at}")
            else:
                print("   Last synced: (never)")
            for reason in r.reasons:
                print(f"   • {reason}")
            for h, subject in zip(r.new_commit_hashes[:5], r.new_commit_subjects[:5]):
                print(f"     - {h[:7]}  {subject}")
            if len(r.new_commit_hashes) > 5:
                print(f"     - ...and {len(r.new_commit_hashes) - 5} more")
            print()
        return 1

    print("\n✅ All pages fresh.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
