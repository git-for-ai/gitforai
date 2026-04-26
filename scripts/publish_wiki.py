#!/usr/bin/env python3
"""Stage a wiki for publication to GitHub Wiki.

Clones (or updates) ``<repo>.wiki.git`` in a working directory, copies the
generated ``*.md`` files in, and commits. **Does not push** — review the
result and ``git push`` from the staging dir yourself.

Usage:
    python scripts/publish_wiki.py SOURCE_DIR WIKI_GIT_URL [--staging DIR]

Examples:
    # Push agentauthvault's wiki
    python scripts/publish_wiki.py /tmp/aav-wiki-llm \\
        git@github.com:bigale/agentauthvault.wiki.git

    # Push gitforai-core's own wiki
    python scripts/publish_wiki.py ./wiki \\
        git@github.com:git-for-ai/gitforai.wiki.git

If GitHub returns "Repository not found" on clone, the wiki repo doesn't
exist yet. Open the repo's Wiki tab in the GitHub UI, create any page (it
becomes Home.md), then re-run — your generated Home.md will overwrite it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_STAGING = Path.home() / ".gitforai" / "wiki-staging"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_dir", help="Generated wiki dir (contains Home.md, *.md, _Sidebar.md).")
    p.add_argument("wiki_git_url", help="The <repo>.wiki.git URL (SSH or HTTPS).")
    p.add_argument(
        "--staging",
        default=None,
        help=f"Working dir for the wiki clone (default: {DEFAULT_STAGING}/<repo>).",
    )
    p.add_argument(
        "--message",
        default="Auto-generated wiki refresh (gitforai)",
        help="Commit message.",
    )
    p.add_argument(
        "--prune",
        action="store_true",
        help="Delete .md files in the wiki that don't exist in source_dir. "
        "Off by default to avoid surprising the user.",
    )
    return p.parse_args()


def _run(cmd: List[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"$ {' '.join(cmd)}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def _ensure_clone(wiki_url: str, staging: Path) -> None:
    if staging.exists() and (staging / ".git").exists():
        existing_url = _run(["git", "remote", "get-url", "origin"], cwd=staging)
        if existing_url != wiki_url:
            print(
                f"❌ Staging dir {staging} is a clone of {existing_url}, not {wiki_url}.\n"
                "   Pass a different --staging dir or remove the existing clone.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(f"  · staging exists, pulling latest...")
        _run(["git", "fetch", "origin"], cwd=staging)
        # Master/main: try both
        try:
            _run(["git", "reset", "--hard", "origin/master"], cwd=staging)
        except SystemExit:
            _run(["git", "reset", "--hard", "origin/main"], cwd=staging)
    else:
        staging.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
        print(f"  · cloning {wiki_url} -> {staging}")
        try:
            _run(["git", "clone", wiki_url, str(staging)])
        except SystemExit:
            print(
                "\n❌ Clone failed. Most likely cause: the GitHub Wiki for this repo\n"
                "   has never been initialized. Open the repo on GitHub, click the\n"
                "   Wiki tab, click 'Create the first page', save any placeholder,\n"
                "   then re-run this script.",
                file=sys.stderr,
            )
            raise


def _copy_files(source: Path, target: Path, prune: bool) -> tuple[int, int, list[str]]:
    md_files = sorted([p for p in source.iterdir() if p.is_file() and p.suffix == ".md"])
    copied = 0
    pruned: list[str] = []
    if not md_files:
        print(f"❌ No .md files in {source}", file=sys.stderr)
        raise SystemExit(2)

    source_names = {p.name for p in md_files}

    for src in md_files:
        dst = target / src.name
        shutil.copy2(src, dst)
        copied += 1

    if prune:
        for existing in target.glob("*.md"):
            if existing.name not in source_names and existing.name != "Home.md":
                # Always keep Home.md (we wrote our own; the prune was for *other* pages)
                # But here Home.md is in source_names, so this branch only catches
                # legitimately removed pages.
                pass
            if existing.name not in source_names:
                existing.unlink()
                pruned.append(existing.name)

    return copied, len(md_files), pruned


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir).resolve()
    if not source.is_dir():
        print(f"❌ Source dir not found: {source}", file=sys.stderr)
        return 2

    # Derive staging path from wiki URL.
    if args.staging:
        staging = Path(args.staging).resolve()
    else:
        # e.g. git@github.com:bigale/agentauthvault.wiki.git -> agentauthvault.wiki
        url_tail = args.wiki_git_url.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        repo_dirname = url_tail.removesuffix(".git")
        staging = DEFAULT_STAGING / repo_dirname

    print(f"Source : {source}")
    print(f"Wiki   : {args.wiki_git_url}")
    print(f"Staging: {staging}")
    print()

    _ensure_clone(args.wiki_git_url, staging)

    print("  · copying markdown files...")
    copied, total, pruned = _copy_files(source, staging, args.prune)
    print(f"    copied {copied}/{total} file(s)")
    if pruned:
        print(f"    pruned {len(pruned)} stale page(s): {pruned}")

    # Stage and commit.
    _run(["git", "add", "-A"], cwd=staging)
    status = _run(["git", "status", "--porcelain"], cwd=staging)
    if not status:
        print("\n✅ No changes — wiki is already up to date on the remote.")
        return 0

    print("\n  · committing...")
    _run(["git", "commit", "-m", args.message], cwd=staging)

    print()
    print("✅ Staged commit ready.")
    print()
    print("Review with:")
    print(f"  git -C {staging} log -1 --stat")
    print(f"  git -C {staging} diff HEAD~1 HEAD -- '*.md'  # if there were prior commits")
    print()
    print("Push with:")
    print(f"  git -C {staging} push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
