#!/usr/bin/env python3
"""Install (or remove) a post-commit hook that auto-syncs a wiki.

The installed hook runs ``sync_wiki.py`` in the background after each commit,
so it never blocks ``git commit``. Output goes to ``<repo>/.gitforai/wiki-sync.log``.

Usage:
    # Install:
    python scripts/install_wiki_hook.py REPO --wiki WIKI_DIR [--auto-push]
                                             [--llm-provider auto|anthropic|openai|none]

    # Status / Uninstall:
    python scripts/install_wiki_hook.py REPO --status
    python scripts/install_wiki_hook.py REPO --uninstall

The hook is identified by a sentinel comment, so the installer refuses to
overwrite an unrelated pre-existing post-commit hook.
"""

from __future__ import annotations

import argparse
import os
import shlex
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = ROOT / "scripts" / "sync_wiki.py"

# Sentinel — anything containing this in the hook is ours and we'll overwrite.
SENTINEL = "# gitforai-wiki-sync-hook v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo_path", help="Path to the git repo whose commits should trigger sync.")
    p.add_argument("--wiki", help="Path to the wiki directory the hook should sync.")
    p.add_argument(
        "--llm-provider",
        choices=["auto", "anthropic", "openai", "none"],
        default="auto",
        help="LLM provider for prose. 'auto' uses ANTHROPIC_API_KEY/OPENAI_API_KEY "
        "from the environment when the hook fires; 'none' produces factual "
        "sections only (no Summary).",
    )
    p.add_argument(
        "--auto-push",
        action="store_true",
        help="After sync, also stage + commit + push the wiki to its remote. "
        "Requires the wiki dir to be a clone of <repo>.wiki.git.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter the hook should use (default: this interpreter).",
    )
    p.add_argument("--uninstall", action="store_true", help="Remove the hook.")
    p.add_argument("--status", action="store_true", help="Just report whether a hook is installed.")
    p.add_argument("--force", action="store_true", help="Overwrite an unrelated existing hook.")
    return p.parse_args()


def _hook_path(repo_path: Path) -> Path:
    """Resolve the hooks dir, honouring core.hooksPath and worktrees."""
    git_dir = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--git-common-dir"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    git_dir_path = Path(git_dir)
    if not git_dir_path.is_absolute():
        git_dir_path = repo_path / git_dir_path

    # Respect a custom hooks path if configured.
    custom = subprocess.run(
        ["git", "-C", str(repo_path), "config", "--get", "core.hooksPath"],
        capture_output=True, text=True,
    ).stdout.strip()
    if custom:
        hooks_dir = Path(custom)
        if not hooks_dir.is_absolute():
            hooks_dir = repo_path / hooks_dir
    else:
        hooks_dir = git_dir_path / "hooks"

    return hooks_dir / "post-commit"


def _build_hook_script(args: argparse.Namespace, repo_path: Path, wiki_path: Path) -> str:
    log_dir = repo_path / ".gitforai"
    log_path = log_dir / "wiki-sync.log"

    sync_cmd = [
        shlex.quote(args.python),
        shlex.quote(str(SYNC_SCRIPT)),
        shlex.quote(str(repo_path)),
        "--wiki", shlex.quote(str(wiki_path)),
        "--llm-provider", args.llm_provider,
    ]

    push_block = ""
    if args.auto_push:
        push_block = textwrap.dedent(f"""
            # auto-push: only proceeds if the wiki dir is a git clone with a remote.
            if [ -d {shlex.quote(str(wiki_path))}/.git ]; then
                cd {shlex.quote(str(wiki_path))}
                git add -A
                if ! git diff --staged --quiet; then
                    git -c user.name="gitforai bot" \\
                        -c user.email="gitforai@localhost" \\
                        commit -m "Auto-sync wiki (post-commit hook)"
                    git push 2>&1 || echo "[hook] push failed (network? auth?); commit kept locally."
                fi
            fi
        """).strip()

    return textwrap.dedent(f"""
        #!/usr/bin/env bash
        {SENTINEL}
        # Installed by scripts/install_wiki_hook.py — do not edit manually.
        # To remove:  python scripts/install_wiki_hook.py {shlex.quote(str(repo_path))} --uninstall

        set -e
        mkdir -p {shlex.quote(str(log_dir))}

        # Background the sync so 'git commit' returns immediately.
        # All output (including LLM cost reports) goes to the log file.
        (
            cd {shlex.quote(str(repo_path))}
            {' '.join(sync_cmd)} >> {shlex.quote(str(log_path))} 2>&1 || \\
                echo "[$(date -Iseconds)] sync failed (see log)." >> {shlex.quote(str(log_path))}
            {push_block}
        ) </dev/null >/dev/null 2>&1 &
        disown 2>/dev/null || true
    """).lstrip()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> int:
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()

    # Validate the target is a git repo.
    if not (repo_path / ".git").exists() and not (repo_path / "HEAD").exists():
        # Could still be a worktree; rev-parse will tell us.
        try:
            subprocess.run(
                ["git", "-C", str(repo_path), "rev-parse"],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            print(f"❌ Not a git repository: {repo_path}", file=sys.stderr)
            return 2

    hook = _hook_path(repo_path)

    if args.status:
        if hook.exists() and SENTINEL in _read(hook):
            print(f"✅ gitforai wiki-sync hook installed at {hook}")
        elif hook.exists():
            print(f"⚠️  A post-commit hook exists at {hook} but it is not ours.")
        else:
            print(f"(none) No post-commit hook at {hook}")
        return 0

    if args.uninstall:
        if not hook.exists():
            print(f"(noop) No post-commit hook at {hook}")
            return 0
        if SENTINEL not in _read(hook):
            print(f"❌ Existing hook at {hook} is not ours; not removing.", file=sys.stderr)
            print("   Inspect and remove it manually if you want.", file=sys.stderr)
            return 2
        hook.unlink()
        print(f"✅ Removed {hook}")
        return 0

    # Install path — needs --wiki.
    if not args.wiki:
        print("❌ --wiki is required for install. (Use --uninstall or --status otherwise.)", file=sys.stderr)
        return 2
    wiki_path = Path(args.wiki).resolve()
    if not wiki_path.is_dir():
        print(f"❌ Wiki dir not found: {wiki_path}", file=sys.stderr)
        return 2

    # Don't clobber an unrelated hook.
    if hook.exists() and SENTINEL not in _read(hook) and not args.force:
        print(f"❌ A post-commit hook already exists at {hook}", file=sys.stderr)
        print("   Re-run with --force to overwrite, or back it up first.", file=sys.stderr)
        return 2

    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(_build_hook_script(args, repo_path, wiki_path), encoding="utf-8")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"✅ Installed post-commit hook at {hook}")
    print(f"   Will sync {wiki_path}")
    print(f"   LLM provider: {args.llm_provider}")
    if args.auto_push:
        print(f"   Auto-push: enabled (requires wiki dir to be a git clone)")
    print(f"   Log: {repo_path / '.gitforai' / 'wiki-sync.log'}")
    print()
    print("To verify, make a commit and then check the log.")
    print(f"To remove: python scripts/install_wiki_hook.py {repo_path} --uninstall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
