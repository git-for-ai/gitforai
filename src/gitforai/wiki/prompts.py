"""Prompt templates for wiki page generation.

Kept in the wiki module rather than ``llm/prompts.py`` because they are
specific to topic-summarization-of-a-cluster and aren't reused elsewhere.

Bumping ``PROMPT_VERSION`` invalidates the LLM cache (see ``LLMRenderer``).
"""

from __future__ import annotations

from typing import List

PROMPT_VERSION = "1"


def topic_summary_prompt(
    files: List[str],
    commit_subjects: List[str],
    max_files: int = 30,
    max_commits: int = 12,
) -> str:
    """Build the prompt that asks an LLM for a wiki page title + summary.

    The output is a structured block we can parse back without JSON, since this
    keeps the prompt short and avoids JSON-encoding issues in commit subjects.
    """
    file_block = "\n".join(f"- {f}" for f in files[:max_files])
    if len(files) > max_files:
        file_block += f"\n- ...and {len(files) - max_files} more"

    commit_block = "\n".join(f"- {s}" for s in commit_subjects[:max_commits])
    if len(commit_subjects) > max_commits:
        commit_block += f"\n- ...and {len(commit_subjects) - max_commits} more"

    return f"""You are documenting a software project's wiki by analyzing a cluster of files that change together in version control.

FILES IN THIS CLUSTER:
{file_block}

RECENT COMMITS THAT TOUCHED THESE FILES (subject lines, newest first):
{commit_block}

Your task: produce a wiki page heading and a short summary for the topic these files share.

Rules:
- The TITLE must be a short noun phrase (2-5 words) naming the topic — what these files collectively are.
- The SUMMARY must be 2-4 sentences in plain prose. Describe what this code/area does, why these files cluster together, and (if clear from the commits) what's been changing recently. Do not list filenames in the summary; the wiki page already shows them separately. Do not mention "this cluster" or "these files" — write as if introducing the topic to a reader.
- Do not invent facts. If the commits don't reveal recent activity, omit that sentence.
- No markdown formatting in either field.

Respond in EXACTLY this format, with no preamble or trailing text:

TITLE: <the title>
SUMMARY: <the summary>
"""


def parse_topic_response(text: str) -> tuple[str, str]:
    """Parse a ``TITLE: ...\\nSUMMARY: ...`` response into (title, summary).

    Tolerates leading/trailing whitespace and missing fields. Returns empty
    strings for fields that can't be extracted, so callers can decide whether
    to fall back to naive output.
    """
    title = ""
    summary = ""
    in_summary = False
    summary_lines: list = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.upper().startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
            in_summary = False
            continue
        if line.upper().startswith("SUMMARY:"):
            summary_lines = [line.split(":", 1)[1].strip()]
            in_summary = True
            continue
        if in_summary:
            summary_lines.append(raw_line.rstrip())

    if summary_lines:
        summary = "\n".join(summary_lines).strip()

    return title, summary
