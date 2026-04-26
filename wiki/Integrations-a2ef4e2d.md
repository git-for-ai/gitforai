# External Tool Integrations

<!-- gitforai:meta page_id="Integrations-a2ef4e2d" cluster_id="cluster-1" synced="7596ec75b04225d870f2a7a9244480a70c5b4c85" synced_at="2026-04-26T20:35:53.271181+00:00" -->

<!-- gitforai:auto section="Summary" synced="7596ec75b04225d870f2a7a9244480a70c5b4c85" commits="7596ec75b04225d870f2a7a9244480a70c5b4c85,8b8f94e4f25a8a4d5e438a3db6ab46e145620bcf,1fdcf62edd5d203f3c9c60d1077a0cdf85f106b8,896fdc7b693d10fdfca11821c64309b72dde83be" -->
## Summary

Wrappers and replay logic that let GitForAI compose with external task-tracking tools — primarily the Beads CLI (`bd`) and Beads Viewer (`bv`) — and that mine git history for task intelligence across four enrichment layers (temporal clustering, commit-body enrichment, task reference linking, and file co-change analysis). The accompanying tests cover the unified-intelligence facade and the thin client wrappers it builds on. Recent work introduced the hierarchical 4-layer enrichment system, deduplication for the history-replay pipeline, and the original Beads + GitForAI integration foundation.

<!-- /gitforai:auto -->

<!-- gitforai:auto section="Overview" synced="7596ec75b04225d870f2a7a9244480a70c5b4c85" commits="7596ec75b04225d870f2a7a9244480a70c5b4c85,8b8f94e4f25a8a4d5e438a3db6ab46e145620bcf,1fdcf62edd5d203f3c9c60d1077a0cdf85f106b8,896fdc7b693d10fdfca11821c64309b72dde83be" -->
## Overview

This page tracks **6 file(s)** that frequently change together (co-change frequency: 0.80).

- Commits touching these files: **4**
- Active range: **2025-12-19 → 2025-12-21**

<!-- /gitforai:auto -->

<!-- gitforai:auto section="Files" synced="7596ec75b04225d870f2a7a9244480a70c5b4c85" -->
## Files

- `src/gitforai/integrations/__init__.py`
- `src/gitforai/integrations/beads.py`
- `src/gitforai/integrations/replay.py`
- `src/gitforai/integrations/unified.py`
- `test_unified_intelligence.py`
- `test_wrapper_classes.py`

<!-- /gitforai:auto -->

<!-- gitforai:auto section="Recent Activity" synced="7596ec75b04225d870f2a7a9244480a70c5b4c85" commits="7596ec75b04225d870f2a7a9244480a70c5b4c85,8b8f94e4f25a8a4d5e438a3db6ab46e145620bcf,1fdcf62edd5d203f3c9c60d1077a0cdf85f106b8,896fdc7b693d10fdfca11821c64309b72dde83be" -->
## Recent Activity

| Date | Commit | Author | Subject |
|---|---|---|---|
| 2025-12-21 | `7596ec7` | Alex Everitt | Add hierarchical task intelligence with 4-layer enrichment system |
| 2025-12-19 | `8b8f94e` | Alex Everitt | Add deduplication logic to git history replay |
| 2025-12-19 | `1fdcf62` | Alex Everitt | Add Git History Replay feature for task extraction from MD files and commits |
| 2025-12-19 | `896fdc7` | Alex Everitt | Add Phase 1: Beads + GitForAI Integration Foundation |

<!-- /gitforai:auto -->

<!-- gitforai:auto section="Authors" synced="7596ec75b04225d870f2a7a9244480a70c5b4c85" -->
## Authors

- **Alex Everitt** — 4 commit(s)

<!-- /gitforai:auto -->

<!-- gitforai:auto section="History" synced="7596ec75b04225d870f2a7a9244480a70c5b4c85" -->
## History

- **Sprint 1: Dec 21, 2025 (1 commits)** (2025-12-21 → 2025-12-21, 1/1 commits in this cluster)
- **Sprint: Dec 19, 2025 (3 commits)** (2025-12-19 → 2025-12-19, 3/3 commits in this cluster)

<!-- /gitforai:auto -->
