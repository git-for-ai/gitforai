"""GitForAI Wiki Module

Generates and self-corrects a GitHub-flavored wiki from git history.

Pipeline:
- identity.py: FileCluster -> WikiPage stubs (page identity + provenance model)
- store.py:   Round-trip WikiPage <-> markdown files (GitHub wiki layout)
- renderer.py: Cluster + epics + tasks -> page sections (naive Step 1: no LLM)

Future stages: drift.py, merge.py, orchestrator.py.
"""

from .drift import DriftDetector, DriftReport
from .identity import (
    PageSource,
    WikiSection,
    WikiPage,
    derive_pages_from_clusters,
)
from .merge import merge_human_edits
from .orchestrator import SyncReport, WikiOrchestrator
from .renderer import LLMRenderer, NaiveRenderer, auto_detect_llm_provider
from .store import WikiStore

__all__ = [
    "PageSource",
    "WikiSection",
    "WikiPage",
    "derive_pages_from_clusters",
    "WikiStore",
    "NaiveRenderer",
    "LLMRenderer",
    "auto_detect_llm_provider",
    "DriftDetector",
    "DriftReport",
    "merge_human_edits",
    "WikiOrchestrator",
    "SyncReport",
]
