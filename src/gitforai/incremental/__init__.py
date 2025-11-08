"""Incremental update system for GitForAI.

This module provides functionality to incrementally update the vector database
with only new commits since the last indexing operation, avoiding full re-indexing.
"""

from gitforai.incremental.state import StateManager, GitForAIState, RepositoryState, BranchState
from gitforai.incremental.delta import DeltaDetector
from gitforai.incremental.manager import IncrementalUpdateManager

__all__ = [
    "StateManager",
    "GitForAIState",
    "RepositoryState",
    "BranchState",
    "DeltaDetector",
    "IncrementalUpdateManager",
]
