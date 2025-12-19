"""
Unified Task Intelligence Module

Combines three intelligence layers:
1. BeadsClient (bd) - Task storage and CRUD operations
2. BeadsViewerClient (bv) - Graph-theoretic intelligence via Robot Protocol
3. GitForAI QueryEngine - Semantic search over commit history

Provides high-level API for AI agents to access comprehensive task context.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UnifiedConfig:
    """Configuration for unified task intelligence

    Attributes:
        repo_path: Path to the repository
        vectordb_path: Path to GitForAI vector database (optional)
        bd_path: Path to bd CLI binary (default: ~/go/bin/bd)
        bv_path: Path to bv CLI binary (default: ~/go/bin/bv)
        timeout: Command timeout in seconds (default: 30)
        enable_gitforai: Enable GitForAI semantic search (default: True if vectordb exists)
    """
    repo_path: Path
    vectordb_path: Optional[Path] = None
    bd_path: Optional[Path] = None
    bv_path: Optional[Path] = None
    timeout: int = 30
    enable_gitforai: bool = True

    def __post_init__(self):
        """Validate configuration and set defaults"""
        if isinstance(self.repo_path, str):
            self.repo_path = Path(self.repo_path)

        # Set default vectordb path if not provided
        if self.vectordb_path is None:
            self.vectordb_path = self.repo_path / ".gitforai" / "vectordb"
        elif isinstance(self.vectordb_path, str):
            self.vectordb_path = Path(self.vectordb_path)

        # Check if GitForAI should be enabled
        if self.enable_gitforai and not self.vectordb_path.exists():
            logger.warning(
                f"GitForAI vector database not found at {self.vectordb_path}. "
                "Layer 3 (semantic search) will be disabled."
            )
            self.enable_gitforai = False


class UnifiedTaskIntelligence:
    """High-level API combining all three intelligence layers

    Provides unified access to:
    - Layer 1: Task storage (BeadsClient)
    - Layer 2: Graph intelligence (BeadsViewerClient)
    - Layer 3: Historical context (GitForAI QueryEngine)

    Example:
        >>> config = UnifiedConfig(repo_path=Path("/path/to/repo"))
        >>> intel = UnifiedTaskIntelligence(config)
        >>>
        >>> # Get complete context for a task
        >>> context = intel.get_task_context("gitforai-core-abc")
        >>> print(context['task']['title'])
        >>> print(f"PageRank: {context['metrics']['pagerank']:.3f}")
        >>> print(f"Similar commits: {len(context['similar_work'])}")
        >>>
        >>> # Get smart recommendations
        >>> recommendations = intel.recommend_next_task()
        >>> top = recommendations[0]
        >>> print(f"Work on: {top['title']} (score: {top['score']:.3f})")
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize unified intelligence with all three layers

        Args:
            config: UnifiedConfig instance
        """
        from .beads import BeadsClient, BeadsViewerClient, BeadsConfig

        self.config = config

        # Initialize Layer 1 & 2: Beads and BeadsViewer
        beads_config = BeadsConfig(
            repo_path=config.repo_path,
            bd_path=config.bd_path,
            bv_path=config.bv_path,
            timeout=config.timeout
        )

        self.bd = BeadsClient(beads_config)
        self.bv = BeadsViewerClient(beads_config)

        # Initialize Layer 3: GitForAI (optional)
        self.gitforai = None
        self.gitforai_available = False

        if config.enable_gitforai:
            try:
                from gitforai.storage.query import QueryEngine
                from gitforai.storage.config import VectorDBConfig
                from gitforai.llm.embeddings import EmbeddingService
                from gitforai.llm.local_provider import LocalProvider

                # Create VectorDBConfig with the vectordb path
                vectordb_config = VectorDBConfig(persist_dir=config.vectordb_path)

                # Initialize local embedding provider (no API key needed)
                embedding_provider = LocalProvider(model='all-MiniLM-L6-v2')

                # Initialize embedding service with the provider
                embedding_service = EmbeddingService(provider=embedding_provider)

                # Create QueryEngine with config and embedding service
                self.gitforai = QueryEngine(
                    config=vectordb_config,
                    embedding_service=embedding_service
                )
                self.gitforai_available = True
                logger.info("GitForAI QueryEngine initialized successfully with local embeddings")
            except ImportError as e:
                logger.warning(
                    f"GitForAI dependencies not installed: {e}. "
                    "Install with: pip install -e .[vectordb,local-embeddings]"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize GitForAI QueryEngine: {e}")

    def get_task_context(self, task_id: str) -> Dict[str, Any]:
        """Get complete context for a task from all three layers

        Combines:
        - Task details from beads (title, description, status, priority, etc.)
        - Graph metrics from beads_viewer (PageRank, betweenness, etc.)
        - Historical correlation from beads_viewer (commits linked to this task)
        - Similar work from gitforai (semantically similar commits)

        Args:
            task_id: Task ID (e.g., 'gitforai-core-abc')

        Returns:
            Dictionary containing:
            - task: Task details from bd
            - metrics: Graph metrics from bv (PageRank, betweenness, etc.)
            - commit_history: Commits linked to this task from bv
            - similar_work: Semantically similar commits from gitforai

        Example:
            >>> context = intel.get_task_context("gitforai-core-abc")
            >>> print(f"Task: {context['task']['title']}")
            >>> print(f"Status: {context['task']['status']}")
            >>> print(f"PageRank: {context['metrics'].get('pagerank', 0):.3f}")
            >>> print(f"Similar commits: {len(context['similar_work'])}")
        """
        logger.debug(f"Getting complete context for task: {task_id}")

        # Layer 1: Task details
        task = self.bd.show(task_id)
        logger.debug(f"Retrieved task details: {task.get('title', 'N/A')}")

        # Layer 2a: Graph metrics
        insights = self.bv.insights()
        task_metrics = insights.get('nodes', {}).get(task_id, {})
        logger.debug(f"Retrieved graph metrics (PageRank: {task_metrics.get('pagerank', 0):.3f})")

        # Layer 2b: Historical correlation (task-to-commit links)
        history = self.bv.history()
        task_commits = history.get('histories', {}).get(task_id, {})
        if task_commits is None:
            task_commits = {}

        # Handle commits field which can be None, list, or missing
        commits_list = task_commits.get('commits') if isinstance(task_commits, dict) else None
        commit_count = len(commits_list) if commits_list else 0
        logger.debug(f"Retrieved commit history ({commit_count} commits)")

        # Layer 3: Semantic search for similar work (if available)
        similar_commits = []
        if self.gitforai_available and 'title' in task:
            try:
                logger.debug(f"Searching for similar commits: '{task['title']}'")

                # Run async search_commits in sync context
                result = asyncio.run(self.gitforai.search_commits(task['title'], n_results=5))

                # Extract commit data from QueryResult
                similar_commits = [
                    {
                        'hash': meta.get('commit_hash', 'N/A')[:7],
                        'message': meta.get('message', 'N/A'),
                        'author': meta.get('author_name', 'N/A'),
                        'date': meta.get('author_date', 'N/A'),
                        'files_changed': len(meta.get('files_changed', []))
                    }
                    for _, _, meta, _ in result
                ]
                logger.debug(f"Found {len(similar_commits)} similar commits")
            except Exception as e:
                logger.warning(f"GitForAI search failed: {e}")

        return {
            'task': task,
            'metrics': task_metrics,
            'commit_history': task_commits,
            'similar_work': similar_commits
        }

    def recommend_next_task(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get smart task recommendations combining graph metrics and historical patterns

        Uses beads_viewer triage for graph-based recommendations, then enriches
        with historical complexity data from gitforai (if available).

        Args:
            limit: Maximum number of recommendations to return (default: 5)

        Returns:
            List of task dictionaries with enhanced scoring:
            - All fields from bv triage (id, title, score, reasons, etc.)
            - historical_complexity: 'low', 'medium', 'high', or 'unknown'
            - avg_files_changed: Average files changed in similar past work
            - similar_commits_found: Number of similar commits found

        Example:
            >>> recommendations = intel.recommend_next_task()
            >>> for i, task in enumerate(recommendations, 1):
            ...     print(f"{i}. {task['title']}")
            ...     print(f"   Score: {task['score']:.3f}")
            ...     print(f"   PageRank: {task.get('pagerank_score', 0):.3f}")
            ...     if 'historical_complexity' in task:
            ...         print(f"   Historical complexity: {task['historical_complexity']}")
            ...         print(f"   Avg files changed: {task['avg_files_changed']}")
        """
        logger.debug(f"Getting task recommendations (limit: {limit})")

        # Get candidates from graph analysis (Layer 2)
        triage = self.bv.triage()
        candidates = triage.get('triage', {}).get('recommendations', [])[:limit * 2]  # Get extra for filtering
        logger.debug(f"Retrieved {len(candidates)} candidates from bv triage")

        # Enrich with historical complexity (Layer 3, if available)
        if self.gitforai_available:
            logger.debug("Enriching recommendations with historical complexity data")

            for task in candidates:
                try:
                    # Search for similar past work (async call in sync context)
                    result = asyncio.run(self.gitforai.search_commits(task['title'], n_results=10))

                    if len(result) > 0:
                        # Calculate average complexity from similar commits
                        total_files = sum(
                            len(meta.get('files_changed', []))
                            for _, _, meta, _ in result
                        )
                        avg_files = total_files / len(result) if len(result) > 0 else 0

                        # Classify complexity
                        if avg_files < 3:
                            complexity = 'low'
                        elif avg_files < 10:
                            complexity = 'medium'
                        else:
                            complexity = 'high'

                        task['historical_complexity'] = complexity
                        task['avg_files_changed'] = round(avg_files, 1)
                        task['similar_commits_found'] = len(result)

                        logger.debug(
                            f"Task {task['id']}: {complexity} complexity "
                            f"({avg_files:.1f} files avg, {len(result)} similar commits)"
                        )
                    else:
                        task['historical_complexity'] = 'unknown'
                        task['avg_files_changed'] = 0
                        task['similar_commits_found'] = 0

                except Exception as e:
                    logger.warning(f"Failed to enrich task {task.get('id')}: {e}")
                    task['historical_complexity'] = 'unknown'
                    task['avg_files_changed'] = 0
                    task['similar_commits_found'] = 0
        else:
            logger.debug("GitForAI not available, skipping historical enrichment")

        # Return top N recommendations
        return candidates[:limit]

    def get_ready_tasks(self) -> List[Dict[str, Any]]:
        """Get all actionable tasks (no blockers)

        Convenience method for Layer 1.

        Returns:
            List of task dictionaries that are ready to work on

        Example:
            >>> ready = intel.get_ready_tasks()
            >>> print(f"Ready tasks: {len(ready)}")
        """
        return self.bd.ready()

    def get_project_health(self) -> Dict[str, Any]:
        """Get overall project health metrics

        Returns project health data from beads_viewer triage.

        Returns:
            Dictionary containing:
            - open_count: Total open tasks
            - actionable_count: Tasks ready to work on
            - blocked_count: Tasks waiting on dependencies
            - in_progress_count: Tasks currently in progress

        Example:
            >>> health = intel.get_project_health()
            >>> print(f"Open: {health['open_count']}, Actionable: {health['actionable_count']}")
        """
        triage = self.bv.triage()
        return triage.get('triage', {}).get('quick_ref', {})

    def get_critical_path(self) -> List[str]:
        """Get tasks on the critical path (longest dependency chain)

        Returns:
            List of task IDs on the critical path

        Example:
            >>> critical = intel.get_critical_path()
            >>> print(f"Critical path length: {len(critical)}")
        """
        insights = self.bv.insights()
        return insights.get('critical_path', [])

    def search_commits(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search commit history semantically (Layer 3 only)

        Args:
            query: Natural language search query
            n_results: Number of results to return

        Returns:
            List of commit dictionaries with hash, message, author, date, files_changed

        Raises:
            RuntimeError: If GitForAI is not available

        Example:
            >>> commits = intel.search_commits("authentication bug fixes")
            >>> for commit in commits:
            ...     print(f"{commit['hash']}: {commit['message']}")
        """
        if not self.gitforai_available:
            raise RuntimeError(
                "GitForAI not available. Install dependencies: pip install -e .[vectordb,local-embeddings]"
            )

        # Run async search in sync context
        result = asyncio.run(self.gitforai.search_commits(query, n_results=n_results))

        return [
            {
                'hash': meta.get('commit_hash', 'N/A')[:7],
                'message': meta.get('message', 'N/A'),
                'author': meta.get('author_name', 'N/A'),
                'date': meta.get('author_date', 'N/A'),
                'files_changed': len(meta.get('files_changed', []))
            }
            for _, _, meta, _ in result
        ]

    def get_execution_plan(self) -> Dict[str, Any]:
        """Get parallel execution plan for multi-agent coordination

        Returns execution tracks from beads_viewer showing which tasks
        can be worked on in parallel.

        Returns:
            Dictionary containing execution tracks

        Example:
            >>> plan = intel.get_execution_plan()
            >>> print(f"Parallel tracks: {len(plan['tracks'])}")
            >>> for track in plan['tracks']:
            ...     print(f"Track {track['track_id']}: {len(track['tasks'])} task(s)")
        """
        return self.bv.plan()

    def get_graph_export(self, format: str = 'json') -> Any:
        """Export dependency graph in specified format

        Args:
            format: Export format ('json', 'dot', 'mermaid')

        Returns:
            Graph data in requested format

        Example:
            >>> graph = intel.get_graph_export('json')
            >>> mermaid = intel.get_graph_export('mermaid')
        """
        return self.bv.graph_export(format)

    def replay_git_history(
        self,
        from_commit: Optional[str] = None,
        create_beads_tasks: bool = False,
        dry_run: bool = True,
        branch: str = "HEAD",
        md_file_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Replay git history to extract task intelligence from MD files and commits

        Walks through git commit history chronologically from the first commit (or a specified
        starting point) to HEAD, extracting task information from:
        - Markdown files (TASK_*.md, TODO.md, README.md sections)
        - Commit messages and references
        - File change patterns

        This reconciles scattered documentation with actual implementation history to
        build a comprehensive tasklist with priorities based on historical patterns.

        Args:
            from_commit: Starting commit hash (default: first commit in repo)
            create_beads_tasks: If True, create beads tasks for extracted tasks (default: False)
            dry_run: If True, only analyze without creating tasks (default: True)
            branch: Branch to analyze (default: HEAD)
            md_file_patterns: Glob patterns for MD files to analyze (default: ['**/*.md'])

        Returns:
            Dictionary containing:
            - tasks_found: Number of tasks extracted
            - commits_analyzed: Number of commits processed
            - md_files_analyzed: List of MD files processed
            - tasks_by_status: Breakdown by status (new, in_progress, completed, abandoned)
            - priority_distribution: Tasks by priority level (1-5)
            - task_timeline: Chronological task events
            - top_priority_tasks: High-priority open tasks
            - all_tasks: Complete list of TaskState objects
            - recommendations: Suggested next steps

        Example:
            >>> result = intel.replay_git_history(dry_run=True)
            >>> print(f"Found {result['summary']['tasks_found']} tasks")
            >>> print(f"High priority tasks: {len(result['top_priority_tasks'])}")
            >>>
            >>> # Create beads tasks from results
            >>> result = intel.replay_git_history(create_beads_tasks=True, dry_run=False)
        """
        from gitforai.extraction.git_extractor import GitExtractor
        from gitforai.models.config import RepositoryConfig
        from gitforai.integrations.replay import GitHistoryReplayEngine

        logger.info(f"Starting git history replay for {self.config.repo_path}")
        logger.info(f"Dry run: {dry_run}, Create tasks: {create_beads_tasks}")

        # Initialize GitExtractor
        repo_config = RepositoryConfig(repo_path=self.config.repo_path)
        git_extractor = GitExtractor(repo_config)

        # Initialize replay engine
        replay_engine = GitHistoryReplayEngine(self.config, git_extractor)

        # Run replay
        result = replay_engine.replay_history(
            from_commit=from_commit,
            branch=branch,
            md_file_patterns=md_file_patterns
        )

        logger.info(f"Replay complete: {result.tasks_found} tasks found")

        # Optionally create beads tasks
        tasks_created = 0
        if create_beads_tasks and not dry_run:
            logger.info("Creating beads tasks from replay results...")

            for task in result.tasks:
                # Only create tasks that are open or in progress
                if task.status in ['new', 'in_progress']:
                    try:
                        self.bd.create(
                            title=task.title,
                            description=task.description or f"Task extracted from git history replay\n\nFirst seen: {task.first_seen_commit[:7]}\nLast updated: {task.last_updated_commit[:7]}\nRelated commits: {len(task.related_commits)}",
                            task_type='task',
                            priority=task.priority or 3
                        )
                        tasks_created += 1
                        logger.debug(f"Created beads task: {task.task_id}")
                    except Exception as e:
                        logger.warning(f"Failed to create beads task for {task.task_id}: {e}")

            logger.info(f"Created {tasks_created} beads tasks")

        result_dict = result.to_dict()
        result_dict['tasks_created'] = tasks_created if create_beads_tasks and not dry_run else 0

        return result_dict
