"""
Beads Integration Module

Provides Python wrappers for beads (bd) and beads_viewer (bv) CLI tools.

Three Intelligence Layers:
1. BeadsClient (bd) - Task storage and CRUD operations
2. BeadsViewerClient (bv) - Graph-theoretic intelligence via Robot Protocol
3. GitForAI (native) - Semantic search over commit history

This module implements Layers 1 and 2 as subprocess wrappers around Go CLI tools.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BeadsError(Exception):
    """Base exception for beads integration errors"""
    pass


class BeadsCommandError(BeadsError):
    """Raised when a beads command fails"""
    def __init__(self, command: List[str], returncode: int, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"Command {' '.join(command)} failed with code {returncode}: {stderr}")


class BeadsConfigError(BeadsError):
    """Raised when beads configuration is invalid"""
    pass


@dataclass
class BeadsConfig:
    """Configuration for beads integration

    Attributes:
        repo_path: Path to the repository containing .beads directory
        bd_path: Path to bd CLI binary (default: ~/go/bin/bd)
        bv_path: Path to bv CLI binary (default: ~/go/bin/bv)
        timeout: Command timeout in seconds (default: 30)
    """
    repo_path: Path
    bd_path: Optional[Path] = None
    bv_path: Optional[Path] = None
    timeout: int = 30

    def __post_init__(self):
        """Validate configuration and set defaults"""
        # Convert string paths to Path objects
        if isinstance(self.repo_path, str):
            self.repo_path = Path(self.repo_path)

        # Set default CLI paths if not provided
        if self.bd_path is None:
            self.bd_path = Path.home() / "go" / "bin" / "bd"
        elif isinstance(self.bd_path, str):
            self.bd_path = Path(self.bd_path)

        if self.bv_path is None:
            self.bv_path = Path.home() / "go" / "bin" / "bv"
        elif isinstance(self.bv_path, str):
            self.bv_path = Path(self.bv_path)

        # Validate paths
        if not self.repo_path.exists():
            raise BeadsConfigError(f"Repository path does not exist: {self.repo_path}")

        if not self.bd_path.exists():
            raise BeadsConfigError(f"bd binary not found at {self.bd_path}")

        if not self.bv_path.exists():
            raise BeadsConfigError(f"bv binary not found at {self.bv_path}")

        # Check for .beads directory
        beads_dir = self.repo_path / ".beads"
        if not beads_dir.exists():
            logger.warning(f"No .beads directory found at {self.repo_path}. Repository may not be initialized.")


class BeadsClient:
    """Layer 1: Beads (bd) - Task Storage and CRUD Operations

    Wraps the bd CLI tool to provide programmatic access to beads issue tracking.
    All methods use the --json flag for structured output.

    Example:
        >>> config = BeadsConfig(repo_path=Path("/path/to/repo"))
        >>> client = BeadsClient(config)
        >>> ready_tasks = client.ready()
        >>> task = client.show("gitforai-core-abc")
        >>> new_task = client.create(
        ...     title="Implement feature X",
        ...     description="Detailed description",
        ...     task_type="task",
        ...     priority=1
        ... )
    """

    def __init__(self, config: BeadsConfig):
        """Initialize BeadsClient

        Args:
            config: BeadsConfig instance with repository and CLI paths
        """
        self.config = config
        self._bd_path = str(config.bd_path)
        self._repo_path = str(config.repo_path)
        self._timeout = config.timeout

    def _run_command(self, args: List[str], expect_json: bool = True) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
        """Execute bd command and return parsed output

        Args:
            args: Command arguments (e.g., ['show', 'task-id', '--json'])
            expect_json: If True, parse stdout as JSON. If False, return raw stdout.

        Returns:
            Parsed JSON (dict or list) if expect_json=True, otherwise raw string

        Raises:
            BeadsCommandError: If command fails
        """
        cmd = [self._bd_path] + args

        logger.debug(f"Running bd command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=True
            )

            if not expect_json:
                return result.stdout

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                logger.debug(f"Command successful, returned {len(data) if isinstance(data, list) else 1} item(s)")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON output: {e}")
                logger.debug(f"Raw output: {result.stdout[:500]}")
                raise BeadsError(f"Invalid JSON response from bd: {e}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"stderr: {e.stderr}")
            raise BeadsCommandError(cmd, e.returncode, e.stderr)

        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {self._timeout}s: {' '.join(cmd)}")
            raise BeadsError(f"Command timed out: {' '.join(cmd)}")

    def show(self, task_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific task

        Args:
            task_id: Task ID (e.g., 'gitforai-core-abc')

        Returns:
            Task details as dictionary with keys: id, title, description, status,
            priority, issue_type, created_at, updated_at, etc.

        Example:
            >>> task = client.show("gitforai-core-abc")
            >>> print(task['title'])
            'Implement feature X'
        """
        result = self._run_command(['show', task_id, '--json'])

        # bd returns a JSON array with one element for show command
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        elif isinstance(result, dict):
            return result
        else:
            raise BeadsError(f"Unexpected response format from 'bd show': {type(result)}")

    def ready(self) -> List[Dict[str, Any]]:
        """Get all actionable tasks (no blockers)

        Returns:
            List of task dictionaries that are ready to work on

        Example:
            >>> ready_tasks = client.ready()
            >>> for task in ready_tasks:
            ...     print(f"{task['id']}: {task['title']}")
        """
        result = self._run_command(['ready', '--json'])

        if not isinstance(result, list):
            raise BeadsError(f"Expected list from 'bd ready', got {type(result)}")

        return result

    def list_all(
        self,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        issue_type: Optional[str] = None,
        assignee: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all tasks with optional filters

        Args:
            status: Filter by status (e.g., 'open', 'in_progress', 'closed')
            priority: Filter by priority (1-5)
            issue_type: Filter by type (e.g., 'task', 'bug', 'epic')
            assignee: Filter by assignee

        Returns:
            List of task dictionaries matching filters

        Example:
            >>> open_tasks = client.list_all(status='open', priority=1)
            >>> bugs = client.list_all(issue_type='bug')
        """
        args = ['list', '--json']

        if status:
            args.extend(['--status', status])
        if priority is not None:
            args.extend(['--priority', str(priority)])
        if issue_type:
            args.extend(['--type', issue_type])
        if assignee:
            args.extend(['--assignee', assignee])

        result = self._run_command(args)

        if not isinstance(result, list):
            raise BeadsError(f"Expected list from 'bd list', got {type(result)}")

        return result

    def create(
        self,
        title: str,
        description: str = "",
        task_type: str = "task",
        priority: int = 3,
        assignee: Optional[str] = None,
        parent: Optional[str] = None,
        deps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new task

        Args:
            title: Task title
            description: Detailed description
            task_type: Type of task ('task', 'bug', 'epic', etc.)
            priority: Priority level (1-5, default: 3)
            assignee: Optional assignee
            parent: Optional parent task ID for hierarchical relationship
            deps: Optional list of dependency specs (e.g., ['blocks:task-id', 'bd-123'])

        Returns:
            Created task details

        Example:
            >>> task = client.create(
            ...     title="Add authentication",
            ...     description="Implement JWT authentication",
            ...     task_type="task",
            ...     priority=1,
            ...     parent="epic-id"
            ... )
        """
        args = [
            'create',
            '--title', title,
            '--type', task_type,
            '--priority', str(priority),
            '--json'
        ]

        if description:
            args.extend(['--description', description])
        if assignee:
            args.extend(['--assignee', assignee])
        if parent:
            args.extend(['--parent', parent])
        if deps:
            args.extend(['--deps', ','.join(deps)])

        result = self._run_command(args)

        if isinstance(result, list) and len(result) > 0:
            return result[0]
        elif isinstance(result, dict):
            return result
        else:
            raise BeadsError(f"Unexpected response from 'bd create': {type(result)}")

    def update(
        self,
        task_id: str,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        assignee: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an existing task

        Args:
            task_id: Task ID to update
            status: New status
            priority: New priority (1-5)
            assignee: New assignee
            title: New title
            description: New description

        Returns:
            Updated task details

        Example:
            >>> task = client.update(
            ...     "gitforai-core-abc",
            ...     status="in_progress",
            ...     assignee="alice"
            ... )
        """
        args = ['update', task_id, '--json']

        if status:
            args.extend(['--status', status])
        if priority is not None:
            args.extend(['--priority', str(priority)])
        if assignee:
            args.extend(['--assignee', assignee])
        if title:
            args.extend(['--title', title])
        if description:
            args.extend(['--description', description])

        result = self._run_command(args)

        if isinstance(result, list) and len(result) > 0:
            return result[0]
        elif isinstance(result, dict):
            return result
        else:
            raise BeadsError(f"Unexpected response from 'bd update': {type(result)}")

    def close(self, task_id: str, reason: str = "Completed") -> Dict[str, Any]:
        """Close a task

        Args:
            task_id: Task ID to close
            reason: Reason for closing (default: "Completed")

        Returns:
            Closed task details

        Example:
            >>> task = client.close("gitforai-core-abc", reason="Fixed in commit xyz")
        """
        args = ['close', task_id, '--reason', reason, '--json']
        result = self._run_command(args)

        if isinstance(result, list) and len(result) > 0:
            return result[0]
        elif isinstance(result, dict):
            return result
        else:
            raise BeadsError(f"Unexpected response from 'bd close': {type(result)}")

    def add_comment(self, task_id: str, comment: str) -> Dict[str, Any]:
        """Add a comment to a task

        Args:
            task_id: Task ID
            comment: Comment text

        Returns:
            Updated task details

        Example:
            >>> task = client.add_comment("gitforai-core-abc", "Work in progress")
        """
        args = ['comment', task_id, comment, '--json']
        result = self._run_command(args)

        if isinstance(result, list) and len(result) > 0:
            return result[0]
        elif isinstance(result, dict):
            return result
        else:
            raise BeadsError(f"Unexpected response from 'bd comment': {type(result)}")

    def add_dependency(
        self,
        task_id: str,
        depends_on_id: str,
        dep_type: str = "blocks"
    ) -> None:
        """Add a dependency between tasks

        Args:
            task_id: Task that depends on another
            depends_on_id: Task that this task depends on (blocks task_id)
            dep_type: Type of dependency ('blocks', 'related', 'parent-child', 'discovered-from')

        Example:
            >>> # Phase 1 blocks Phase 2
            >>> client.add_dependency("phase2-id", "phase1-id", dep_type="blocks")
        """
        args = ['dep', 'add', task_id, depends_on_id, '--type', dep_type]
        self._run_command(args, expect_json=False)

    def sync(self) -> str:
        """Sync tasks with remote

        Returns:
            Sync output message

        Example:
            >>> result = client.sync()
        """
        return self._run_command(['sync'], expect_json=False)


class BeadsViewerClient:
    """Layer 2: Beads Viewer (bv) - Graph-Theoretic Intelligence via Robot Protocol

    Wraps the bv CLI tool to provide programmatic access to graph metrics and intelligence.
    All robot protocol commands return structured JSON.

    Provides 9 graph-theoretic metrics:
    - PageRank: Recursive dependency importance
    - Betweenness Centrality: Bottleneck identification
    - HITS (Hubs & Authorities): Task influence analysis
    - Eigenvector Centrality: Global importance
    - Degree Centrality: Direct connections (in/out)
    - Critical Path: Longest dependency chains
    - Graph Density: Overall connectivity
    - Cycle Detection: Circular dependencies
    - Topological Sort: Execution order

    Example:
        >>> config = BeadsConfig(repo_path=Path("/path/to/repo"))
        >>> client = BeadsViewerClient(config)
        >>> triage = client.triage()
        >>> top_task = triage['triage']['recommendations'][0]
        >>> print(f"Work on: {top_task['title']} (score: {top_task['score']:.3f})")
    """

    def __init__(self, config: BeadsConfig):
        """Initialize BeadsViewerClient

        Args:
            config: BeadsConfig instance with repository and CLI paths
        """
        self.config = config
        self._bv_path = str(config.bv_path)
        self._repo_path = str(config.repo_path)
        self._timeout = config.timeout

    def _run_robot_command(self, args: List[str]) -> Dict[str, Any]:
        """Execute bv robot protocol command and return parsed JSON

        Args:
            args: Command arguments (e.g., ['--robot-triage'])

        Returns:
            Parsed JSON response

        Raises:
            BeadsCommandError: If command fails
        """
        cmd = [self._bv_path] + args

        logger.debug(f"Running bv command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=True
            )

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                logger.debug(f"Robot command successful")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON output: {e}")
                logger.debug(f"Raw output: {result.stdout[:500]}")
                raise BeadsError(f"Invalid JSON response from bv: {e}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"stderr: {e.stderr}")
            raise BeadsCommandError(cmd, e.returncode, e.stderr)

        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {self._timeout}s: {' '.join(cmd)}")
            raise BeadsError(f"Command timed out: {' '.join(cmd)}")

    def triage(self) -> Dict[str, Any]:
        """Get strategic task recommendations with graph metrics

        Returns prioritized task recommendations based on graph analysis.
        Includes quick wins, blockers to clear, and project health metrics.

        Returns:
            Dictionary containing:
            - quick_ref: Summary stats (open/actionable/blocked counts)
            - recommendations: Ranked tasks with scores and reasoning
            - quick_wins: Low-effort, high-impact tasks
            - blockers_to_clear: High-impact blocking tasks
            - project_health: Overall project metrics
            - suggested_commands: Actionable next steps

        Example:
            >>> triage = client.triage()
            >>> print(f"Actionable tasks: {triage['triage']['quick_ref']['actionable_count']}")
            >>> for rec in triage['triage']['recommendations'][:3]:
            ...     print(f"{rec['id']}: {rec['title']} (score: {rec['score']:.3f})")
        """
        return self._run_robot_command(['--robot-triage'])

    def insights(self) -> Dict[str, Any]:
        """Get full graph analysis with 9 metrics for all tasks

        Returns comprehensive graph-theoretic analysis including all metrics
        for every node in the dependency graph.

        Returns:
            Dictionary containing:
            - metrics: Graph-level stats (node_count, edge_count, density, has_cycles)
            - nodes: Per-task metrics (pagerank, betweenness, degree, HITS, etc.)
            - critical_path: Tasks on the longest dependency chain

        Example:
            >>> insights = client.insights()
            >>> print(f"Graph density: {insights['metrics']['density']:.3f}")
            >>> for task_id, metrics in insights['nodes'].items():
            ...     if metrics['pagerank'] > 0.1:
            ...         print(f"{task_id}: PageRank={metrics['pagerank']:.3f}")
        """
        return self._run_robot_command(['--robot-insights'])

    def history(self) -> Dict[str, Any]:
        """Get task-to-commit correlations

        Returns historical analysis linking beads tasks to git commits.
        Useful for understanding which commits addressed which tasks.

        Returns:
            Dictionary containing:
            - stats: Summary (total_issues, issues_with_commits, total_commits)
            - histories: Per-task commit history
            - commit_index: Reverse lookup (commit -> tasks)

        Example:
            >>> history = client.history()
            >>> print(f"Issues with commits: {history['stats']['issues_with_commits']}")
            >>> for task_id, task_history in history['histories'].items():
            ...     if task_history['commits']:
            ...         print(f"{task_id}: {len(task_history['commits'])} commit(s)")
        """
        return self._run_robot_command(['--robot-history'])

    def plan(self) -> Dict[str, Any]:
        """Get parallel execution tracks for multi-agent coordination

        Generates execution plan with parallel work streams based on
        dependency graph structure.

        Returns:
            Dictionary containing:
            - tracks: List of parallel execution tracks
            - Each track contains: track_id, tasks, dependencies

        Example:
            >>> plan = client.plan()
            >>> print(f"Total tracks: {len(plan['tracks'])}")
            >>> for track in plan['tracks']:
            ...     print(f"Track {track['track_id']}: {len(track['tasks'])} task(s)")
        """
        return self._run_robot_command(['--robot-plan'])

    def graph_export(self, format: str = 'json') -> Union[Dict[str, Any], str]:
        """Export dependency graph in specified format

        Args:
            format: Export format ('json', 'dot', 'mermaid')

        Returns:
            Graph data in requested format:
            - 'json': Dict with nodes and edges
            - 'dot': GraphViz DOT format string
            - 'mermaid': Mermaid diagram string

        Raises:
            ValueError: If format is not supported

        Example:
            >>> # Export as JSON for programmatic processing
            >>> graph = client.graph_export('json')
            >>> print(f"Nodes: {len(graph['nodes'])}, Edges: {len(graph['edges'])}")
            >>>
            >>> # Export as Mermaid for documentation
            >>> mermaid = client.graph_export('mermaid')
            >>> print(mermaid)
        """
        valid_formats = ['json', 'dot', 'mermaid']
        if format not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Must be one of: {valid_formats}")

        result = self._run_robot_command(['--robot-graph', f'--graph-format={format}'])

        # For JSON format, result is already a dict
        # For DOT and Mermaid, it's returned as a string in the response
        if format == 'json':
            return result
        else:
            # DOT and Mermaid are returned as strings
            # The robot protocol may wrap them, or return them directly
            if isinstance(result, dict) and 'graph' in result:
                return result['graph']
            elif isinstance(result, str):
                return result
            else:
                # Fallback: convert result to string
                return str(result)
