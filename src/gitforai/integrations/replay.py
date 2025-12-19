"""
Git History Replay Module

Replays git history to extract task intelligence from MD files and commits.
Reconciles scattered documentation with implementation history.
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Iterator

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TaskEvent:
    """Represents a single event in a task's lifecycle"""
    event_type: str  # created, updated, referenced, completed, abandoned
    commit_hash: str
    timestamp: datetime
    file_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type,
            'commit_hash': self.commit_hash[:7],
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'file_path': self.file_path,
            'details': self.details
        }


@dataclass
class TaskState:
    """Represents the complete state of a task throughout its lifecycle"""
    task_id: str
    title: str
    description: str = ""
    status: str = "new"  # new, in_progress, completed, abandoned
    priority: Optional[int] = None
    first_seen_commit: str = ""
    last_updated_commit: str = ""
    first_seen_date: Optional[datetime] = None
    last_updated_date: Optional[datetime] = None
    related_commits: List[str] = field(default_factory=list)
    file_locations: List[str] = field(default_factory=list)
    lifecycle_events: List[TaskEvent] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    files_changed: int = 0  # Total files changed across related commits

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'title': self.title,
            'description': self.description[:200] + '...' if len(self.description) > 200 else self.description,
            'status': self.status,
            'priority': self.priority,
            'first_seen_commit': self.first_seen_commit[:7] if self.first_seen_commit else None,
            'last_updated_commit': self.last_updated_commit[:7] if self.last_updated_commit else None,
            'first_seen_date': self.first_seen_date.isoformat() if self.first_seen_date else None,
            'last_updated_date': self.last_updated_date.isoformat() if self.last_updated_date else None,
            'related_commits_count': len(self.related_commits),
            'file_locations': self.file_locations,
            'tags': self.tags,
            'files_changed': self.files_changed
        }


@dataclass
class MDTask:
    """Represents a task extracted from a markdown file"""
    task_id: str
    title: str
    description: str = ""
    priority: Optional[int] = None
    status: Optional[str] = None
    file_path: str = ""
    line_number: int = 0
    tags: List[str] = field(default_factory=list)
    is_completed: bool = False


@dataclass
class ReplayResult:
    """Results from replaying git history"""
    tasks_found: int
    commits_analyzed: int
    md_files_analyzed: List[str]
    tasks: List[TaskState]
    tasks_by_status: Dict[str, int]
    priority_distribution: Dict[int, int]
    task_timeline: List[TaskEvent]
    commit_task_map: Dict[str, List[str]]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP response"""
        # Get top priority tasks (priority 1-2, status open/in_progress)
        top_tasks = [
            t for t in self.tasks
            if t.priority in [1, 2] and t.status in ['new', 'in_progress']
        ]
        top_tasks.sort(key=lambda t: (t.priority or 999, -(len(t.related_commits))))

        return {
            'summary': {
                'tasks_found': self.tasks_found,
                'commits_analyzed': self.commits_analyzed,
                'md_files_analyzed': self.md_files_analyzed
            },
            'tasks_by_status': self.tasks_by_status,
            'priority_distribution': self.priority_distribution,
            'top_priority_tasks': [t.to_dict() for t in top_tasks[:10]],
            'all_tasks': [t.to_dict() for t in self.tasks],
            'recommendations': self.recommendations
        }


# ============================================================================
# Markdown Task Parser
# ============================================================================

class MarkdownTaskParser:
    """Parses tasks from various markdown file formats"""

    # Regex patterns for detecting various task formats
    PATTERNS = {
        'task_files': re.compile(r'TASK[_-](\d+)[_-](.+)\.md', re.IGNORECASE),
        'todo_sections': re.compile(r'^##?\s+(TODO|Tasks?|Backlog)', re.MULTILINE | re.IGNORECASE),
        'task_items': re.compile(r'^\s*[-*]\s+\[([ xX])\]\s+(.+)$', re.MULTILINE),
        'priority_markers': re.compile(r'\[P(\d)\]|\(priority:\s*(\d)\)', re.IGNORECASE),
        'task_ids': re.compile(r'\[([A-Z]+-\d+)\]|#(\d+)'),
        'header_tasks': re.compile(r'^##\s+(.+)$', re.MULTILINE),
    }

    def parse_file(self, content: str, file_path: str, commit_hash: str) -> List[MDTask]:
        """
        Parse a markdown file for task information

        Args:
            content: File content
            file_path: Path to the file
            commit_hash: Commit hash where this file version exists

        Returns:
            List of MDTask objects
        """
        tasks = []

        # Extract task ID from filename if it matches TASK_*.md pattern
        filename_match = self.PATTERNS['task_files'].search(file_path)
        if filename_match:
            task_id = filename_match.group(1)
            task_title = filename_match.group(2).replace('_', ' ').replace('-', ' ')

            # Extract description from content (first paragraph after title)
            lines = content.split('\n')
            description_lines = []
            in_description = False

            for line in lines:
                if line.startswith('#'):
                    in_description = True
                    continue
                if in_description and line.strip():
                    description_lines.append(line)
                elif in_description and len(description_lines) > 0:
                    break

            description = '\n'.join(description_lines[:5])  # First 5 lines

            # Extract priority if present
            priority = self._extract_priority(content)

            tasks.append(MDTask(
                task_id=f"TASK-{task_id}",
                title=task_title,
                description=description,
                priority=priority,
                file_path=file_path,
                line_number=1
            ))

        # Parse TODO sections and task lists
        tasks.extend(self._parse_task_lists(content, file_path))

        return tasks

    def _parse_task_lists(self, content: str, file_path: str) -> List[MDTask]:
        """Parse task lists with checkboxes"""
        tasks = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            match = self.PATTERNS['task_items'].match(line)
            if match:
                is_completed = match.group(1).lower() == 'x'
                task_text = match.group(2).strip()

                # Extract priority
                priority = self._extract_priority(task_text)

                # Extract task ID if present
                task_id_match = self.PATTERNS['task_ids'].search(task_text)
                if task_id_match:
                    task_id = task_id_match.group(1) or task_id_match.group(2)
                    # Remove task ID from title
                    task_text = self.PATTERNS['task_ids'].sub('', task_text).strip()
                else:
                    # Generate task ID from content
                    task_id = self._generate_task_id(task_text, file_path)

                # Remove priority markers from title
                task_text = self.PATTERNS['priority_markers'].sub('', task_text).strip()

                tasks.append(MDTask(
                    task_id=task_id,
                    title=task_text,
                    priority=priority,
                    status='completed' if is_completed else 'new',
                    is_completed=is_completed,
                    file_path=file_path,
                    line_number=i + 1
                ))

        return tasks

    def _extract_priority(self, text: str) -> Optional[int]:
        """Extract priority from text"""
        match = self.PATTERNS['priority_markers'].search(text)
        if match:
            priority_str = match.group(1) or match.group(2)
            try:
                return int(priority_str)
            except (ValueError, TypeError):
                pass
        return None

    def _generate_task_id(self, title: str, file_path: str) -> str:
        """Generate a unique task ID from title and file path"""
        # Create a hash from title + file path for uniqueness
        content = f"{title}{file_path}".encode('utf-8')
        hash_short = hashlib.md5(content).hexdigest()[:6]

        # Extract meaningful prefix from title
        words = re.findall(r'\b[A-Z][a-z]*\b|\b[a-z]+\b', title)
        prefix = ''.join(w[0].upper() for w in words[:3]) if words else 'TASK'

        return f"{prefix}-{hash_short}"


# ============================================================================
# Commit Message Analyzer
# ============================================================================

class CommitMessageAnalyzer:
    """Analyzes commit messages for task references and patterns"""

    # Common patterns in commit messages
    PATTERNS = {
        'task_refs': re.compile(
            r'(?:refs?|closes?|fixes?|resolves?|implements?)\s+(?:#(\d+)|([A-Z]+-[\w\d]+))',
            re.IGNORECASE
        ),
        'task_brackets': re.compile(r'\[([A-Z]+-[\w\d]+)\]'),
        'task_hashtags': re.compile(r'#(\d+)'),
        'wip': re.compile(r'\b(WIP|work in progress)\b', re.IGNORECASE),
        'done': re.compile(r'\b(DONE|completed|finished)\b', re.IGNORECASE),
        'start': re.compile(r'\b(START|begin|starting)\b', re.IGNORECASE),
    }

    def extract_task_references(self, commit_message: str) -> List[str]:
        """Extract all task references from commit message"""
        references = []

        # Extract from formal references (fixes #123, closes TASK-456)
        for match in self.PATTERNS['task_refs'].finditer(commit_message):
            ref = match.group(1) or match.group(2)
            if ref:
                # Normalize to include prefix if just a number
                if ref.isdigit():
                    ref = f"TASK-{ref}"
                references.append(ref)

        # Extract from brackets [TASK-123]
        for match in self.PATTERNS['task_brackets'].finditer(commit_message):
            references.append(match.group(1))

        # Extract from hashtags #123
        for match in self.PATTERNS['task_hashtags'].finditer(commit_message):
            references.append(f"TASK-{match.group(1)}")

        return list(set(references))  # Deduplicate

    def classify_commit_intent(self, commit_message: str) -> str:
        """
        Classify commit intent regarding tasks

        Returns: 'task_start', 'task_progress', 'task_complete', 'task_reference', 'no_task'
        """
        message_lower = commit_message.lower()

        # Check for completion indicators
        if self.PATTERNS['done'].search(commit_message):
            return 'task_complete'

        # Check for formal task closures
        if re.search(r'\b(closes?|fixes?|resolves?)\b', message_lower):
            return 'task_complete'

        # Check for WIP
        if self.PATTERNS['wip'].search(commit_message):
            return 'task_progress'

        # Check for start indicators
        if self.PATTERNS['start'].search(commit_message):
            return 'task_start'

        # Check if any task references exist
        if self.extract_task_references(commit_message):
            return 'task_reference'

        return 'no_task'


# ============================================================================
# Task Evolution Tracker
# ============================================================================

class TaskEvolutionTracker:
    """Tracks task state changes throughout git history replay"""

    def __init__(self):
        self.tasks: Dict[str, TaskState] = {}
        self.events: List[TaskEvent] = []
        self._md_file_cache: Dict[str, Set[str]] = {}  # commit_hash -> set of MD files

    def process_task_creation(
        self,
        task_id: str,
        title: str,
        description: str,
        commit_hash: str,
        commit_date: datetime,
        file_path: str,
        priority: Optional[int] = None,
        tags: List[str] = None
    ) -> None:
        """Record a new task being created"""
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskState(
                task_id=task_id,
                title=title,
                description=description,
                priority=priority,
                first_seen_commit=commit_hash,
                last_updated_commit=commit_hash,
                first_seen_date=commit_date,
                last_updated_date=commit_date,
                file_locations=[file_path],
                tags=tags or []
            )

            event = TaskEvent(
                event_type='created',
                commit_hash=commit_hash,
                timestamp=commit_date,
                file_path=file_path,
                details={'title': title, 'priority': priority}
            )
            self.tasks[task_id].lifecycle_events.append(event)
            self.events.append(event)

            logger.debug(f"Created task {task_id}: {title}")
        else:
            # Task already exists, update it
            self.process_task_update(task_id, {
                'description': description,
                'priority': priority
            }, commit_hash, commit_date, file_path)

    def process_task_update(
        self,
        task_id: str,
        updates: Dict[str, Any],
        commit_hash: str,
        commit_date: datetime,
        file_path: Optional[str] = None
    ) -> None:
        """Record task being modified"""
        if task_id not in self.tasks:
            logger.warning(f"Attempted to update non-existent task: {task_id}")
            return

        task = self.tasks[task_id]
        task.last_updated_commit = commit_hash
        task.last_updated_date = commit_date

        if file_path and file_path not in task.file_locations:
            task.file_locations.append(file_path)

        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key) and value is not None:
                setattr(task, key, value)

        event = TaskEvent(
            event_type='updated',
            commit_hash=commit_hash,
            timestamp=commit_date,
            file_path=file_path,
            details=updates
        )
        task.lifecycle_events.append(event)
        self.events.append(event)

        logger.debug(f"Updated task {task_id}: {updates}")

    def process_task_reference(
        self,
        task_id: str,
        commit_hash: str,
        commit_date: datetime,
        commit_intent: str,
        files_changed: int = 0
    ) -> None:
        """Record a commit referencing a task"""
        if task_id not in self.tasks:
            # Create placeholder task if it doesn't exist
            self.tasks[task_id] = TaskState(
                task_id=task_id,
                title=f"Task {task_id}",
                description="Task inferred from commit reference",
                first_seen_commit=commit_hash,
                last_updated_commit=commit_hash,
                first_seen_date=commit_date,
                last_updated_date=commit_date
            )

        task = self.tasks[task_id]

        if commit_hash not in task.related_commits:
            task.related_commits.append(commit_hash)
            task.files_changed += files_changed

        task.last_updated_commit = commit_hash
        task.last_updated_date = commit_date

        # Update status based on commit intent
        if commit_intent == 'task_complete':
            task.status = 'completed'
        elif commit_intent == 'task_progress' and task.status == 'new':
            task.status = 'in_progress'
        elif commit_intent == 'task_start' and task.status == 'new':
            task.status = 'in_progress'

        event = TaskEvent(
            event_type='referenced',
            commit_hash=commit_hash,
            timestamp=commit_date,
            details={'commit_intent': commit_intent, 'files_changed': files_changed}
        )
        task.lifecycle_events.append(event)
        self.events.append(event)

        logger.debug(f"Referenced task {task_id} in commit {commit_hash[:7]} (intent: {commit_intent})")

    def mark_stale_tasks_abandoned(self, cutoff_date: datetime) -> int:
        """Mark tasks with no updates since cutoff_date as abandoned"""
        count = 0
        for task in self.tasks.values():
            if task.status in ['new', 'in_progress']:
                if task.last_updated_date and task.last_updated_date < cutoff_date:
                    task.status = 'abandoned'
                    count += 1
                    logger.debug(f"Marked task {task.task_id} as abandoned (no updates since {task.last_updated_date})")
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked tasks"""
        tasks_by_status = {}
        priority_dist = {}

        for task in self.tasks.values():
            # Count by status
            tasks_by_status[task.status] = tasks_by_status.get(task.status, 0) + 1

            # Count by priority
            if task.priority:
                priority_dist[task.priority] = priority_dist.get(task.priority, 0) + 1

        return {
            'total_tasks': len(self.tasks),
            'total_events': len(self.events),
            'tasks_by_status': tasks_by_status,
            'priority_distribution': priority_dist
        }


# ============================================================================
# Task Priority Calculator
# ============================================================================

class TaskPriorityCalculator:
    """Calculates task priorities based on historical patterns"""

    def calculate_priority(self, task_state: TaskState) -> int:
        """
        Calculate priority score (1-5, where 1 is highest)

        Factors:
        1. Explicit priority in MD files (weight: 40%)
        2. Activity level - number of related commits (weight: 25%)
        3. Recency - when last updated (weight: 20%)
        4. File change volume - how much code changed (weight: 15%)

        Special rules:
        - Completed tasks get priority 5 (lowest)
        - Abandoned tasks get priority 4
        """
        # Completed and abandoned tasks get low priority
        if task_state.status == 'completed':
            return 5
        if task_state.status == 'abandoned':
            return 4

        # If explicit priority exists, give it heavy weight
        if task_state.priority:
            explicit_score = task_state.priority * 0.4
        else:
            explicit_score = 3.0 * 0.4  # Default to medium priority

        # Activity score (more commits = higher priority)
        activity_score = self.calculate_activity_score(task_state) * 0.25

        # Recency score (more recent = higher priority)
        recency_score = self.calculate_recency_score(task_state) * 0.20

        # File change score (more changes = higher priority)
        change_score = self.calculate_change_score(task_state) * 0.15

        # Combine scores (invert for priority scale where 1 is highest)
        combined_score = explicit_score + activity_score + recency_score + change_score

        # Normalize to 1-5 scale
        priority = max(1, min(5, round(combined_score)))

        return priority

    def calculate_activity_score(self, task_state: TaskState) -> float:
        """
        Calculate activity score (0-5) based on commit count
        More commits = lower score (higher priority)
        """
        commit_count = len(task_state.related_commits)

        if commit_count == 0:
            return 3.0  # No commits = medium priority
        elif commit_count >= 10:
            return 1.0  # Many commits = high priority (active task)
        elif commit_count >= 5:
            return 2.0
        else:
            return 3.0

    def calculate_recency_score(self, task_state: TaskState) -> float:
        """
        Calculate recency score (0-5) based on last update
        More recent = lower score (higher priority)
        """
        if not task_state.last_updated_date:
            return 4.0  # No date = low priority

        now = datetime.now()
        days_since_update = (now - task_state.last_updated_date).days

        if days_since_update <= 7:
            return 1.0  # Updated in last week = high priority
        elif days_since_update <= 30:
            return 2.0  # Updated in last month
        elif days_since_update <= 90:
            return 3.0  # Updated in last quarter
        else:
            return 4.0  # Old task = lower priority

    def calculate_change_score(self, task_state: TaskState) -> float:
        """
        Calculate change volume score (0-5) based on files changed
        More files = lower score (higher priority/complexity)
        """
        if task_state.files_changed == 0:
            return 3.0
        elif task_state.files_changed >= 20:
            return 1.0  # High complexity
        elif task_state.files_changed >= 10:
            return 2.0
        else:
            return 3.0


# ============================================================================
# Git History Replay Engine
# ============================================================================

class GitHistoryReplayEngine:
    """Replays git history to extract task intelligence from MD files and commits"""

    def __init__(self, config, git_extractor):
        """
        Initialize replay engine

        Args:
            config: UnifiedConfig instance
            git_extractor: GitExtractor instance
        """
        self.config = config
        self.git_extractor = git_extractor
        self.task_tracker = TaskEvolutionTracker()
        self.md_parser = MarkdownTaskParser()
        self.commit_analyzer = CommitMessageAnalyzer()
        self.priority_calculator = TaskPriorityCalculator()

        self.commits_processed = 0
        self.md_files_processed: Set[str] = set()

    def replay_history(
        self,
        from_commit: Optional[str] = None,
        branch: str = "HEAD",
        md_file_patterns: List[str] = None
    ) -> ReplayResult:
        """
        Replay git history from first commit (or from_commit) to HEAD

        Args:
            from_commit: Starting commit hash (default: first commit)
            branch: Branch to analyze (default: HEAD)
            md_file_patterns: Glob patterns for MD files (default: ['**/*.md'])

        Returns:
            ReplayResult with complete task intelligence
        """
        if md_file_patterns is None:
            md_file_patterns = ['**/*.md']

        logger.info(f"Starting git history replay for {self.config.repo_path}")
        logger.info(f"Branch: {branch}, From commit: {from_commit or 'first commit'}")

        # Extract commits chronologically (oldest first)
        # GitPython iter_commits returns newest first, so we'll reverse
        commits_iter = self.git_extractor.extract_all_commits(branch=branch)
        commits = list(commits_iter)
        commits.reverse()  # Now oldest first

        # If from_commit specified, find starting point
        if from_commit:
            try:
                start_idx = next(i for i, c in enumerate(commits) if c.hash.startswith(from_commit))
                commits = commits[start_idx:]
                logger.info(f"Starting from commit {commits[0].hash[:7]}")
            except StopIteration:
                logger.warning(f"Starting commit {from_commit} not found, processing all commits")

        logger.info(f"Processing {len(commits)} commits chronologically")

        # Process each commit
        for commit in commits:
            self._process_commit(commit, md_file_patterns)
            self.commits_processed += 1

            if self.commits_processed % 100 == 0:
                logger.info(f"Processed {self.commits_processed} commits...")

        logger.info(f"Completed processing {self.commits_processed} commits")
        logger.info(f"Found {len(self.md_files_processed)} unique MD files")

        # Mark stale tasks as abandoned (no updates in 90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        abandoned_count = self.task_tracker.mark_stale_tasks_abandoned(cutoff_date)
        logger.info(f"Marked {abandoned_count} stale tasks as abandoned")

        # Calculate final priorities
        for task in self.task_tracker.tasks.values():
            if task.priority is None or task.priority == 0:
                task.priority = self.priority_calculator.calculate_priority(task)

        # Generate result
        return self._generate_result()

    def _process_commit(self, commit, md_file_patterns: List[str]) -> None:
        """Process a single commit"""
        # Extract task references from commit message
        task_refs = self.commit_analyzer.extract_task_references(commit.message)
        commit_intent = self.commit_analyzer.classify_commit_intent(commit.message)

        # Process task references
        for task_ref in task_refs:
            self.task_tracker.process_task_reference(
                task_id=task_ref,
                commit_hash=commit.hash,
                commit_date=commit.timestamp,
                commit_intent=commit_intent,
                files_changed=len(commit.files_changed)
            )

        # Check if any MD files were changed in this commit
        md_files_changed = [
            f for f in commit.files_changed
            if f.endswith('.md') and self._matches_patterns(f, md_file_patterns)
        ]

        # Process MD files
        for md_file in md_files_changed:
            self._process_md_file(md_file, commit)

    def _process_md_file(self, file_path: str, commit) -> None:
        """Process a markdown file from a commit"""
        try:
            # Extract file snapshot at this commit
            snapshot = self.git_extractor.extract_file_snapshot(commit.hash, file_path)

            if not snapshot:
                return

            self.md_files_processed.add(file_path)

            # Parse tasks from MD file
            md_tasks = self.md_parser.parse_file(
                content=snapshot.content,
                file_path=file_path,
                commit_hash=commit.hash
            )

            # Process each extracted task
            for md_task in md_tasks:
                if md_task.is_completed:
                    # Task is marked completed in MD
                    if md_task.task_id in self.task_tracker.tasks:
                        self.task_tracker.process_task_update(
                            task_id=md_task.task_id,
                            updates={'status': 'completed'},
                            commit_hash=commit.hash,
                            commit_date=commit.timestamp,
                            file_path=file_path
                        )
                    else:
                        # Create as completed task
                        self.task_tracker.process_task_creation(
                            task_id=md_task.task_id,
                            title=md_task.title,
                            description=md_task.description,
                            commit_hash=commit.hash,
                            commit_date=commit.timestamp,
                            file_path=file_path,
                            priority=md_task.priority,
                            tags=md_task.tags
                        )
                        self.task_tracker.tasks[md_task.task_id].status = 'completed'
                else:
                    # New or updated task
                    self.task_tracker.process_task_creation(
                        task_id=md_task.task_id,
                        title=md_task.title,
                        description=md_task.description,
                        commit_hash=commit.hash,
                        commit_date=commit.timestamp,
                        file_path=file_path,
                        priority=md_task.priority,
                        tags=md_task.tags
                    )

        except Exception as e:
            logger.warning(f"Error processing MD file {file_path} at commit {commit.hash[:7]}: {e}")

    def _matches_patterns(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file path matches any of the glob patterns"""
        from pathlib import Path
        file_path_obj = Path(file_path)

        for pattern in patterns:
            # Simple glob matching (** for recursive, * for wildcard)
            if pattern == '**/*.md':
                return True
            elif file_path_obj.match(pattern):
                return True

        return False

    def _generate_result(self) -> ReplayResult:
        """Generate final replay result"""
        stats = self.task_tracker.get_statistics()

        # Generate recommendations
        recommendations = self._generate_recommendations(stats)

        # Build commit-task map
        commit_task_map = {}
        for task in self.task_tracker.tasks.values():
            for commit_hash in task.related_commits:
                if commit_hash not in commit_task_map:
                    commit_task_map[commit_hash] = []
                commit_task_map[commit_hash].append(task.task_id)

        return ReplayResult(
            tasks_found=stats['total_tasks'],
            commits_analyzed=self.commits_processed,
            md_files_analyzed=sorted(list(self.md_files_processed)),
            tasks=list(self.task_tracker.tasks.values()),
            tasks_by_status=stats['tasks_by_status'],
            priority_distribution=stats['priority_distribution'],
            task_timeline=self.task_tracker.events,
            commit_task_map=commit_task_map,
            recommendations=recommendations
        )

    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        tasks_by_status = stats['tasks_by_status']
        priority_dist = stats['priority_distribution']

        # Recommendations based on status
        open_count = tasks_by_status.get('new', 0) + tasks_by_status.get('in_progress', 0)
        high_priority_open = sum(
            1 for t in self.task_tracker.tasks.values()
            if t.status in ['new', 'in_progress'] and t.priority in [1, 2]
        )

        if high_priority_open > 0:
            recommendations.append(
                f"Found {high_priority_open} high-priority tasks (P1-P2) ready to work on"
            )

        # Abandoned tasks
        abandoned_count = tasks_by_status.get('abandoned', 0)
        if abandoned_count > 0:
            recommendations.append(
                f"{abandoned_count} tasks appear abandoned (no updates in >90 days) - consider closing"
            )

        # In-progress tasks
        in_progress_count = tasks_by_status.get('in_progress', 0)
        if in_progress_count > 0:
            recommendations.append(
                f"{in_progress_count} tasks are in progress - check if they need completion"
            )

        # Correlation analysis
        tasks_with_commits = sum(
            1 for t in self.task_tracker.tasks.values()
            if len(t.related_commits) > 0
        )
        correlation_pct = (tasks_with_commits / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0

        recommendations.append(
            f"Correlation: {correlation_pct:.0f}% of tasks have related commits"
        )

        return recommendations
