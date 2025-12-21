"""
Git History Replay Module

Replays git history to extract task intelligence from MD files and commits.
Reconciles scattered documentation with implementation history.
"""

import re
import hashlib
import logging
from collections import defaultdict, Counter
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
    # Hierarchy support
    parent_section: Optional[str] = None  # Section ID or title that this task belongs to
    section_level: int = 0  # Header level (2=##, 3=###, etc.)
    section_order: int = 0  # Position within section


@dataclass
class MDSection:
    """Represents a markdown section that becomes a parent task"""
    section_id: str
    title: str
    description: str = ""
    level: int = 2  # Header level (## = 2, ### = 3)
    file_path: str = ""
    line_number: int = 0
    tasks: List[MDTask] = field(default_factory=list)
    subsections: List['MDSection'] = field(default_factory=list)
    order_in_doc: int = 0  # Position in document for sequential ordering


@dataclass
class FileCluster:
    """Group of files that frequently change together"""
    cluster_id: str
    files: List[str]
    co_change_frequency: float  # 0.0 to 1.0
    commits_touching_cluster: List[str]
    description: str = ""  # e.g., "Authentication Module"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'files': self.files,
            'co_change_frequency': self.co_change_frequency,
            'commits_count': len(self.commits_touching_cluster),
            'description': self.description
        }


@dataclass
class CoChangeResult:
    """Result of file co-change analysis"""
    clusters: List[FileCluster]
    co_change_matrix: Dict[Tuple[str, str], int]
    file_to_cluster: Dict[str, str]  # file_path -> cluster_id


@dataclass
class CommitTaskLink:
    """Link between commit and task (Layer 3)"""
    commit_hash: str
    task_id: str
    link_type: str  # "explicit", "phase", "inferred"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'commit_hash': self.commit_hash,
            'task_id': self.task_id,
            'link_type': self.link_type,
            'confidence': self.confidence
        }


@dataclass
class TemporalEpic:
    """Time-based epic representing a sprint or burst period (Layer 1)"""
    sprint_id: str
    title: str
    start_date: datetime
    end_date: datetime
    commits: List[str]  # List of commit hashes
    tasks: List[str] = field(default_factory=list)  # List of task_ids

    @property
    def duration_days(self) -> int:
        """Calculate duration in days"""
        return (self.end_date.date() - self.start_date.date()).days + 1

    @property
    def commit_count(self) -> int:
        """Get number of commits in this sprint"""
        return len(self.commits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sprint_id': self.sprint_id,
            'title': self.title,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'duration_days': self.duration_days,
            'commit_count': self.commit_count,
            'commits': self.commits,
            'tasks': self.tasks
        }


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
    sections: List[MDSection] = field(default_factory=list)  # Hierarchical sections
    # Layer 1: Temporal clustering
    temporal_epics: List[TemporalEpic] = field(default_factory=list)
    task_sprint_map: Dict[str, str] = field(default_factory=dict)  # task_id -> sprint_id
    # Layer 3: Task reference linking
    commit_task_links: List[CommitTaskLink] = field(default_factory=list)
    # Layer 4: File co-change analysis
    file_clusters: List[FileCluster] = field(default_factory=list)
    file_dependencies: List[Tuple[str, str, str]] = field(default_factory=list)  # (task1, task2, dep_type)

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
            'recommendations': self.recommendations,
            # Layer 1: Temporal clustering
            'temporal_epics': [epic.to_dict() for epic in self.temporal_epics],
            'task_sprint_map': self.task_sprint_map,
            # Layer 3: Task reference linking
            'commit_task_links': [link.to_dict() for link in self.commit_task_links],
            # Layer 4: File co-change analysis
            'file_clusters': [c.to_dict() for c in self.file_clusters],
            'file_dependencies': [
                {'task1': t1, 'task2': t2, 'type': dtype}
                for t1, t2, dtype in self.file_dependencies
            ]
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
# Markdown Hierarchy Parser
# ============================================================================

class MarkdownHierarchyParser:
    """Parses markdown files with hierarchical structure awareness"""

    # Pattern for detecting headers
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    TASK_PATTERN = re.compile(r'^\s*[-*]\s+\[([ xX])\]\s+(.+)$')
    PRIORITY_PATTERN = re.compile(r'\[P(\d)\]|\(priority:\s*(\d)\)', re.IGNORECASE)

    def __init__(self):
        self.md_parser = MarkdownTaskParser()

    def parse_file_with_hierarchy(
        self,
        content: str,
        file_path: str,
        commit_hash: str
    ) -> Tuple[List[MDSection], List[MDTask]]:
        """
        Parse markdown file with hierarchical structure

        Returns:
            Tuple of (sections, tasks_with_parents)
        """
        sections = []
        all_tasks = []
        current_section = None
        section_stack = []  # Track nested sections
        order_in_doc = 0

        lines = content.split('\n')

        for line_num, line in enumerate(lines):
            # Check for headers (sections/phases)
            header_match = self.HEADER_PATTERN.match(line)
            if header_match:
                header_level = len(header_match.group(1))  # Count # characters
                title = header_match.group(2).strip()

                # Create section
                section_id = self._generate_section_id(title, file_path, commit_hash)

                # Extract description (lines following header until next header or task)
                description = self._extract_section_description(lines, line_num + 1)

                section = MDSection(
                    section_id=section_id,
                    title=title,
                    description=description,
                    level=header_level,
                    file_path=file_path,
                    line_number=line_num + 1,
                    order_in_doc=order_in_doc
                )

                order_in_doc += 1

                # Handle section nesting
                # Pop sections from stack that are at same or deeper level
                while section_stack and section_stack[-1].level >= header_level:
                    section_stack.pop()

                # If we have a parent section, add as subsection
                if section_stack:
                    parent_section = section_stack[-1]
                    parent_section.subsections.append(section)
                else:
                    # Top-level section
                    sections.append(section)

                # Push current section onto stack
                section_stack.append(section)
                current_section = section

                continue

            # Check for task items
            task_match = self.TASK_PATTERN.match(line)
            if task_match and current_section:
                is_completed = task_match.group(1).lower() == 'x'
                task_text = task_match.group(2).strip()

                # Extract priority
                priority = self._extract_priority(task_text)

                # Generate task ID
                task_id = self.md_parser._generate_task_id(task_text, file_path)

                # Remove priority markers from title
                task_text = self.PRIORITY_PATTERN.sub('', task_text).strip()

                # Create task with parent section reference
                task = MDTask(
                    task_id=task_id,
                    title=task_text,
                    priority=priority,
                    status='completed' if is_completed else 'new',
                    is_completed=is_completed,
                    file_path=file_path,
                    line_number=line_num + 1,
                    parent_section=current_section.section_id,
                    section_level=current_section.level,
                    section_order=len(current_section.tasks)
                )

                current_section.tasks.append(task)
                all_tasks.append(task)

        return sections, all_tasks

    def _generate_section_id(self, title: str, file_path: str, commit_hash: str) -> str:
        """Generate unique ID for section"""
        content = f"{title}{file_path}{commit_hash}".encode('utf-8')
        hash_short = hashlib.md5(content).hexdigest()[:8]

        # Extract prefix from title
        words = re.findall(r'\b[A-Z][a-z]*\b|\b[a-z]+\b', title)
        prefix = ''.join(w[0].upper() for w in words[:3]) if words else 'SEC'

        return f"{prefix}-{hash_short}"

    def _extract_priority(self, text: str) -> Optional[int]:
        """Extract priority from text"""
        match = self.PRIORITY_PATTERN.search(text)
        if match:
            priority_str = match.group(1) or match.group(2)
            try:
                return int(priority_str)
            except (ValueError, TypeError):
                pass
        return None

    def _extract_section_description(self, lines: List[str], start_idx: int) -> str:
        """Extract description lines following a header"""
        description_lines = []

        for i in range(start_idx, len(lines)):
            line = lines[i]

            # Stop at next header or task
            if self.HEADER_PATTERN.match(line) or self.TASK_PATTERN.match(line):
                break

            # Add non-empty lines to description
            stripped = line.strip()
            if stripped:
                description_lines.append(stripped)
            elif description_lines:
                # Empty line after content - stop
                break

        return '\n'.join(description_lines[:3])  # First 3 lines


# ============================================================================
# Phase Order Analyzer
# ============================================================================

class PhaseOrderAnalyzer:
    """Analyzes sequential dependencies between phases/sections"""

    PHASE_PATTERN = re.compile(r'(?:Phase|Step|Stage)\s+(\d+)', re.IGNORECASE)

    def analyze_phase_order(self, sections: List[MDSection]) -> List[Tuple[str, str]]:
        """
        Detect implicit sequential dependencies between phases

        Returns:
            List of (from_section_title, to_section_title) where from blocks to
        """
        dependencies = []

        # Collect all sections (including nested subsections)
        all_sections = []
        def collect_sections(section_list):
            for section in section_list:
                all_sections.append(section)
                if section.subsections:
                    collect_sections(section.subsections)

        collect_sections(sections)

        # Sort by document order
        sorted_sections = sorted(all_sections, key=lambda s: s.order_in_doc)

        # Find phases with numeric identifiers
        phases_with_numbers = []
        for section in sorted_sections:
            match = self.PHASE_PATTERN.search(section.title)
            if match:
                phase_num = int(match.group(1))
                phases_with_numbers.append((phase_num, section))

        # Sort by phase number
        phases_with_numbers.sort(key=lambda x: x[0])

        # Create sequential dependencies (using titles for stability across commits)
        for i in range(len(phases_with_numbers) - 1):
            current_phase = phases_with_numbers[i][1]
            next_phase = phases_with_numbers[i + 1][1]

            # Current phase blocks next phase (return titles, not IDs)
            dependencies.append((current_phase.title, next_phase.title))

        return dependencies


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
        # Layer 3: New patterns for enhanced task reference extraction
        'task_number': re.compile(r'\b(?:Task|Issue|Feature|Bug)\s+(\d+(?:\.\d+)?)', re.IGNORECASE),
        'phase_ref': re.compile(r'\bPhase\s+(\d+(?:\.\d+)?)', re.IGNORECASE),
        # Lifecycle patterns
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

    def extract_all_references(self, commit_message: str) -> Dict[str, List[str]]:
        """
        Extract all types of task references (Layer 3 enhancement)

        Returns:
            Dict with keys:
                - 'task_ids': Formal task references (TASK-123, #456)
                - 'phases': Phase references (Phase 2, Phase 3.1)
                - 'numbers': Task numbers (Task 2.5, Issue 42)
        """
        result = {
            'task_ids': self.extract_task_references(commit_message),
            'phases': [],
            'numbers': []
        }

        # Extract phase references
        for match in self.PATTERNS['phase_ref'].finditer(commit_message):
            result['phases'].append(match.group(1))

        # Extract task numbers
        for match in self.PATTERNS['task_number'].finditer(commit_message):
            result['numbers'].append(match.group(1))

        return result


# ============================================================================
# Task Reference Linker (Layer 3)
# ============================================================================

class TaskReferenceLinker:
    """Links commits to tasks using pattern matching and file overlap inference"""

    def __init__(self, commit_analyzer: CommitMessageAnalyzer):
        """
        Args:
            commit_analyzer: CommitMessageAnalyzer for extracting references
        """
        self.commit_analyzer = commit_analyzer
        self.links: List[CommitTaskLink] = []

    def link_commits_to_tasks(
        self,
        commits: List[Any],
        tasks: Dict[str, 'TaskState'],
        sections: Dict[str, MDSection]
    ) -> List[CommitTaskLink]:
        """
        Link commits to tasks using multiple strategies

        Args:
            commits: List of CommitMetadata objects
            tasks: Dict mapping task_id -> TaskState
            sections: Dict mapping section_id -> MDSection

        Returns:
            List of CommitTaskLink objects

        Strategies:
            1. Explicit: Direct task ID references in commit message
            2. Phase: Phase references linked to phase sections
            3. Inferred: File overlap between commit and task files
        """
        links = []

        for commit in commits:
            # Extract all reference types from commit message
            refs = self.commit_analyzer.extract_all_references(commit.message)

            # Strategy 1: Explicit task IDs
            for task_id in refs['task_ids']:
                if task_id in tasks:
                    links.append(CommitTaskLink(
                        commit_hash=commit.hash,
                        task_id=task_id,
                        link_type="explicit",
                        confidence=1.0
                    ))

            # Strategy 2: Phase references
            for phase_num in refs['phases']:
                # Find section matching phase
                phase_title_pattern = f"Phase {phase_num}"
                for section_id, section in sections.items():
                    if phase_title_pattern in section.title:
                        links.append(CommitTaskLink(
                            commit_hash=commit.hash,
                            task_id=section_id,
                            link_type="phase",
                            confidence=0.9
                        ))

            # Strategy 3: Task numbers (Task 2.5, Issue 42)
            # Try to match task numbers to existing task IDs
            for task_num in refs['numbers']:
                # Look for tasks with this number in their ID or title
                for task_id, task in tasks.items():
                    if task_num in task.title or task_num in task_id:
                        links.append(CommitTaskLink(
                            commit_hash=commit.hash,
                            task_id=task_id,
                            link_type="explicit",
                            confidence=0.8
                        ))
                        break  # Only link to first match

            # Strategy 4: Inferred from file overlap
            commit_files = set(commit.files_changed)
            for task_id, task in tasks.items():
                task_files = set(task.file_locations)
                overlap = commit_files & task_files

                # Significant overlap = at least 2 files in common
                if len(overlap) >= 2:
                    # Check if we don't already have an explicit link
                    existing_links = [
                        link for link in links
                        if link.commit_hash == commit.hash and link.task_id == task_id
                    ]
                    if not existing_links:
                        links.append(CommitTaskLink(
                            commit_hash=commit.hash,
                            task_id=task_id,
                            link_type="inferred",
                            confidence=min(0.7, len(overlap) * 0.2)  # More overlap = higher confidence
                        ))

        self.links = links
        return links


# ============================================================================
# Commit Body Enricher (Layer 2)
# ============================================================================

class CommitBodyEnricher:
    """Enriches task descriptions with commit body content"""

    def __init__(self):
        """Initialize enricher"""
        self.enriched_tasks: Set[str] = set()
        self.commit_bodies_added: Dict[str, Set[str]] = defaultdict(set)  # task_id -> set of commit hashes

    def enrich_tasks_from_links(
        self,
        tasks: Dict[str, 'TaskState'],
        commits_by_hash: Dict[str, Any],
        commit_task_links: List[CommitTaskLink],
        min_confidence: float = 0.8
    ) -> int:
        """
        Enrich task descriptions using commit-task links

        Args:
            tasks: Dict mapping task_id -> TaskState
            commits_by_hash: Dict mapping commit_hash -> CommitMetadata
            commit_task_links: List of CommitTaskLink objects from Layer 3
            min_confidence: Minimum confidence threshold for including commit bodies

        Returns:
            Number of tasks enriched
        """
        enriched_count = 0

        # Group links by task
        links_by_task: Dict[str, List[CommitTaskLink]] = defaultdict(list)
        for link in commit_task_links:
            # Only use high-confidence links (explicit and phase, not inferred)
            if link.confidence >= min_confidence:
                links_by_task[link.task_id].append(link)

        # Enrich each task
        for task_id, task_links in links_by_task.items():
            if task_id not in tasks:
                continue

            task = tasks[task_id]

            # Sort links by commit timestamp (chronological order)
            sorted_links = sorted(
                task_links,
                key=lambda link: commits_by_hash.get(link.commit_hash).timestamp if link.commit_hash in commits_by_hash else datetime.min
            )

            # Add commit bodies
            for link in sorted_links:
                commit = commits_by_hash.get(link.commit_hash)
                if not commit:
                    continue

                # Check if commit has a body (not just subject line)
                commit_body = self._extract_commit_body(commit.message)
                if not commit_body:
                    continue

                # Check if we've already added this commit to this task
                if link.commit_hash in self.commit_bodies_added[task_id]:
                    continue

                # Add commit body to task description
                self._add_commit_body_to_task(task, commit, commit_body, link.link_type)
                self.commit_bodies_added[task_id].add(link.commit_hash)

                if task_id not in self.enriched_tasks:
                    enriched_count += 1
                    self.enriched_tasks.add(task_id)

        return enriched_count

    def _extract_commit_body(self, commit_message: str) -> Optional[str]:
        """
        Extract body from commit message (everything after first line)

        Args:
            commit_message: Full commit message

        Returns:
            Commit body or None if no body exists
        """
        lines = commit_message.split('\n', 1)
        if len(lines) < 2:
            return None

        body = lines[1].strip()
        # Only return if body is substantial (more than just whitespace)
        return body if len(body) > 10 else None

    def _add_commit_body_to_task(
        self,
        task: 'TaskState',
        commit: Any,
        commit_body: str,
        link_type: str
    ) -> None:
        """
        Add commit body to task description with formatting

        Args:
            task: TaskState to enrich
            commit: CommitMetadata object
            commit_body: Extracted commit body text
            link_type: Type of link (explicit, phase, inferred)
        """
        # Format commit entry
        timestamp_str = commit.timestamp.strftime('%Y-%m-%d')
        commit_hash_short = commit.hash[:7]

        entry = f"\n\n---\n**Commit {commit_hash_short}** ({timestamp_str}) [{link_type}]\n{commit_body}"

        # Initialize or append to description
        if not task.description or task.description == "":
            # Start with header
            task.description = f"## Implementation Details\n{entry}"
        elif "## Implementation Details" in task.description:
            # Append to existing implementation details
            task.description += entry
        else:
            # Prepend implementation details section
            task.description = f"## Implementation Details\n{entry}\n\n---\n\n{task.description}"


# ============================================================================
# Temporal Cluster Analyzer (Layer 1 - Time-Based Organization)
# ============================================================================

class TemporalClusterAnalyzer:
    """Analyzes temporal patterns to detect sprints and burst periods"""

    def __init__(self, burst_threshold: int = 5, merge_gap_days: int = 1):
        """
        Args:
            burst_threshold: Minimum commits per day to be considered a burst
            merge_gap_days: Maximum gap between burst days to merge into same sprint
        """
        self.burst_threshold = burst_threshold
        self.merge_gap_days = merge_gap_days
        self.sprints: List[TemporalEpic] = []

    def analyze_temporal_patterns(self, commits: List[Any]) -> List[TemporalEpic]:
        """
        Detect sprint/burst periods from commit timestamps

        Args:
            commits: List of CommitMetadata objects

        Returns:
            List of TemporalEpic objects representing sprints
        """
        if not commits:
            return []

        logger.info("Analyzing temporal patterns for sprint detection...")

        # Group commits by day
        by_day: Dict[datetime, List[Any]] = defaultdict(list)
        for commit in commits:
            day = commit.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            by_day[day].append(commit)

        # Find burst days (days with >threshold commits)
        burst_days = [
            (day, day_commits)
            for day, day_commits in by_day.items()
            if len(day_commits) >= self.burst_threshold
        ]

        logger.info(f"  Found {len(burst_days)} burst days (>={self.burst_threshold} commits/day)")

        # Merge consecutive burst days into sprints
        sprints = self._merge_consecutive_bursts(burst_days)

        logger.info(f"  Created {len(sprints)} sprint periods")

        self.sprints = sprints
        return sprints

    def _merge_consecutive_bursts(
        self,
        burst_days: List[Tuple[datetime, List[Any]]]
    ) -> List[TemporalEpic]:
        """
        Merge consecutive burst days into sprint periods

        Args:
            burst_days: List of (day, commits) tuples

        Returns:
            List of TemporalEpic objects
        """
        if not burst_days:
            return []

        sprints = []
        sorted_days = sorted(burst_days, key=lambda x: x[0])

        current_sprint_commits = []
        current_start_date = None
        current_end_date = None

        for day, day_commits in sorted_days:
            if current_start_date is None:
                # Start new sprint
                current_start_date = day
                current_end_date = day
                current_sprint_commits.extend(day_commits)
            else:
                # Check if this day is consecutive (within merge_gap_days)
                days_gap = (day.date() - current_end_date.date()).days

                if days_gap <= self.merge_gap_days:
                    # Extend current sprint
                    current_end_date = day
                    current_sprint_commits.extend(day_commits)
                else:
                    # Finalize current sprint
                    sprint = self._create_sprint(
                        current_start_date,
                        current_end_date,
                        current_sprint_commits
                    )
                    sprints.append(sprint)

                    # Start new sprint
                    current_start_date = day
                    current_end_date = day
                    current_sprint_commits = list(day_commits)

        # Finalize last sprint
        if current_start_date is not None:
            sprint = self._create_sprint(
                current_start_date,
                current_end_date,
                current_sprint_commits
            )
            sprints.append(sprint)

        return sprints

    def _create_sprint(
        self,
        start_date: datetime,
        end_date: datetime,
        commits: List[Any]
    ) -> TemporalEpic:
        """
        Create a TemporalEpic for a sprint period

        Args:
            start_date: Sprint start date
            end_date: Sprint end date
            commits: List of commits in this sprint

        Returns:
            TemporalEpic object
        """
        # Generate sprint ID
        sprint_id = f"SPRINT-{start_date.strftime('%Y%m%d')}"

        # Generate title
        title = self._generate_sprint_title(start_date, end_date, commits)

        # Get commit hashes
        commit_hashes = [c.hash for c in commits]

        return TemporalEpic(
            sprint_id=sprint_id,
            title=title,
            start_date=start_date,
            end_date=end_date,
            commits=commit_hashes,
            tasks=[]  # Will be populated by link_tasks_to_sprints
        )

    def _generate_sprint_title(
        self,
        start_date: datetime,
        end_date: datetime,
        commits: List[Any]
    ) -> str:
        """
        Generate descriptive sprint title

        Args:
            start_date: Sprint start date
            end_date: Sprint end date
            commits: List of commits in sprint

        Returns:
            Sprint title string
        """
        # Format date range
        if start_date.date() == end_date.date():
            date_range = start_date.strftime("%b %d, %Y")
        else:
            # Same month
            if start_date.month == end_date.month:
                date_range = f"{start_date.strftime('%b %d')}-{end_date.strftime('%d, %Y')}"
            else:
                date_range = f"{start_date.strftime('%b %d')}-{end_date.strftime('%b %d, %Y')}"

        commit_count = len(commits)

        # Try to extract sprint name from commit messages
        for commit in commits:
            message_lower = commit.message.lower()
            # Look for "Sprint N" in commit messages
            match = re.search(r'sprint\s+(\d+)', message_lower)
            if match:
                sprint_num = match.group(1)
                return f"Sprint {sprint_num}: {date_range} ({commit_count} commits)"

        # Default title
        return f"Sprint: {date_range} ({commit_count} commits)"

    def link_tasks_to_sprints(
        self,
        tasks: Dict[str, 'TaskState'],
        sprints: List[TemporalEpic]
    ) -> Dict[str, str]:
        """
        Link tasks to sprints based on when they were worked on

        A task belongs to a sprint if majority of its commits fall in sprint period.

        Args:
            tasks: Dict mapping task_id -> TaskState
            sprints: List of TemporalEpic objects

        Returns:
            Dict mapping task_id -> sprint_id
        """
        task_to_sprint = {}

        for task_id, task in tasks.items():
            # Count commits in each sprint based on lifecycle events
            sprint_commits: Dict[str, int] = defaultdict(int)

            for event in task.lifecycle_events:
                event_date = event.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

                # Find which sprint this event falls into
                for sprint in sprints:
                    sprint_start = sprint.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    sprint_end = sprint.end_date.replace(hour=0, minute=0, second=0, microsecond=0)

                    if sprint_start <= event_date <= sprint_end:
                        sprint_commits[sprint.sprint_id] += 1
                        break

            # Assign to sprint with most commits
            if sprint_commits:
                best_sprint_id = max(sprint_commits.items(), key=lambda x: x[1])[0]
                task_to_sprint[task_id] = best_sprint_id

                # Add task to sprint's task list
                for sprint in sprints:
                    if sprint.sprint_id == best_sprint_id:
                        sprint.tasks.append(task_id)
                        break

        return task_to_sprint


# ============================================================================
# File Co-Change Analyzer (Layer 4 - Concept Foundation)
# ============================================================================

class FileCoChangeAnalyzer:
    """Analyzes file co-change patterns to detect architectural coupling"""

    def __init__(self, co_change_threshold: int = 3):
        """
        Args:
            co_change_threshold: Minimum times files must change together to be considered coupled
        """
        self.co_change_threshold = co_change_threshold
        self.co_change_matrix: Dict[Tuple[str, str], int] = {}
        self.file_to_tasks: Dict[str, List[str]] = defaultdict(list)

    def analyze_commits(self, commits: List[Any]) -> CoChangeResult:
        """
        Build co-change matrix from all commits

        Args:
            commits: List of CommitMetadata objects

        Returns:
            CoChangeResult with clusters and matrix
        """
        logger.info("Analyzing file co-change patterns...")

        # Build co-change matrix
        for commit in commits:
            files = commit.files_changed

            # Record all file pairs changed together
            for i, file1 in enumerate(files):
                for file2 in files[i+1:]:
                    # Normalize file extensions (ignore certain files)
                    if self._should_skip_file(file1) or self._should_skip_file(file2):
                        continue

                    pair = tuple(sorted([file1, file2]))
                    self.co_change_matrix[pair] = self.co_change_matrix.get(pair, 0) + 1

        # Identify clusters
        clusters = self._identify_clusters()

        # Build file->cluster mapping
        file_to_cluster = {}
        for cluster in clusters:
            for file_path in cluster.files:
                file_to_cluster[file_path] = cluster.cluster_id

        logger.info(f"Found {len(clusters)} file clusters with {len(self.co_change_matrix)} co-change pairs")

        return CoChangeResult(
            clusters=clusters,
            co_change_matrix=self.co_change_matrix,
            file_to_cluster=file_to_cluster
        )

    def _should_skip_file(self, file_path: str) -> bool:
        """Skip files that aren't meaningful for co-change analysis"""
        skip_patterns = [
            '.gitignore',
            'package-lock.json',
            'yarn.lock',
            'poetry.lock',
            '.env',
            '__pycache__',
        ]
        return any(pattern in file_path for pattern in skip_patterns)

    def _identify_clusters(self) -> List[FileCluster]:
        """
        Group files that frequently change together into clusters

        Uses simple connected components approach:
        - Files that co-change >= threshold times are connected
        - Find all connected components
        """
        # Build adjacency list
        adjacency: Dict[str, Set[str]] = defaultdict(set)

        for (file1, file2), count in self.co_change_matrix.items():
            if count >= self.co_change_threshold:
                adjacency[file1].add(file2)
                adjacency[file2].add(file1)

        # Find connected components (file clusters)
        visited = set()
        clusters = []

        for file_path in adjacency.keys():
            if file_path in visited:
                continue

            # BFS to find connected component
            cluster_files = set()
            queue = [file_path]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster_files.add(current)

                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Create cluster
            if len(cluster_files) >= 2:  # At least 2 files to form a cluster
                cluster_id = f"cluster-{len(clusters)+1}"

                # Calculate average co-change frequency
                total_pairs = len(cluster_files) * (len(cluster_files) - 1) / 2
                actual_pairs = sum(
                    1 for (f1, f2) in self.co_change_matrix.keys()
                    if f1 in cluster_files and f2 in cluster_files
                )
                frequency = actual_pairs / total_pairs if total_pairs > 0 else 0

                # Generate description from common path prefix
                description = self._generate_cluster_description(list(cluster_files))

                clusters.append(FileCluster(
                    cluster_id=cluster_id,
                    files=sorted(list(cluster_files)),
                    co_change_frequency=frequency,
                    commits_touching_cluster=[],  # Will be filled later if needed
                    description=description
                ))

        return clusters

    def _generate_cluster_description(self, files: List[str]) -> str:
        """Generate a descriptive name for a file cluster"""
        if not files:
            return "Empty cluster"

        # Find common path prefix
        if len(files) == 1:
            return f"Single file: {files[0]}"

        # Split paths
        paths = [Path(f).parts for f in files]

        # Find common prefix
        common = []
        for parts in zip(*paths):
            if len(set(parts)) == 1:
                common.append(parts[0])
            else:
                break

        if common:
            prefix = '/'.join(common)
            return f"Module: {prefix}"

        # Check for common file extensions
        extensions = Counter([Path(f).suffix for f in files])
        most_common_ext = extensions.most_common(1)[0] if extensions else ('', 0)

        if most_common_ext[0]:
            return f"{len(files)} {most_common_ext[0]} files"

        return f"Cluster of {len(files)} files"

    def create_dependencies(
        self,
        tasks: Dict[str, TaskState],
        threshold: Optional[int] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Create file-based dependencies between tasks

        Args:
            tasks: Dict of task_id -> TaskState
            threshold: Override co-change threshold (default: use instance threshold)

        Returns:
            List of (task_id_1, task_id_2, dep_type) tuples
        """
        if threshold is None:
            threshold = self.co_change_threshold

        dependencies = []

        # Build file->tasks mapping
        file_to_tasks = defaultdict(list)
        for task_id, task in tasks.items():
            for file_path in task.file_locations:
                file_to_tasks[file_path].append(task_id)

        # Create dependencies between tasks touching co-changed files
        for (file1, file2), count in self.co_change_matrix.items():
            if count >= threshold:
                tasks_file1 = file_to_tasks.get(file1, [])
                tasks_file2 = file_to_tasks.get(file2, [])

                for task1_id in tasks_file1:
                    for task2_id in tasks_file2:
                        if task1_id != task2_id:
                            # Avoid duplicates
                            dep = tuple(sorted([task1_id, task2_id])) + ("related",)
                            if dep not in dependencies:
                                dependencies.append((task1_id, task2_id, "related"))

        logger.info(f"Created {len(dependencies)} file-based dependencies")
        return dependencies


# ============================================================================
# Task Evolution Tracker
# ============================================================================

class TaskEvolutionTracker:
    """Tracks task state changes throughout git history replay"""

    def __init__(self):
        self.tasks: Dict[str, TaskState] = {}
        self.events: List[TaskEvent] = []
        self._md_file_cache: Dict[str, Set[str]] = {}  # commit_hash -> set of MD files
        # Hierarchy support
        self.sections: Dict[str, MDSection] = {}  # section_id -> MDSection
        self.section_dependencies: List[Tuple[str, str]] = []  # (from_id, to_id) dependencies

    def process_task_creation(
        self,
        task_id: str,
        title: str,
        description: str,
        commit_hash: str,
        commit_date: datetime,
        file_path: str,
        priority: Optional[int] = None,
        tags: List[str] = None,
        code_files: List[str] = None  # NEW: Code files changed in same commit
    ) -> None:
        """
        Record a new task being created

        Args:
            task_id: Task identifier
            title: Task title
            description: Task description
            commit_hash: Commit where task was created
            commit_date: Date of commit
            file_path: Markdown file where task is defined
            priority: Task priority (1-5)
            tags: List of tags
            code_files: Code files changed in same commit (Layer 4 enhancement)
        """
        if task_id not in self.tasks:
            # Start with markdown file, add code files
            file_locations = [file_path]
            if code_files:
                file_locations.extend(code_files)

            self.tasks[task_id] = TaskState(
                task_id=task_id,
                title=title,
                description=description,
                priority=priority,
                first_seen_commit=commit_hash,
                last_updated_commit=commit_hash,
                first_seen_date=commit_date,
                last_updated_date=commit_date,
                file_locations=file_locations,
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
            }, commit_hash, commit_date, file_path, code_files=code_files)

    def process_task_update(
        self,
        task_id: str,
        updates: Dict[str, Any],
        commit_hash: str,
        commit_date: datetime,
        file_path: Optional[str] = None,
        code_files: List[str] = None  # NEW: Code files from commit
    ) -> None:
        """
        Record task being modified

        Args:
            task_id: Task identifier
            updates: Dictionary of field updates
            commit_hash: Commit hash
            commit_date: Commit date
            file_path: Markdown file path
            code_files: Code files changed in same commit (Layer 4 enhancement)
        """
        if task_id not in self.tasks:
            logger.warning(f"Attempted to update non-existent task: {task_id}")
            return

        task = self.tasks[task_id]
        task.last_updated_commit = commit_hash
        task.last_updated_date = commit_date

        # Add markdown file if new
        if file_path and file_path not in task.file_locations:
            task.file_locations.append(file_path)

        # NEW: Add code files if provided (Layer 4 enhancement)
        if code_files:
            for code_file in code_files:
                if code_file not in task.file_locations:
                    task.file_locations.append(code_file)

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
        files_changed: int = 0,
        changed_file_paths: List[str] = None  # NEW: Actual file paths changed
    ) -> None:
        """
        Record a commit referencing a task

        Args:
            task_id: Task identifier
            commit_hash: Commit that references the task
            commit_date: Date of commit
            commit_intent: Intent classification (task_start, task_progress, task_complete, etc.)
            files_changed: Number of files changed (deprecated - use len(changed_file_paths))
            changed_file_paths: List of actual file paths changed in this commit
        """
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

        # NEW: Add changed files to task's file_locations (Layer 4 enhancement)
        if changed_file_paths:
            for file_path in changed_file_paths:
                if file_path not in task.file_locations:
                    task.file_locations.append(file_path)
                    logger.debug(f"  Added file to task {task_id}: {file_path}")

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

    def process_section_creation(
        self,
        section: MDSection,
        commit_hash: str,
        commit_date: datetime
    ) -> None:
        """Record a new section being created"""
        if section.section_id not in self.sections:
            self.sections[section.section_id] = section
            logger.debug(f"Created section {section.section_id}: {section.title}")

    def add_section_dependency(self, from_section_id: str, to_section_id: str) -> None:
        """Add a dependency between sections (from blocks to)"""
        dependency = (from_section_id, to_section_id)
        if dependency not in self.section_dependencies:
            self.section_dependencies.append(dependency)
            logger.debug(f"Added section dependency: {from_section_id} blocks {to_section_id}")


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
        self.hierarchy_parser = MarkdownHierarchyParser()  # Hierarchy support
        self.phase_analyzer = PhaseOrderAnalyzer()  # Phase dependencies
        self.commit_analyzer = CommitMessageAnalyzer()
        self.temporal_analyzer = TemporalClusterAnalyzer(burst_threshold=5)  # Layer 1: Temporal clustering
        self.body_enricher = CommitBodyEnricher()  # Layer 2: Commit body enrichment
        self.ref_linker = TaskReferenceLinker(self.commit_analyzer)  # Layer 3: Task reference linking
        self.priority_calculator = TaskPriorityCalculator()
        self.file_analyzer = FileCoChangeAnalyzer(co_change_threshold=3)  # Layer 4: File co-changes

        self.commits_processed = 0
        self.md_files_processed: Set[str] = set()

        # Layer 1: Temporal clustering results
        self.temporal_epics: List[TemporalEpic] = []
        self.task_sprint_map: Dict[str, str] = {}
        # Layer 2: Commit body enrichment results
        self.tasks_enriched_count: int = 0
        # Layer 3: Task reference linking results
        self.commit_task_links: List[CommitTaskLink] = []
        # Layer 4: File co-change analysis results
        self.co_change_result: Optional[CoChangeResult] = None
        self.file_dependencies: List[Tuple[str, str, str]] = []

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

        # ===== LAYER 1 PREPROCESSING: Temporal Pattern Analysis =====
        logger.info("Layer 1: Detecting temporal patterns and sprint periods...")
        self.temporal_epics = self.temporal_analyzer.analyze_temporal_patterns(commits)

        # ===== LAYER 4 PREPROCESSING: File Co-Change Analysis =====
        logger.info("Layer 4: Analyzing file co-change patterns...")
        self.co_change_result = self.file_analyzer.analyze_commits(commits)
        logger.info(f"  Found {len(self.co_change_result.clusters)} file clusters")

        # ===== MAIN PROCESSING: Extract tasks from commits and markdown =====
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

        # ===== LAYER 3 POST-PROCESSING: Link commits to tasks =====
        logger.info("Layer 3: Linking commits to tasks via references and file overlap...")
        self.commit_task_links = self.ref_linker.link_commits_to_tasks(
            commits=commits,
            tasks=self.task_tracker.tasks,
            sections=self.task_tracker.sections
        )
        logger.info(f"  Created {len(self.commit_task_links)} commit-task links")

        # ===== LAYER 2 POST-PROCESSING: Enrich tasks with commit body content =====
        logger.info("Layer 2: Enriching task descriptions with commit body content...")
        # Build commits index for quick lookup
        commits_by_hash = {commit.hash: commit for commit in commits}
        self.tasks_enriched_count = self.body_enricher.enrich_tasks_from_links(
            tasks=self.task_tracker.tasks,
            commits_by_hash=commits_by_hash,
            commit_task_links=self.commit_task_links,
            min_confidence=0.2  # Include all links: explicit (1.0), phase (0.9), explicit-match (0.8), and all inferred (0.2+)
        )
        logger.info(f"  Enriched {self.tasks_enriched_count} task descriptions with commit bodies")

        # ===== LAYER 1 POST-PROCESSING: Link tasks to sprints =====
        logger.info("Layer 1: Linking tasks to sprint periods...")
        self.task_sprint_map = self.temporal_analyzer.link_tasks_to_sprints(
            tasks=self.task_tracker.tasks,
            sprints=self.temporal_epics
        )
        logger.info(f"  Linked {len(self.task_sprint_map)} tasks to {len(self.temporal_epics)} sprints")

        # ===== LAYER 4 POST-PROCESSING: Create file-based dependencies =====
        logger.info("Layer 4: Creating file-based dependencies...")
        self.file_dependencies = self.file_analyzer.create_dependencies(
            tasks=self.task_tracker.tasks,
            threshold=3
        )

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
                files_changed=len(commit.files_changed),
                changed_file_paths=commit.files_changed  # NEW: Pass actual file paths for Layer 4
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

            # NEW (Layer 4): Extract code files changed in this commit
            # Associate tasks in this markdown file with code files changed alongside it
            code_files = [f for f in commit.files_changed if not f.endswith('.md')]

            # Parse with hierarchy awareness
            sections, md_tasks_with_parents = self.hierarchy_parser.parse_file_with_hierarchy(
                content=snapshot.content,
                file_path=file_path,
                commit_hash=commit.hash
            )

            # Process sections first
            for section in sections:
                self.task_tracker.process_section_creation(
                    section=section,
                    commit_hash=commit.hash,
                    commit_date=commit.timestamp
                )

            # Analyze phase dependencies
            if sections:
                phase_deps = self.phase_analyzer.analyze_phase_order(sections)
                for from_section_id, to_section_id in phase_deps:
                    self.task_tracker.add_section_dependency(from_section_id, to_section_id)

            # Process tasks with parent relationships
            for md_task in md_tasks_with_parents:
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
                            tags=md_task.tags,
                            code_files=code_files  # NEW: Associate code files
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
                        tags=md_task.tags,
                        code_files=code_files  # NEW: Associate code files
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
            recommendations=recommendations,
            sections=list(self.task_tracker.sections.values()),  # Hierarchical sections
            temporal_epics=self.temporal_epics,  # Layer 1
            task_sprint_map=self.task_sprint_map,  # Layer 1
            commit_task_links=self.commit_task_links,  # Layer 3
            file_clusters=self.co_change_result.clusters if self.co_change_result else [],  # Layer 4
            file_dependencies=self.file_dependencies  # Layer 4
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
