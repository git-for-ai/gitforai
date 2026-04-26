"""
Document Classifier for Markdown Files

Categorizes markdown files by their extraction value for task intelligence.
Designed to reduce over-extraction by filtering out low-value documents.
"""

import os
import re
from enum import Enum
from typing import Tuple
from dataclasses import dataclass


class DocumentType(Enum):
    """Document classification for extraction value"""
    SKIP = 0          # Don't extract anything (agent instructions, meta-docs)
    LOW_VALUE = 1     # Extract minimally (README, specs, analysis)
    MEDIUM_VALUE = 2  # Extract conservatively (design docs, research)
    HIGH_VALUE = 3    # Extract comprehensively (completions, implementation plans)


@dataclass
class DocumentClassification:
    """Result of document classification"""
    doc_type: DocumentType
    confidence: float  # 0.0 to 1.0
    reason: str
    suggested_layers: list[int]  # Which extraction layers to apply


class MarkdownDocumentClassifier:
    """
    Classifies markdown documents by analyzing filename and content structure
    to determine extraction value for task intelligence.
    """

    # Filename patterns for each category
    SKIP_PATTERNS = [
        r'@?agents?\.md$',
        r'claude\.md$',
        r'cursor\.md$',
        r'copilot\.md$',
    ]

    HIGH_VALUE_PATTERNS = [
        r'task[_-]\d+',         # TASK_2.1_COMPLETION.md
        r'complete',            # SCHEMA-ANALYZER-COMPLETE.md
        r'implementation[_-]plan',  # IMPLEMENTATION-PLAN.md
        r'integration[_-]plan',
        r'phase\d+',            # PHASE2-COMPLETE.md
    ]

    MEDIUM_VALUE_PATTERNS = [
        r'design',              # ARCHITECTURE-DESIGN-REVIEW.md
        r'architecture',
        r'integration[^-]',     # DRAKON-INTEGRATION-DESIGN.md (not -plan)
        r'strategy',
    ]

    LOW_VALUE_PATTERNS = [
        r'readme',
        r'spec',
        r'analysis',
        r'research',
        r'summary',
        r'session',
        r'morning',
        r'night',
        r'update',
    ]

    # Content patterns for HIGH_VALUE documents
    COMPLETION_MARKERS = [
        r'\*\*Status:\*\*\s*[✅✓]\s*COMPLETE',
        r'Status:\s*COMPLETE',
        r'##\s*Deliverables',
        r'##\s*Completed\s+Components',
    ]

    IMPLEMENTATION_MARKERS = [
        r'##\s*Phase\s+\d+',
        r'##\s*Step\s+\d+',
        r'##\s*Stage\s+\d+',
        r'##\s*Sprint\s+\d+',
    ]

    # Content patterns for sections to SKIP
    SKIP_SECTIONS = [
        'installation',
        'setup',
        'usage',
        'api reference',
        'examples',
        'code samples',
        'technical details',
        'troubleshooting',
    ]

    def classify(self, file_path: str, content: str) -> DocumentClassification:
        """
        Classify a markdown document

        Args:
            file_path: Path to the markdown file
            content: File content

        Returns:
            DocumentClassification with type, confidence, and reasoning
        """
        filename = os.path.basename(file_path).lower()
        first_lines = '\n'.join(content.split('\n')[:50])

        # Check SKIP patterns first
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return DocumentClassification(
                    doc_type=DocumentType.SKIP,
                    confidence=1.0,
                    reason=f"Agent instructions file: {pattern}",
                    suggested_layers=[]
                )

        # Check HIGH_VALUE patterns
        for pattern in self.HIGH_VALUE_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                # Verify with content markers
                has_completion = any(
                    re.search(marker, first_lines, re.IGNORECASE)
                    for marker in self.COMPLETION_MARKERS
                )
                has_implementation = any(
                    re.search(marker, content, re.IGNORECASE)
                    for marker in self.IMPLEMENTATION_MARKERS
                )

                if has_completion or has_implementation:
                    return DocumentClassification(
                        doc_type=DocumentType.HIGH_VALUE,
                        confidence=0.9,
                        reason=f"Completion report or implementation plan: {pattern}",
                        suggested_layers=[1, 3]  # Layer 1 (temporal) and Layer 3 (tasks)
                    )

        # Check LOW_VALUE patterns
        for pattern in self.LOW_VALUE_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                # Check if it has any high-value sections anyway
                has_todo = re.search(r'##\s*(TODO|Next Steps|Action Items)', content, re.IGNORECASE)
                if has_todo:
                    return DocumentClassification(
                        doc_type=DocumentType.LOW_VALUE,
                        confidence=0.7,
                        reason=f"Low-value doc but has TODO section: {pattern}",
                        suggested_layers=[3]  # Only Layer 3 (extract TODO items)
                    )
                return DocumentClassification(
                    doc_type=DocumentType.LOW_VALUE,
                    confidence=0.8,
                    reason=f"Reference documentation: {pattern}",
                    suggested_layers=[]
                )

        # Check MEDIUM_VALUE patterns
        for pattern in self.MEDIUM_VALUE_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return DocumentClassification(
                    doc_type=DocumentType.MEDIUM_VALUE,
                    confidence=0.6,
                    reason=f"Design or architecture document: {pattern}",
                    suggested_layers=[1]  # Only Layer 1 (high-level epics)
                )

        # Default: MEDIUM_VALUE with low confidence
        return DocumentClassification(
            doc_type=DocumentType.MEDIUM_VALUE,
            confidence=0.4,
            reason="Unclassified markdown file (default to medium value)",
            suggested_layers=[3]  # Layer 3 with conservative extraction
        )

    def should_extract_from_section(self, section_title: str, doc_type: DocumentType) -> bool:
        """
        Determine if a section should be extracted based on title and document type

        Args:
            section_title: Section heading text
            doc_type: Document classification

        Returns:
            True if section should be extracted
        """
        section_lower = section_title.lower()

        # Always skip these sections regardless of document type
        if any(skip in section_lower for skip in self.SKIP_SECTIONS):
            return False

        # For HIGH_VALUE docs, extract from most sections except skips
        if doc_type == DocumentType.HIGH_VALUE:
            return True

        # For MEDIUM/LOW_VALUE, only extract from specific sections
        high_value_sections = [
            'deliverables',
            'components',
            'next steps',
            'todo',
            'action items',
            'tasks',
            'backlog',
            'phase',
            'step',
            'sprint',
        ]

        return any(hvs in section_lower for hvs in high_value_sections)


# Example usage
if __name__ == '__main__':
    classifier = MarkdownDocumentClassifier()

    # Test cases
    test_files = [
        ('TASK_2.2_COMPLETION.md', '# Task 2.2\n**Status:** ✅ COMPLETE\n## Deliverables\n- ✅ Tool A'),
        ('AGENTS.md', '# Agent Instructions\n## Quick Reference\nbd ready'),
        ('README.md', '# Project\n## Installation\npip install'),
        ('DRAKON-AI-INTEGRATION-DESIGN.md', '# Design\n## Architecture\n## Integration'),
        ('SESSION-SUMMARY.md', '# Summary\n## What We Did\n## Next Steps\n- [ ] Task 1'),
    ]

    for filename, content in test_files:
        result = classifier.classify(filename, content)
        print(f"\n{filename}:")
        print(f"  Type: {result.doc_type.name}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Reason: {result.reason}")
        print(f"  Suggested Layers: {result.suggested_layers}")
