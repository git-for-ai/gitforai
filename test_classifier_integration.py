#!/usr/bin/env python3
"""
Test the classifier integration in replay.py

This script validates that the document classifier is working correctly
within the MarkdownHierarchyParser to reduce over-extraction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from gitforai.integrations.replay import MarkdownHierarchyParser
from gitforai.integrations.document_classifier import DocumentType


def test_single_file(file_path: Path):
    """Test classification and extraction on a single file"""
    parser = MarkdownHierarchyParser()

    try:
        content = file_path.read_text(encoding='utf-8')

        # Parse with classifier
        sections, tasks = parser.parse_file_with_hierarchy(
            content=content,
            file_path=str(file_path),
            commit_hash="test-commit"
        )

        return {
            'file': file_path.name,
            'sections': len(sections),
            'tasks': len(tasks),
            'total_items': len(sections) + len(tasks)
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def test_repository(repo_path: Path):
    """Test classifier integration on all markdown files in a repository"""
    md_files = list(repo_path.glob("*.md"))

    print(f"\n{'='*80}")
    print(f"Testing Classifier Integration on: {repo_path.name}")
    print(f"{'='*80}\n")

    print(f"Found {len(md_files)} markdown files\n")

    results = []
    total_sections = 0
    total_tasks = 0

    for md_file in sorted(md_files):
        result = test_single_file(md_file)
        if result:
            results.append(result)
            total_sections += result['sections']
            total_tasks += result['tasks']

            # Print file-level results
            print(f"{result['file']:<50} {result['sections']:3d}S + {result['tasks']:3d}T = {result['total_items']:3d} items")

    print(f"\n{'-'*80}")
    print(f"TOTAL EXTRACTION:")
    print(f"  Sections: {total_sections}")
    print(f"  Tasks:    {total_tasks}")
    print(f"  TOTAL:    {total_sections + total_tasks}")
    print(f"{'-'*80}\n")

    return total_sections + total_tasks


def main():
    """Run tests on tas-sdk"""
    tas_sdk_path = Path('/home/bigale/repos/tas-sdk')

    if not tas_sdk_path.exists():
        print(f"Error: {tas_sdk_path} not found")
        sys.exit(1)

    print("Testing Document Classifier Integration in MarkdownHierarchyParser")
    print("=" * 80)
    print()
    print("This test validates that:")
    print("  1. SKIP files return 0 items")
    print("  2. HIGH_VALUE files extract ## headings + tasks from high-value sections")
    print("  3. MEDIUM_VALUE files extract only ## headings (no tasks)")
    print("  4. LOW_VALUE files extract only TODO sections")
    print()

    total_items = test_repository(tas_sdk_path)

    # Expected reduction
    original_extraction = 2696  # From validation results
    expected_extraction = 478   # From validation results

    print(f"\n{'='*80}")
    print("COMPARISON TO BASELINE:")
    print(f"{'='*80}\n")
    print(f"Original extraction (no filtering):  {original_extraction} items")
    print(f"Expected extraction (with filtering): {expected_extraction} items")
    print(f"Actual extraction (this test):        {total_items} items")
    print()

    if total_items <= expected_extraction * 1.1:  # Allow 10% tolerance
        print("✅ SUCCESS: Extraction is within expected range!")
        print(f"   Reduction: {(1 - total_items/original_extraction)*100:.1f}%")
    else:
        print("⚠️  WARNING: Extraction is higher than expected")
        print(f"   Expected: {expected_extraction} items")
        print(f"   Actual: {total_items} items")
        print(f"   Difference: {total_items - expected_extraction} items")


if __name__ == '__main__':
    main()
