"""
Pytest configuration and shared fixtures for HTML RAG Pipeline tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from html_rag.core.pipeline import RAGPipeline, create_pipeline
from html_rag.core.config import PipelineConfig
from html_rag.utils.logging import setup_logging


# Setup test logging
setup_logging(level="WARNING", enable_console=False)


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    return PipelineConfig(
        collection_name="test_collection",
        persist_directory="./test_chroma_db",
        prefer_basic_cleaning=True,  # Faster for tests
        enable_metrics=False,  # Disable metrics for faster tests
        log_level="WARNING"
    )


@pytest.fixture
def pipeline(test_config):
    """Create a test pipeline instance."""
    pipeline = create_pipeline(config=test_config)
    yield pipeline
    # Cleanup after test
    try:
        pipeline.cleanup()
    except Exception:
        pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """<!DOCTYPE html>
    <html>
    <head>
        <title>Test Document</title>
    </head>
    <body>
        <h1>Test Heading</h1>
        <p>This is a test paragraph with some content.</p>
        <h2>Section 2</h2>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>
        <h3>Subsection</h3>
        <p>Another paragraph with different content for testing.</p>
    </body>
    </html>"""


@pytest.fixture
def ukrainian_html():
    """Ukrainian HTML content for testing."""
    return """<!DOCTYPE html>
    <html>
    <head>
        <title>"5AB C:@0W=AL:>3> :>=B5=BC</title>
    </head>
    <body>
        <h1>@57845=B 5;5=AL:89</h1>
        <p>>;VB8G=0 ?0@BVO "!;C30 =0@>4C" ?@54AB02;OT V=B5@5A8 C:@0W=AL:>3> =0@>4C.</p>
        <h2>5<>:@0BVO</h2>
        <ul>
            <li>570;56=VABL</li>
            <li> >728B>:</li>
            <li> 5D>@<8</li>
        </ul>
        <p>8 ?@0FNT<> 4;O @>728B:C #:@0W=8.</p>
    </body>
    </html>"""


@pytest.fixture
def large_html():
    """Large HTML content for performance testing."""
    content = """<!DOCTYPE html>
    <html>
    <head>
        <title>Large Test Document</title>
    </head>
    <body>
        <h1>Large Document for Testing</h1>
        <p>This document contains a lot of content for performance testing.</p>
    """
    
    # Add many sections
    for i in range(50):
        content += f"""
        <h2>Section {i+1}</h2>
        <p>This is section {i+1} with detailed content for testing performance and scalability of the HTML RAG pipeline.</p>
        <ul>
            <li>Item {i+1}.1 with detailed description</li>
            <li>Item {i+1}.2 with comprehensive information</li>
            <li>Item {i+1}.3 with extensive details</li>
        </ul>
        """
    
    content += """
    </body>
    </html>"""
    
    return content


@pytest.fixture
def malformed_html():
    """Malformed HTML for error testing."""
    return """<!DOCTYPE html>
    <html>
    <head>
        <title>Malformed Document
    </head>
    <body>
        <h1>Unclosed heading
        <p>Paragraph without closing tag
        <ul>
            <li>Item 1
            <li>Item 2</li>
        </ul>
        <div>Unclosed div
        Random text without tags
    </body>
    """


@pytest.fixture
def empty_html():
    """Empty HTML for edge case testing."""
    return """<!DOCTYPE html>
    <html>
    <head>
        <title></title>
    </head>
    <body>
    </body>
    </html>"""


@pytest.fixture
def wayback_metadata():
    """Sample Wayback metadata."""
    return {
        'timestamp': '20201130051600',
        'domain': 'example.com',
        'original_url': 'https://example.com/test',
        'archive_url': 'https://web.archive.org/web/20201130051600/https://example.com/test',
        'title': 'Test Page'
    }


@pytest.fixture
def sample_documents():
    """Sample documents for batch testing."""
    return [
        {
            'html': """<!DOCTYPE html>
            <html><head><title>Doc 1</title></head>
            <body><h1>Document 1</h1><p>Content 1</p></body></html>""",
            'url': 'https://example.com/doc1'
        },
        {
            'html': """<!DOCTYPE html>
            <html><head><title>Doc 2</title></head>
            <body><h1>Document 2</h1><p>Content 2</p></body></html>""",
            'url': 'https://example.com/doc2'
        },
        {
            'html': """<!DOCTYPE html>
            <html><head><title>Doc 3</title></head>
            <body><h1>Document 3</h1><p>Content 3</p></body></html>""",
            'url': 'https://example.com/doc3'
        }
    ]


@pytest.fixture
def wayback_directory(temp_dir):
    """Create a sample Wayback directory structure."""
    wayback_dir = temp_dir / "wayback_test"
    wayback_dir.mkdir()
    
    # Create sample snapshots
    snapshots = [
        {
            'dir': 'snapshot_001',
            'html': """<!DOCTYPE html>
            <html><head><title>Snapshot 1</title></head>
            <body><h1>First Snapshot</h1><p>Content from first snapshot.</p></body></html>""",
            'metadata': {
                'timestamp': '20200101120000',
                'domain': 'test.com',
                'original_url': 'https://test.com/page1',
                'title': 'Snapshot 1'
            }
        },
        {
            'dir': 'snapshot_002',
            'html': """<!DOCTYPE html>
            <html><head><title>Snapshot 2</title></head>
            <body><h1>Second Snapshot</h1><p>Content from second snapshot.</p></body></html>""",
            'metadata': {
                'timestamp': '20210601150000',
                'domain': 'test.com',
                'original_url': 'https://test.com/page2',
                'title': 'Snapshot 2'
            }
        }
    ]
    
    for snapshot in snapshots:
        snapshot_dir = wayback_dir / snapshot['dir']
        snapshot_dir.mkdir()
        
        # Create HTML file
        html_file = snapshot_dir / "index.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(snapshot['html'])
        
        # Create metadata file
        meta_file = snapshot_dir / "meta.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot['metadata'], f, indent=2)
    
    return wayback_dir


@pytest.fixture
def search_test_data():
    """Data for search testing."""
    return {
        'queries': [
            'test content',
            'machine learning',
            'artificial intelligence',
            'data science',
            'programming'
        ],
        'expected_results': {
            'test content': {'min_results': 1, 'min_score': 0.5},
            'machine learning': {'min_results': 0, 'min_score': 0.0},
        }
    }


# Test data generators
def generate_test_html_documents(count: int = 10) -> List[Dict[str, Any]]:
    """Generate test HTML documents."""
    documents = []
    
    topics = ['technology', 'science', 'business', 'education', 'health']
    
    for i in range(count):
        topic = topics[i % len(topics)]
        
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>{topic.title()} Article {i+1}</title>
        </head>
        <body>
            <h1>{topic.title()} Article {i+1}</h1>
            <p>This is an article about {topic} with content number {i+1}.</p>
            <h2>Section 1</h2>
            <p>Detailed information about {topic} topics and related concepts.</p>
            <ul>
                <li>Point 1 about {topic}</li>
                <li>Point 2 about {topic}</li>
                <li>Point 3 about {topic}</li>
            </ul>
            <h3>Conclusion</h3>
            <p>Summary of the {topic} article with key takeaways.</p>
        </body>
        </html>"""
        
        documents.append({
            'html': html,
            'url': f'https://example.com/{topic}/{i+1}',
            'topic': topic,
            'id': i+1
        })
    
    return documents


# Utility functions for tests
def assert_processing_success(result: Dict[str, Any]) -> None:
    """Assert that processing was successful."""
    assert result is not None
    assert isinstance(result, dict)
    assert result.get('success') is True
    assert 'embedded_blocks_count' in result
    assert result['embedded_blocks_count'] > 0


def assert_search_results_valid(results: List[Dict[str, Any]]) -> None:
    """Assert that search results are valid."""
    assert isinstance(results, list)
    for result in results:
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'similarity_score' in result
        assert isinstance(result['similarity_score'], (int, float))
        assert -1.0 <= result['similarity_score'] <= 1.0


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "ukrainian: marks tests related to Ukrainian content"
    )
    config.addinivalue_line(
        "markers", "wayback: marks tests related to Wayback Machine processing"
    )


# Cleanup function
def cleanup_test_data():
    """Cleanup test data and databases."""
    test_dirs = [
        "./test_chroma_db",
        "./test_db",
        "./batch_test_db",
        "./perf_test_default_db",
        "./perf_test_optimized_db"
    ]
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            shutil.rmtree(test_path, ignore_errors=True)


# Pytest hooks
def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("\n>ê Starting HTML RAG Pipeline tests...")


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    print(f"\n>ê Test session finished with exit status: {exitstatus}")
    
    # Cleanup test data
    cleanup_test_data()
    print(">ù Test cleanup completed")