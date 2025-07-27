"""
Integration tests for the complete HTML RAG Pipeline.

These tests verify the entire pipeline workflow from HTML input to search results.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from html_rag.core.pipeline import create_pipeline
from html_rag.core.config import PipelineConfig, WaybackConfig
from html_rag.utils.metrics import track_processing
from tests.conftest import assert_processing_success, assert_search_results_valid, generate_test_html_documents


@pytest.mark.integration
class TestFullPipelineWorkflow:
    """Test complete pipeline workflows."""
    
    def test_basic_workflow(self, temp_dir):
        """Test basic HTML processing to search workflow."""
        # Setup pipeline
        config = PipelineConfig(
            collection_name="integration_test",
            persist_directory=str(temp_dir / "test_db"),
            prefer_basic_cleaning=True
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Step 1: Process HTML content
            html_content = """<!DOCTYPE html>
            <html>
            <head><title>Integration Test Document</title></head>
            <body>
                <h1>Machine Learning in Healthcare</h1>
                <p>Machine learning applications in healthcare are transforming patient care.</p>
                <h2>Applications</h2>
                <ul>
                    <li>Medical image analysis</li>
                    <li>Drug discovery</li>
                    <li>Predictive analytics</li>
                </ul>
                <h3>Future Prospects</h3>
                <p>AI-driven healthcare solutions will continue to evolve and improve outcomes.</p>
            </body>
            </html>"""
            
            result = pipeline.process_html(html_content, url="https://example.com/ml-healthcare")
            assert_processing_success(result)
            
            # Step 2: Verify content is searchable
            search_results = pipeline.search("machine learning healthcare", n_results=5)
            assert_search_results_valid(search_results)
            assert len(search_results) > 0
            
            # Step 3: Verify specific searches work
            specific_searches = [
                "medical image analysis",
                "drug discovery",
                "predictive analytics",
                "AI healthcare"
            ]
            
            for query in specific_searches:
                results = pipeline.search(query, n_results=3)
                assert_search_results_valid(results)
                # At least some searches should return results
            
            # Step 4: Test semantic similarity
            semantic_results = pipeline.search("artificial intelligence in medicine", n_results=3)
            assert_search_results_valid(semantic_results)
            
        finally:
            pipeline.cleanup()
    
    def test_batch_processing_workflow(self, temp_dir):
        """Test batch processing workflow."""
        config = PipelineConfig(
            collection_name="batch_integration_test",
            persist_directory=str(temp_dir / "batch_db"),
            prefer_basic_cleaning=True
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Generate test documents
            documents = generate_test_html_documents(count=5)
            
            # Process batch
            results = pipeline.process_multiple_html(documents)
            
            # Verify batch results
            assert len(results) == 5
            successful = [r for r in results if r.get('success', False)]
            assert len(successful) >= 4  # At least 80% success rate
            
            # Test search across all documents
            search_queries = ['technology', 'science', 'business', 'education', 'health']
            
            for query in search_queries:
                results = pipeline.search(query, n_results=10)
                assert_search_results_valid(results)
                # Should find relevant content for each topic
            
            # Test cross-document search
            cross_results = pipeline.search("article information", n_results=10)
            assert_search_results_valid(cross_results)
            assert len(cross_results) > 0  # Should match multiple documents
            
        finally:
            pipeline.cleanup()
    
    @pytest.mark.ukrainian
    def test_ukrainian_content_workflow(self, temp_dir):
        """Test Ukrainian content processing workflow."""
        config = PipelineConfig(
            collection_name="ukrainian_integration_test",
            persist_directory=str(temp_dir / "ukrainian_db"),
            prefer_basic_cleaning=True,
            cyrillic_detection_threshold=0.1
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Ukrainian content
            ukrainian_content = """<!DOCTYPE html>
            <html>
            <head><title>Українська політика</title></head>
            <body>
                <h1>Політична партія "Слуга народу"</h1>
                <p>Партія працює для розвитку України та добробуту українського народу.</p>
                <h2>Основні цінності</h2>
                <ul>
                    <li>Демократія та прозорість</li>
                    <li>Європейська інтеграція</li>
                    <li>Економічний розвиток</li>
                    <li>Соціальна справедливість</li>
                </ul>
                <h3>Програма реформ</h3>
                <p>Партія підтримує комплексні реформи в усіх сферах життя країни.</p>
                <h2>Президент і команда</h2>
                <p>Команда президента Зеленського працює над впровадженням реформ.</p>
            </body>
            </html>"""
            
            # Process Ukrainian content
            result = pipeline.process_html(ukrainian_content, url="https://sluga-narodu.com")
            assert_processing_success(result)
            
            # Test Ukrainian searches
            ukrainian_searches = [
                "партія",
                "політична партія",
                "Слуга народу",
                "президент Зеленський",
                "українського народу",
                "демократія",
                "реформи",
                "економічний розвиток"
            ]
            
            successful_searches = 0
            for query in ukrainian_searches:
                results = pipeline.search(query, n_results=5)
                assert_search_results_valid(results)
                
                if results and len(results) > 0:
                    successful_searches += 1
                    # Check that top result has good similarity
                    best_score = results[0].get('similarity_score', 0)
                    print(f"Ukrainian search '{query}': {len(results)} results, best score: {best_score:.3f}")
            
            # Should find most Ukrainian terms
            success_rate = successful_searches / len(ukrainian_searches)
            assert success_rate >= 0.7, f"Ukrainian search success rate too low: {success_rate:.1%}"
            
            # Test semantic Ukrainian searches
            semantic_ukrainian = [
                "українська демократія",
                "політичні реформи",
                "європейська інтеграція"
            ]
            
            for query in semantic_ukrainian:
                results = pipeline.search(query, n_results=3)
                assert_search_results_valid(results)
            
        finally:
            pipeline.cleanup()
    
    @pytest.mark.wayback
    def test_wayback_processing_workflow(self, temp_dir):
        """Test complete Wayback Machine processing workflow."""
        # Create wayback directory structure
        wayback_dir = temp_dir / "wayback_integration"
        wayback_dir.mkdir()
        
        # Create sample snapshots
        snapshots = [
            {
                'dir': 'snapshot_001',
                'html': """<!DOCTYPE html>
                <html>
                <head><title>Політична партія 2020</title></head>
                <body>
                    <h1>Слуга народу - 2020</h1>
                    <p>Партія продовжує роботу для українського народу.</p>
                    <h2>Досягнення року</h2>
                    <ul>
                        <li>Антикорупційні реформи</li>
                        <li>Цифровізація послуг</li>
                        <li>Підтримка підприємництва</li>
                    </ul>
                </body>
                </html>""",
                'metadata': {
                    'timestamp': '20201130051600',
                    'domain': 'sluga-narodu.com',
                    'original_url': 'https://sluga-narodu.com/',
                    'title': 'Слуга народу - 2020'
                }
            },
            {
                'dir': 'snapshot_002',
                'html': """<!DOCTYPE html>
                <html>
                <head><title>Партійні новини 2021</title></head>
                <body>
                    <h1>Новини партії - 2021</h1>
                    <p>Важливі події та рішення політичної партії "Слуга народу".</p>
                    <h2>Ключові ініціативи</h2>
                    <ul>
                        <li>Зелена енергетика</li>
                        <li>Освітні реформи</li>
                        <li>Медична реформа</li>
                    </ul>
                    <h3>Президент і команда</h3>
                    <p>Президент Зеленський та команда працюють над новими проектами.</p>
                </body>
                </html>""",
                'metadata': {
                    'timestamp': '20210615143000',
                    'domain': 'sluga-narodu.com',
                    'original_url': 'https://sluga-narodu.com/news',
                    'title': 'Партійні новини 2021'
                }
            }
        ]
        
        # Create snapshot files
        for snapshot in snapshots:
            snapshot_dir = wayback_dir / snapshot['dir']
            snapshot_dir.mkdir()
            
            # HTML file
            with open(snapshot_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(snapshot['html'])
            
            # Metadata file
            with open(snapshot_dir / "meta.json", 'w', encoding='utf-8') as f:
                json.dump(snapshot['metadata'], f, indent=2, ensure_ascii=False)
        
        # Setup pipeline for wayback processing
        config = PipelineConfig(
            collection_name="wayback_integration_test",
            persist_directory=str(temp_dir / "wayback_db"),
            prefer_basic_cleaning=True
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Process wayback snapshots
            wayback_config = WaybackConfig(
                require_metadata=True,
                force_basic_cleaning=True
            )
            
            results = pipeline.process_wayback_snapshots(
                str(wayback_dir),
                wayback_config=wayback_config
            )
            
            # Verify processing results
            assert len(results) == 2
            successful = [r for r in results if r.get('success', False)]
            assert len(successful) == 2  # Both should succeed
            
            # Test wayback-specific searches
            wayback_searches = [
                "партія",
                "Слуга народу",
                "президент Зеленський",
                "реформи",
                "українського народу"
            ]
            
            for query in wayback_searches:
                # Regular search
                results = pipeline.search(query, n_results=5)
                assert_search_results_valid(results)
                
                # Wayback-specific search with domain filter
                wayback_results = pipeline.search_wayback_snapshots(
                    query=query,
                    n_results=5,
                    domain_filter="sluga-narodu.com"
                )
                assert_search_results_valid(wayback_results)
            
            # Test temporal searches
            timestamp_results = pipeline.search_wayback_snapshots(
                query="новини",
                n_results=5,
                timestamp_filter="20210615143000"
            )
            assert_search_results_valid(timestamp_results)
            
            # Test timeline analysis
            all_results = pipeline.search("партія", n_results=10)
            wayback_results = [r for r in all_results if 'wayback_timestamp' in r.get('metadata', {})]
            
            # Should have results from different time periods
            timestamps = set()
            for result in wayback_results:
                timestamp = result.get('metadata', {}).get('wayback_timestamp')
                if timestamp:
                    timestamps.add(timestamp[:4])  # Year
            
            assert len(timestamps) >= 2  # Results from different years
            
        finally:
            pipeline.cleanup()
    
    @pytest.mark.slow
    def test_performance_integration(self, temp_dir):
        """Test pipeline performance with realistic load."""
        config = PipelineConfig(
            collection_name="performance_integration_test",
            persist_directory=str(temp_dir / "performance_db"),
            prefer_basic_cleaning=True,
            enable_metrics=True
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Generate larger dataset
            documents = generate_test_html_documents(count=20)
            
            # Process with metrics tracking
            with track_processing(enable_resource_monitoring=True) as metrics:
                results = pipeline.process_multiple_html(documents)
            
            # Verify performance metrics
            assert len(results) == 20
            successful = [r for r in results if r.get('success', False)]
            success_rate = len(successful) / len(results)
            assert success_rate >= 0.8  # At least 80% success rate
            
            # Check metrics
            if metrics:
                metrics_data = metrics.get_metrics_dict()
                assert metrics_data['documents']['processed'] == 20
                assert metrics_data['documents']['success_rate'] >= 0.8
                assert metrics_data['processing']['total_duration'] > 0
            
            # Test search performance
            search_queries = [
                'technology', 'science', 'business', 'education', 'health',
                'artificial intelligence', 'machine learning', 'data science',
                'research', 'development'
            ]
            
            search_results = {}
            for query in search_queries:
                results = pipeline.search(query, n_results=5)
                search_results[query] = results
                assert_search_results_valid(results)
            
            # Verify search coverage
            total_searches = len(search_queries)
            successful_searches = sum(1 for results in search_results.values() if results)
            search_success_rate = successful_searches / total_searches
            assert search_success_rate >= 0.5  # At least 50% of searches should return results
            
        finally:
            pipeline.cleanup()
    
    def test_error_recovery_workflow(self, temp_dir):
        """Test pipeline error recovery and resilience."""
        config = PipelineConfig(
            collection_name="error_recovery_test",
            persist_directory=str(temp_dir / "error_db"),
            prefer_basic_cleaning=True
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Mix of good and bad documents
            mixed_documents = [
                # Good document
                {
                    'html': '<html><body><h1>Good Document 1</h1><p>Content</p></body></html>',
                    'url': 'https://good1.com'
                },
                # Empty document (should fail)
                {
                    'html': '',
                    'url': 'https://empty.com'
                },
                # Another good document
                {
                    'html': '<html><body><h1>Good Document 2</h1><p>More content</p></body></html>',
                    'url': 'https://good2.com'
                },
                # Minimal document (might fail or succeed)
                {
                    'html': '<html><body></body></html>',
                    'url': 'https://minimal.com'
                },
                # Good document
                {
                    'html': '<html><body><h1>Good Document 3</h1><p>Final content</p></body></html>',
                    'url': 'https://good3.com'
                }
            ]
            
            # Process mixed batch
            results = pipeline.process_multiple_html(mixed_documents)
            
            # Should handle errors gracefully
            assert len(results) == 5
            
            # Count successes and failures
            successes = [r for r in results if r.get('success', False)]
            failures = [r for r in results if not r.get('success', False)]
            
            # Should have both successes and failures
            assert len(successes) >= 2  # At least some should succeed
            assert len(failures) >= 1   # At least some should fail
            
            # Pipeline should remain functional after errors
            good_result = pipeline.process_html(
                '<html><body><h1>Recovery Test</h1><p>Pipeline still works</p></body></html>',
                url='https://recovery.com'
            )
            assert_processing_success(good_result)
            
            # Search should still work
            search_results = pipeline.search("recovery test", n_results=5)
            assert_search_results_valid(search_results)
            
        finally:
            pipeline.cleanup()
    
    def test_multilingual_workflow(self, temp_dir):
        """Test multilingual content processing workflow."""
        config = PipelineConfig(
            collection_name="multilingual_test",
            persist_directory=str(temp_dir / "multilingual_db"),
            prefer_basic_cleaning=True
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Multilingual documents
            multilingual_docs = [
                # English
                {
                    'html': '''<html><body>
                        <h1>Machine Learning Research</h1>
                        <p>Artificial intelligence and machine learning are transforming technology.</p>
                    </body></html>''',
                    'url': 'https://en.example.com',
                    'language': 'en'
                },
                # Ukrainian
                {
                    'html': '''<html><body>
                        <h1>Дослідження машинного навчання</h1>
                        <p>Штучний інтелект та машинне навчання трансформують технології.</p>
                    </body></html>''',
                    'url': 'https://ua.example.com',
                    'language': 'uk'
                },
                # Mixed content
                {
                    'html': '''<html><body>
                        <h1>Research / Дослідження</h1>
                        <p>International collaboration in AI research. Міжнародна співпраця у дослідженнях ШІ.</p>
                    </body></html>''',
                    'url': 'https://mixed.example.com',
                    'language': 'mixed'
                }
            ]
            
            # Process multilingual content
            for doc in multilingual_docs:
                result = pipeline.process_html(doc['html'], url=doc['url'])
                assert_processing_success(result)
            
            # Test searches in different languages
            test_searches = [
                # English searches
                ('machine learning', 'en'),
                ('artificial intelligence', 'en'),
                ('research', 'en'),
                
                # Ukrainian searches
                ('машинне навчання', 'uk'),
                ('штучний інтелект', 'uk'),
                ('дослідження', 'uk'),
                
                # Cross-lingual semantic searches
                ('AI technology', 'cross'),
                ('технології ШІ', 'cross')
            ]
            
            search_success_count = 0
            for query, lang_type in test_searches:
                results = pipeline.search(query, n_results=5)
                assert_search_results_valid(results)
                
                if results:
                    search_success_count += 1
                    print(f"Multilingual search '{query}' ({lang_type}): {len(results)} results")
            
            # Should successfully search in multiple languages
            search_success_rate = search_success_count / len(test_searches)
            assert search_success_rate >= 0.5, f"Multilingual search success rate too low: {search_success_rate:.1%}"
            
        finally:
            pipeline.cleanup()


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    def test_news_analysis_scenario(self, temp_dir):
        """Test news article analysis scenario."""
        config = PipelineConfig(
            collection_name="news_analysis",
            persist_directory=str(temp_dir / "news_db")
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Simulate news articles
            news_articles = [
                {
                    'html': '''<html><body>
                        <h1>Tech Company Reports Strong Q4 Earnings</h1>
                        <p>Major technology company announced record quarterly earnings driven by AI product sales.</p>
                        <h2>Key Highlights</h2>
                        <ul>
                            <li>Revenue increased 25% year-over-year</li>
                            <li>AI division contributed 30% of total revenue</li>
                            <li>Strong demand for machine learning services</li>
                        </ul>
                    </body></html>''',
                    'url': 'https://news.com/tech-earnings'
                },
                {
                    'html': '''<html><body>
                        <h1>New AI Regulations Proposed by Government</h1>
                        <p>Government officials unveiled new regulatory framework for artificial intelligence applications.</p>
                        <h2>Proposed Rules</h2>
                        <ul>
                            <li>Mandatory AI safety assessments</li>
                            <li>Transparency requirements for AI systems</li>
                            <li>Data privacy protections</li>
                        </ul>
                    </body></html>''',
                    'url': 'https://news.com/ai-regulations'
                }
            ]
            
            # Process articles
            for article in news_articles:
                result = pipeline.process_html(article['html'], url=article['url'])
                assert_processing_success(result)
            
            # Perform analysis queries
            analysis_queries = [
                'technology earnings revenue',
                'AI artificial intelligence',
                'government regulations',
                'machine learning services',
                'data privacy'
            ]
            
            for query in analysis_queries:
                results = pipeline.search(query, n_results=3)
                assert_search_results_valid(results)
            
        finally:
            pipeline.cleanup()
    
    def test_research_database_scenario(self, temp_dir):
        """Test academic research database scenario."""
        config = PipelineConfig(
            collection_name="research_database",
            persist_directory=str(temp_dir / "research_db")
        )
        pipeline = create_pipeline(config=config)
        
        try:
            # Research papers
            papers = generate_test_html_documents(count=10)
            
            # Process research collection
            results = pipeline.process_multiple_html(papers)
            successful = [r for r in results if r.get('success', False)]
            assert len(successful) >= 8  # Most should process successfully
            
            # Research-style queries
            research_queries = [
                'methodology approach',
                'experimental results',
                'statistical analysis',
                'literature review',
                'future work'
            ]
            
            for query in research_queries:
                results = pipeline.search(query, n_results=5)
                assert_search_results_valid(results)
            
        finally:
            pipeline.cleanup()