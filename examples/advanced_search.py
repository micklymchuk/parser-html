#!/usr/bin/env python3
"""
Advanced search example for HTML RAG Pipeline.

This example demonstrates:
- Complex metadata filtering
- Similarity threshold tuning
- Result ranking and scoring
- Search result analysis
- Export search results
- Semantic similarity exploration
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from html_rag.core.pipeline import create_pipeline
from html_rag.core.config import PipelineConfig
from html_rag.utils.logging import setup_logging, PipelineLogger
from html_rag.exceptions.pipeline_exceptions import PipelineError

# Setup logging
setup_logging(level="INFO", log_file="logs/advanced_search.log")
logger = PipelineLogger("AdvancedSearch")


def create_diverse_content():
    """Create diverse HTML content for search testing."""
    
    content_samples = [
        {
            'html': """<!DOCTYPE html>
            <html><head><title>Machine Learning Basics</title></head>
            <body>
                <h1>Introduction to Machine Learning</h1>
                <p>Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.</p>
                <h2>Types of Machine Learning</h2>
                <ul>
                    <li>Supervised Learning: Uses labeled data</li>
                    <li>Unsupervised Learning: Finds patterns in unlabeled data</li>
                    <li>Reinforcement Learning: Learns through rewards and penalties</li>
                </ul>
                <h3>Applications</h3>
                <p>Machine learning is used in computer vision, natural language processing, and recommendation systems.</p>
            </body></html>""",
            'url': 'https://example.com/ml-basics',
            'metadata': {
                'category': 'technology',
                'difficulty': 'beginner',
                'topic': 'machine_learning',
                'author': 'Dr. Smith',
                'publication_date': '2024-01-15',
                'word_count': 150
            }
        },
        {
            'html': """<!DOCTYPE html>
            <html><head><title>Deep Learning Architecture</title></head>
            <body>
                <h1>Neural Network Architectures</h1>
                <p>Deep learning utilizes artificial neural networks with multiple layers to model complex patterns.</p>
                <h2>Convolutional Neural Networks (CNNs)</h2>
                <p>CNNs are particularly effective for image recognition and computer vision tasks.</p>
                <h2>Recurrent Neural Networks (RNNs)</h2>
                <p>RNNs excel at processing sequential data like text and time series.</p>
                <h2>Transformer Architecture</h2>
                <p>Transformers have revolutionized natural language processing and are the foundation of models like GPT and BERT.</p>
            </body></html>""",
            'url': 'https://example.com/deep-learning',
            'metadata': {
                'category': 'technology',
                'difficulty': 'advanced',
                'topic': 'deep_learning',
                'author': 'Prof. Johnson',
                'publication_date': '2024-02-20',
                'word_count': 120
            }
        },
        {
            'html': """<!DOCTYPE html>
            <html><head><title>Data Science Career Guide</title></head>
            <body>
                <h1>Building a Career in Data Science</h1>
                <p>Data science combines statistics, programming, and domain expertise to extract insights from data.</p>
                <h2>Essential Skills</h2>
                <ul>
                    <li>Programming: Python, R, SQL</li>
                    <li>Statistics and Mathematics</li>
                    <li>Machine Learning Algorithms</li>
                    <li>Data Visualization</li>
                    <li>Communication Skills</li>
                </ul>
                <h2>Career Paths</h2>
                <p>Data scientists can work in various roles including analyst, engineer, and researcher positions.</p>
            </body></html>""",
            'url': 'https://example.com/data-science-career',
            'metadata': {
                'category': 'career',
                'difficulty': 'intermediate',
                'topic': 'data_science',
                'author': 'Career Expert',
                'publication_date': '2024-01-30',
                'word_count': 100
            }
        },
        {
            'html': """<!DOCTYPE html>
            <html><head><title>Climate Change Research</title></head>
            <body>
                <h1>Climate Change and Environmental Science</h1>
                <p>Climate change research involves studying long-term changes in global weather patterns and their impacts.</p>
                <h2>Research Methods</h2>
                <p>Scientists use satellite data, ice core samples, and computer models to study climate trends.</p>
                <h2>Impact Assessment</h2>
                <p>Research focuses on understanding impacts on ecosystems, agriculture, and human societies.</p>
                <h3>Machine Learning in Climate Science</h3>
                <p>AI and machine learning are increasingly used to analyze climate data and improve prediction models.</p>
            </body></html>""",
            'url': 'https://example.com/climate-research',
            'metadata': {
                'category': 'science',
                'difficulty': 'intermediate',
                'topic': 'climate_science',
                'author': 'Dr. Green',
                'publication_date': '2024-03-10',
                'word_count': 130
            }
        },
        {
            'html': """<!DOCTYPE html>
            <html><head><title>Business Intelligence Tools</title></head>
            <body>
                <h1>Modern Business Intelligence Platforms</h1>
                <p>Business intelligence tools help organizations make data-driven decisions through analytics and reporting.</p>
                <h2>Popular BI Tools</h2>
                <ul>
                    <li>Tableau: Visual analytics platform</li>
                    <li>Power BI: Microsoft's business analytics solution</li>
                    <li>Looker: Modern BI and data platform</li>
                </ul>
                <h2>Integration with Machine Learning</h2>
                <p>Modern BI platforms increasingly integrate machine learning capabilities for predictive analytics.</p>
            </body></html>""",
            'url': 'https://example.com/business-intelligence',
            'metadata': {
                'category': 'business',
                'difficulty': 'intermediate',
                'topic': 'business_intelligence',
                'author': 'Business Analyst',
                'publication_date': '2024-02-05',
                'word_count': 90
            }
        }
    ]
    
    return content_samples


def perform_basic_search(pipeline, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Perform basic search and return results with analysis."""
    
    logger.info(f"🔍 Searching for: '{query}'")
    
    try:
        results = pipeline.search(query, n_results=n_results)
        
        if results:
            logger.info(f"✅ Found {len(results)} results")
            
            # Analyze score distribution
            scores = [r.get('similarity_score', 0) for r in results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            logger.info(f"   📊 Score range: {min_score:.3f} - {max_score:.3f} (avg: {avg_score:.3f})")
            
            return results
        else:
            logger.warning(f"❌ No results found for '{query}'")
            return []
            
    except Exception as e:
        logger.error(f"❌ Search failed: {str(e)}")
        return []


def perform_filtered_search(
    pipeline, 
    query: str, 
    metadata_filter: Dict[str, Any],
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """Perform search with metadata filtering."""
    
    filter_desc = ", ".join([f"{k}={v}" for k, v in metadata_filter.items()])
    logger.info(f"🔍 Filtered search: '{query}' | Filters: {filter_desc}")
    
    try:
        results = pipeline.search(
            query=query,
            n_results=n_results,
            metadata_filter=metadata_filter
        )
        
        if results:
            logger.info(f"✅ Found {len(results)} filtered results")
            return results
        else:
            logger.warning(f"❌ No filtered results found")
            return []
            
    except Exception as e:
        logger.error(f"❌ Filtered search failed: {str(e)}")
        return []


def perform_threshold_search(
    pipeline,
    query: str,
    similarity_threshold: float,
    n_results: int = 10
) -> List[Dict[str, Any]]:
    """Perform search with similarity threshold."""
    
    logger.info(f"🔍 Threshold search: '{query}' | Min similarity: {similarity_threshold:.3f}")
    
    try:
        results = pipeline.search(
            query=query,
            n_results=n_results,
            similarity_threshold=similarity_threshold
        )
        
        if results:
            logger.info(f"✅ Found {len(results)} results above threshold")
            return results
        else:
            logger.warning(f"❌ No results above threshold {similarity_threshold:.3f}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Threshold search failed: {str(e)}")
        return []


def analyze_search_results(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Analyze search results in detail."""
    
    if not results:
        return {'query': query, 'result_count': 0}
    
    # Score analysis
    scores = [r.get('similarity_score', 0) for r in results]
    
    # Metadata analysis
    categories = [r.get('metadata', {}).get('category', 'unknown') for r in results]
    difficulties = [r.get('metadata', {}).get('difficulty', 'unknown') for r in results]
    topics = [r.get('metadata', {}).get('topic', 'unknown') for r in results]
    
    # Text length analysis
    text_lengths = [len(r.get('text', '')) for r in results]
    
    analysis = {
        'query': query,
        'result_count': len(results),
        'scores': {
            'min': min(scores),
            'max': max(scores),
            'average': sum(scores) / len(scores),
            'distribution': scores
        },
        'metadata_distribution': {
            'categories': {cat: categories.count(cat) for cat in set(categories)},
            'difficulties': {diff: difficulties.count(diff) for diff in set(difficulties)},
            'topics': {topic: topics.count(topic) for topic in set(topics)}
        },
        'text_stats': {
            'min_length': min(text_lengths),
            'max_length': max(text_lengths),
            'avg_length': sum(text_lengths) / len(text_lengths)
        }
    }
    
    return analysis


def export_search_results(
    results: List[Dict[str, Any]], 
    analysis: Dict[str, Any],
    filename: str
) -> None:
    """Export search results and analysis."""
    
    export_data = {
        'search_analysis': analysis,
        'results': results,
        'export_metadata': {
            'timestamp': str(Path().cwd()),
            'result_count': len(results)
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📁 Search results exported to {filename}")


def main():
    """Advanced search demonstration."""
    
    logger.info("="*60)
    logger.info("HTML RAG Pipeline - Advanced Search Example")
    logger.info("="*60)
    
    try:
        # Example 1: Setup pipeline and load diverse content
        logger.info("\n1. Setting up pipeline and loading diverse content...")
        
        config = PipelineConfig(
            collection_name="advanced_search_collection",
            persist_directory="./advanced_search_db",
            enable_metrics=True
        )
        
        pipeline = create_pipeline(config=config)
        
        # Load diverse content
        content_samples = create_diverse_content()
        
        for sample in content_samples:
            result = pipeline.process_html(
                raw_html=sample['html'],
                url=sample['url']
            )
            
            if result['success']:
                # Add custom metadata to vector store if needed
                logger.debug(f"Processed: {sample['url']}")
            else:
                logger.warning(f"Failed to process: {sample['url']}")
        
        logger.info(f"✅ Loaded {len(content_samples)} diverse documents")
        
        # Example 2: Basic semantic search
        logger.info("\n2. Basic semantic search examples...")
        
        basic_queries = [
            "artificial intelligence and machine learning",
            "neural networks and deep learning",
            "data science programming skills",
            "climate change and environmental research",
            "business analytics and intelligence tools"
        ]
        
        basic_results = {}
        for query in basic_queries:
            results = perform_basic_search(pipeline, query, n_results=3)
            basic_results[query] = results
        
        # Example 3: Metadata filtering
        logger.info("\n3. Search with metadata filtering...")
        
        # Filter by category
        tech_results = perform_filtered_search(
            pipeline,
            "machine learning",
            {'category': 'technology'},
            n_results=5
        )
        
        # Filter by difficulty
        beginner_results = perform_filtered_search(
            pipeline,
            "learning algorithms",
            {'difficulty': 'beginner'},
            n_results=5
        )
        
        # Complex filter
        advanced_tech_results = perform_filtered_search(
            pipeline,
            "neural networks",
            {'category': 'technology', 'difficulty': 'advanced'},
            n_results=5
        )
        
        # Example 4: Similarity threshold experiments
        logger.info("\n4. Similarity threshold experiments...")
        
        query = "machine learning applications"
        
        # Test different thresholds
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.5]
        threshold_results = {}
        
        for threshold in thresholds:
            results = perform_threshold_search(pipeline, query, threshold, n_results=10)
            threshold_results[threshold] = results
            
            if results:
                logger.info(f"   Threshold {threshold:.1f}: {len(results)} results")
            else:
                logger.info(f"   Threshold {threshold:.1f}: No results")
        
        # Example 5: Semantic similarity exploration
        logger.info("\n5. Semantic similarity exploration...")
        
        # Related terms exploration
        similarity_tests = [
            ("machine learning", "artificial intelligence"),
            ("neural networks", "deep learning"),
            ("data science", "business intelligence"),
            ("climate research", "environmental science"),
            ("programming", "software development")
        ]
        
        for term1, term2 in similarity_tests:
            results1 = perform_basic_search(pipeline, term1, n_results=1)
            results2 = perform_basic_search(pipeline, term2, n_results=1)
            
            if results1 and results2:
                score1 = results1[0].get('similarity_score', 0)
                score2 = results2[0].get('similarity_score', 0)
                logger.info(f"   '{term1}' vs '{term2}': {score1:.3f} vs {score2:.3f}")
        
        # Example 6: Advanced result analysis
        logger.info("\n6. Advanced result analysis...")
        
        # Analyze results from different search types
        analyses = {}
        
        # Analyze basic search results
        for query, results in basic_results.items():
            if results:
                analysis = analyze_search_results(results, query)
                analyses[f"basic_{hash(query) % 1000}"] = analysis
                
                logger.info(f"   Query: '{query[:30]}...'")
                logger.info(f"   Results: {analysis['result_count']}")
                logger.info(f"   Avg score: {analysis['scores']['average']:.3f}")
                logger.info(f"   Categories: {list(analysis['metadata_distribution']['categories'].keys())}")
        
        # Example 7: Query expansion and refinement
        logger.info("\n7. Query expansion and refinement...")
        
        base_query = "machine learning"
        
        # Expanded queries
        expanded_queries = [
            "machine learning algorithms",
            "machine learning applications",
            "machine learning and artificial intelligence",
            "supervised machine learning",
            "machine learning in business"
        ]
        
        expansion_results = {}
        for expanded_query in expanded_queries:
            results = perform_basic_search(pipeline, expanded_query, n_results=3)
            expansion_results[expanded_query] = results
            
            if results:
                best_score = results[0].get('similarity_score', 0)
                logger.info(f"   '{expanded_query}': Best score {best_score:.3f}")
        
        # Example 8: Export comprehensive results
        logger.info("\n8. Exporting comprehensive search results...")
        
        # Compile all results
        comprehensive_results = {
            'basic_searches': basic_results,
            'filtered_searches': {
                'technology_ml': tech_results,
                'beginner_learning': beginner_results,
                'advanced_tech': advanced_tech_results
            },
            'threshold_experiments': threshold_results,
            'expansion_experiments': expansion_results,
            'analyses': analyses
        }
        
        # Export to file
        export_file = Path("advanced_search_results.json")
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Comprehensive results exported to {export_file}")
        
        # Example 9: Search performance summary
        logger.info("\n9. Search performance summary...")
        
        total_searches = (
            len(basic_results) + 
            3 +  # filtered searches
            len(threshold_results) +
            len(expansion_results)
        )
        
        successful_searches = sum([
            len([r for r in basic_results.values() if r]),
            len([r for r in [tech_results, beginner_results, advanced_tech_results] if r]),
            len([r for r in threshold_results.values() if r]),
            len([r for r in expansion_results.values() if r])
        ])
        
        logger.info(f"📊 Search Performance Summary:")
        logger.info(f"   Total searches performed: {total_searches}")
        logger.info(f"   Successful searches: {successful_searches}")
        logger.info(f"   Success rate: {successful_searches/total_searches:.1%}")
        
        # Pipeline statistics
        stats = pipeline.get_pipeline_stats()
        if stats and 'vector_store' in stats:
            doc_count = stats['vector_store'].get('document_count', 0)
            logger.info(f"   Documents in database: {doc_count}")
        
        # Clean up
        logger.info("\n10. Cleaning up...")
        pipeline.cleanup()
        logger.info("✅ Cleanup completed")
        
        logger.info("\n" + "="*60)
        logger.info("Advanced search example completed successfully! 🎉")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced search example failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)