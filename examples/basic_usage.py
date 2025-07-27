#!/usr/bin/env python3
"""
Basic usage example for HTML RAG Pipeline.

This example demonstrates:
- Configuration loading and setup
- Processing a single HTML document
- Basic search functionality
- Error handling and logging
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from html_rag.core.pipeline import create_pipeline
from html_rag.core.config import PipelineConfig
from html_rag.utils.logging import setup_logging, PipelineLogger
from html_rag.exceptions.pipeline_exceptions import PipelineError

# Setup logging
setup_logging(level="INFO", log_file="logs/basic_usage.log")
logger = PipelineLogger("BasicUsage")


def main():
    """Basic usage demonstration."""
    
    logger.info("="*60)
    logger.info("HTML RAG Pipeline - Basic Usage Example")
    logger.info("="*60)
    
    try:
        # Example 1: Create pipeline with default configuration
        logger.info("\n1. Creating pipeline with default configuration...")
        
        pipeline = create_pipeline()
        logger.info("✅ Pipeline created successfully")
        
        # Example 2: Process HTML content
        logger.info("\n2. Processing HTML content...")
        
        # Sample HTML content
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sample Article</title>
        </head>
        <body>
            <h1>Introduction to Machine Learning</h1>
            <p>Machine learning is a subset of artificial intelligence that focuses on algorithms.</p>
            
            <h2>Key Concepts</h2>
            <ul>
                <li>Supervised Learning</li>
                <li>Unsupervised Learning</li>
                <li>Reinforcement Learning</li>
            </ul>
            
            <h2>Applications</h2>
            <p>Machine learning has many applications in various fields including:</p>
            <ul>
                <li>Natural Language Processing</li>
                <li>Computer Vision</li>
                <li>Recommendation Systems</li>
            </ul>
            
            <h3>Conclusion</h3>
            <p>Understanding machine learning is crucial for modern data science.</p>
        </body>
        </html>
        """
        
        # Process the HTML
        result = pipeline.process_html(
            raw_html=sample_html,
            url="https://example.com/ml-article"
        )
        
        if result['success']:
            logger.info(f"✅ Processing successful!")
            logger.info(f"   📄 Original HTML: {result['original_html_length']} characters")
            logger.info(f"   🧹 Cleaned HTML: {result['cleaned_html_length']} characters")
            logger.info(f"   📝 Text blocks: {result['text_blocks_count']}")
            logger.info(f"   🔗 Embedded blocks: {result['embedded_blocks_count']}")
            logger.info(f"   📊 Embedding dimension: {result['embedding_dimension']}")
        else:
            logger.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Example 3: Search functionality
        logger.info("\n3. Testing search functionality...")
        
        # Perform various searches
        search_queries = [
            "machine learning",
            "supervised learning algorithms",
            "applications of ML",
            "computer vision",
            "data science"
        ]
        
        for query in search_queries:
            logger.info(f"\nSearching for: '{query}'")
            
            try:
                search_results = pipeline.search(query, n_results=3)
                
                if search_results:
                    logger.info(f"✅ Found {len(search_results)} results")
                    for i, result in enumerate(search_results, 1):
                        score = result.get('similarity_score', 0)
                        text_preview = result.get('text', '')[:80] + '...' if len(result.get('text', '')) > 80 else result.get('text', '')
                        logger.info(f"   {i}. Score: {score:.3f} | {text_preview}")
                else:
                    logger.warning(f"❌ No results found for '{query}'")
                    
            except Exception as e:
                logger.error(f"❌ Search failed for '{query}': {str(e)}")
        
        # Example 4: Pipeline statistics
        logger.info("\n4. Getting pipeline statistics...")
        
        stats = pipeline.get_pipeline_stats()
        if stats:
            logger.info("📊 Pipeline Statistics:")
            logger.info(f"   Collection: {stats.get('vector_store', {}).get('collection_name', 'N/A')}")
            logger.info(f"   Documents: {stats.get('vector_store', {}).get('document_count', 0)}")
            logger.info(f"   Embedding model: {stats.get('embedding_model', {}).get('model_name', 'N/A')}")
            logger.info(f"   HTML pruner: {stats.get('html_pruner_model', 'N/A')}")
        
        # Example 5: Demonstrate error handling
        logger.info("\n5. Demonstrating error handling...")
        
        try:
            # Try to process invalid HTML
            invalid_result = pipeline.process_html(
                raw_html="<invalid>Not really HTML</invalid>",
                url="https://example.com/invalid"
            )
            
            if not invalid_result['success']:
                logger.info(f"✅ Error handling working: {invalid_result.get('error', 'Unknown error')}")
            
        except PipelineError as e:
            logger.info(f"✅ Pipeline error caught: {str(e)}")
        except Exception as e:
            logger.warning(f"⚠️  Unexpected error: {str(e)}")
        
        # Clean up
        logger.info("\n6. Cleaning up resources...")
        pipeline.cleanup()
        logger.info("✅ Cleanup completed")
        
        logger.info("\n" + "="*60)
        logger.info("Basic usage example completed successfully! 🎉")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Example failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)