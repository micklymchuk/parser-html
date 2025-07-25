"""
Example usage of the RAG Pipeline with sample HTML content
"""

import logging
from rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the RAG pipeline with sample HTML content."""
    
    # Sample HTML content for testing
    sample_html_1 = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Machine Learning Fundamentals</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .highlight { background-color: yellow; }
        </style>
        <script>
            function highlightText() {
                console.log("Highlighting text");
            }
        </script>
    </head>
    <body>
        <h1>Introduction to Machine Learning</h1>
        <p>Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience.</p>
        
        <h2>Types of Machine Learning</h2>
        <p>There are three main types of machine learning:</p>
        <ul>
            <li>Supervised Learning: Uses labeled training data to learn a mapping from inputs to outputs</li>
            <li>Unsupervised Learning: Finds patterns in data without labeled examples</li>
            <li>Reinforcement Learning: Learns through interaction with an environment</li>
        </ul>
        
        <h2>Popular Algorithms</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Type</th>
                <th>Use Case</th>
            </tr>
            <tr>
                <td>Linear Regression</td>
                <td>Supervised</td>
                <td>Predicting continuous values</td>
            </tr>
            <tr>
                <td>K-Means</td>
                <td>Unsupervised</td>
                <td>Clustering data points</td>
            </tr>
        </table>
        
        <blockquote>
            "Machine learning is the future of technology, enabling systems to learn and adapt without being explicitly programmed for every scenario."
        </blockquote>
        
        <div class="footer">
            <p>For more information, visit our comprehensive ML guide.</p>
        </div>
    </body>
    </html>
    """
    
    sample_html_2 = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Python Programming Guide</title>
    </head>
    <body>
        <h1>Python Programming Basics</h1>
        <p>Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation.</p>
        
        <h2>Key Features</h2>
        <ul>
            <li>Easy to learn and use</li>
            <li>Extensive standard library</li>
            <li>Cross-platform compatibility</li>
            <li>Strong community support</li>
        </ul>
        
        <h2>Data Types</h2>
        <p>Python supports various data types including:</p>
        <ul>
            <li>Numbers (int, float, complex)</li>
            <li>Strings</li>
            <li>Lists</li>
            <li>Tuples</li>
            <li>Dictionaries</li>
            <li>Sets</li>
        </ul>
        
        <h3>Example Code</h3>
        <p>Here's a simple Python example:</p>
        <div class="code-block">
            def greet(name):
                return f"Hello, {name}!"
            
            message = greet("World")
            print(message)
        </div>
    </body>
    </html>
    """
    
    try:
        # Initialize the RAG pipeline
        logger.info("Initializing RAG Pipeline...")
        pipeline = RAGPipeline(
            collection_name="example_documents",
            persist_directory="./example_chroma_db"
        )
        
        # Pre-load models (optional but recommended for better performance)
        logger.info("Loading models...")
        pipeline.load_models()
        
        # Process sample HTML documents
        logger.info("Processing sample HTML documents...")
        
        html_documents = [
            {"html": sample_html_1, "url": "https://example.com/ml-fundamentals"},
            {"html": sample_html_2, "url": "https://example.com/python-guide"}
        ]
        
        # Process multiple documents
        results = pipeline.process_multiple_html(html_documents)
        
        # Display processing results
        logger.info("\n" + "="*50)
        logger.info("PROCESSING RESULTS")
        logger.info("="*50)
        
        for i, result in enumerate(results):
            if result['success']:
                logger.info(f"\nDocument {i+1}: {result['url']}")
                logger.info(f"  Original HTML length: {result['original_html_length']:,} chars")
                logger.info(f"  Cleaned HTML length: {result['cleaned_html_length']:,} chars")
                logger.info(f"  Text blocks extracted: {result['text_blocks_count']}")
                logger.info(f"  Final embedded blocks: {result['embedded_blocks_count']}")
                logger.info(f"  Total processing time: {result['processing_times']['total']:.2f}s")
            else:
                logger.error(f"Document {i+1} failed: {result.get('error', 'Unknown error')}")
        
        # Get pipeline statistics
        stats = pipeline.get_pipeline_stats()
        logger.info(f"\nPipeline Statistics:")
        logger.info(f"  Total documents in database: {stats['vector_store']['document_count']}")
        logger.info(f"  Embedding model: {stats['embedding_model']['model_name']}")
        logger.info(f"  Embedding dimension: {stats['embedding_model']['embedding_dimension']}")
        
        # Demonstrate search functionality
        logger.info("\n" + "="*50)
        logger.info("SEARCH EXAMPLES")
        logger.info("="*50)
        
        search_queries = [
            "What is machine learning?",
            "Python data types",
            "supervised learning algorithms",
            "programming language features"
        ]
        
        for query in search_queries:
            logger.info(f"\nSearching for: '{query}'")
            search_results = pipeline.search(query, n_results=3)
            
            if search_results:
                for j, result in enumerate(search_results):
                    logger.info(f"  Result {j+1} (similarity: {result['similarity_score']:.3f}):")
                    logger.info(f"    Text: {result['text'][:100]}...")
                    logger.info(f"    Type: {result['metadata']['element_type']}")
                    logger.info(f"    URL: {result['metadata']['url']}")
            else:
                logger.info("  No results found")
        
        # Demonstrate metadata filtering
        logger.info("\n" + "="*50)
        logger.info("METADATA FILTERING EXAMPLES")
        logger.info("="*50)
        
        # Search for headings only
        logger.info("\nSearching for headings only...")
        heading_results = pipeline.search_by_metadata(
            metadata_filter={"element_type": "heading"},
            limit=5
        )
        
        for result in heading_results:
            logger.info(f"  Heading: {result['text']}")
            logger.info(f"    Level: {result['metadata'].get('hierarchy_level', 'N/A')}")
        
        # Search for content from specific URL
        logger.info(f"\nSearching for content from ML fundamentals page...")
        ml_results = pipeline.search_by_metadata(
            metadata_filter={"url": "https://example.com/ml-fundamentals"},
            limit=3
        )
        
        for result in ml_results:
            logger.info(f"  {result['metadata']['element_type']}: {result['text'][:80]}...")
        
        # Export documents (optional)
        logger.info("\nExporting documents...")
        try:
            pipeline.export_documents("exported_documents.json", format="json")
            logger.info("Documents exported to exported_documents.json")
        except Exception as e:
            logger.warning(f"Export failed: {e}")
        
        logger.info("\n" + "="*50)
        logger.info("EXAMPLE COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info("The RAG pipeline has successfully:")
        logger.info("1. Pruned HTML content to remove noise")
        logger.info("2. Parsed HTML into structured text blocks")
        logger.info("3. Generated embeddings for semantic search")
        logger.info("4. Stored everything in ChromaDB for retrieval")
        logger.info("5. Demonstrated search and filtering capabilities")
        
    except Exception as e:
        logger.error(f"Error in example execution: {e}")
        raise
    
    finally:
        # Clean up resources
        try:
            pipeline.cleanup()
            logger.info("Pipeline resources cleaned up")
        except:
            pass


if __name__ == "__main__":
    main()