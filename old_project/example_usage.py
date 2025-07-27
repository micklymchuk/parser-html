"""
Example usage of the RAG Pipeline with Wayback Machine snapshots
"""

import logging
import os
from pathlib import Path
from rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the RAG pipeline with Wayback Machine snapshots."""

    logger.info("\n" + "="*60)
    logger.info("WAYBACK MACHINE SNAPSHOT PROCESSING DEMONSTRATION")
    logger.info("="*60)

    # Check if wayback snapshots directory exists
    wayback_directory = "./20201130051600"  # Example directory from user's description

    if not os.path.exists(wayback_directory):
        logger.error(f"Wayback directory '{wayback_directory}' not found.")
        logger.error("Please provide a directory with Wayback Machine snapshots.")
        return

    try:
        # Initialize pipeline for wayback processing
        pipeline = RAGPipeline(
            collection_name="wayback_documents",
            persist_directory="./wayback_chroma_db"
        )

        # Pre-load models
        logger.info("Loading models...")
        pipeline.load_models()

        # Validate wayback directory
        logger.info("Validating Wayback snapshots directory...")
        validation = pipeline.validate_wayback_directory(wayback_directory)

        logger.info(f"Directory validation results:")
        logger.info(f"  Directory exists: {validation.get('directory_exists', False)}")
        logger.info(f"  HTML files: {validation.get('html_files_count', 0)}")
        logger.info(f"  Meta files: {validation.get('meta_files_count', 0)}")
        logger.info(f"  Complete pairs: {validation.get('paired_files_count', 0)}")

        if validation.get('errors'):
            logger.warning(f"  Errors: {'; '.join(validation['errors'])}")

        # Check if directory has HTML files (don't require meta.json files)
        if not validation.get('directory_exists', False):
            logger.error("Directory does not exist, stopping")
            return
            
        if validation.get('html_files_count', 0) == 0:
            logger.error("No HTML files found in directory, stopping")
            return
            
        logger.info(f"Found {validation.get('html_files_count', 0)} HTML files to process")
        if validation.get('paired_files_count', 0) > 0:
            logger.info(f"  {validation.get('paired_files_count', 0)} have corresponding meta.json files")
        orphaned_html = len(validation.get('orphaned_html_files', []))
        if orphaned_html > 0:
            logger.info(f"  {orphaned_html} will use synthetic metadata")

        # Process wayback snapshots (all HTML files, not just those with metadata)
        logger.info("Processing ALL Wayback HTML files (with or without metadata)...")
        wayback_results = pipeline.process_wayback_snapshots(
            wayback_directory,
            require_metadata=False,  # Process ALL HTML files, not just those with meta.json
            # Optional filters - uncomment to test
            # domain_filter="sluga-narodu.com",
            # year_filter=2020,
            # min_content_length=1000
        )

        # Display wayback processing results
        logger.info("\nWayback Processing Results:")
        successful_wayback = sum(1 for r in wayback_results if r.get('success', False))
        failed_wayback = len(wayback_results) - successful_wayback

        logger.info(f"  Processed: {successful_wayback} successful, {failed_wayback} failed")

        for i, result in enumerate(wayback_results):
            if result['success']:
                logger.info(f"\nWayback Document {i+1}:")
                logger.info(f"  URL: {result['url']}")
                logger.info(f"  Wayback processed: {result.get('wayback_processed', False)}")
                logger.info(f"  Text blocks: {result['embedded_blocks_count']}")
                logger.info(f"  Processing time: {result['processing_times']['total']:.2f}s")
                if 'stage0_wayback' in result['processing_times']:
                    logger.info(f"  Stage 0 (Wayback): {result['processing_times']['stage0_wayback']:.2f}s")

        # Get pipeline statistics
        stats = pipeline.get_pipeline_stats()
        logger.info(f"\nPipeline Statistics:")
        logger.info(f"  Total documents in database: {stats['vector_store']['document_count']}")
        logger.info(f"  Embedding model: {stats['embedding_model']['model_name']}")
        logger.info(f"  Embedding dimension: {stats['embedding_model']['embedding_dimension']}")

        # Demonstrate wayback-specific search
        logger.info("\n" + "="*50)
        logger.info("WAYBACK-SPECIFIC SEARCH EXAMPLES")
        logger.info("="*50)

        wayback_queries = [
            "партія",
            "політична",
            "народу",
            "слуга"
        ]

        for query in wayback_queries:
            logger.info(f"\nWayback search for: '{query}'")
            wayback_search_results = pipeline.search_wayback_snapshots(
                query,
                n_results=3,
                # timestamp_filter="20201130051600",  # Uncomment to filter by timestamp
                # domain_filter="sluga-narodu.com"        # Uncomment to filter by domain
            )

            if wayback_search_results:
                for j, result in enumerate(wayback_search_results):
                    metadata = result['metadata']
                    logger.info(f"  Result {j+1} (similarity: {result['similarity_score']:.3f}):")
                    logger.info(f"    Text: {result['text'][:80]}...")
                    logger.info(f"    Wayback timestamp: {metadata.get('wayback_timestamp', 'N/A')}")
                    logger.info(f"    Original URL: {metadata.get('wayback_original_url', 'N/A')}")
                    logger.info(f"    Archive URL: {metadata.get('wayback_archive_url', 'N/A')}")
                    logger.info(f"    Domain: {metadata.get('wayback_domain', 'N/A')}")
            else:
                logger.info("  No wayback results found")

        # Search by wayback metadata
        logger.info("\n" + "="*50)
        logger.info("WAYBACK METADATA FILTERING")
        logger.info("="*50)

        # Find all documents from 2020
        logger.info("\nSearching for documents from timestamp 20201130051600...")
        wayback_2020 = pipeline.search_by_metadata(
            metadata_filter={"wayback_timestamp": "20201130051600"},
            limit=5
        )

        for result in wayback_2020:
            metadata = result['metadata']
            logger.info(f"  Found: {result['text'][:60]}...")
            logger.info(f"    Timestamp: {metadata.get('wayback_timestamp', 'N/A')}")
            logger.info(f"    Domain: {metadata.get('wayback_domain', 'N/A')}")

        # Search for headings only from wayback
        logger.info("\nSearching for headings from wayback snapshots...")
        wayback_headings = pipeline.search_by_metadata(
            metadata_filter={
                "element_type": "heading",
                "wayback_domain": "sluga-narodu.com"
            },
            limit=5
        )

        for result in wayback_headings:
            metadata = result['metadata']
            logger.info(f"  Heading: {result['text']}")
            logger.info(f"    Level: {metadata.get('hierarchy_level', 'N/A')}")
            logger.info(f"    URL: {metadata.get('wayback_original_url', 'N/A')}")

        # Export wayback documents
        logger.info("\nExporting wayback documents...")
        try:
            pipeline.export_documents("wayback_documents.json", format="json")
            logger.info("Wayback documents exported to wayback_documents.json")
        except Exception as e:
            logger.warning(f"Export failed: {e}")

        logger.info("\n" + "="*50)
        logger.info("WAYBACK DEMONSTRATION COMPLETED!")
        logger.info("="*50)
        logger.info("Successfully demonstrated:")
        logger.info("0. Wayback snapshot directory validation")
        logger.info("1. Processing Wayback Machine snapshots")
        logger.info("2. Wayback-specific search capabilities")
        logger.info("3. Metadata filtering for archived content")
        logger.info("4. Export of processed wayback content")

    except Exception as e:
        logger.error(f"Error in wayback demonstration: {e}")
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