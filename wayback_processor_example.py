#!/usr/bin/env python3
"""
Wayback HTML Files Database Processor

This script processes a folder of HTML files from Wayback Machine archives
and fills the vector database for subsequent searching.

Usage:
    python wayback_processor_example.py /path/to/wayback/html/files
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.html_rag import create_pipeline
from src.html_rag.core.config import PipelineConfig, WaybackConfig
from src.html_rag.utils.logging import setup_logging, PipelineLogger
from src.html_rag.utils.metrics import track_processing

# Setup logging
setup_logging(level="INFO", log_file="logs/wayback_processing.log", enable_console=True)
logger = PipelineLogger("WaybackProcessor")


def find_html_files(directory: Path) -> List[Dict[str, Any]]:
    """
    Find all HTML files in a directory and prepare them for processing.
    
    Args:
        directory: Path to directory containing HTML files
        
    Returns:
        List of document dictionaries with html content and metadata
    """
    logger.info(f"Scanning directory: {directory}")
    
    documents = []
    html_files = list(directory.glob("**/*.html")) + list(directory.glob("**/*.htm"))
    
    logger.info(f"Found {len(html_files)} HTML files")
    
    for html_file in html_files:
        try:
            # Read HTML content
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Skip very small files (likely not content)
            if len(html_content) < 100:
                logger.debug(f"Skipping small file: {html_file}")
                continue
            
            # Try to extract metadata from filename/path
            # Expected patterns: timestamp_domain_url.html or similar
            metadata = extract_metadata_from_path(html_file)
            
            document = {
                'html': html_content,
                'url': metadata.get('original_url', f"file://{html_file}"),
                'wayback_metadata': metadata,
                'file_path': str(html_file),
                'file_size': len(html_content)
            }
            
            documents.append(document)
            logger.debug(f"Added file: {html_file.name} ({len(html_content)} chars)")
            
        except Exception as e:
            logger.warning(f"Failed to read {html_file}: {e}")
            continue
    
    logger.info(f"Prepared {len(documents)} documents for processing")
    return documents


def extract_metadata_from_path(file_path: Path) -> Dict[str, Any]:
    """
    Extract Wayback metadata from file path and name.
    
    Args:
        file_path: Path to HTML file
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    # Get filename and path parts
    filename = file_path.stem
    path_parts = file_path.parts
    
    # Pattern 1: Check if timestamp is in filename (timestamp_domain_path.html)
    parts = filename.split('_')
    if len(parts) >= 2:
        potential_timestamp = parts[0]
        if len(potential_timestamp) == 14 and potential_timestamp.isdigit():
            metadata['timestamp'] = potential_timestamp
            year = int(potential_timestamp[:4])
            metadata['year'] = year
            
            # Second part might be domain
            if len(parts) > 1:
                potential_domain = parts[1]
                if '.' in potential_domain:
                    metadata['domain'] = potential_domain
                    metadata['original_url'] = f"https://{potential_domain}"
    
    # Pattern 2: Check if timestamp is in directory name (common Wayback pattern)
    # Look for 14-digit timestamp in any path component
    if 'timestamp' not in metadata:
        for part in path_parts:
            if part.isdigit() and len(part) == 14:
                metadata['timestamp'] = part
                year = int(part[:4])
                metadata['year'] = year
                break
    
    # Pattern 3: Extract domain from filename (domain.com_page.html pattern)
    if 'domain' not in metadata:
        # Look for domain.com pattern in filename
        if '.' in filename:
            # Split by underscore and look for domain-like patterns
            for part in parts:
                if '.' in part and len(part) > 3:
                    # Check if it looks like a domain
                    if part.count('.') >= 1 and not part.startswith('.') and not part.endswith('.'):
                        metadata['domain'] = part
                        break
    
    # Pattern 4: Extract domain from directory structure
    if 'domain' not in metadata:
        for part in path_parts:
            if '.' in part and len(part) > 3 and part.count('.') >= 1:
                metadata['domain'] = part
                break
    
    # Pattern 5: Extract year from path if no timestamp found
    if 'year' not in metadata:
        for part in path_parts:
            if part.isdigit() and len(part) == 4:
                year = int(part)
                if 1990 <= year <= 2030:
                    metadata['year'] = year
                    break
    
    # Set required defaults
    if 'timestamp' not in metadata:
        if 'year' in metadata:
            metadata['timestamp'] = f"{metadata['year']}0101000000"
        else:
            # Extract year from any 4-digit number in path
            for part in path_parts:
                if len(part) >= 4:
                    year_match = part[:4]
                    if year_match.isdigit():
                        year = int(year_match)
                        if 1990 <= year <= 2030:
                            metadata['year'] = year
                            metadata['timestamp'] = f"{year}0101000000"
                            break
            else:
                # Ultimate fallback
                metadata['timestamp'] = "20200101000000"
                metadata['year'] = 2020
    
    if 'domain' not in metadata:
        metadata['domain'] = 'unknown.com'
    
    if 'original_url' not in metadata:
        domain = metadata.get('domain', 'unknown.com')
        # Build URL from filename if possible
        if domain != 'unknown.com':
            # Extract path from filename
            path_suffix = filename.replace(domain, '').strip('_')
            if path_suffix:
                path_suffix = path_suffix.replace('_', '/')
                metadata['original_url'] = f"https://{domain}/{path_suffix}"
            else:
                metadata['original_url'] = f"https://{domain}"
        else:
            metadata['original_url'] = f"https://unknown.com"
    
    # Add archive URL
    if 'timestamp' in metadata:
        metadata['archive_url'] = f"https://web.archive.org/web/{metadata['timestamp']}/{metadata['original_url']}"
    
    metadata['title'] = f"Archived page from {metadata.get('domain', 'unknown')}"
    
    return metadata


def process_wayback_files(
    html_directory: str,
    collection_name: str = "wayback_html_files",
    persist_directory: str = "./wayback_html_db"
) -> Dict[str, Any]:
    """
    Process Wayback HTML files and store in vector database.
    
    Args:
        html_directory: Directory containing HTML files
        collection_name: Name for the ChromaDB collection
        persist_directory: Directory to store the database
        
    Returns:
        Processing results and statistics
    """
    logger.info("="*60)
    logger.info("WAYBACK HTML FILES PROCESSOR")
    logger.info("="*60)
    
    # Validate input directory
    html_dir = Path(html_directory)
    if not html_dir.exists():
        raise ValueError(f"Directory does not exist: {html_directory}")
    
    if not html_dir.is_dir():
        raise ValueError(f"Path is not a directory: {html_directory}")
    
    logger.info(f"Processing HTML files from: {html_dir}")
    logger.info(f"Database collection: {collection_name}")
    logger.info(f"Database location: {persist_directory}")
    
    # Configure pipeline for Wayback processing
    config = PipelineConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        prefer_basic_cleaning=True,  # Better for preserving Ukrainian/multilingual content
        cyrillic_detection_threshold=0.1,  # More sensitive for Ukrainian detection
        enable_metrics=True,
        log_level="INFO"
    )
    
    logger.info("🔧 Creating RAG pipeline...")
    pipeline = create_pipeline(config=config)
    
    try:
        # Find and prepare HTML files
        logger.info("📁 Scanning for HTML files...")
        documents = find_html_files(html_dir)
        
        if not documents:
            logger.error("❌ No HTML files found to process")
            return {'success': False, 'error': 'No HTML files found'}
        
        # Process documents with metrics tracking
        logger.info(f"🚀 Processing {len(documents)} HTML files...")
        
        with track_processing(enable_resource_monitoring=True) as metrics:
            start_time = time.time()
            
            # Process in batches for better performance
            batch_size = 10  # Adjust based on file sizes and memory
            results = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} files)")
                
                # Process batch
                batch_results = pipeline.process_multiple_html(
                    batch, 
                    force_basic_cleaning=True  # Ensure Ukrainian content preservation
                )
                results.extend(batch_results)
                
                # Log batch progress
                successful_in_batch = sum(1 for r in batch_results if r.get('success', False))
                logger.info(f"Batch completed: {successful_in_batch}/{len(batch)} successful")
            
            total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        # Calculate statistics
        total_text_blocks = sum(r.get('embedded_blocks_count', 0) for r in successful)
        total_html_size = sum(d['file_size'] for d in documents)
        
        # Get pipeline statistics
        pipeline_stats = pipeline.get_pipeline_stats()
        
        # Compile final results
        processing_results = {
            'success': True,
            'processing_summary': {
                'total_files': len(documents),
                'successful_files': len(successful),
                'failed_files': len(failed),
                'success_rate': len(successful) / len(documents) if documents else 0,
                'total_processing_time': total_time,
                'average_time_per_file': total_time / len(documents) if documents else 0,
                'files_per_second': len(documents) / total_time if total_time > 0 else 0
            },
            'content_statistics': {
                'total_html_size_mb': total_html_size / (1024 * 1024),
                'total_text_blocks': total_text_blocks,
                'average_blocks_per_file': total_text_blocks / len(successful) if successful else 0,
                'database_documents': pipeline_stats.get('vector_store', {}).get('document_count', 0)
            },
            'performance_metrics': metrics.get_metrics_dict() if metrics else {},
            'failed_files': [
                {
                    'file': documents[i].get('file_path', 'unknown'),
                    'error': r.get('error', 'Unknown error')
                }
                for i, r in enumerate(results) if not r.get('success', False)
            ]
        }
        
        # Log final summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETED!")
        logger.info("="*60)
        logger.info(f"📊 Files processed: {processing_results['processing_summary']['total_files']}")
        logger.info(f"✅ Successful: {processing_results['processing_summary']['successful_files']}")
        logger.info(f"❌ Failed: {processing_results['processing_summary']['failed_files']}")
        logger.info(f"📈 Success rate: {processing_results['processing_summary']['success_rate']:.1%}")
        logger.info(f"⏱️  Total time: {processing_results['processing_summary']['total_processing_time']:.2f} seconds")
        logger.info(f"📝 Text blocks created: {processing_results['content_statistics']['total_text_blocks']}")
        logger.info(f"🗄️  Database documents: {processing_results['content_statistics']['database_documents']}")
        
        if metrics:
            perf = processing_results['performance_metrics']
            logger.info(f"💾 Peak memory: {perf['resources']['peak_memory_mb']:.1f} MB")
            logger.info(f"🖥️  Average CPU: {perf['resources']['avg_cpu_percent']:.1f}%")
        
        # Show failed files if any
        if processing_results['failed_files']:
            logger.warning(f"\n⚠️  {len(processing_results['failed_files'])} files failed to process:")
            for failed_file in processing_results['failed_files'][:5]:  # Show first 5
                logger.warning(f"   • {Path(failed_file['file']).name}: {failed_file['error']}")
            if len(processing_results['failed_files']) > 5:
                logger.warning(f"   ... and {len(processing_results['failed_files']) - 5} more")
        
        logger.info(f"\n🎉 Database ready for searching!")
        logger.info(f"   Collection: {collection_name}")
        logger.info(f"   Location: {persist_directory}")
        
        return processing_results
        
    finally:
        # Always cleanup
        logger.info("🧹 Cleaning up resources...")
        pipeline.cleanup()


def main():
    """Main function to run the Wayback HTML processor."""
    parser = argparse.ArgumentParser(
        description="Process Wayback HTML files into a searchable vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python wayback_processor_example.py /path/to/wayback/html/files
    python wayback_processor_example.py ./wayback_snapshots --collection my_wayback --db-dir ./my_db
    python wayback_processor_example.py /data/wayback --collection ukrainian_sites --db-dir ./ukrainian_db
        """
    )
    
    parser.add_argument(
        'html_directory',
        help='Directory containing HTML files from Wayback Machine'
    )
    parser.add_argument(
        '--collection',
        default='wayback_html_files',
        help='Name for the ChromaDB collection (default: wayback_html_files)'
    )
    parser.add_argument(
        '--db-dir',
        default='./wayback_html_db',
        help='Directory to store the vector database (default: ./wayback_html_db)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    setup_logging(level=args.log_level, log_file="logs/wayback_processing.log", enable_console=True)
    
    try:
        # Process the HTML files
        results = process_wayback_files(
            html_directory=args.html_directory,
            collection_name=args.collection,
            persist_directory=args.db_dir
        )
        
        if results['success']:
            print(f"\n✅ Successfully processed {results['processing_summary']['successful_files']} files!")
            print(f"🔍 Ready to search with: python wayback_search_example.py --collection {args.collection} --db-dir {args.db_dir}")
            return 0
        else:
            print(f"\n❌ Processing failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())