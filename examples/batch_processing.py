#!/usr/bin/env python3
"""
Batch processing example for HTML RAG Pipeline.

This example demonstrates:
- Processing multiple HTML files from a directory
- Progress tracking with tqdm
- Parallel processing capabilities
- Batch statistics and reporting
- Error handling for batch operations
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from html_rag.core.pipeline import create_pipeline
from html_rag.core.config import PipelineConfig
from html_rag.utils.logging import setup_logging, PipelineLogger
from html_rag.utils.metrics import track_processing
from html_rag.exceptions.pipeline_exceptions import PipelineError

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available. Install with: pip install tqdm")

# Setup logging
setup_logging(level="INFO", log_file="logs/batch_processing.log")
logger = PipelineLogger("BatchProcessing")


def create_sample_html_files(temp_dir: Path, num_files: int = 10) -> List[Path]:
    """Create sample HTML files for batch processing."""
    
    sample_templates = [
        """<!DOCTYPE html>
        <html><head><title>Tech Article {i}</title></head>
        <body>
            <h1>Technology Trends in {year}</h1>
            <p>This article discusses the latest trends in technology for {year}.</p>
            <h2>Artificial Intelligence</h2>
            <p>AI continues to evolve with new breakthroughs in machine learning and deep learning.</p>
            <h2>Cloud Computing</h2>
            <p>Cloud services are becoming more accessible and powerful.</p>
            <ul>
                <li>Scalability</li>
                <li>Cost-effectiveness</li>
                <li>Reliability</li>
            </ul>
        </body></html>""",
        
        """<!DOCTYPE html>
        <html><head><title>Science News {i}</title></head>
        <body>
            <h1>Recent Discoveries in Science</h1>
            <p>Scientists have made remarkable discoveries in various fields.</p>
            <h2>Space Exploration</h2>
            <p>New missions to Mars and beyond are planned for the coming years.</p>
            <h2>Medical Research</h2>
            <p>Breakthrough treatments for various diseases are being developed.</p>
            <blockquote>Science is the key to understanding our universe.</blockquote>
        </body></html>""",
        
        """<!DOCTYPE html>
        <html><head><title>Business Report {i}</title></head>
        <body>
            <h1>Market Analysis for Q{quarter}</h1>
            <p>The market shows strong performance across multiple sectors.</p>
            <h2>Technology Sector</h2>
            <p>Tech companies continue to drive innovation and growth.</p>
            <h2>Healthcare Sector</h2>
            <p>Healthcare investments are increasing due to aging populations.</p>
            <table>
                <tr><th>Sector</th><th>Growth</th></tr>
                <tr><td>Tech</td><td>12%</td></tr>
                <tr><td>Healthcare</td><td>8%</td></tr>
            </table>
        </body></html>"""
    ]
    
    html_files = []
    
    for i in range(num_files):
        template = sample_templates[i % len(sample_templates)]
        
        # Fill template variables
        content = template.format(
            i=i+1,
            year=2024 + (i % 3),
            quarter=(i % 4) + 1
        )
        
        # Create file
        file_path = temp_dir / f"document_{i+1:03d}.html"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        html_files.append(file_path)
    
    return html_files


def process_files_with_progress(
    pipeline,
    html_files: List[Path],
    batch_size: int = 5,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """Process HTML files with progress tracking."""
    
    documents = []
    results = []
    
    # Prepare documents
    logger.info(f"Preparing {len(html_files)} documents for processing...")
    
    file_iterator = tqdm(html_files, desc="Reading files") if TQDM_AVAILABLE and show_progress else html_files
    
    for html_file in file_iterator:
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            documents.append({
                'html': html_content,
                'url': f"file://{html_file.absolute()}",
                'filename': html_file.name
            })
            
        except Exception as e:
            logger.warning(f"Failed to read {html_file}: {e}")
    
    if not documents:
        logger.error("No valid documents found")
        return []
    
    # Process in batches
    logger.info(f"Processing {len(documents)} documents in batches of {batch_size}...")
    
    batch_iterator = range(0, len(documents), batch_size)
    if TQDM_AVAILABLE and show_progress:
        batch_iterator = tqdm(batch_iterator, desc="Processing batches")
    
    for start_idx in batch_iterator:
        end_idx = min(start_idx + batch_size, len(documents))
        batch = documents[start_idx:end_idx]
        
        logger.info(f"Processing batch {start_idx//batch_size + 1}: documents {start_idx+1}-{end_idx}")
        
        try:
            batch_results = pipeline.process_multiple_html(batch)
            results.extend(batch_results)
            
            # Log batch statistics
            batch_successful = sum(1 for r in batch_results if r.get('success', False))
            logger.info(f"Batch completed: {batch_successful}/{len(batch)} successful")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Add error results for failed batch
            for doc in batch:
                results.append({
                    'success': False,
                    'error': f"Batch processing error: {str(e)}",
                    'url': doc.get('url', 'unknown')
                })
    
    return results


def analyze_batch_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze batch processing results."""
    
    total_docs = len(results)
    successful_docs = sum(1 for r in results if r.get('success', False))
    failed_docs = total_docs - successful_docs
    
    # Calculate processing statistics
    total_text_blocks = sum(r.get('text_blocks_count', 0) for r in results if r.get('success'))
    total_embedded_blocks = sum(r.get('embedded_blocks_count', 0) for r in results if r.get('success'))
    total_html_length = sum(r.get('original_html_length', 0) for r in results if r.get('success'))
    total_cleaned_length = sum(r.get('cleaned_html_length', 0) for r in results if r.get('success'))
    
    # Error analysis
    error_types = {}
    for result in results:
        if not result.get('success') and 'error' in result:
            error_msg = result['error']
            error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    # Calculate averages
    avg_text_blocks = total_text_blocks / successful_docs if successful_docs > 0 else 0
    avg_embedded_blocks = total_embedded_blocks / successful_docs if successful_docs > 0 else 0
    compression_ratio = total_cleaned_length / total_html_length if total_html_length > 0 else 0
    
    return {
        'summary': {
            'total_documents': total_docs,
            'successful_documents': successful_docs,
            'failed_documents': failed_docs,
            'success_rate': successful_docs / total_docs if total_docs > 0 else 0
        },
        'processing_stats': {
            'total_text_blocks': total_text_blocks,
            'total_embedded_blocks': total_embedded_blocks,
            'avg_text_blocks_per_doc': avg_text_blocks,
            'avg_embedded_blocks_per_doc': avg_embedded_blocks,
            'total_html_length': total_html_length,
            'total_cleaned_length': total_cleaned_length,
            'compression_ratio': compression_ratio
        },
        'errors': {
            'error_count': failed_docs,
            'error_types': error_types
        }
    }


def main():
    """Batch processing demonstration."""
    
    logger.info("="*60)
    logger.info("HTML RAG Pipeline - Batch Processing Example")
    logger.info("="*60)
    
    # Create temporary directory for sample files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Example 1: Create sample HTML files
            logger.info("\n1. Creating sample HTML files...")
            
            num_files = 15
            html_files = create_sample_html_files(temp_path, num_files)
            logger.info(f"✅ Created {len(html_files)} sample HTML files in {temp_path}")
            
            # Example 2: Setup pipeline with batch-optimized configuration
            logger.info("\n2. Setting up pipeline for batch processing...")
            
            # Use performance preset for better batch processing
            config = PipelineConfig(
                collection_name="batch_test_collection",
                persist_directory="./batch_test_db",
                batch_size=32,
                num_workers=4,
                enable_metrics=True,
                prefer_basic_cleaning=True  # Faster for batch processing
            )
            
            pipeline = create_pipeline(config=config)
            logger.info("✅ Pipeline configured for batch processing")
            
            # Example 3: Process files with metrics tracking
            logger.info("\n3. Processing files with metrics tracking...")
            
            with track_processing(enable_resource_monitoring=True) as metrics:
                start_time = time.time()
                
                results = process_files_with_progress(
                    pipeline=pipeline,
                    html_files=html_files,
                    batch_size=5,
                    show_progress=TQDM_AVAILABLE
                )
                
                processing_time = time.time() - start_time
            
            logger.info(f"✅ Batch processing completed in {processing_time:.2f} seconds")
            
            # Example 4: Analyze results
            logger.info("\n4. Analyzing batch results...")
            
            analysis = analyze_batch_results(results)
            
            logger.info("📊 Batch Processing Summary:")
            summary = analysis['summary']
            logger.info(f"   📄 Total documents: {summary['total_documents']}")
            logger.info(f"   ✅ Successful: {summary['successful_documents']}")
            logger.info(f"   ❌ Failed: {summary['failed_documents']}")
            logger.info(f"   📈 Success rate: {summary['success_rate']:.1%}")
            
            stats = analysis['processing_stats']
            logger.info(f"\n📊 Processing Statistics:")
            logger.info(f"   📝 Total text blocks: {stats['total_text_blocks']}")
            logger.info(f"   🔗 Total embedded blocks: {stats['total_embedded_blocks']}")
            logger.info(f"   📊 Avg blocks per document: {stats['avg_text_blocks_per_doc']:.1f}")
            logger.info(f"   🗜️  Compression ratio: {stats['compression_ratio']:.1%}")
            
            # Resource metrics
            if metrics:
                resource_metrics = metrics.get_metrics_dict()
                logger.info(f"\n🔧 Resource Usage:")
                logger.info(f"   💾 Peak memory: {resource_metrics['resources']['peak_memory_mb']:.1f} MB")
                logger.info(f"   🖥️  Avg CPU: {resource_metrics['resources']['avg_cpu_percent']:.1f}%")
                logger.info(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
                logger.info(f"   📈 Throughput: {len(html_files)/processing_time:.1f} docs/sec")
            
            # Error analysis
            errors = analysis['errors']
            if errors['error_count'] > 0:
                logger.warning(f"\n⚠️  Error Analysis:")
                logger.warning(f"   Total errors: {errors['error_count']}")
                for error_type, count in errors['error_types'].items():
                    logger.warning(f"   {error_type}: {count}")
            
            # Example 5: Test search on processed documents
            logger.info("\n5. Testing search on processed documents...")
            
            search_queries = ["technology", "science", "market analysis", "AI"]
            
            for query in search_queries:
                search_results = pipeline.search(query, n_results=3)
                if search_results:
                    logger.info(f"🔍 '{query}': {len(search_results)} results")
                    best_match = search_results[0]
                    score = best_match.get('similarity_score', 0)
                    logger.info(f"   Best match: {score:.3f} score")
                else:
                    logger.warning(f"❌ No results for '{query}'")
            
            # Example 6: Export results
            logger.info("\n6. Exporting batch results...")
            
            # Export to JSON
            import json
            results_file = Path("batch_processing_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': results,
                    'analysis': analysis,
                    'processing_time': processing_time
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Results exported to {results_file}")
            
            # Clean up
            logger.info("\n7. Cleaning up...")
            pipeline.cleanup()
            logger.info("✅ Cleanup completed")
            
            logger.info("\n" + "="*60)
            logger.info("Batch processing example completed successfully! 🎉")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch processing example failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)