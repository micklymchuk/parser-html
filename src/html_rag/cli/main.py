"""
Command-line interface for HTML RAG Pipeline.
"""

import sys
import json
import click
from pathlib import Path
from typing import Optional, Dict, Any

from ..core.pipeline import RAGPipeline, create_pipeline
from ..core.config import PipelineConfig, get_config_preset, save_config
from ..utils.logging import setup_logging, PipelineLogger
from ..exceptions.pipeline_exceptions import PipelineError


# Setup logging for CLI
setup_logging(level="INFO", enable_console=True)
logger = PipelineLogger("CLI")


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', help='Log file path')
@click.pass_context
def main(ctx, config, log_level, log_file):
    """HTML RAG Pipeline CLI - Process HTML content into searchable vector database."""
    
    # Ensure context dict exists
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(level=log_level, log_file=log_file, enable_console=True)
    
    # Store config path in context
    ctx.obj['config_path'] = config


@main.command()
@click.argument('html_file', type=click.Path(exists=True))
@click.option('--url', help='Source URL of the HTML content')
@click.option('--output', '-o', help='Output file for results (JSON)')
@click.option('--collection', help='ChromaDB collection name')
@click.option('--preset', help='Configuration preset (ukrainian, english, wayback, performance)')
@click.option('--force-basic-cleaning', is_flag=True, help='Force basic HTML cleaning')
@click.pass_context
def process(ctx, html_file, url, output, collection, preset, force_basic_cleaning):
    """Process a single HTML file through the RAG pipeline."""
    
    try:
        # Load configuration
        config_path = ctx.obj.get('config_path')
        
        if preset:
            pipeline = create_pipeline(preset=preset)
        elif config_path:
            pipeline = create_pipeline(config_path=config_path)
        else:
            kwargs = {}
            if collection:
                kwargs['collection_name'] = collection
            pipeline = create_pipeline(**kwargs)
        
        # Read HTML file
        html_path = Path(html_file)
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Use filename as URL if not provided
        if not url:
            url = f"file://{html_path.absolute()}"
        
        logger.info(f"Processing HTML file: {html_file}")
        
        # Process the HTML
        result = pipeline.process_html(
            html_content, 
            url=url, 
            force_basic_cleaning=force_basic_cleaning
        )
        
        # Output results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output}")
        else:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result['success']:
            logger.info(f"✅ Successfully processed {result['embedded_blocks_count']} text blocks")
        else:
            logger.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)


@main.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--pattern', default='*.html', help='File pattern to match')
@click.option('--output', '-o', help='Output file for results (JSON)')
@click.option('--collection', help='ChromaDB collection name')
@click.option('--preset', help='Configuration preset')
@click.option('--force-basic-cleaning', is_flag=True, help='Force basic HTML cleaning')
@click.option('--parallel', '-p', default=1, help='Number of parallel workers')
@click.pass_context
def batch(ctx, directory, pattern, output, collection, preset, force_basic_cleaning, parallel):
    """Process multiple HTML files from a directory."""
    
    try:
        # Load configuration
        config_path = ctx.obj.get('config_path')
        
        if preset:
            pipeline = create_pipeline(preset=preset)
        elif config_path:
            pipeline = create_pipeline(config_path=config_path)
        else:
            kwargs = {}
            if collection:
                kwargs['collection_name'] = collection
            if parallel > 1:
                kwargs['num_workers'] = parallel
            pipeline = create_pipeline(**kwargs)
        
        # Find HTML files
        dir_path = Path(directory)
        html_files = list(dir_path.glob(pattern))
        
        if not html_files:
            logger.warning(f"No HTML files found matching pattern '{pattern}' in {directory}")
            return
        
        logger.info(f"Found {len(html_files)} HTML files to process")
        
        # Prepare documents
        documents = []
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                documents.append({
                    'html': html_content,
                    'url': f"file://{html_file.absolute()}"
                })
            except Exception as e:
                logger.warning(f"Failed to read {html_file}: {e}")
        
        if not documents:
            logger.error("No valid HTML documents found")
            sys.exit(1)
        
        # Process documents
        logger.info(f"Processing {len(documents)} documents...")
        results = pipeline.process_multiple_html(documents, force_basic_cleaning=force_basic_cleaning)
        
        # Calculate statistics
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        # Output results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output}")
        else:
            summary = {
                'total_documents': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(results) if results else 0.0
            }
            click.echo(json.dumps(summary, indent=2))
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        sys.exit(1)


@main.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--output', '-o', help='Output file for results (JSON)')
@click.option('--collection', help='ChromaDB collection name')
@click.option('--require-metadata', is_flag=True, help='Only process files with meta.json')
@click.option('--domain-filter', multiple=True, help='Filter by domain')
@click.option('--year-filter', multiple=True, type=int, help='Filter by year')
@click.option('--min-content-length', type=int, help='Minimum content length')
@click.pass_context
def wayback(ctx, directory, output, collection, require_metadata, domain_filter, year_filter, min_content_length):
    """Process Wayback Machine snapshots from a directory."""
    
    try:
        # Load configuration
        config_path = ctx.obj.get('config_path')
        
        kwargs = {'preset': 'wayback'}
        if collection:
            kwargs['collection_name'] = collection
        
        pipeline = create_pipeline(**kwargs)
        
        # Setup wayback configuration
        from ..core.config import WaybackConfig
        wayback_config = WaybackConfig(
            require_metadata=require_metadata,
            domain_filters=list(domain_filter) if domain_filter else [],
            year_filters=list(year_filter) if year_filter else [],
            min_content_length=min_content_length if min_content_length else 100
        )
        
        logger.info(f"Processing Wayback snapshots from: {directory}")
        
        # Process snapshots
        results = pipeline.process_wayback_snapshots(directory, wayback_config)
        
        # Calculate statistics
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        logger.info(f"Wayback processing completed: {successful} successful, {failed} failed")
        
        # Output results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output}")
        else:
            summary = {
                'total_snapshots': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(results) if results else 0.0
            }
            click.echo(json.dumps(summary, indent=2))
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Wayback processing failed: {str(e)}")
        sys.exit(1)


@main.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Number of results to return')
@click.option('--collection', help='ChromaDB collection name')
@click.option('--threshold', type=float, help='Similarity threshold')
@click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'table']), help='Output format')
@click.option('--metadata-filter', help='Metadata filter (JSON string)')
@click.option('--output', '-o', help='Output file for results')
@click.pass_context
def search(ctx, query, limit, collection, threshold, output_format, metadata_filter, output):
    """Search the vector database for relevant documents."""
    
    try:
        # Load configuration
        config_path = ctx.obj.get('config_path')
        
        kwargs = {}
        if collection:
            kwargs['collection_name'] = collection
        
        if config_path:
            pipeline = create_pipeline(config_path=config_path, **kwargs)
        else:
            pipeline = create_pipeline(**kwargs)
        
        # Parse metadata filter
        metadata_filter_dict = None
        if metadata_filter:
            try:
                metadata_filter_dict = json.loads(metadata_filter)
            except json.JSONDecodeError:
                logger.error("Invalid JSON in metadata filter")
                sys.exit(1)
        
        # Perform search
        logger.info(f"Searching for: '{query}'")
        
        search_kwargs = {
            'n_results': limit,
            'metadata_filter': metadata_filter_dict
        }
        if threshold is not None:
            search_kwargs['similarity_threshold'] = threshold
        
        results = pipeline.search(query, **search_kwargs)
        
        if not results:
            logger.info("No results found")
            return
        
        logger.info(f"Found {len(results)} results")
        
        # Format output
        if output_format == 'table':
            # Simple table output
            click.echo(f"\nSearch results for: '{query}'\n")
            click.echo("-" * 80)
            for i, result in enumerate(results, 1):
                score = result.get('similarity_score', 0)
                text = result.get('text', '')[:100] + '...' if len(result.get('text', '')) > 100 else result.get('text', '')
                url = result.get('metadata', {}).get('url', 'N/A')
                click.echo(f"{i:2d}. Score: {score:.3f} | URL: {url}")
                click.echo(f"    {text}")
                click.echo("-" * 80)
        else:
            # JSON output
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to: {output}")
            else:
                click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--collection', help='ChromaDB collection name')
@click.option('--output', '-o', help='Output file for stats (JSON)')
@click.pass_context
def stats(ctx, collection, output):
    """Show pipeline and database statistics."""
    
    try:
        # Load configuration
        config_path = ctx.obj.get('config_path')
        
        kwargs = {}
        if collection:
            kwargs['collection_name'] = collection
        
        if config_path:
            pipeline = create_pipeline(config_path=config_path, **kwargs)
        else:
            pipeline = create_pipeline(**kwargs)
        
        # Get statistics
        stats_data = pipeline.get_pipeline_stats()
        
        # Output statistics
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Statistics saved to: {output}")
        else:
            click.echo(json.dumps(stats_data, indent=2, ensure_ascii=False))
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--preset', help='Configuration preset to use as base')
@click.option('--output', '-o', required=True, help='Output configuration file path')
@click.option('--collection', help='ChromaDB collection name')
@click.option('--embedding-model', help='Embedding model name')
@click.option('--html-pruner-model', help='HTML pruner model name')
@click.option('--max-chunk-size', type=int, help='Maximum chunk size')
def config(preset, output, collection, embedding_model, html_pruner_model, max_chunk_size):
    """Generate a configuration file."""
    
    try:
        # Start with preset or default config
        if preset:
            config = get_config_preset(preset)
        else:
            config = PipelineConfig()
        
        # Apply overrides
        if collection:
            config.collection_name = collection
        if embedding_model:
            config.embedding_model = embedding_model
        if html_pruner_model:
            config.html_pruner_model = html_pruner_model
        if max_chunk_size:
            config.max_chunk_size = max_chunk_size
        
        # Save configuration
        output_path = Path(output)
        save_config(config, str(output_path))
        
        logger.info(f"Configuration saved to: {output}")
        
        # Show configuration
        click.echo("Generated configuration:")
        click.echo(json.dumps(config.dict(), indent=2))
        
    except Exception as e:
        logger.error(f"Failed to generate configuration: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--collection', help='ChromaDB collection name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def reset(ctx, collection, confirm):
    """Reset the vector database (delete all documents)."""
    
    try:
        if not confirm:
            click.confirm('This will delete all documents in the database. Continue?', abort=True)
        
        # Load configuration
        config_path = ctx.obj.get('config_path')
        
        kwargs = {}
        if collection:
            kwargs['collection_name'] = collection
        
        if config_path:
            pipeline = create_pipeline(config_path=config_path, **kwargs)
        else:
            pipeline = create_pipeline(**kwargs)
        
        # Reset database
        pipeline.reset_database()
        logger.info("Database reset completed")
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Database reset failed: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--collection', help='ChromaDB collection name')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--format', 'export_format', default='json', type=click.Choice(['json', 'csv']), help='Export format')
@click.pass_context
def export(ctx, collection, output, export_format):
    """Export all documents to a file."""
    
    try:
        # Load configuration
        config_path = ctx.obj.get('config_path')
        
        kwargs = {}
        if collection:
            kwargs['collection_name'] = collection
        
        if config_path:
            pipeline = create_pipeline(config_path=config_path, **kwargs)
        else:
            pipeline = create_pipeline(**kwargs)
        
        # Export documents
        pipeline.export_documents(output, export_format)
        logger.info(f"Documents exported to: {output}")
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()