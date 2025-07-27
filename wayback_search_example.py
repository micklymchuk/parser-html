#!/usr/bin/env python3
"""
Wayback HTML Database Search Tool

This script searches the vector database created by wayback_processor_example.py
and provides various search capabilities including Ukrainian content search.

Usage:
    python wayback_search_example.py "search query"
    python wayback_search_example.py "партія" --ukrainian
    python wayback_search_example.py "machine learning" --domain example.com --year 2020
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.html_rag import create_pipeline
from src.html_rag.core.config import PipelineConfig
from src.html_rag.utils.logging import setup_logging, PipelineLogger

# Setup logging
setup_logging(level="WARNING", enable_console=False)  # Quiet for search
logger = PipelineLogger("WaybackSearcher")


class WaybackSearcher:
    """Search interface for Wayback HTML database."""
    
    def __init__(self, collection_name: str = "wayback_html_files", persist_directory: str = "./wayback_html_db"):
        """
        Initialize the searcher.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory containing the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Configure pipeline for searching
        config = PipelineConfig(
            collection_name=collection_name,
            persist_directory=persist_directory,
            prefer_basic_cleaning=True,  # Same as processing
            enable_metrics=False,  # Disable for faster search
            log_level="WARNING"  # Quiet logging
        )
        
        self.pipeline = create_pipeline(config=config)
        
        # Verify database exists and has content
        stats = self.pipeline.get_pipeline_stats()
        doc_count = stats.get('vector_store', {}).get('document_count', 0)
        
        if doc_count == 0:
            raise ValueError(
                f"No documents found in database!\n"
                f"Collection: {collection_name}\n"
                f"Directory: {persist_directory}\n"
                f"Please run wayback_processor_example.py first to populate the database."
            )
        
        print(f"🔍 Connected to database with {doc_count:,} documents")
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        domain_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        similarity_threshold: float = 0.0,
        show_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search the database.
        
        Args:
            query: Search query
            n_results: Number of results to return
            domain_filter: Filter by domain (e.g., "sluga-narodu.com")
            year_filter: Filter by year (e.g., 2020)
            similarity_threshold: Minimum similarity score
            show_metadata: Include full metadata in results
            
        Returns:
            List of search results
        """
        # Build metadata filter
        metadata_filter = {}
        
        if domain_filter:
            metadata_filter['wayback_domain'] = domain_filter
        
        if year_filter:
            metadata_filter['year'] = year_filter
        
        # Perform search
        try:
            if metadata_filter:
                results = self.pipeline.search_wayback_snapshots(
                    query=query,
                    n_results=n_results,
                    domain_filter=domain_filter,
                    # Note: search_wayback_snapshots doesn't have year filter, so we'll filter manually
                )
                
                # Manual year filtering if needed
                if year_filter and results:
                    results = [
                        r for r in results 
                        if r.get('metadata', {}).get('year') == year_filter
                    ][:n_results]
            else:
                results = self.pipeline.search(
                    query=query,
                    n_results=n_results,
                    similarity_threshold=similarity_threshold
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_ukrainian_content(
        self,
        query: str,
        n_results: int = 10,
        domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for Ukrainian content.
        
        Args:
            query: Ukrainian search query
            n_results: Number of results
            domain_filter: Optional domain filter
            
        Returns:
            Search results optimized for Ukrainian content
        """
        print(f"🇺🇦 Searching Ukrainian content for: '{query}'")
        
        # Ukrainian-specific search with lower threshold
        return self.search(
            query=query,
            n_results=n_results,
            domain_filter=domain_filter,
            similarity_threshold=0.0,  # More permissive for Ukrainian
            show_metadata=True
        )
    
    def browse_by_domain(self, domain: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Browse documents by domain.
        
        Args:
            domain: Domain to browse
            limit: Maximum number of documents
            
        Returns:
            Documents from the specified domain
        """
        print(f"🌐 Browsing domain: {domain}")
        
        try:
            results = self.pipeline.search_by_metadata(
                metadata_filter={'wayback_domain': domain},
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Browse failed: {e}")
            return []
    
    def browse_by_year(self, year: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Browse documents by year.
        
        Args:
            year: Year to browse
            limit: Maximum number of documents
            
        Returns:
            Documents from the specified year
        """
        print(f"📅 Browsing year: {year}")
        
        try:
            results = self.pipeline.search_by_metadata(
                metadata_filter={'year': year},
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Browse failed: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Database statistics
        """
        stats = self.pipeline.get_pipeline_stats()
        
        # Get additional stats by querying metadata
        try:
            # Get all documents to analyze
            all_docs = self.pipeline.search_by_metadata({}, limit=10000)
            
            # Analyze domains
            domains = {}
            years = {}
            
            for doc in all_docs:
                metadata = doc.get('metadata', {})
                
                domain = metadata.get('wayback_domain', 'unknown')
                domains[domain] = domains.get(domain, 0) + 1
                
                year = metadata.get('year')
                if year:
                    years[year] = years.get(year, 0) + 1
            
            return {
                'total_documents': len(all_docs),
                'unique_domains': len(domains),
                'unique_years': len(years),
                'top_domains': sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10],
                'years_coverage': sorted(years.items()),
                'database_info': stats.get('vector_store', {}),
                'embedding_model': stats.get('embedding_model', {}).get('model_name', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Stats collection failed: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        self.pipeline.cleanup()


def format_search_results(results: List[Dict[str, Any]], show_metadata: bool = False, max_text_length: int = 200) -> None:
    """
    Format and display search results.
    
    Args:
        results: Search results
        show_metadata: Whether to show metadata
        max_text_length: Maximum text length to display
    """
    if not results:
        print("❌ No results found")
        return
    
    print(f"\n📊 Found {len(results)} results:")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        score = result.get('similarity_score', 0)
        text = result.get('text', '')
        metadata = result.get('metadata', {})
        
        # Truncate text
        display_text = text[:max_text_length] + "..." if len(text) > max_text_length else text
        
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Text: {display_text}")
        
        if show_metadata and metadata:
            # Show key metadata
            url = metadata.get('url', 'N/A')
            domain = metadata.get('wayback_domain', 'N/A')
            year = metadata.get('year', 'N/A')
            timestamp = metadata.get('wayback_timestamp', 'N/A')
            
            print(f"   URL: {url}")
            print(f"   Domain: {domain}")
            print(f"   Year: {year}")
            print(f"   Timestamp: {timestamp}")
        
        print("-" * 40)


def interactive_search_session(searcher: WaybackSearcher) -> None:
    """
    Run an interactive search session.
    
    Args:
        searcher: WaybackSearcher instance
    """
    print("\n🔍 Interactive Search Session")
    print("Commands:")
    print("  search <query>                 - Basic search")
    print("  ukrainian <query>              - Ukrainian content search")
    print("  domain <domain>                - Browse by domain")
    print("  year <year>                    - Browse by year")
    print("  stats                          - Show database statistics")
    print("  help                           - Show this help")
    print("  quit                           - Exit")
    print("\nExamples:")
    print("  search machine learning")
    print("  ukrainian партія")
    print("  domain sluga-narodu.com")
    print("  year 2020")
    
    while True:
        try:
            command = input("\n🔍 Enter command: ").strip()
            
            if not command or command.lower() == 'quit':
                break
            
            parts = command.split(' ', 1)
            cmd = parts[0].lower()
            
            if cmd == 'help':
                print("Available commands: search, ukrainian, domain, year, stats, help, quit")
                continue
            
            if cmd == 'stats':
                print("📊 Collecting database statistics...")
                stats = searcher.get_database_stats()
                
                if 'error' in stats:
                    print(f"❌ Error: {stats['error']}")
                else:
                    print(f"\n📊 Database Statistics:")
                    print(f"   Total documents: {stats['total_documents']:,}")
                    print(f"   Unique domains: {stats['unique_domains']}")
                    print(f"   Unique years: {stats['unique_years']}")
                    print(f"   Years range: {min(y for y, _ in stats['years_coverage'])} - {max(y for y, _ in stats['years_coverage'])}")
                    
                    print(f"\n🌐 Top domains:")
                    for domain, count in stats['top_domains'][:5]:
                        print(f"   {domain}: {count:,} documents")
                
                continue
            
            if len(parts) < 2:
                print("❌ Please provide a query/parameter")
                continue
            
            query_or_param = parts[1]
            
            if cmd == 'search':
                results = searcher.search(query_or_param, n_results=5)
                format_search_results(results, show_metadata=True)
            
            elif cmd == 'ukrainian':
                results = searcher.search_ukrainian_content(query_or_param, n_results=5)
                format_search_results(results, show_metadata=True)
            
            elif cmd == 'domain':
                results = searcher.browse_by_domain(query_or_param, limit=10)
                format_search_results(results, show_metadata=True)
            
            elif cmd == 'year':
                try:
                    year = int(query_or_param)
                    results = searcher.browse_by_year(year, limit=10)
                    format_search_results(results, show_metadata=True)
                except ValueError:
                    print("❌ Invalid year format")
            
            else:
                print(f"❌ Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function for the search tool."""
    parser = argparse.ArgumentParser(
        description="Search Wayback HTML database created by wayback_processor_example.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python wayback_search_example.py "machine learning"
    python wayback_search_example.py "партія" --ukrainian
    python wayback_search_example.py "politics" --domain sluga-narodu.com
    python wayback_search_example.py "technology" --year 2020 --limit 5
    python wayback_search_example.py --interactive
    python wayback_search_example.py --stats
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Search query (optional if using --interactive or --stats)'
    )
    parser.add_argument(
        '--collection',
        default='wayback_html_files',
        help='ChromaDB collection name (default: wayback_html_files)'
    )
    parser.add_argument(
        '--db-dir',
        default='./wayback_html_db',
        help='Database directory (default: ./wayback_html_db)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    parser.add_argument(
        '--domain',
        help='Filter by domain (e.g., sluga-narodu.com)'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Filter by year (e.g., 2020)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Minimum similarity threshold (default: 0.0)'
    )
    parser.add_argument(
        '--ukrainian',
        action='store_true',
        help='Optimize search for Ukrainian content'
    )
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Show full metadata in results'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive search session'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher
        print(f"🔗 Connecting to database...")
        searcher = WaybackSearcher(
            collection_name=args.collection,
            persist_directory=args.db_dir
        )
        
        try:
            # Handle different modes
            if args.stats:
                print("📊 Collecting database statistics...")
                stats = searcher.get_database_stats()
                
                if args.json:
                    print(json.dumps(stats, indent=2, ensure_ascii=False))
                else:
                    if 'error' in stats:
                        print(f"❌ Error: {stats['error']}")
                    else:
                        print(f"\n📊 Database Statistics:")
                        print(f"   Total documents: {stats['total_documents']:,}")
                        print(f"   Unique domains: {stats['unique_domains']}")
                        print(f"   Unique years: {stats['unique_years']}")
                        
                        if stats['years_coverage']:
                            years = [y for y, _ in stats['years_coverage']]
                            print(f"   Years range: {min(years)} - {max(years)}")
                        
                        print(f"\n🌐 Top domains:")
                        for domain, count in stats['top_domains'][:10]:
                            print(f"   {domain}: {count:,} documents")
                        
                        print(f"\n📅 Year distribution:")
                        for year, count in stats['years_coverage']:
                            print(f"   {year}: {count:,} documents")
            
            elif args.interactive:
                interactive_search_session(searcher)
            
            elif args.query:
                # Perform search
                if args.ukrainian:
                    results = searcher.search_ukrainian_content(
                        args.query,
                        n_results=args.limit,
                        domain_filter=args.domain
                    )
                else:
                    results = searcher.search(
                        query=args.query,
                        n_results=args.limit,
                        domain_filter=args.domain,
                        year_filter=args.year,
                        similarity_threshold=args.threshold,
                        show_metadata=args.metadata
                    )
                
                if args.json:
                    print(json.dumps(results, indent=2, ensure_ascii=False))
                else:
                    print(f"🔍 Searching for: '{args.query}'")
                    if args.domain:
                        print(f"   Domain filter: {args.domain}")
                    if args.year:
                        print(f"   Year filter: {args.year}")
                    if args.ukrainian:
                        print(f"   Mode: Ukrainian content optimization")
                    
                    format_search_results(results, show_metadata=args.metadata)
            
            else:
                print("❌ Please provide a query, use --interactive, or --stats")
                return 1
        
        finally:
            searcher.cleanup()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())