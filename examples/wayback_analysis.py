#!/usr/bin/env python3
"""
Wayback Machine analysis example for HTML RAG Pipeline.

This example demonstrates:
- Wayback-specific processing with metadata preservation
- Temporal analysis of content changes
- Domain-based filtering and analysis
- Archive timeline visualization
- Historical content comparison
- Ukrainian content preservation
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import tempfile
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from html_rag.core.pipeline import create_pipeline
from html_rag.core.config import PipelineConfig, WaybackConfig
from html_rag.utils.logging import setup_logging, PipelineLogger
from html_rag.exceptions.pipeline_exceptions import PipelineError

# Setup logging
setup_logging(level="INFO", log_file="logs/wayback_analysis.log")
logger = PipelineLogger("WaybackAnalysis")


def create_sample_wayback_structure(temp_dir: Path) -> Path:
    """Create a sample Wayback Machine directory structure with metadata."""
    
    # Create wayback directory structure
    wayback_dir = temp_dir / "wayback_snapshots"
    wayback_dir.mkdir(parents=True)
    
    # Sample snapshots with different timestamps and domains
    snapshots = [
        {
            'timestamp': '20201130051600',
            'domain': 'sluga-narodu.com',
            'original_url': 'https://sluga-narodu.com/',
            'title': 'Політична партія "Слуга народу"',
            'html': '''<!DOCTYPE html>
            <html><head><title>Слуга народу</title></head>
            <body>
                <h1>Політична партія "Слуга народу"</h1>
                <p>Ми працюємо для розвитку України та добробуту українського народу.</p>
                <h2>Наші цінності</h2>
                <ul>
                    <li>Демократія</li>
                    <li>Прозорість</li>
                    <li>Відповідальність</li>
                </ul>
                <h3>Програма партії</h3>
                <p>Партія працює над створенням сучасної європейської держави.</p>
            </body></html>'''
        },
        {
            'timestamp': '20210615143000',
            'domain': 'sluga-narodu.com',
            'original_url': 'https://sluga-narodu.com/about',
            'title': 'Про партію - Слуга народу',
            'html': '''<!DOCTYPE html>
            <html><head><title>Про партію</title></head>
            <body>
                <h1>Про партію</h1>
                <p>Політична партія "Слуга народу" була створена для служіння українському народу.</p>
                <h2>Історія</h2>
                <p>Партія заснована у 2018 році з метою реформування політичної системи України.</p>
                <h2>Керівництво</h2>
                <p>Голова партії працює разом з командою професіоналів.</p>
            </body></html>'''
        },
        {
            'timestamp': '20220301120000',
            'domain': 'example.com',
            'original_url': 'https://example.com/tech-news',
            'title': 'Technology News',
            'html': '''<!DOCTYPE html>
            <html><head><title>Tech News</title></head>
            <body>
                <h1>Latest Technology News</h1>
                <p>Stay updated with the latest developments in technology and innovation.</p>
                <h2>Artificial Intelligence</h2>
                <p>AI continues to transform industries with new applications and breakthroughs.</p>
                <h2>Machine Learning</h2>
                <p>Machine learning algorithms are becoming more sophisticated and accessible.</p>
            </body></html>'''
        },
        {
            'timestamp': '20230815090000',
            'domain': 'news-site.com',
            'original_url': 'https://news-site.com/climate',
            'title': 'Climate Research Updates',
            'html': '''<!DOCTYPE html>
            <html><head><title>Climate Research</title></head>
            <body>
                <h1>Climate Change Research</h1>
                <p>Recent studies show the impact of climate change on global ecosystems.</p>
                <h2>Research Findings</h2>
                <p>Scientists are using machine learning to analyze climate data patterns.</p>
                <h2>Future Predictions</h2>
                <p>Models predict significant changes in weather patterns over the next decade.</p>
            </body></html>'''
        },
        {
            'timestamp': '20220801180000',
            'domain': 'sluga-narodu.com',
            'original_url': 'https://sluga-narodu.com/news',
            'title': 'Новини партії',
            'html': '''<!DOCTYPE html>
            <html><head><title>Новини</title></head>
            <body>
                <h1>Останні новини</h1>
                <p>Партія "Слуга народу" продовжує працювати над важливими реформами.</p>
                <h2>Економічні реформи</h2>
                <p>Розроблено нові підходи до економічного розвитку країни.</p>
                <h2>Соціальна політика</h2>
                <p>Приділяємо особливу увагу підтримці соціально незахищених верств населення.</p>
            </body></html>'''
        }
    ]
    
    # Create snapshot files
    for i, snapshot in enumerate(snapshots):
        # Create snapshot directory
        snapshot_dir = wayback_dir / f"snapshot_{i+1:03d}"
        snapshot_dir.mkdir()
        
        # Create HTML file
        html_file = snapshot_dir / "index.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(snapshot['html'])
        
        # Create metadata file
        meta_file = snapshot_dir / "meta.json"
        metadata = {
            'timestamp': snapshot['timestamp'],
            'domain': snapshot['domain'],
            'original_url': snapshot['original_url'],
            'title': snapshot['title'],
            'archive_url': f"https://web.archive.org/web/{snapshot['timestamp']}/{snapshot['original_url']}"
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(snapshots)} sample Wayback snapshots in {wayback_dir}")
    return wayback_dir


def analyze_temporal_distribution(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze temporal distribution of processed snapshots."""
    
    timestamps = []
    domains = []
    years = []
    
    for result in results:
        if result.get('success') and 'metrics' in result:
            # Extract wayback metadata if available
            metadata = result.get('wayback_metadata', {})
            if 'timestamp' in metadata:
                timestamp = metadata['timestamp']
                timestamps.append(timestamp)
                
                # Extract year
                year = int(timestamp[:4])
                years.append(year)
                
                # Extract domain
                domain = metadata.get('domain', 'unknown')
                domains.append(domain)
    
    # Analyze distribution
    year_counts = Counter(years)
    domain_counts = Counter(domains)
    
    analysis = {
        'total_snapshots': len(timestamps),
        'year_distribution': dict(year_counts),
        'domain_distribution': dict(domain_counts),
        'time_range': {
            'earliest': min(timestamps) if timestamps else None,
            'latest': max(timestamps) if timestamps else None
        },
        'unique_domains': len(set(domains)),
        'unique_years': len(set(years))
    }
    
    return analysis


def perform_wayback_specific_searches(pipeline) -> Dict[str, List[Dict[str, Any]]]:
    """Perform searches specific to Wayback content."""
    
    search_results = {}
    
    # Ukrainian political searches
    ukrainian_queries = [
        "партія",
        "політична партія",
        "Слуга народу",
        "українського народу",
        "демократія",
        "реформи",
        "економічні реформи"
    ]
    
    logger.info("🔍 Performing Ukrainian political content searches...")
    for query in ukrainian_queries:
        results = pipeline.search(query, n_results=5)
        search_results[f"ukrainian_{query}"] = results
        
        if results:
            logger.info(f"   '{query}': {len(results)} results, best score: {results[0].get('similarity_score', 0):.3f}")
        else:
            logger.warning(f"   '{query}': No results found")
    
    # Temporal searches
    logger.info("\n🔍 Performing temporal searches...")
    
    # Search by domain
    sluga_results = pipeline.search_wayback_snapshots(
        query="партія",
        domain_filter="sluga-narodu.com",
        n_results=10
    )
    search_results['domain_sluga'] = sluga_results
    logger.info(f"   Domain 'sluga-narodu.com': {len(sluga_results)} results")
    
    # Search by timestamp
    timestamp_results = pipeline.search_wayback_snapshots(
        query="новини",
        timestamp_filter="20220801180000",
        n_results=5
    )
    search_results['timestamp_specific'] = timestamp_results
    logger.info(f"   Specific timestamp: {len(timestamp_results)} results")
    
    # Technology searches
    tech_queries = ["technology", "artificial intelligence", "machine learning"]
    
    logger.info("\n🔍 Performing technology content searches...")
    for query in tech_queries:
        results = pipeline.search(query, n_results=3)
        search_results[f"tech_{query.replace(' ', '_')}"] = results
        
        if results:
            logger.info(f"   '{query}': {len(results)} results")
    
    return search_results


def analyze_content_evolution(search_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze how content evolved over time."""
    
    # Group results by domain and analyze temporal changes
    domain_timeline = defaultdict(list)
    
    for search_key, results in search_results.items():
        for result in results:
            metadata = result.get('metadata', {})
            wayback_timestamp = metadata.get('wayback_timestamp')
            wayback_domain = metadata.get('wayback_domain')
            
            if wayback_timestamp and wayback_domain:
                domain_timeline[wayback_domain].append({
                    'timestamp': wayback_timestamp,
                    'search_key': search_key,
                    'similarity_score': result.get('similarity_score', 0),
                    'text_preview': result.get('text', '')[:100]
                })
    
    # Sort by timestamp for each domain
    for domain in domain_timeline:
        domain_timeline[domain].sort(key=lambda x: x['timestamp'])
    
    # Analyze evolution
    evolution_analysis = {
        'domains_tracked': list(domain_timeline.keys()),
        'timeline_data': dict(domain_timeline),
        'content_changes': {}
    }
    
    # Analyze content changes for each domain
    for domain, timeline in domain_timeline.items():
        if len(timeline) > 1:
            # Compare first and last entries
            first_entry = timeline[0]
            last_entry = timeline[-1]
            
            evolution_analysis['content_changes'][domain] = {
                'first_snapshot': first_entry['timestamp'],
                'last_snapshot': last_entry['timestamp'],
                'time_span_years': (int(last_entry['timestamp'][:4]) - int(first_entry['timestamp'][:4])),
                'snapshot_count': len(timeline)
            }
    
    return evolution_analysis


def create_timeline_visualization_data(
    temporal_analysis: Dict[str, Any],
    search_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Create data for timeline visualization."""
    
    # Extract timeline events
    events = []
    
    # Process search results to create timeline events
    for search_key, results in search_results.items():
        for result in results:
            metadata = result.get('metadata', {})
            wayback_timestamp = metadata.get('wayback_timestamp')
            wayback_domain = metadata.get('wayback_domain')
            wayback_title = metadata.get('wayback_title', 'Unknown Title')
            
            if wayback_timestamp:
                # Parse timestamp
                year = int(wayback_timestamp[:4])
                month = int(wayback_timestamp[4:6])
                day = int(wayback_timestamp[6:8])
                
                events.append({
                    'date': f"{year}-{month:02d}-{day:02d}",
                    'timestamp': wayback_timestamp,
                    'domain': wayback_domain,
                    'title': wayback_title,
                    'search_context': search_key,
                    'similarity_score': result.get('similarity_score', 0),
                    'text_preview': result.get('text', '')[:150]
                })
    
    # Sort events by timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    # Create domain-based timelines
    domain_timelines = defaultdict(list)
    for event in events:
        if event['domain']:
            domain_timelines[event['domain']].append(event)
    
    visualization_data = {
        'all_events': events,
        'domain_timelines': dict(domain_timelines),
        'summary': {
            'total_events': len(events),
            'domains_count': len(domain_timelines),
            'date_range': {
                'start': events[0]['date'] if events else None,
                'end': events[-1]['date'] if events else None
            }
        }
    }
    
    return visualization_data


def main():
    """Wayback Machine analysis demonstration."""
    
    logger.info("="*60)
    logger.info("HTML RAG Pipeline - Wayback Machine Analysis Example")
    logger.info("="*60)
    
    # Create temporary directory for sample Wayback structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Example 1: Create sample Wayback structure
            logger.info("\n1. Creating sample Wayback Machine structure...")
            
            wayback_dir = create_sample_wayback_structure(temp_path)
            logger.info(f"✅ Sample Wayback structure created at {wayback_dir}")
            
            # Example 2: Setup pipeline for Wayback processing
            logger.info("\n2. Setting up pipeline for Wayback processing...")
            
            # Use wayback preset for optimal Ukrainian content processing
            pipeline = create_pipeline(preset="wayback")
            
            # Configure wayback-specific settings
            wayback_config = WaybackConfig(
                require_metadata=True,  # Use metadata files
                force_basic_cleaning=True,  # Preserve Ukrainian content
                min_content_length=50
            )
            
            logger.info("✅ Pipeline configured for Wayback processing with Ukrainian support")
            
            # Example 3: Process Wayback snapshots
            logger.info("\n3. Processing Wayback snapshots...")
            
            results = pipeline.process_wayback_snapshots(
                snapshots_directory=str(wayback_dir),
                wayback_config=wayback_config
            )
            
            # Analyze processing results
            successful = sum(1 for r in results if r.get('success', False))
            failed = len(results) - successful
            
            logger.info(f"✅ Wayback processing completed:")
            logger.info(f"   📄 Total snapshots: {len(results)}")
            logger.info(f"   ✅ Successful: {successful}")
            logger.info(f"   ❌ Failed: {failed}")
            logger.info(f"   📈 Success rate: {successful/len(results):.1%}")
            
            # Example 4: Temporal analysis
            logger.info("\n4. Performing temporal analysis...")
            
            temporal_analysis = analyze_temporal_distribution(results)
            
            logger.info("📊 Temporal Distribution:")
            logger.info(f"   Total snapshots processed: {temporal_analysis['total_snapshots']}")
            logger.info(f"   Unique domains: {temporal_analysis['unique_domains']}")
            logger.info(f"   Unique years: {temporal_analysis['unique_years']}")
            logger.info(f"   Time range: {temporal_analysis['time_range']['earliest']} - {temporal_analysis['time_range']['latest']}")
            
            # Domain distribution
            logger.info("   Domain distribution:")
            for domain, count in temporal_analysis['domain_distribution'].items():
                logger.info(f"     {domain}: {count} snapshots")
            
            # Year distribution
            logger.info("   Year distribution:")
            for year, count in temporal_analysis['year_distribution'].items():
                logger.info(f"     {year}: {count} snapshots")
            
            # Example 5: Wayback-specific searches
            logger.info("\n5. Performing Wayback-specific searches...")
            
            search_results = perform_wayback_specific_searches(pipeline)
            
            # Analyze search success
            total_searches = len(search_results)
            successful_searches = sum(1 for results in search_results.values() if results)
            
            logger.info(f"📊 Search Summary:")
            logger.info(f"   Total searches: {total_searches}")
            logger.info(f"   Successful searches: {successful_searches}")
            logger.info(f"   Search success rate: {successful_searches/total_searches:.1%}")
            
            # Example 6: Content evolution analysis
            logger.info("\n6. Analyzing content evolution...")
            
            evolution_analysis = analyze_content_evolution(search_results)
            
            logger.info("📈 Content Evolution Analysis:")
            logger.info(f"   Domains tracked: {len(evolution_analysis['domains_tracked'])}")
            
            for domain, changes in evolution_analysis['content_changes'].items():
                logger.info(f"   {domain}:")
                logger.info(f"     Time span: {changes['time_span_years']} years")
                logger.info(f"     Snapshots: {changes['snapshot_count']}")
                logger.info(f"     First: {changes['first_snapshot']}")
                logger.info(f"     Last: {changes['last_snapshot']}")
            
            # Example 7: Timeline visualization data
            logger.info("\n7. Creating timeline visualization data...")
            
            timeline_data = create_timeline_visualization_data(temporal_analysis, search_results)
            
            logger.info("📅 Timeline Visualization:")
            logger.info(f"   Total events: {timeline_data['summary']['total_events']}")
            logger.info(f"   Domains in timeline: {timeline_data['summary']['domains_count']}")
            logger.info(f"   Date range: {timeline_data['summary']['date_range']['start']} to {timeline_data['summary']['date_range']['end']}")
            
            # Example 8: Ukrainian content preservation validation
            logger.info("\n8. Validating Ukrainian content preservation...")
            
            ukrainian_test_queries = ["партія", "політична", "українського", "демократія"]
            ukrainian_validation = {}
            
            for query in ukrainian_test_queries:
                results = pipeline.search(query, n_results=3)
                if results:
                    best_score = results[0].get('similarity_score', 0)
                    ukrainian_validation[query] = {
                        'found': True,
                        'best_score': best_score,
                        'result_count': len(results)
                    }
                    logger.info(f"   ✅ '{query}': Found {len(results)} results, best score: {best_score:.3f}")
                else:
                    ukrainian_validation[query] = {'found': False}
                    logger.warning(f"   ❌ '{query}': No results found")
            
            # Calculate preservation success rate
            preserved_terms = sum(1 for v in ukrainian_validation.values() if v.get('found'))
            preservation_rate = preserved_terms / len(ukrainian_test_queries)
            
            logger.info(f"📊 Ukrainian Content Preservation: {preservation_rate:.1%} ({preserved_terms}/{len(ukrainian_test_queries)} terms)")
            
            # Example 9: Export comprehensive analysis
            logger.info("\n9. Exporting comprehensive Wayback analysis...")
            
            comprehensive_analysis = {
                'processing_results': results,
                'temporal_analysis': temporal_analysis,
                'search_results': search_results,
                'evolution_analysis': evolution_analysis,
                'timeline_data': timeline_data,
                'ukrainian_validation': ukrainian_validation,
                'metadata': {
                    'total_snapshots': len(results),
                    'successful_processing': successful,
                    'preservation_rate': preservation_rate,
                    'analysis_timestamp': str(Path().cwd())
                }
            }
            
            # Export to file
            export_file = Path("wayback_analysis_results.json")
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📁 Comprehensive analysis exported to {export_file}")
            
            # Example 10: Generate summary report
            logger.info("\n10. Generating summary report...")
            
            logger.info("📋 Wayback Analysis Summary Report:")
            logger.info(f"   📄 Snapshots processed: {len(results)}")
            logger.info(f"   ✅ Processing success rate: {successful/len(results):.1%}")
            logger.info(f"   🌐 Unique domains: {temporal_analysis['unique_domains']}")
            logger.info(f"   📅 Time span: {temporal_analysis['unique_years']} years")
            logger.info(f"   🔍 Searches performed: {total_searches}")
            logger.info(f"   📊 Search success rate: {successful_searches/total_searches:.1%}")
            logger.info(f"   🇺🇦 Ukrainian preservation: {preservation_rate:.1%}")
            
            # Pipeline statistics
            stats = pipeline.get_pipeline_stats()
            if stats and 'vector_store' in stats:
                doc_count = stats['vector_store'].get('document_count', 0)
                logger.info(f"   🗃️  Documents in database: {doc_count}")
            
            # Clean up
            logger.info("\n11. Cleaning up...")
            pipeline.cleanup()
            logger.info("✅ Cleanup completed")
            
            logger.info("\n" + "="*60)
            logger.info("Wayback Machine analysis completed successfully! 🎉")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Wayback analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)