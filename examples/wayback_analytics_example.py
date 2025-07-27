#!/usr/bin/env python3
"""
Wayback Snapshots Content Analytics Example

This example demonstrates a complete workflow for analyzing Wayback Machine snapshots:
1. Process snapshots from a specific folder (20201130051600)
2. Store them in ChromaDB with content analytics
3. Query and analyze the stored data
4. Search for controversial content about specific people/entities
5. Generate analytics reports and insights

This script shows real-world usage for analyzing historical Ukrainian web content.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.html_rag.core.pipeline import RAGPipeline
from src.html_rag.core.config import PipelineConfig, ContentAnalyticsConfig, WaybackConfig


def setup_logging():
    """Set up detailed logging for the analysis process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'wayback_analytics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def create_analytics_pipeline() -> RAGPipeline:
    """Create a specialized pipeline for Wayback snapshot analysis."""
    print("🚀 Setting up Wayback Analytics Pipeline...")
    
    # Configure content analytics with optimized settings for historical content
    analytics_config = ContentAnalyticsConfig(
        enabled=True,
        controversy_threshold=0.5,  # Lower threshold for historical analysis
        max_entities_per_document=100,  # More entities for comprehensive analysis
        sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        classification_model="facebook/bart-large-mnli",
        batch_size=8,  # Smaller batches for memory efficiency
        cache_results=True
        # cache_ttl=7200,  # 2 hours cache for repeated analysis
        # min_text_length=50,  # Analyze even shorter content
        # language_detection_threshold=0.7  # More lenient for historical content
    )
    
    # Configure pipeline for Ukrainian/historical content
    pipeline_config = PipelineConfig(
        collection_name="wayback_20201130051600",  # Specific collection for this dataset
        prefer_basic_cleaning=True,  # Better for Ukrainian/non-English content
        enable_metrics=True,
        max_chunk_size=2000,  # Smaller chunks for better analysis granularity
        cyrillic_detection_threshold=0.3  # Sensitive Cyrillic detection
    )
    
    # Configure Wayback processing
    wayback_config = WaybackConfig(
        require_metadata=True,
        force_basic_cleaning=True,  # Preserve Ukrainian content
        min_content_length=100  # Filter very short content
        # domain_filters=None,  # Process all domains
        # year_filters=None  # Process all years in folder
    )
    
    # Create pipeline with analytics enabled
    pipeline = RAGPipeline(
        config=pipeline_config,
        enable_content_analytics=True,
        analytics_config=analytics_config
    )
    
    print("✅ Pipeline created successfully!")
    print(f"   📁 Collection: {pipeline_config.collection_name}")
    print(f"   🧠 Analytics enabled with {analytics_config.max_entities_per_document} max entities")
    print(f"   🔍 Controversy threshold: {analytics_config.controversy_threshold}")
    
    return pipeline, wayback_config


def validate_snapshots_folder(folder_path: str) -> bool:
    """Validate that the Wayback snapshots folder exists and contains data."""
    print(f"\n📂 Validating snapshots folder: {folder_path}")
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder does not exist: {folder_path}")
        print("   Please ensure you have the Wayback snapshots folder '20201130051600'")
        return False
    
    if not folder.is_dir():
        print(f"❌ Path is not a directory: {folder_path}")
        return False
    
    # Check for typical Wayback snapshot files
    snapshot_files = list(folder.glob("*.html")) + list(folder.glob("*.htm"))
    metadata_files = list(folder.glob("*.json")) + list(folder.glob("*.meta"))
    
    print(f"   📄 Found {len(snapshot_files)} HTML snapshot files")
    print(f"   📋 Found {len(metadata_files)} metadata files")
    
    if len(snapshot_files) == 0:
        print("⚠️  No HTML snapshot files found")
        return False
    
    print("✅ Folder validation successful!")
    return True


def process_wayback_snapshots(pipeline: RAGPipeline, wayback_config: WaybackConfig, 
                             folder_path: str) -> List[Dict[str, Any]]:
    """Process all Wayback snapshots from the specified folder."""
    print(f"\n🔄 Processing Wayback snapshots from: {folder_path}")
    print("=" * 60)
    
    try:
        # Validate the folder first
        validation_result = pipeline.validate_wayback_directory(folder_path)
        if not validation_result.get('is_valid'):
            print(f"❌ Folder validation failed: {validation_result.get('errors')}")
            return []
        
        print(f"✅ Folder validation passed")
        print(f"   📊 {validation_result.get('snapshot_count', 0)} snapshots found")
        
        # Process snapshots with analytics
        print("\n🚀 Starting snapshot processing...")
        results = pipeline.process_wayback_snapshots(
            snapshots_directory=folder_path,
            wayback_config=wayback_config
        )

        # Analyze results
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        total_analytics = sum(1 for r in results if r.get('content_analytics'))
        
        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 With Analytics: {total_analytics}")
        print(f"   📁 Total Documents: {len(results)}")
        print(results)

        # Show some sample analytics results
        analytics_samples = [r for r in results if r.get('content_analytics')][:3]
        if analytics_samples:
            print(f"\n🎯 Sample Analytics Results:")
            for i, sample in enumerate(analytics_samples, 1):
                analytics = sample['content_analytics']
                print(f"   Document {i}:")
                print(f"     URL: {sample.get('url', 'Unknown')[:60]}...")
                print(f"     Language: {analytics.get('language', 'Unknown')}")
                print(f"     Sentiment: {analytics['sentiment']['label']} ({analytics['sentiment']['score']:.2f})")
                print(f"     Controversy: {analytics['controversy']['level']} ({analytics['controversy']['score']:.2f})")
                print(f"     Topic: {analytics['topics']['primary_topic']}")
                print(f"     Entities: {len(analytics['entities'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing snapshots: {e}")
        import traceback
        traceback.print_exc()
        return []


def analyze_stored_data(pipeline: RAGPipeline) -> Dict[str, Any]:
    """Analyze the stored data and generate comprehensive insights."""
    print(f"\n📊 Analyzing Stored Data")
    print("=" * 40)
    
    try:
        # Get overall analytics summary
        summary = pipeline.get_analytics_summary()
        
        if 'total_documents_analyzed' not in summary:
            print("❌ No analytics data found in the database")
            return {}
        
        print(f"📈 Analytics Summary:")
        print(f"   📄 Total Documents: {summary['total_documents_analyzed']}")
        print(f"   💭 Avg Sentiment: {summary.get('average_sentiment_score', 0):.3f}")
        print(f"   ⚠️  Avg Controversy: {summary.get('average_controversy_score', 0):.3f}")
        print(f"   ⭐ Avg Quality: {summary.get('average_quality_score', 0):.3f}")
        
        # Sentiment distribution
        sentiment_dist = summary.get('sentiment_distribution', {})
        print(f"\n😊 Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / summary['total_documents_analyzed']) * 100
            print(f"   {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Controversy distribution
        controversy_dist = summary.get('controversy_distribution', {})
        print(f"\n🔥 Controversy Distribution:")
        for level, count in controversy_dist.items():
            percentage = (count / summary['total_documents_analyzed']) * 100
            print(f"   {level.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Topic distribution
        topic_dist = summary.get('topic_distribution', {})
        print(f"\n🏷️  Top Topics:")
        for i, (topic, count) in enumerate(list(topic_dist.items())[:10], 1):
            percentage = (count / summary['total_documents_analyzed']) * 100
            print(f"   {i}. {topic}: {count} ({percentage:.1f}%)")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return {}


def search_controversial_content_about_person(pipeline: RAGPipeline, person_name: str) -> List[Dict[str, Any]]:
    """Search for controversial content mentioning a specific person."""
    print(f"\n🔍 Searching for controversial content about: '{person_name}'")
    print("-" * 50)
    
    try:
        # Search 1: Direct name search with controversy filter
        controversial_mentions = pipeline.search(
            query=person_name,
            n_results=20,
            metadata_filter={'controversy_level': 'high'}
        )
        
        print(f"🔥 High controversy mentions: {len(controversial_mentions)}")
        
        # Search 2: Medium + High controversy
        all_controversial = pipeline.search(
            query=person_name,
            n_results=50
        )
        
        # Filter for medium+ controversy
        medium_high_controversy = [
            doc for doc in all_controversial 
            if doc.get('metadata', {}).get('controversy_level') in ['medium', 'high', 'critical']
        ]
        
        print(f"⚠️  Medium+ controversy mentions: {len(medium_high_controversy)}")
        
        # Search 3: Negative sentiment mentions
        negative_mentions = pipeline.search(
            query=person_name,
            n_results=30,
            metadata_filter={'sentiment_label': 'negative'}
        )
        
        print(f"😞 Negative sentiment mentions: {len(negative_mentions)}")
        
        # Combine and analyze results
        all_results = controversial_mentions + medium_high_controversy + negative_mentions
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for doc in all_results:
            url = doc.get('metadata', {}).get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(doc)
        
        print(f"📋 Total unique controversial documents: {len(unique_results)}")
        
        # Show detailed analysis of top results
        if unique_results:
            print(f"\n🎯 Top Controversial Mentions of '{person_name}':")
            
            for i, doc in enumerate(unique_results[:5], 1):
                metadata = doc.get('metadata', {})
                print(f"\n   [{i}] Document Analysis:")
                print(f"       URL: {metadata.get('url', 'Unknown')}")
                print(f"       Wayback Date: {metadata.get('wayback_timestamp', 'Unknown')}")
                print(f"       Domain: {metadata.get('wayback_domain', 'Unknown')}")
                print(f"       Sentiment: {metadata.get('sentiment_label', 'Unknown')} "
                      f"({metadata.get('sentiment_score', 0):.2f})")
                print(f"       Controversy: {metadata.get('controversy_level', 'Unknown')} "
                      f"({metadata.get('controversy_score', 0):.2f})")
                print(f"       Topic: {metadata.get('primary_topic', 'Unknown')}")
                print(f"       Quality: {metadata.get('quality_score', 0):.2f}")
                print(f"       Similarity: {doc.get('similarity_score', 0):.3f}")
                
                # Show text snippet
                text = doc.get('text', '')
                # Find the person's name in the text for context
                name_lower = person_name.lower()
                text_lower = text.lower()
                if name_lower in text_lower:
                    start_pos = text_lower.find(name_lower)
                    snippet_start = max(0, start_pos - 100)
                    snippet_end = min(len(text), start_pos + len(person_name) + 100)
                    snippet = text[snippet_start:snippet_end]
                    print(f"       Context: ...{snippet}...")
                else:
                    snippet = text[:200]
                    print(f"       Snippet: {snippet}...")
        
        return unique_results
        
    except Exception as e:
        print(f"❌ Error searching for controversial content: {e}")
        return []


def search_by_topic_and_controversy(pipeline: RAGPipeline, topic: str, min_controversy: str = 'medium') -> List[Dict[str, Any]]:
    """Search for content by topic with controversy filtering."""
    print(f"\n🏷️  Searching for {min_controversy}+ controversial content in topic: '{topic}'")
    print("-" * 60)
    
    try:
        # Get controversy levels to include
        controversy_levels = {
            'low': ['low', 'medium', 'high', 'critical'],
            'medium': ['medium', 'high', 'critical'],
            'high': ['high', 'critical'],
            'critical': ['critical']
        }
        
        levels_to_include = controversy_levels.get(min_controversy, ['medium', 'high', 'critical'])
        
        results = []
        for level in levels_to_include:
            level_results = pipeline.search_by_topic(topic, n_results=20)
            # Filter by controversy level
            filtered_results = [
                doc for doc in level_results
                if doc.get('metadata', {}).get('controversy_level') == level
            ]
            results.extend(filtered_results)
        
        # Remove duplicates
        seen_urls = set()
        unique_results = []
        for doc in results:
            url = doc.get('metadata', {}).get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(doc)
        
        print(f"📊 Found {len(unique_results)} {min_controversy}+ controversial documents in '{topic}' topic")
        
        if unique_results:
            # Group by controversy level
            by_level = {}
            for doc in unique_results:
                level = doc.get('metadata', {}).get('controversy_level', 'unknown')
                if level not in by_level:
                    by_level[level] = []
                by_level[level].append(doc)
            
            print(f"\n📈 Breakdown by controversy level:")
            for level in ['critical', 'high', 'medium', 'low']:
                if level in by_level:
                    count = len(by_level[level])
                    print(f"   {level.capitalize()}: {count} documents")
            
            # Show top results
            print(f"\n🎯 Top Results:")
            sorted_results = sorted(unique_results, 
                                  key=lambda x: x.get('metadata', {}).get('controversy_score', 0), 
                                  reverse=True)
            
            for i, doc in enumerate(sorted_results[:3], 1):
                metadata = doc.get('metadata', {})
                print(f"\n   [{i}] High Controversy in {topic.capitalize()}:")
                print(f"       URL: {metadata.get('url', 'Unknown')}")
                print(f"       Controversy: {metadata.get('controversy_level', 'Unknown')} "
                      f"({metadata.get('controversy_score', 0):.2f})")
                print(f"       Sentiment: {metadata.get('sentiment_label', 'Unknown')}")
                print(f"       Entities: {metadata.get('entity_count', 0)}")
                
                text_snippet = doc.get('text', '')[:150]
                print(f"       Snippet: {text_snippet}...")
        
        return unique_results
        
    except Exception as e:
        print(f"❌ Error searching by topic and controversy: {e}")
        return []


def generate_analytics_report(pipeline: RAGPipeline, summary: Dict[str, Any]) -> None:
    """Generate a comprehensive analytics report."""
    print(f"\n📋 Generating Comprehensive Analytics Report")
    print("=" * 50)
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_filename = f"wayback_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Get pipeline statistics
        pipeline_stats = pipeline.get_pipeline_stats()
        
        # Gather additional insights
        insights = {
            "report_metadata": {
                "generated_at": timestamp,
                "wayback_folder": "20201130051600",
                "pipeline_version": pipeline_stats.get('pipeline_version', 'Unknown')
            },
            "collection_stats": pipeline_stats.get('vector_store', {}),
            "analytics_summary": summary,
            "content_insights": {},
            "controversy_analysis": {},
            "entity_analysis": {},
            "topic_analysis": {}
        }
        
        # Content insights
        if summary.get('total_documents_analyzed', 0) > 0:
            total_docs = summary['total_documents_analyzed']
            
            # Controversy insights
            controversy_dist = summary.get('controversy_distribution', {})
            high_controversy_count = controversy_dist.get('high', 0) + controversy_dist.get('critical', 0)
            controversy_percentage = (high_controversy_count / total_docs) * 100
            
            insights["controversy_analysis"] = {
                "high_controversy_documents": high_controversy_count,
                "controversy_percentage": round(controversy_percentage, 2),
                "most_controversial_level": max(controversy_dist.keys(), key=lambda k: controversy_dist[k]) if controversy_dist else "unknown"
            }
            
            # Sentiment insights
            sentiment_dist = summary.get('sentiment_distribution', {})
            negative_count = sentiment_dist.get('negative', 0)
            negative_percentage = (negative_count / total_docs) * 100
            
            insights["content_insights"] = {
                "negative_sentiment_documents": negative_count,
                "negative_percentage": round(negative_percentage, 2),
                "dominant_sentiment": max(sentiment_dist.keys(), key=lambda k: sentiment_dist[k]) if sentiment_dist else "unknown",
                "average_quality_score": round(summary.get('average_quality_score', 0), 3)
            }
            
            # Topic insights
            topic_dist = summary.get('topic_distribution', {})
            insights["topic_analysis"] = {
                "most_common_topic": max(topic_dist.keys(), key=lambda k: topic_dist[k]) if topic_dist else "unknown",
                "topic_diversity": len(topic_dist),
                "top_5_topics": dict(list(topic_dist.items())[:5])
            }
        
        # Save report
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Analytics report saved to: {report_filename}")
        
        # Print key insights
        print(f"\n🎯 Key Insights:")
        if insights["controversy_analysis"]:
            print(f"   🔥 High Controversy: {insights['controversy_analysis']['controversy_percentage']}% of documents")
        if insights["content_insights"]:
            print(f"   😞 Negative Sentiment: {insights['content_insights']['negative_percentage']}% of documents")
            print(f"   ⭐ Average Quality: {insights['content_insights']['average_quality_score']}")
        if insights["topic_analysis"]:
            print(f"   🏷️  Most Common Topic: {insights['topic_analysis']['most_common_topic']}")
            print(f"   📚 Topic Diversity: {insights['topic_analysis']['topic_diversity']} different topics")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")


def main():
    """Main function orchestrating the complete Wayback analytics workflow."""
    print("🎯 Wayback Snapshots Content Analytics")
    print("=" * 60)
    print("This script will:")
    print("1. 📂 Process Wayback snapshots from folder '20201130051600'")
    print("2. 🧠 Perform content analytics on all documents")
    print("3. 💾 Store everything in ChromaDB")
    print("4. 🔍 Query and analyze the stored data")
    print("5. 📊 Generate comprehensive reports")
    print("=" * 60)
    
    setup_logging()
    
    # Configuration
    SNAPSHOTS_FOLDER = "../20201130051600"  # The specific Wayback folder to process
    
    # People to search for controversial content (Ukrainian politicians/figures)
    PEOPLE_TO_ANALYZE = [
        "Зеленський",
    ]
    
    # Topics to analyze for controversy
    TOPICS_TO_ANALYZE = ["politics", "economics", "social"]
    
    try:
        # Step 1: Validate snapshots folder
        if not validate_snapshots_folder(SNAPSHOTS_FOLDER):
            print("\n❌ Cannot proceed without valid snapshots folder")
            print("Please ensure you have the '20201130051600' folder with Wayback snapshots")
            return
        
        # Step 2: Create analytics pipeline
        pipeline, wayback_config = create_analytics_pipeline()
        
        # Step 3: Process Wayback snapshots
        processing_results = process_wayback_snapshots(pipeline, wayback_config, SNAPSHOTS_FOLDER)
        
        if not processing_results:
            print("❌ No documents were processed successfully")
            return
        
        # Step 4: Analyze stored data
        analytics_summary = analyze_stored_data(pipeline)
        
        if not analytics_summary:
            print("❌ No analytics data available for analysis")
            return
        
        # Step 5: Search for controversial content about specific people
        print(f"\n👥 Analyzing Controversial Content About Specific People")
        print("=" * 60)
        
        all_controversial_results = {}
        for person in PEOPLE_TO_ANALYZE:
            print(f"\n🔍 Analyzing: {person}")
            controversial_docs = search_controversial_content_about_person(pipeline, person)
            if controversial_docs:
                all_controversial_results[person] = controversial_docs
                print(f"   📊 Found {len(controversial_docs)} controversial mentions")
            else:
                print(f"   ℹ️  No controversial content found")
        
        # Step 6: Search by topic and controversy
        print(f"\n🏷️  Analyzing Controversial Content by Topic")
        print("=" * 50)
        
        topic_controversy_results = {}
        for topic in TOPICS_TO_ANALYZE:
            print(f"\n📂 Analyzing topic: {topic}")
            topic_docs = search_by_topic_and_controversy(pipeline, topic, min_controversy='medium')
            if topic_docs:
                topic_controversy_results[topic] = topic_docs
        
        # Step 7: Generate comprehensive report
        generate_analytics_report(pipeline, analytics_summary)
        
        # Step 8: Summary and recommendations
        print(f"\n🎉 Analysis Complete!")
        print("=" * 30)
        
        total_processed = len(processing_results)
        total_with_analytics = sum(1 for r in processing_results if r.get('content_analytics'))
        total_controversial_people = sum(len(docs) for docs in all_controversial_results.values())
        
        print(f"📊 Final Summary:")
        print(f"   📄 Documents Processed: {total_processed}")
        print(f"   🧠 Documents with Analytics: {total_with_analytics}")
        print(f"   👥 Controversial People Mentions: {total_controversial_people}")
        print(f"   🏷️  Topics Analyzed: {len(TOPICS_TO_ANALYZE)}")
        
        if all_controversial_results:
            print(f"\n🔥 Most Controversial People Found:")
            sorted_people = sorted(all_controversial_results.items(), 
                                 key=lambda x: len(x[1]), reverse=True)
            for person, docs in sorted_people[:3]:
                print(f"   {person}: {len(docs)} controversial documents")
        
        # Usage examples
        print(f"\n💡 Usage Examples:")
        print(f"   # Search for more people:")
        print(f"   controversial_docs = pipeline.search('Клітковський', metadata_filter={{'controversy_level': 'high'}})")
        print(f"   ")
        print(f"   # Get analytics summary anytime:")
        print(f"   summary = pipeline.get_analytics_summary()")
        print(f"   ")
        print(f"   # Search by multiple criteria:")
        print(f"   results = pipeline.search('корупція', metadata_filter={{'sentiment_label': 'negative', 'primary_topic': 'politics'}})")
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        pipeline.cleanup()
        
        print(f"\n✅ Wayback Analytics Complete!")
        print(f"   📁 All data stored in collection: wayback_20201130051600")
        print(f"   📋 Report saved to analytics report file")
        print(f"   🔍 You can now query the database for specific analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()