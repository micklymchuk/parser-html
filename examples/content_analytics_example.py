#!/usr/bin/env python3
"""
Content Analytics Example

This example demonstrates how to use the content analytics features
of the HTML RAG Pipeline to analyze Ukrainian and English content.

Features demonstrated:
- Basic content analytics setup
- Document processing with analytics
- Sentiment analysis
- Controversy detection
- Entity extraction
- Topic classification
- Analytics search and filtering
- Trend analysis
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.html_rag.core.pipeline import RAGPipeline
from src.html_rag.core.config import PipelineConfig, ContentAnalyticsConfig


# Sample content for testing
SAMPLE_DOCUMENTS = [
    {
        "html": """
        <html>
        <head><title>Political News</title></head>
        <body>
            <h1>Президент України зустрівся з лідерами ЄС</h1>
            <p>Володимир Зеленський провів важливі переговори з представниками Європейського Союзу у Києві. 
            Під час зустрічі обговорювалися питання національної безпеки, економічної підтримки та 
            європейської інтеграції України.</p>
            <p>Президент підкреслив важливість міжнародної співпраці та висловив вдячність партнерам 
            за постійну підтримку української держави.</p>
        </body>
        </html>
        """,
        "url": "https://example.com/ukraine-eu-meeting"
    },
    {
        "html": """
        <html>
        <head><title>Economic Update</title></head>
        <body>
            <h1>Government Announces New Economic Reforms</h1>
            <p>The Prime Minister unveiled a comprehensive package of economic reforms aimed at 
            promoting sustainable growth and innovation. The reforms include tax incentives for 
            small businesses, investment in renewable energy, and modernization of infrastructure.</p>
            <p>Financial experts have praised the initiative, calling it a positive step towards 
            economic recovery and long-term prosperity.</p>
        </body>
        </html>
        """,
        "url": "https://example.com/economic-reforms"
    },
    {
        "html": """
        <html>
        <head><title>Controversial Statement</title></head>
        <body>
            <h1>Скандальна заява політика викликала критику</h1>
            <p>Вчорашня заява одного з провідних політиків країни викликала хвилю критики та протестів 
            з боку опозиції та громадських організацій. Опозиція звинуватила політика у поширенні 
            неправдивої інформації та підриві довіри до демократичних інститутів.</p>
            <p>Цей інцидент призвів до політичної напруженості та вимог щодо публічних вибачень.</p>
        </body>
        </html>
        """,
        "url": "https://example.com/political-controversy"
    }
]


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    

def create_analytics_pipeline():
    """Create a RAG pipeline with content analytics enabled."""
    print("🚀 Setting up Content Analytics Pipeline...")
    
    # Configure content analytics
    analytics_config = ContentAnalyticsConfig(
        enabled=True,
        controversy_threshold=0.5,  # Lower threshold for demo
        max_entities_per_document=20,
        sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        classification_model="facebook/bart-large-mnli"
    )
    
    # Configure main pipeline
    pipeline_config = PipelineConfig(
        collection_name="analytics_demo",
        prefer_basic_cleaning=True,  # Better for Ukrainian content
        enable_metrics=True
    )
    
    # Create pipeline with analytics enabled
    pipeline = RAGPipeline(
        config=pipeline_config,
        enable_content_analytics=True,
        analytics_config=analytics_config
    )
    
    print("✅ Pipeline created successfully!")
    return pipeline


def process_documents_with_analytics(pipeline):
    """Process sample documents and perform content analytics."""
    print("\n📄 Processing documents with content analytics...")
    
    results = []
    
    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n--- Processing Document {i} ---")
        print(f"URL: {doc['url']}")
        
        try:
            # Process with analytics enabled
            result = pipeline.process_html(
                raw_html=doc['html'],
                url=doc['url'],
                analyze_content=True  # Force analytics for this document
            )
            
            if result.get('success'):
                print("✅ Document processed successfully")
                
                # Display analytics results if available
                analytics = result.get('content_analytics')
                if analytics:
                    print("📊 Analytics Results:")
                    
                    # Sentiment
                    sentiment = analytics['sentiment']
                    print(f"  Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
                    
                    # Controversy
                    controversy = analytics['controversy']
                    print(f"  Controversy: {controversy['level']} (score: {controversy['score']:.2f})")
                    
                    # Topics
                    topics = analytics['topics']
                    print(f"  Primary Topic: {topics['primary_topic']} (confidence: {topics['confidence']:.2f})")
                    
                    # Entities
                    entities = analytics['entities']
                    print(f"  Entities Found: {len(entities)}")
                    for entity in entities[:3]:  # Show first 3 entities
                        print(f"    - {entity['text']} ({entity['entity_type']})")
                    
                    # Language and Quality
                    print(f"  Language: {analytics['language']}")
                    print(f"  Quality Score: {analytics.get('quality_score', 0):.2f}")
                
                results.append(result)
            else:
                print(f"❌ Failed to process document: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Error processing document: {e}")
    
    print(f"\n✅ Processed {len(results)} documents successfully")
    return results


def demonstrate_analytics_search(pipeline):
    """Demonstrate content analytics search capabilities."""
    print("\n🔍 Demonstrating Analytics Search Features...")
    
    try:
        # Get analytics summary
        print("\n--- Analytics Summary ---")
        summary = pipeline.get_analytics_summary()
        
        if 'total_documents_analyzed' in summary:
            print(f"📈 Total Documents Analyzed: {summary['total_documents_analyzed']}")
            print(f"📊 Sentiment Distribution: {summary['sentiment_distribution']}")
            print(f"⚠️  Controversy Distribution: {summary['controversy_distribution']}")
            print(f"🏷️  Top Topics: {list(summary['topic_distribution'].keys())[:5]}")
            print(f"💯 Average Quality Score: {summary.get('average_quality_score', 0):.2f}")
        else:
            print("📝 No analytics data available yet")
        
        # Search by sentiment
        print("\n--- Search by Sentiment ---")
        positive_docs = pipeline.search_by_sentiment('positive', n_results=5)
        print(f"🙂 Found {len(positive_docs)} positive documents")
        
        negative_docs = pipeline.search_by_sentiment('negative', n_results=5)
        print(f"😞 Found {len(negative_docs)} negative documents")
        
        # Search by controversy level
        print("\n--- Search by Controversy Level ---")
        high_controversy = pipeline.search_by_controversy_level('high', n_results=5)
        print(f"🔥 Found {len(high_controversy)} high controversy documents")
        
        low_controversy = pipeline.search_by_controversy_level('low', n_results=5)
        print(f"😊 Found {len(low_controversy)} low controversy documents")
        
        # Search by topic
        print("\n--- Search by Topic ---")
        political_docs = pipeline.search_by_topic('politics', n_results=5)
        print(f"🏛️  Found {len(political_docs)} political documents")
        
        economic_docs = pipeline.search_by_topic('economics', n_results=5)
        print(f"💰 Found {len(economic_docs)} economic documents")
        
    except Exception as e:
        print(f"❌ Error in analytics search: {e}")


def demonstrate_regular_search(pipeline):
    """Demonstrate regular search with analytics metadata."""
    print("\n🔎 Demonstrating Enhanced Search with Analytics...")
    
    search_queries = [
        "Чи анонсував уряд нову економічну реформу?"
    ]
    
    for query in search_queries:
        print(f"\n--- Searching for: '{query}' ---")
        
        try:
            results = pipeline.search(query, n_results=3)
            
            print(f"🎯 Found {len(results)} results")
            
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    URL: {result.get('metadata', {}).get('url', 'Unknown')}")
                print(f"    Similarity: {result.get('similarity_score', 0):.3f}")
                
                # Show analytics metadata if available
                metadata = result.get('metadata', {})
                if metadata.get('analytics_processed'):
                    print(f"    📊 Sentiment: {metadata.get('sentiment_label', 'Unknown')}")
                    print(f"    📊 Controversy: {metadata.get('controversy_level', 'Unknown')}")
                    print(f"    📊 Topic: {metadata.get('primary_topic', 'Unknown')}")
                    print(f"    📊 Quality: {metadata.get('quality_score', 0):.2f}")
                
                # Show snippet
                text_snippet = result.get('text', '')[:150]
                print(f"    📝 Snippet: {text_snippet}...")
        
        except Exception as e:
            print(f"❌ Search error: {e}")


def demonstrate_analytics_on_existing(pipeline):
    """Demonstrate running analytics on existing documents."""
    print("\n🔄 Running Analytics on Existing Documents...")
    
    try:
        # Analyze existing documents that might not have analytics yet
        result = pipeline.analyze_existing_documents(limit=10)
        
        print(f"📈 Analysis Results:")
        print(f"  Total Documents: {result.total_documents}")
        print(f"  Successfully Analyzed: {result.successful}")
        print(f"  Failed: {result.failed}")
        print(f"  Processing Time: {result.processing_time:.2f} seconds")
        
        if result.errors:
            print(f"  ❌ Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
        
        if result.warnings:
            print(f"  ⚠️  Warnings: {len(result.warnings)}")
        
    except Exception as e:
        print(f"❌ Error analyzing existing documents: {e}")


def demonstrate_export_functionality(pipeline):
    """Demonstrate exporting documents with analytics."""
    print("\n💾 Demonstrating Export Functionality...")
    
    try:
        # Export to JSON
        json_file = "analytics_export.json"
        pipeline.export_documents(json_file, format='json')
        print(f"✅ Exported documents to {json_file}")
        
        # Check if file was created
        if Path(json_file).exists():
            file_size = Path(json_file).stat().st_size
            print(f"📁 File size: {file_size} bytes")
        
    except Exception as e:
        print(f"❌ Export error: {e}")


def main():
    """Main demonstration function."""
    print("🎯 Content Analytics Demo")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # Create pipeline with analytics
        pipeline = create_analytics_pipeline()
        
        # Process documents with analytics
        results = process_documents_with_analytics(pipeline)
        
        if results:
            # Demonstrate analytics search features
            demonstrate_analytics_search(pipeline)
            
            # Demonstrate enhanced search
            demonstrate_regular_search(pipeline)
            
            # Demonstrate analytics on existing documents
            demonstrate_analytics_on_existing(pipeline)
            
            # Demonstrate export
            demonstrate_export_functionality(pipeline)
            
            # Show pipeline statistics
            print("\n📈 Pipeline Statistics:")
            stats = pipeline.get_pipeline_stats()
            print(f"  Vector Store: {stats.get('vector_store', {}).get('document_count', 0)} documents")
            
            analytics_info = stats.get('content_analytics', {})
            print(f"  Analytics Enabled: {analytics_info.get('enabled', False)}")
            print(f"  Analytics Available: {analytics_info.get('available', False)}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        pipeline.cleanup()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- ✅ Multi-language content analytics (Ukrainian & English)")
        print("- ✅ Sentiment analysis with confidence scores")
        print("- ✅ Controversy detection with evidence")
        print("- ✅ Named entity extraction and linking")
        print("- ✅ Topic classification")
        print("- ✅ Content quality scoring")
        print("- ✅ Analytics-enhanced search capabilities")
        print("- ✅ Export functionality with analytics metadata")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()