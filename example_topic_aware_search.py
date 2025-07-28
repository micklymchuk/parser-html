#!/usr/bin/env python3
"""
Example usage of the Topic-Aware Search system in the HTML RAG Pipeline.

This example demonstrates how to use the new topic-aware search functionality
to perform intelligent queries with Ukrainian political content.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from html_rag.core.pipeline import RAGPipeline


async def demonstrate_topic_aware_search():
    """Demonstrate topic-aware search functionality."""
    
    # Initialize the RAG pipeline
    print("🚀 Initializing RAG Pipeline with Topic-Aware Search...")
    pipeline = RAGPipeline()
    
    # Example queries that showcase different search strategies
    example_queries = [
        # Contradiction search
        "знайди суперечності Тимошенко про приватизацію",
        "покажи зміни позиції щодо освіти",
        
        # Topic-focused search
        "що говорили про економічні реформи",
        "позиція щодо енергетичної політики",
        
        # Person-specific search
        "думка Тимошенко про соціальні програми",
        
        # General analysis
        "аналіз політичних позицій за 2024 рік"
    ]
    
    print("\n📊 Pipeline Statistics:")
    stats = pipeline.get_pipeline_stats()
    if 'topic_aware_search' in stats:
        search_stats = stats['topic_aware_search']
        print(f"  - Ollama Service: {'✅ Available' if search_stats.get('service_available') else '❌ Unavailable'}")
        print(f"  - Model: {search_stats.get('model', 'N/A')}")
        print(f"  - Supported Intents: {', '.join(search_stats.get('supported_intents', []))}")
        print(f"  - Supported Strategies: {', '.join(search_stats.get('supported_strategies', []))}")
    
    print(f"\n🔍 Running example searches...")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Example {i}: {query} ---")
        
        try:
            # Perform topic-aware search
            results = pipeline.topic_aware_search(query, n_results=3)
            
            if results:
                print(f"Found {len(results)} results:")
                
                for j, result in enumerate(results, 1):
                    # Extract key information
                    text_preview = result.get('text', '')[:200] + '...' if len(result.get('text', '')) > 200 else result.get('text', '')
                    
                    # Check for query context
                    query_context = result.get('query_context', {})
                    matched_intent = query_context.get('matched_intent', 'general')
                    matched_topics = query_context.get('matched_topics', [])
                    search_strategy = query_context.get('search_strategy', 'general')
                    relevance_score = query_context.get('relevance_score', 0.0)
                    
                    print(f"  {j}. Intent: {matched_intent} | Strategy: {search_strategy}")
                    print(f"     Topics: {', '.join(matched_topics) if matched_topics else 'None'}")
                    print(f"     Relevance: {relevance_score:.2f}")
                    print(f"     Text: {text_preview}")
                    
                    # Show topic analysis if available
                    metadata = result.get('metadata', {})
                    topic_analysis = metadata.get('topic_analysis', {})
                    if topic_analysis.get('topics'):
                        topics = topic_analysis.get('topics', [])
                        sentiments = topic_analysis.get('sentiment_by_topic', {})
                        print(f"     Document Topics: {topics}")
                        if sentiments:
                            sentiment_info = [f"{topic}: {sentiment}" for topic, sentiment in sentiments.items()]
                            print(f"     Sentiments: {', '.join(sentiment_info)}")
                    print()
            else:
                print("  No results found.")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print("\n✅ Topic-aware search demonstration completed!")
    
    # Cleanup
    pipeline.cleanup()


def demonstrate_query_analysis():
    """Demonstrate query analysis functionality."""
    
    print("\n🔬 Query Analysis Demonstration")
    print("=" * 50)
    
    # Initialize just the searcher for testing
    from html_rag.search.topic_aware_search import TopicAwareSearcher
    
    # Mock components for testing
    class MockVectorStore:
        def filter_by_metadata(self, filter_dict, limit=None):
            return []
        def search_by_similarity(self, query_embedding, n_results, metadata_filter=None):
            return []
    
    class MockTextEmbedder:
        def embed_query(self, query):
            return [0.1] * 768
    
    searcher = TopicAwareSearcher(
        vector_store=MockVectorStore(),
        text_embedder=MockTextEmbedder()
    )
    
    # Test query analysis
    test_queries = [
        "знайди суперечності Тимошенко про приватизацію",
        "що говорив Зеленський про освітні реформи в 2024 році",
        "покажи зміни позиції щодо енергетичної політики",
        "аналіз економічної програми"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        analysis = searcher.analyze_query(query)
        print(f"Analysis: {analysis}")


if __name__ == "__main__":
    print("🎯 HTML RAG Pipeline - Topic-Aware Search Example")
    print("=" * 60)
    
    # Check if we should run the full demo or just query analysis
    if len(sys.argv) > 1 and sys.argv[1] == "--analysis-only":
        demonstrate_query_analysis()
    else:
        print("\nNote: This example requires a configured RAG pipeline with processed documents.")
        print("Use --analysis-only to test just the query analysis functionality.")
        print("\nRunning full demonstration...")
        
        try:
            import asyncio
            asyncio.run(demonstrate_topic_aware_search())
        except Exception as e:
            print(f"❌ Error running full demo: {str(e)}")
            print("\nTrying query analysis demonstration instead...")
            demonstrate_query_analysis()