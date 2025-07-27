from rag_pipeline import RAGPipeline

# Підключитися до існуючої БД
pipeline = RAGPipeline(
    collection_name="wayback_documents",
    persist_directory="./wayback_chroma_db"
)
stats = pipeline.get_pipeline_stats()
print(f"Documents in DB: {stats['vector_store']['document_count']}")
# Шукати без повторного завантаження
all_results = pipeline.search("цитата", n_results=20)
for i, result in enumerate(all_results):
    print(f"{i+1}. Score: {result['similarity_score']:.3f}")
    print(f"   Text: {result['text']}")
    print(f"   Type: {result['metadata']['element_type']}")
    print()