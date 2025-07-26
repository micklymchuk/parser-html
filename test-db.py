from rag_pipeline import RAGPipeline

# Підключитися до існуючої БД
pipeline = RAGPipeline(
    collection_name="wayback_documents",
    persist_directory="./wayback_chroma_db"
)

# Шукати без повторного завантаження
results = pipeline.search("зеленський", n_results=5)

print(results)