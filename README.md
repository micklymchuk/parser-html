# HTML RAG Pipeline

A complete Python RAG (Retrieval-Augmented Generation) pipeline for processing HTML content into a searchable vector database. The pipeline consists of 4 stages that clean, parse, embed, and store HTML content for semantic search applications.

## Features

- **Stage 1: HTML Pruning** - Uses `zstanjj/HTML-Pruner-Phi-3.8B` to clean raw HTML and remove noise
- **Stage 2: HTML Parsing** - Extracts structured text blocks with metadata using BeautifulSoup
- **Stage 3: Text Embedding** - Converts text to 768-dimensional vectors using `paraphrase-multilingual-mpnet-base-v2`
- **Stage 4: ChromaDB Storage** - Stores documents, embeddings, and metadata for fast retrieval

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The first run will download the required models (this may take some time):
   - HTML Pruner model: ~7GB
   - Embedding model: ~420MB

## Quick Start

```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline()

# Process HTML content
raw_html = "<html><body><h1>Title</h1><p>Content...</p></body></html>"
result = pipeline.process_html(raw_html, url="https://example.com")

# Search for relevant content
results = pipeline.search("your search query", n_results=5)

# Print results
for result in results:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")
```

## Usage Examples

### Basic Usage

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline with custom settings
pipeline = RAGPipeline(
    collection_name="my_documents",
    persist_directory="./my_chroma_db",
    max_chunk_size=512
)

# Process single HTML document
html_content = "..."  # Your HTML content
result = pipeline.process_html(html_content, url="https://example.com")

if result['success']:
    print(f"Processed {result['embedded_blocks_count']} text blocks")
```

### Batch Processing

```python
# Process multiple HTML documents
documents = [
    {"html": html1, "url": "https://example.com/page1"},
    {"html": html2, "url": "https://example.com/page2"}
]

results = pipeline.process_multiple_html(documents)
```

### Search and Retrieval

```python
# Semantic search
search_results = pipeline.search("machine learning algorithms", n_results=10)

# Search with metadata filtering
filtered_results = pipeline.search(
    "python programming", 
    n_results=5,
    metadata_filter={"element_type": "heading"}
)

# Metadata-only filtering
headings = pipeline.search_by_metadata(
    {"element_type": "heading"},
    limit=10
)
```

### Advanced Features

```python
# Get pipeline statistics
stats = pipeline.get_pipeline_stats()
print(f"Documents in database: {stats['vector_store']['document_count']}")

# Export documents
pipeline.export_documents("backup.json", format="json")

# Reset database
pipeline.reset_database()

# Clean up resources
pipeline.cleanup()
```

## Pipeline Stages

### Stage 1: HTML Pruning
- **Input**: Raw HTML with scripts, styles, and noise
- **Process**: Uses AI model to clean and extract main content
- **Output**: Clean HTML with headings, paragraphs, lists, tables

### Stage 2: HTML Parsing
- **Input**: Cleaned HTML from Stage 1
- **Process**: Parses structure and extracts text blocks with metadata
- **Output**: List of dictionaries with text, element type, hierarchy, position, URL

### Stage 3: Text Embedding
- **Input**: Text blocks from Stage 2
- **Process**: Converts text to 768-dimensional vectors
- **Output**: Text blocks with embedding vectors for semantic search

### Stage 4: ChromaDB Storage
- **Input**: Embedded text blocks from Stage 3
- **Process**: Stores in ChromaDB vector database
- **Output**: Searchable database with similarity search capabilities

## Output Format

Each processed text block contains:

```python
{
    'text': 'The actual text content',
    'element_type': 'heading|paragraph|list_item|table_cell|quote|...',
    'hierarchy_level': 1,  # For headings (h1=1, h2=2, etc.)
    'position': 5,  # Order on the page
    'url': 'https://example.com',
    # Additional metadata preserved from processing
}
```

## Search Results Format

Search results include:

```python
{
    'text': 'Matching text content',
    'metadata': {
        'element_type': 'paragraph',
        'hierarchy_level': None,
        'position': 3,
        'url': 'https://example.com'
    },
    'distance': 0.23,  # Lower is more similar
    'similarity_score': 0.77  # Higher is more similar (1 - distance)
}
```

## Configuration

### Pipeline Parameters

- `html_pruner_model`: HTML pruning model name (default: "zstanjj/HTML-Pruner-Phi-3.8B")
- `embedding_model`: Text embedding model name (default: "paraphrase-multilingual-mpnet-base-v2")
- `collection_name`: ChromaDB collection name
- `persist_directory`: Database storage directory
- `max_chunk_size`: Maximum characters per text chunk

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only processing
- **Recommended**: 16GB+ RAM, CUDA-compatible GPU for faster processing
- **Storage**: ~8GB for models + database storage

## Error Handling

The pipeline includes comprehensive error handling:

- Automatic fallback if HTML pruning fails
- Graceful handling of parsing errors
- Embedding batch processing with error recovery
- Database transaction safety

## Performance

Processing times (approximate, varies by hardware):

- **HTML Pruning**: 2-10 seconds per document (depends on length)
- **HTML Parsing**: < 1 second per document
- **Text Embedding**: 0.1-1 second per text block
- **ChromaDB Storage**: < 0.1 second per block

## Example Output

Run the example script to see the pipeline in action:

```bash
python example_usage.py
```

This will process sample HTML documents and demonstrate search capabilities.

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Ensure stable internet connection for initial model download
2. **Out of Memory**: Reduce `max_chunk_size` or process documents in smaller batches
3. **CUDA Errors**: Install appropriate PyTorch version for your CUDA version
4. **ChromaDB Issues**: Ensure write permissions for the persist directory

### Performance Optimization

1. **Pre-load Models**: Call `pipeline.load_models()` once at startup
2. **Batch Processing**: Process multiple documents together
3. **GPU Usage**: Ensure CUDA is available for faster embedding generation
4. **Chunk Size**: Adjust `max_chunk_size` based on your use case

## License

This project uses models and libraries with various licenses. Please check individual component licenses before commercial use.