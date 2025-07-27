# HTML RAG Pipeline

[![CI](https://github.com/yourusername/html-rag-pipeline/workflows/CI/badge.svg)](https://github.com/yourusername/html-rag-pipeline/actions)
[![codecov](https://codecov.io/gh/yourusername/html-rag-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/html-rag-pipeline)
[![PyPI version](https://badge.fury.io/py/html-rag-pipeline.svg)](https://badge.fury.io/py/html-rag-pipeline)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, production-ready HTML RAG (Retrieval-Augmented Generation) pipeline for processing web content into searchable vector databases. Features comprehensive support for Ukrainian content, Wayback Machine archives, and enterprise-grade performance monitoring.

## 🚀 Features

### Core Capabilities
- **5-Stage Processing Pipeline**: Wayback → HTML Pruning → HTML Parsing → Text Embedding → Vector Storage
- **Ukrainian Content Support**: Specialized handling for Cyrillic text with automatic language detection
- **Wayback Machine Integration**: Process historical web archives with metadata preservation
- **Enterprise-Grade Performance**: Batch processing, parallel execution, and comprehensive monitoring
- **Production Ready**: Full error handling, logging, metrics, and CI/CD integration

### Advanced Features
- **Smart HTML Cleaning**: AI-powered and basic cleaning with automatic selection
- **Semantic Search**: High-quality text embeddings with similarity scoring
- **Configurable Pipeline**: Pydantic-based configuration with environment variable support
- **CLI Interface**: Complete command-line tool for all operations
- **Comprehensive Testing**: Unit and integration tests with Ukrainian content validation
- **Docker Support**: Multi-stage builds for development and production

## 📦 Installation

### Quick Install
```bash
pip install html-rag-pipeline
```

### Development Install
```bash
git clone https://github.com/yourusername/html-rag-pipeline.git
cd html-rag-pipeline
pip install -e ".[dev]"
```

### Docker Install
```bash
docker pull yourusername/html-rag-pipeline:latest
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from html_rag import create_pipeline

# Create pipeline with default configuration
pipeline = create_pipeline()

# Process HTML content
html_content = """
<html>
<body>
    <h1>Machine Learning in Healthcare</h1>
    <p>AI applications are transforming patient care...</p>
</body>
</html>
"""

result = pipeline.process_html(html_content, url="https://example.com/article")

if result['success']:
    print(f"✅ Processed {result['embedded_blocks_count']} text blocks")
    
    # Search the content
    results = pipeline.search("machine learning healthcare", n_results=5)
    for result in results:
        print(f"Score: {result['similarity_score']:.3f} | {result['text'][:100]}...")

pipeline.cleanup()
```

### Ukrainian Content Processing

```python
from html_rag import create_pipeline

# Use Ukrainian preset for optimal Cyrillic handling
pipeline = create_pipeline(preset="ukrainian")

ukrainian_html = """
<html>
<body>
    <h1>Політична партія "Слуга народу"</h1>
    <p>Партія працює для розвитку України та добробуту українського народу.</p>
</body>
</html>
"""

result = pipeline.process_html(ukrainian_html, url="https://sluga-narodu.com")

# Search in Ukrainian
results = pipeline.search("партія", n_results=3)
print(f"Found {len(results)} Ukrainian results")

pipeline.cleanup()
```

### Wayback Machine Processing

```python
from html_rag import create_pipeline
from html_rag.core.config import WaybackConfig

pipeline = create_pipeline(preset="wayback")

# Configure wayback processing
wayback_config = WaybackConfig(
    require_metadata=True,
    force_basic_cleaning=True,  # Preserve Ukrainian content
    min_content_length=100
)

# Process entire wayback directory
results = pipeline.process_wayback_snapshots(
    "path/to/wayback/snapshots",
    wayback_config=wayback_config
)

# Search with temporal filters
results = pipeline.search_wayback_snapshots(
    query="політична партія",
    domain_filter="sluga-narodu.com",
    n_results=10
)

pipeline.cleanup()
```

## 🖥️ Command Line Interface

The package includes a comprehensive CLI tool:

```bash
# Process single HTML file
html-rag process document.html --url https://example.com

# Batch process directory
html-rag batch ./html_files --pattern "*.html" --output results.json

# Process Wayback snapshots
html-rag wayback ./wayback_snapshots --domain-filter example.com

# Search documents
html-rag search "machine learning" --limit 10 --format table

# Get pipeline statistics
html-rag stats --output stats.json

# Generate configuration
html-rag config --preset ukrainian --output config.json
```

## 📊 Performance Monitoring

Built-in performance monitoring and metrics collection:

```python
from html_rag import create_pipeline
from html_rag.utils.metrics import track_processing
from html_rag.core.config import PipelineConfig

config = PipelineConfig(enable_metrics=True)
pipeline = create_pipeline(config=config)

# Process with metrics tracking
with track_processing() as metrics:
    results = pipeline.process_multiple_html(documents)

# Analyze performance
metrics_data = metrics.get_metrics_dict()
print(f"Processed {metrics_data['documents']['processed']} documents")
print(f"Success rate: {metrics_data['documents']['success_rate']:.1%}")
print(f"Peak memory: {metrics_data['resources']['peak_memory_mb']:.1f} MB")
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
RAG_COLLECTION_NAME=my_documents
RAG_PREFER_BASIC_CLEANING=true
RAG_CYRILLIC_DETECTION_THRESHOLD=0.1
RAG_LOG_LEVEL=INFO
```

### Programmatic Configuration

```python
from html_rag.core.config import PipelineConfig

config = PipelineConfig(
    collection_name="my_documents",
    prefer_basic_cleaning=True,
    cyrillic_detection_threshold=0.1,
    max_chunk_size=256,
    enable_metrics=True
)

pipeline = create_pipeline(config=config)
```

### Configuration Presets

```python
# Ukrainian content optimization
pipeline = create_pipeline(preset="ukrainian")

# English content optimization  
pipeline = create_pipeline(preset="english")

# Wayback Machine processing
pipeline = create_pipeline(preset="wayback")

# Performance optimization
pipeline = create_pipeline(preset="performance")
```

## 🏗️ Architecture

### Pipeline Stages

1. **Stage 0: Wayback Processing** (Optional)
   - Validates directory structure
   - Processes metadata files
   - Handles archived content

2. **Stage 1: HTML Pruning**
   - AI-powered content cleaning
   - Basic cleaning fallback
   - Cyrillic content detection

3. **Stage 2: HTML Parsing**
   - BeautifulSoup-based parsing
   - Text block extraction
   - Metadata preservation

4. **Stage 3: Text Embedding**
   - Multilingual embeddings
   - Batch processing
   - 768-dimensional vectors

5. **Stage 4: Vector Storage**
   - ChromaDB integration
   - Metadata filtering
   - Similarity search

### Project Structure

```
src/html_rag/
├── core/
│   ├── pipeline.py          # Main RAG pipeline
│   └── config.py            # Pydantic configuration
├── processors/
│   ├── wayback.py           # Wayback Machine processing
│   ├── html_pruner.py       # HTML cleaning
│   ├── html_parser.py       # HTML parsing
│   ├── text_embedder.py     # Text embedding
│   └── vector_store.py      # Vector storage
├── utils/
│   ├── logging.py           # Loguru-based logging
│   ├── metrics.py           # Performance metrics
│   └── validators.py        # Data validation
├── exceptions/
│   └── pipeline_exceptions.py  # Custom exceptions
└── cli/
    └── main.py              # CLI interface
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run Ukrainian content tests
pytest -m ukrainian

# Run performance tests
pytest -m slow

# Run with coverage
pytest --cov=html_rag --cov-report=html
```

## 📈 Examples

The `examples/` directory contains comprehensive demonstrations:

- **`basic_usage.py`**: Getting started with the pipeline
- **`batch_processing.py`**: Processing multiple documents with progress tracking
- **`advanced_search.py`**: Complex search scenarios and analysis
- **`wayback_analysis.py`**: Wayback Machine content analysis
- **`performance_monitoring.py`**: Performance optimization and monitoring

Run examples:

```bash
python examples/basic_usage.py
python examples/batch_processing.py
python examples/wayback_analysis.py
```

## 🐳 Docker Usage

### Development

```bash
# Build development image
docker build --target development -t html-rag-dev .

# Run with mounted code
docker run -v $(pwd):/app -it html-rag-dev bash
```

### Production

```bash
# Run production container
docker run -v $(pwd)/data:/app/data html-rag-pipeline:latest

# Process files
docker run -v $(pwd)/html_files:/app/data \
  html-rag-pipeline:latest \
  python -m html_rag.cli.main batch /app/data
```

### Jupyter Notebooks

```bash
# Run Jupyter development environment
docker build --target jupyter -t html-rag-jupyter .
docker run -p 8888:8888 -v $(pwd):/app html-rag-jupyter
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/html-rag-pipeline.git
cd html-rag-pipeline

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Quality

We use comprehensive code quality tools:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning
- **pre-commit**: Git hooks

## 📋 Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ recommended
- **Storage**: 2GB+ for models and data
- **OS**: Linux, macOS, Windows

### Dependencies

- `torch>=1.11.0` - PyTorch for ML models
- `transformers>=4.21.0` - Hugging Face transformers
- `sentence-transformers>=2.2.0` - Text embeddings
- `beautifulsoup4>=4.11.0` - HTML parsing
- `chromadb>=0.4.0` - Vector database
- `pydantic>=2.0.0` - Configuration management
- `loguru>=0.7.0` - Advanced logging
- `click>=8.0.0` - CLI framework

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentence Transformers** for multilingual embeddings
- **ChromaDB** for vector storage
- **BeautifulSoup** for HTML parsing
- **Hugging Face** for transformer models
- **Ukrainian RAG Community** for testing and feedback

## 📞 Support

- **Documentation**: [Read the Docs](https://html-rag-pipeline.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/html-rag-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/html-rag-pipeline/discussions)
- **Email**: contact@html-rag-pipeline.dev

## 🗺️ Roadmap

- [ ] Support for additional vector databases (Pinecone, Weaviate)
- [ ] Advanced chunking strategies
- [ ] Multi-modal content support (images, tables)
- [ ] Real-time processing capabilities
- [ ] Web UI for pipeline management
- [ ] Integration with popular RAG frameworks

---

**Made with ❤️ for the RAG community**