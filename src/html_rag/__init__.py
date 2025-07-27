"""
HTML RAG Pipeline - A modern, production-ready HTML processing pipeline.

This package provides a comprehensive solution for processing HTML content into
searchable vector databases, with special support for Ukrainian content and
Wayback Machine archives.
"""

__version__ = "2.0.0"
__author__ = "HTML RAG Pipeline Team"
__email__ = "contact@html-rag-pipeline.dev"

# Core imports
from .core.pipeline import RAGPipeline, create_pipeline
from .core.config import PipelineConfig, WaybackConfig, SearchConfig

# Utility imports
from .utils.logging import setup_logging, PipelineLogger
from .utils.metrics import MetricsCollector, PerformanceProfiler

# Exception imports
from .exceptions.pipeline_exceptions import (
    PipelineError,
    ConfigurationError,
    HTMLProcessingError,
    EmbeddingError,
    VectorStoreError,
    WaybackProcessingError,
    SearchError,
    ValidationError,
    ModelLoadError,
    ResourceError,
)

# Public API
__all__ = [
    # Core classes
    "RAGPipeline",
    "create_pipeline",
    
    # Configuration
    "PipelineConfig",
    "WaybackConfig", 
    "SearchConfig",
    
    # Utilities
    "setup_logging",
    "PipelineLogger",
    "MetricsCollector",
    "PerformanceProfiler",
    
    # Exceptions
    "PipelineError",
    "ConfigurationError",
    "HTMLProcessingError",
    "EmbeddingError",
    "VectorStoreError",
    "WaybackProcessingError",
    "SearchError",
    "ValidationError",
    "ModelLoadError",
    "ResourceError",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
]


def get_version() -> str:
    """Get the package version."""
    return __version__


def get_info() -> dict:
    """Get package information."""
    return {
        "name": "html-rag-pipeline",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "A modern, production-ready HTML RAG pipeline",
        "python_requires": ">=3.8",
    }


# Package-level configuration
import logging

# Set up default logging configuration
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Suppress warnings from dependencies during import
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")