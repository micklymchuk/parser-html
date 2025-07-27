"""
Exception classes for HTML RAG Pipeline.
"""

from .pipeline_exceptions import (
    PipelineError,
    ConfigurationError,
    HTMLProcessingError,
    HTMLPruningError,
    HTMLParsingError,
    EmbeddingError,
    VectorStoreError,
    WaybackProcessingError,
    SearchError,
    ValidationError,
    ModelLoadError,
    ResourceError,
    get_pipeline_exception,
    handle_pipeline_error,
)

__all__ = [
    "PipelineError",
    "ConfigurationError", 
    "HTMLProcessingError",
    "HTMLPruningError",
    "HTMLParsingError",
    "EmbeddingError",
    "VectorStoreError",
    "WaybackProcessingError",
    "SearchError",
    "ValidationError",
    "ModelLoadError",
    "ResourceError",
    "get_pipeline_exception",
    "handle_pipeline_error",
]