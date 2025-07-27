"""
Custom exceptions for HTML RAG Pipeline.
"""

from typing import Optional, Dict, Any


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class ConfigurationError(PipelineError):
    """Raised when there's an error in configuration."""
    pass


class HTMLProcessingError(PipelineError):
    """Raised when HTML processing fails."""
    pass


class HTMLPruningError(HTMLProcessingError):
    """Raised when HTML pruning fails."""
    pass


class HTMLParsingError(HTMLProcessingError):
    """Raised when HTML parsing fails."""
    pass


class EmbeddingError(PipelineError):
    """Raised when text embedding fails."""
    pass


class VectorStoreError(PipelineError):
    """Raised when vector store operations fail."""
    pass


class WaybackProcessingError(PipelineError):
    """Raised when Wayback Machine processing fails."""
    pass


class SearchError(PipelineError):
    """Raised when search operations fail."""
    pass


class ValidationError(PipelineError):
    """Raised when data validation fails."""
    pass


class ModelLoadError(PipelineError):
    """Raised when model loading fails."""
    pass


class ResourceError(PipelineError):
    """Raised when resource allocation/cleanup fails."""
    pass


# Exception mapping for better error handling
EXCEPTION_MAPPING = {
    "html_pruning": HTMLPruningError,
    "html_parsing": HTMLParsingError,
    "embedding": EmbeddingError,
    "vector_store": VectorStoreError,
    "wayback": WaybackProcessingError,
    "search": SearchError,
    "validation": ValidationError,
    "model_load": ModelLoadError,
    "resource": ResourceError,
    "config": ConfigurationError,
}


def get_pipeline_exception(error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> PipelineError:
    """
    Get appropriate pipeline exception based on error type.
    
    Args:
        error_type: Type of error
        message: Error message
        details: Optional error details
        
    Returns:
        Appropriate PipelineError subclass instance
    """
    exception_class = EXCEPTION_MAPPING.get(error_type, PipelineError)
    return exception_class(message, details)


def handle_pipeline_error(func):
    """
    Decorator to handle pipeline errors and convert them to appropriate exceptions.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PipelineError:
            # Re-raise pipeline errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to PipelineError
            raise PipelineError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                details={"function": func.__name__, "error_type": type(e).__name__}
            ) from e
    
    return wrapper