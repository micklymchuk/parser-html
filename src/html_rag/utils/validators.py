"""
Data validation utilities for HTML RAG Pipeline.
"""

import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import validators as url_validators
from urllib.parse import urlparse

try:
    # Pydantic v2
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_V2 = True
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, validator
    PYDANTIC_V2 = False
    
    # Create a wrapper to make v1 validator work like v2
    def field_validator(field_name, **kwargs):
        def decorator(func):
            return validator(field_name, **kwargs)(func)
        return decorator


class HTMLDocument(BaseModel):
    """Validation model for HTML documents."""
    
    html: str = Field(..., min_length=1, description="HTML content")
    url: str = Field(..., description="Source URL")
    wayback_metadata: Optional[Dict[str, Any]] = Field(None, description="Wayback metadata")
    
    @field_validator('html')
    @classmethod
    def validate_html_content(cls, v):
        """Validate HTML content."""
        if not v or v.isspace():
            raise ValueError("HTML content cannot be empty or whitespace only")
        
        # Check for basic HTML structure
        if not any(tag in v.lower() for tag in ['<html', '<body', '<div', '<p', '<h1', '<h2', '<h3']):
            raise ValueError("Content does not appear to be valid HTML")
        
        return v
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format."""
        if not v:
            raise ValueError("URL cannot be empty")
        
        # Allow test URLs and localhost
        if v.startswith(('http://', 'https://', 'file://')):
            return v
        
        # Try to parse as URL
        try:
            parsed = urlparse(v)
            if not parsed.scheme and not parsed.netloc:
                # If no scheme, assume it's a relative URL or identifier
                if not v.startswith('/') and '.' not in v:
                    return f"document://{v}"
        except Exception:
            raise ValueError(f"Invalid URL format: {v}")
        
        return v


class WaybackMetadata(BaseModel):
    """Validation model for Wayback metadata."""
    
    timestamp: str = Field(
        ...,
        **({"pattern": r'\d{14}'} if PYDANTIC_V2 else {"regex": r'\d{14}'}),
        description="Wayback timestamp (YYYYMMDDHHMMSS)"
    )
    original_url: str = Field(..., description="Original URL")
    archive_url: Optional[str] = Field(None, description="Archive URL")
    domain: Optional[str] = Field(None, description="Domain")
    title: Optional[str] = Field(None, description="Page title")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate Wayback timestamp format."""
        if not re.match(r'^\d{14}$', v):
            raise ValueError("Timestamp must be in format YYYYMMDDHHMMSS")
        
        # Validate date components
        year = int(v[:4])
        month = int(v[4:6])
        day = int(v[6:8])
        hour = int(v[8:10])
        minute = int(v[10:12])
        second = int(v[12:14])
        
        if not (1990 <= year <= 2030):
            raise ValueError("Year must be between 1990 and 2030")
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
        if not (0 <= hour <= 23):
            raise ValueError("Hour must be between 0 and 23")
        if not (0 <= minute <= 59):
            raise ValueError("Minute must be between 0 and 59")
        if not (0 <= second <= 59):
            raise ValueError("Second must be between 0 and 59")
        
        return v


class TextBlock(BaseModel):
    """Validation model for text blocks."""
    
    text: str = Field(..., min_length=1, description="Text content")
    element_type: str = Field(..., description="HTML element type")
    hierarchy_level: Optional[int] = Field(None, ge=1, le=6, description="Heading level")
    position: int = Field(..., ge=1, description="Position on page")
    url: str = Field(..., description="Source URL")
    chunk_index: Optional[int] = Field(None, ge=0, description="Chunk index")
    total_chunks: Optional[int] = Field(None, ge=1, description="Total chunks")
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        """Validate text content."""
        if not v or v.isspace():
            raise ValueError("Text content cannot be empty or whitespace only")
        
        # Check for reasonable length limits
        if len(v) > 100000:  # 100KB limit
            raise ValueError("Text content too long (max 100KB)")
        
        return v.strip()
    
    @field_validator('element_type')
    @classmethod
    def validate_element_type(cls, v):
        """Validate element type."""
        valid_types = [
            'heading', 'paragraph', 'list_item', 'table_cell', 
            'table_header', 'quote', 'division', 'span', 'text'
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid element type. Must be one of: {valid_types}")
        return v


class SearchQuery(BaseModel):
    """Validation model for search queries."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    n_results: int = Field(default=10, ge=1, le=100, description="Number of results")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    similarity_threshold: float = Field(default=0.0, ge=-1.0, le=1.0, description="Similarity threshold")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate search query."""
        if not v or v.isspace():
            raise ValueError("Search query cannot be empty")
        
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Check for minimum meaningful length
        if len(v) < 2:
            raise ValueError("Search query too short (minimum 2 characters)")
        
        return v


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """
    Validate file path exists and is readable.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    if not path.stat().st_size > 0:
        raise ValueError(f"File is empty: {path}")
    
    return path


def validate_directory_path(dir_path: Union[str, Path]) -> Path:
    """
    Validate directory path exists and is readable.
    
    Args:
        dir_path: Directory path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If directory is invalid
    """
    path = Path(dir_path)
    
    if not path.exists():
        raise ValueError(f"Directory does not exist: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    return path


def validate_html_content(html: str) -> str:
    """
    Validate HTML content.
    
    Args:
        html: HTML content to validate
        
    Returns:
        Validated HTML content
        
    Raises:
        ValueError: If HTML is invalid
    """
    document = HTMLDocument(html=html, url="validation://test")
    return document.html


def validate_wayback_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Wayback metadata.
    
    Args:
        metadata: Metadata to validate
        
    Returns:
        Validated metadata
        
    Raises:
        ValueError: If metadata is invalid
    """
    wayback_meta = WaybackMetadata(**metadata)
    return wayback_meta.dict()


def validate_text_blocks(text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate list of text blocks.
    
    Args:
        text_blocks: Text blocks to validate
        
    Returns:
        Validated text blocks
        
    Raises:
        ValueError: If any text block is invalid
    """
    validated_blocks = []
    
    for i, block in enumerate(text_blocks):
        try:
            text_block = TextBlock(**block)
            validated_blocks.append(text_block.dict())
        except Exception as e:
            raise ValueError(f"Invalid text block at index {i}: {e}")
    
    return validated_blocks


def validate_search_query(query: str, **kwargs) -> Dict[str, Any]:
    """
    Validate search query and parameters.
    
    Args:
        query: Search query
        **kwargs: Additional search parameters
        
    Returns:
        Validated search parameters
        
    Raises:
        ValueError: If query is invalid
    """
    search_query = SearchQuery(query=query, **kwargs)
    return search_query.dict()


def validate_embedding_dimension(embeddings: List[List[float]], expected_dim: int) -> None:
    """
    Validate embedding dimensions.
    
    Args:
        embeddings: List of embeddings
        expected_dim: Expected dimension
        
    Raises:
        ValueError: If dimensions don't match
    """
    if not embeddings:
        raise ValueError("Embeddings list is empty")
    
    for i, embedding in enumerate(embeddings):
        if len(embedding) != expected_dim:
            raise ValueError(
                f"Embedding {i} has dimension {len(embedding)}, expected {expected_dim}"
            )


def validate_url_format(url: str) -> str:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        Validated URL
        
    Raises:
        ValueError: If URL is invalid
    """
    if not url:
        raise ValueError("URL cannot be empty")
    
    # Allow test and local URLs
    if url.startswith(('http://', 'https://', 'file://', 'document://')):
        return url
    
    # Try standard URL validation for others
    if url_validators.url(url):
        return url
    
    # If validation fails, check if it's a reasonable identifier
    if re.match(r'^[a-zA-Z0-9._/-]+$', url):
        return f"document://{url}"
    
    raise ValueError(f"Invalid URL format: {url}")


def validate_collection_name(name: str) -> str:
    """
    Validate ChromaDB collection name.
    
    Args:
        name: Collection name
        
    Returns:
        Validated collection name
        
    Raises:
        ValueError: If name is invalid
    """
    if not name:
        raise ValueError("Collection name cannot be empty")
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError("Collection name can only contain alphanumeric characters, underscores, and hyphens")
    
    if len(name) > 63:
        raise ValueError("Collection name too long (max 63 characters)")
    
    return name


class PipelineValidator:
    """Main validator class for pipeline operations."""
    
    @staticmethod
    def validate_pipeline_input(
        html: str,
        url: str,
        wayback_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate complete pipeline input.
        
        Args:
            html: HTML content
            url: Source URL
            wayback_metadata: Optional Wayback metadata
            
        Returns:
            Validated input data
            
        Raises:
            ValueError: If validation fails
        """
        # Validate HTML document
        document = HTMLDocument(html=html, url=url, wayback_metadata=wayback_metadata)
        
        # Validate wayback metadata if provided
        if wayback_metadata:
            validate_wayback_metadata(wayback_metadata)
        
        return document.dict()
    
    @staticmethod
    def validate_batch_input(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate batch processing input.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Validated documents
            
        Raises:
            ValueError: If validation fails
        """
        if not documents:
            raise ValueError("Document list cannot be empty")
        
        validated_documents = []
        
        for i, doc in enumerate(documents):
            try:
                validated_doc = PipelineValidator.validate_pipeline_input(
                    html=doc.get('html', ''),
                    url=doc.get('url', f'document_{i}'),
                    wayback_metadata=doc.get('wayback_metadata')
                )
                validated_documents.append(validated_doc)
            except Exception as e:
                raise ValueError(f"Invalid document at index {i}: {e}")
        
        return validated_documents