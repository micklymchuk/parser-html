"""
Core configuration management for HTML RAG Pipeline using Pydantic.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import os

from sqlalchemy.sql.operators import truediv

try:
    # Try pydantic-settings first (for Pydantic v2)
    from pydantic_settings import BaseSettings
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    PYDANTIC_V2 = True
except ImportError:
    try:
        # Pydantic v1
        from pydantic import BaseSettings, BaseModel, Field, validator
        PYDANTIC_V2 = False
        ConfigDict = dict  # Fallback for v1
        field_validator = validator  # Alias for v1
    except ImportError:
        # Fallback without pydantic
        PYDANTIC_V2 = False
        
        class BaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def dict(self):
                return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        class BaseSettings(BaseModel):
            pass
        
        def Field(default=None, **kwargs):
            return default
        
        def field_validator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        def validator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        ConfigDict = dict


class PipelineConfig(BaseSettings):
    """Main configuration for the HTML RAG Pipeline."""
    
    # Model configurations
    html_pruner_model: str = Field(
        default="zstanjj/HTML-Pruner-Phi-3.8B",
        description="Model name for HTML pruning"
    )
    embedding_model: str = Field(
        default="paraphrase-multilingual-mpnet-base-v2",
        description="Model name for text embedding"
    )
    
    # Processing configurations
    max_chunk_size: int = Field(
        default=512,
        description="Maximum characters per text chunk",
        ge=50,
        le=2048
    )
    prefer_basic_cleaning: bool = Field(
        default=False,
        description="If True, default to basic cleaning over AI model"
    )
    cyrillic_detection_threshold: float = Field(
        default=0.2,
        description="Threshold for Cyrillic content detection (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Database configurations
    collection_name: str = Field(
        default="html_documents",
        description="ChromaDB collection name"
    )
    persist_directory: str = Field(
        default="./chroma_db",
        description="Directory to persist ChromaDB"
    )
    
    # Logging configurations
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default="logs/pipeline.log",
        description="Log file path"
    )
    log_rotation: str = Field(
        default="10 MB",
        description="Log file rotation size"
    )
    
    # Performance configurations
    batch_size: int = Field(
        default=32,
        description="Batch size for processing",
        ge=1,
        le=256
    )
    num_workers: int = Field(
        default=4,
        description="Number of worker processes",
        ge=1,
        le=32
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable performance metrics collection"
    )
    
    # Device configuration
    device: Optional[str] = Field(
        default=None,
        description="Device for model inference (cpu, cuda, mps)"
    )
    
    
    if PYDANTIC_V2:
        @field_validator('persist_directory')
        @classmethod
        def validate_persist_directory(cls, v):
            """Ensure persist directory exists."""
            Path(v).mkdir(parents=True, exist_ok=True)
            return v
        
        @field_validator('log_file')
        @classmethod
        def validate_log_file(cls, v):
            """Ensure log directory exists."""
            if v:
                Path(v).parent.mkdir(parents=True, exist_ok=True)
            return v
        
        model_config = ConfigDict(
            env_prefix='RAG_',
            env_file='.env',
            case_sensitive=False
        )
    else:
        @validator('persist_directory')
        def validate_persist_directory(cls, v):
            """Ensure persist directory exists."""
            Path(v).mkdir(parents=True, exist_ok=True)
            return v
        
        @validator('log_file')
        def validate_log_file(cls, v):
            """Ensure log directory exists."""
            if v:
                Path(v).parent.mkdir(parents=True, exist_ok=True)
            return v
        
        class Config:
            env_prefix = "RAG_"
            env_file = ".env"
            case_sensitive = False


class WaybackConfig(BaseSettings):
    """Configuration for Wayback Machine processing."""
    
    require_metadata: bool = Field(
        default=False,
        description="If True, only process HTML files with meta.json files"
    )
    force_basic_cleaning: bool = Field(
        default=True,
        description="Force basic cleaning for wayback content"
    )
    min_content_length: int = Field(
        default=100,
        description="Minimum content length for processing",
        ge=0
    )
    domain_filters: List[str] = Field(
        default_factory=list,
        description="List of domain filters"
    )
    year_filters: List[int] = Field(
        default_factory=list,
        description="List of year filters"
    )
    
    if PYDANTIC_V2:
        model_config = ConfigDict(
            env_prefix='WAYBACK_',
            env_file='.env',
            case_sensitive=False
        )
    else:
        class Config:
            env_prefix = "WAYBACK_"
            env_file = ".env"
            case_sensitive = False


class SearchConfig(BaseSettings):
    """Configuration for search operations."""
    
    default_n_results: int = Field(
        default=10,
        description="Default number of search results",
        ge=1,
        le=100
    )
    similarity_threshold: float = Field(
        default=0.0,
        description="Minimum similarity threshold for results",
        ge=-1.0,
        le=1.0
    )
    enable_metadata_filter: bool = Field(
        default=True,
        description="Enable metadata filtering in search"
    )
    
    if PYDANTIC_V2:
        model_config = ConfigDict(
            env_prefix='SEARCH_',
            env_file='.env',
            case_sensitive=False
        )
    else:
        class Config:
            env_prefix = "SEARCH_"
            env_file = ".env"
            case_sensitive = False


class ContentAnalyticsConfig(BaseSettings):
    """Configuration for content analytics processing."""
    
    enabled: bool = Field(
        default=True,
        description="Enable content analytics processing"
    )
    entities_db_path: str = Field(
        default="src/html_rag/data/entities_db.json",
        description="Path to entities database file"
    )
    controversy_threshold: float = Field(
        default=0.6,
        description="Controversy detection threshold (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="Sentiment analysis model"
    )
    classification_model: str = Field(
        default="facebook/bart-large-mnli",
        description="Zero-shot classification model"
    )
    ukrainian_model: str = Field(
        default="lang-uk/roberta-base-uk",
        description="Ukrainian language model"
    )
    enable_topic_classification: bool = Field(
        default=True,
        description="Enable topic classification"
    )
    enable_trend_analysis: bool = Field(
        default=True,
        description="Enable trend analysis over time"
    )
    enable_readability_analysis: bool = Field(
        default=True,
        description="Enable readability and complexity analysis"
    )
    cache_results: bool = Field(
        default=True,
        description="Cache analytics results"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for analytics processing",
        ge=1,
        le=256
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["uk", "ru", "en"],
        description="Supported languages for analysis"
    )
    min_confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence threshold for results",
        ge=0.0,
        le=1.0
    )
    enable_entity_linking: bool = Field(
        default=True,
        description="Enable entity linking and relationship detection"
    )
    max_entities_per_document: int = Field(
        default=50,
        description="Maximum entities to extract per document",
        ge=1,
        le=200
    )
    
    if PYDANTIC_V2:
        @field_validator('entities_db_path')
        @classmethod
        def validate_entities_db_path(cls, v):
            """Ensure entities database directory exists."""
            Path(v).parent.mkdir(parents=True, exist_ok=True)
            return v
        
        model_config = ConfigDict(
            env_prefix='ANALYTICS_',
            env_file='.env',
            case_sensitive=False
        )
    else:
        @validator('entities_db_path')
        def validate_entities_db_path(cls, v):
            """Ensure entities database directory exists."""
            Path(v).parent.mkdir(parents=True, exist_ok=True)
            return v
        
        class Config:
            env_prefix = "ANALYTICS_"
            env_file = ".env"
            case_sensitive = False


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        PipelineConfig instance
    """
    if config_path and Path(config_path).exists():
        # Load from file if provided
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return PipelineConfig(**config_data)
    else:
        # Load from environment variables
        return PipelineConfig()


def save_config(config: PipelineConfig, config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: PipelineConfig instance
        config_path: Path to save configuration
    """
    import json
    
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config.dict(), f, indent=2)


# Default configurations for different use cases
DEFAULT_CONFIGS = {
    "ukrainian": PipelineConfig(
        prefer_basic_cleaning=True,
        cyrillic_detection_threshold=0.1,
        collection_name="ukrainian_documents"
    ),
    "english": PipelineConfig(
        prefer_basic_cleaning=False,
        cyrillic_detection_threshold=0.5,
        collection_name="english_documents"
    ),
    "wayback": PipelineConfig(
        prefer_basic_cleaning=True,
        cyrillic_detection_threshold=0.1,
        collection_name="wayback_documents"
    ),
    "performance": PipelineConfig(
        batch_size=64,
        num_workers=8,
        max_chunk_size=1024,
        enable_metrics=True
    )
}


def get_config_preset(preset: str) -> PipelineConfig:
    """
    Get a predefined configuration preset.
    
    Args:
        preset: Configuration preset name
        
    Returns:
        PipelineConfig instance
        
    Raises:
        ValueError: If preset not found
    """
    if preset not in DEFAULT_CONFIGS:
        available = ", ".join(DEFAULT_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    return DEFAULT_CONFIGS[preset]