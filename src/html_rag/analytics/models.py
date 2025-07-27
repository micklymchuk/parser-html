"""
Data models for content analytics using Pydantic.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_V2 = False
    field_validator = validator


class EntityType(str, Enum):
    """Types of named entities."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    TOPIC = "topic"
    PRODUCT = "product"
    MISC = "misc"


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ControversyLevel(str, Enum):
    """Controversy level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NamedEntity(BaseModel):
    """Named entity extracted from content."""
    
    text: str = Field(description="Entity text as it appears in content")
    entity_type: EntityType = Field(description="Type of entity")
    start_position: int = Field(description="Start character position in text", ge=0)
    end_position: int = Field(description="End character position in text", ge=0)
    confidence: float = Field(description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
    normalized_form: Optional[str] = Field(default=None, description="Normalized entity name")
    aliases: List[str] = Field(default_factory=list, description="Known aliases for this entity")
    description: Optional[str] = Field(default=None, description="Entity description")
    
    # Additional metadata
    language: Optional[str] = Field(default=None, description="Language of the entity")
    external_ids: Dict[str, str] = Field(default_factory=dict, description="External database IDs")
    
    if PYDANTIC_V2:
        @field_validator('end_position')
        @classmethod
        def validate_positions(cls, v, info):
            """Ensure end position is after start position."""
            if 'start_position' in info.data and v <= info.data['start_position']:
                raise ValueError("End position must be greater than start position")
            return v
    else:
        @validator('end_position')
        def validate_positions(cls, v, values):
            """Ensure end position is after start position."""
            if 'start_position' in values and v <= values['start_position']:
                raise ValueError("End position must be greater than start position")
            return v


class ControversyScore(BaseModel):
    """Controversy analysis results."""
    
    score: float = Field(description="Controversy score (0.0-1.0)", ge=0.0, le=1.0)
    level: ControversyLevel = Field(description="Controversy level classification")
    confidence: float = Field(description="Confidence in the assessment", ge=0.0, le=1.0)
    
    # Analysis details
    indicators: List[str] = Field(default_factory=list, description="Controversy indicators found")
    keyword_matches: List[str] = Field(default_factory=list, description="Matched controversy keywords")
    sentiment_factor: Optional[float] = Field(default=None, description="Sentiment contribution to score")
    context_factor: Optional[float] = Field(default=None, description="Context contribution to score")
    
    # Supporting evidence
    evidence_snippets: List[str] = Field(default_factory=list, description="Text snippets supporting the score")
    related_entities: List[str] = Field(default_factory=list, description="Entities contributing to controversy")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""
    
    label: SentimentLabel = Field(description="Overall sentiment label")
    score: float = Field(description="Sentiment score (-1.0 to 1.0)", ge=-1.0, le=1.0)
    confidence: float = Field(description="Confidence in the classification", ge=0.0, le=1.0)
    
    # Detailed scores
    positive_score: float = Field(description="Positive sentiment score", ge=0.0, le=1.0)
    negative_score: float = Field(description="Negative sentiment score", ge=0.0, le=1.0)
    neutral_score: float = Field(description="Neutral sentiment score", ge=0.0, le=1.0)
    
    # Model information
    model_name: Optional[str] = Field(default=None, description="Model used for analysis")
    language: Optional[str] = Field(default=None, description="Detected language")


class TopicClassification(BaseModel):
    """Topic classification results."""
    
    primary_topic: str = Field(description="Primary topic classification")
    confidence: float = Field(description="Confidence in primary topic", ge=0.0, le=1.0)
    
    # All topics with scores
    topics: Dict[str, float] = Field(description="All topics with confidence scores")
    categories: List[str] = Field(default_factory=list, description="Broader categories")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    
    # Domain-specific classifications
    domain: Optional[str] = Field(default=None, description="Content domain (politics, business, etc.)")
    subdomain: Optional[str] = Field(default=None, description="Content subdomain")


class ContentStatistics(BaseModel):
    """Content statistics and readability metrics."""
    
    # Basic statistics
    character_count: int = Field(description="Total character count", ge=0)
    word_count: int = Field(description="Total word count", ge=0)
    sentence_count: int = Field(description="Total sentence count", ge=0)
    paragraph_count: int = Field(description="Total paragraph count", ge=0)
    
    # Readability scores
    flesch_reading_ease: Optional[float] = Field(default=None, description="Flesch Reading Ease score")
    flesch_kincaid_grade: Optional[float] = Field(default=None, description="Flesch-Kincaid Grade Level")
    coleman_liau_index: Optional[float] = Field(default=None, description="Coleman-Liau Index")
    
    # Language statistics
    language: Optional[str] = Field(default=None, description="Detected primary language")
    language_confidence: Optional[float] = Field(default=None, description="Language detection confidence")
    
    # Complexity metrics
    avg_words_per_sentence: float = Field(description="Average words per sentence", ge=0.0)
    avg_sentence_length: float = Field(description="Average sentence length", ge=0.0)
    unique_words_ratio: float = Field(description="Ratio of unique words", ge=0.0, le=1.0)
    
    # Entity statistics
    entity_count: int = Field(description="Total named entities found", ge=0)
    unique_entities: int = Field(description="Count of unique entities", ge=0)
    entity_density: float = Field(description="Entities per 100 words", ge=0.0)


class TrendAnalysis(BaseModel):
    """Trend analysis over time."""
    
    timeframe: str = Field(description="Analysis timeframe (e.g., '2023-01-01 to 2024-01-01')")
    data_points: int = Field(description="Number of data points analyzed", ge=0)
    
    # Trend metrics
    sentiment_trend: Optional[Dict[str, float]] = Field(default=None, description="Sentiment trend over time")
    controversy_trend: Optional[Dict[str, float]] = Field(default=None, description="Controversy trend over time")
    entity_trends: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Entity mention trends")
    topic_trends: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Topic trend analysis")
    
    # Statistical measures
    trend_direction: Optional[str] = Field(default=None, description="Overall trend direction")
    volatility_score: Optional[float] = Field(default=None, description="Trend volatility score")
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(default=None, description="Correlation between metrics")


class ContentAnalytics(BaseModel):
    """Complete content analytics result for a document."""
    
    document_id: str = Field(description="Document identifier")
    url: Optional[str] = Field(default=None, description="Source URL")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    # Core analytics
    entities: List[NamedEntity] = Field(default_factory=list, description="Extracted named entities")
    controversy: ControversyScore = Field(description="Controversy analysis")
    sentiment: SentimentAnalysis = Field(description="Sentiment analysis")
    topics: TopicClassification = Field(description="Topic classification")
    statistics: ContentStatistics = Field(description="Content statistics")
    
    # Optional analysis
    trends: Optional[TrendAnalysis] = Field(default=None, description="Trend analysis")
    
    # Metadata
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    model_versions: Dict[str, str] = Field(default_factory=dict, description="Model versions used")
    language: Optional[str] = Field(default=None, description="Detected language")
    
    # Additional metrics
    confidence_score: float = Field(description="Overall analysis confidence", ge=0.0, le=1.0)
    quality_score: float = Field(description="Content quality score", ge=0.0, le=1.0)


class AnalyticsResult(BaseModel):
    """Batch analytics processing result."""
    
    total_documents: int = Field(description="Total documents processed", ge=0)
    successful: int = Field(description="Successfully processed documents", ge=0)
    failed: int = Field(description="Failed document processing", ge=0)
    
    # Aggregated results
    analytics: List[ContentAnalytics] = Field(default_factory=list, description="Individual document analytics")
    
    # Batch statistics
    processing_time: float = Field(description="Total processing time in seconds", ge=0.0)
    average_processing_time: float = Field(description="Average time per document", ge=0.0)
    
    # Summary metrics
    avg_sentiment: Optional[float] = Field(default=None, description="Average sentiment score")
    avg_controversy: Optional[float] = Field(default=None, description="Average controversy score")
    top_entities: List[str] = Field(default_factory=list, description="Most frequent entities")
    top_topics: List[str] = Field(default_factory=list, description="Most common topics")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_documents == 0:
            return 0.0
        return self.successful / self.total_documents
    
    if PYDANTIC_V2:
        @field_validator('successful', 'failed')
        @classmethod
        def validate_counts(cls, v, info):
            """Ensure counts don't exceed total."""
            if 'total_documents' in info.data:
                if v > info.data['total_documents']:
                    raise ValueError("Count cannot exceed total documents")
            return v
    else:
        @validator('successful', 'failed')
        def validate_counts(cls, v, values):
            """Ensure counts don't exceed total."""
            if 'total_documents' in values:
                if v > values['total_documents']:
                    raise ValueError("Count cannot exceed total documents")
            return v