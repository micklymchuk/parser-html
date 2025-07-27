"""
Content Analytics Module for HTML RAG Pipeline.

This module provides comprehensive content analysis capabilities including:
- Named entity extraction and linking
- Controversy detection and scoring
- Sentiment analysis
- Topic classification
- Content statistics and readability analysis
- Trend analysis over time
"""

from .models import (
    NamedEntity,
    ControversyScore,
    SentimentAnalysis,
    TopicClassification,
    ContentAnalytics,
    ContentStatistics,
    TrendAnalysis,
    AnalyticsResult
)

__all__ = [
    "NamedEntity",
    "ControversyScore", 
    "SentimentAnalysis",
    "TopicClassification",
    "ContentAnalytics",
    "ContentStatistics",
    "TrendAnalysis",
    "AnalyticsResult"
]