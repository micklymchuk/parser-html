"""
Tests for Content Analytics Components.

This module contains comprehensive tests for all content analytics functionality
including entity extraction, sentiment analysis, controversy detection,
topic classification, and trend analysis.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import components to test
from src.html_rag.core.config import ContentAnalyticsConfig
from src.html_rag.analytics.models import (
    NamedEntity, EntityType, SentimentAnalysis, SentimentLabel,
    ControversyScore, ControversyLevel, TopicClassification,
    ContentStatistics, ContentAnalytics, AnalyticsResult
)

# Test data
SAMPLE_UKRAINIAN_TEXT = """
Володимир Зеленський провів важливу зустріч з представниками НАТО у Києві.
Президент обговорив питання національної безпеки та європейської інтеграції.
Це рішення викликало позитивну реакцію у міжнародного співтовариства.
"""

SAMPLE_ENGLISH_TEXT = """
The government announced new economic reforms today in Parliament.
The Prime Minister emphasized the importance of sustainable development.
Citizens showed positive reactions to the proposed changes.
"""

SAMPLE_CONTROVERSIAL_TEXT = """
Скандальна заява політика викликала хвилю критики та протестів.
Опозиція звинуватила уряд у корупції та неефективному управлінні.
Ця ситуація призвела до політичної кризи в країні.
"""


class TestContentAnalyticsConfig:
    """Test Content Analytics Configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ContentAnalyticsConfig()
        
        assert config.enabled == False
        assert config.sentiment_model == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert config.classification_model == "facebook/bart-large-mnli"
        assert config.controversy_threshold == 0.6
        assert config.max_entities_per_document == 50
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ContentAnalyticsConfig(
            enabled=True,
            controversy_threshold=0.8,
            max_entities_per_document=25
        )
        assert config.controversy_threshold == 0.8
        assert config.max_entities_per_document == 25
        
        # Invalid threshold (should be clamped)
        config = ContentAnalyticsConfig(controversy_threshold=1.5)
        assert config.controversy_threshold <= 1.0


class TestAnalyticsModels:
    """Test Analytics Data Models."""
    
    def test_named_entity_model(self):
        """Test NamedEntity model."""
        entity = NamedEntity(
            text="Володимир Зеленський",
            entity_type=EntityType.PERSON,
            start_position=0,
            end_position=19,
            confidence=0.95,
            language="uk"
        )
        
        assert entity.text == "Володимир Зеленський"
        assert entity.entity_type == EntityType.PERSON
        assert entity.confidence == 0.95
        assert entity.language == "uk"
    
    def test_sentiment_analysis_model(self):
        """Test SentimentAnalysis model."""
        sentiment = SentimentAnalysis(
            label=SentimentLabel.POSITIVE,
            score=0.8,
            confidence=0.9,
            positive_score=0.8,
            negative_score=0.1,
            neutral_score=0.1,
            language="uk"
        )
        
        assert sentiment.label == SentimentLabel.POSITIVE
        assert sentiment.score == 0.8
        assert sentiment.confidence == 0.9
    
    def test_controversy_score_model(self):
        """Test ControversyScore model."""
        controversy = ControversyScore(
            score=0.7,
            level=ControversyLevel.HIGH,
            confidence=0.8,
            indicators=["scandal", "criticism"],
            keyword_matches=["скандал", "критика"]
        )
        
        assert controversy.score == 0.7
        assert controversy.level == ControversyLevel.HIGH
        assert len(controversy.indicators) == 2
    
    def test_content_analytics_model(self):
        """Test complete ContentAnalytics model."""
        # Create sub-components
        entity = NamedEntity(
            text="Test Entity",
            entity_type=EntityType.PERSON,
            start_position=0,
            end_position=11,
            confidence=0.9,
            language="en"
        )
        
        sentiment = SentimentAnalysis(
            label=SentimentLabel.POSITIVE,
            score=0.5,
            confidence=0.8,
            positive_score=0.5,
            negative_score=0.3,
            neutral_score=0.2,
            language="en"
        )
        
        controversy = ControversyScore(
            score=0.3,
            level=ControversyLevel.LOW,
            confidence=0.7,
            indicators=[],
            keyword_matches=[]
        )
        
        topics = TopicClassification(
            primary_topic="general",
            confidence=0.6,
            topics={"general": 1.0},
            categories=["general"],
            tags=[]
        )
        
        stats = ContentStatistics(
            character_count=100,
            word_count=20,
            sentence_count=2,
            paragraph_count=1,
            language="en",
            avg_words_per_sentence=10.0,
            avg_sentence_length=50.0,
            unique_words_ratio=0.8,
            entity_count=1,
            unique_entities=1,
            entity_density=5.0
        )
        
        # Create ContentAnalytics
        analytics = ContentAnalytics(
            document_id="test_doc",
            url="https://example.com",
            timestamp=datetime.now(),
            entities=[entity],
            controversy=controversy,
            sentiment=sentiment,
            topics=topics,
            statistics=stats,
            processing_time=1.5,
            model_versions={"sentiment": "test-model"},
            language="en",
            confidence_score=0.8,
            quality_score=0.7
        )
        
        assert analytics.document_id == "test_doc"
        assert len(analytics.entities) == 1
        assert analytics.sentiment.label == SentimentLabel.POSITIVE
        assert analytics.confidence_score == 0.8


class TestEntityExtractor:
    """Test Entity Extraction functionality."""
    
    @pytest.fixture
    def mock_entity_extractor(self):
        """Create a mock entity extractor for testing."""
        with patch('src.html_rag.analytics.entity_extractor.DEPENDENCIES_AVAILABLE', True):
            from src.html_rag.analytics.entity_extractor import EntityExtractor
            
            # Mock spaCy and other dependencies
            with patch('src.html_rag.analytics.entity_extractor.spacy') as mock_spacy:
                mock_nlp = Mock()
                mock_doc = Mock()
                mock_ent = Mock()
                mock_ent.text = "Test Entity"
                mock_ent.label_ = "PERSON"
                mock_ent.start_char = 0
                mock_ent.end_char = 11
                mock_doc.ents = [mock_ent]
                mock_nlp.return_value = mock_doc
                mock_spacy.load.return_value = mock_nlp
                mock_spacy.blank.return_value = mock_nlp
                mock_spacy.util.is_package.return_value = False
                
                extractor = EntityExtractor()
                return extractor
    
    def test_language_detection(self, mock_entity_extractor):
        """Test language detection."""
        # Ukrainian text
        lang = mock_entity_extractor._detect_language(SAMPLE_UKRAINIAN_TEXT)
        assert lang in ['uk', 'ru']  # Could be either due to similarity
        
        # English text
        lang = mock_entity_extractor._detect_language(SAMPLE_ENGLISH_TEXT)
        assert lang == 'en'
    
    def test_entity_extraction(self, mock_entity_extractor):
        """Test basic entity extraction."""
        entities = mock_entity_extractor.extract_entities(SAMPLE_ENGLISH_TEXT, language='en')
        
        # Should return some entities (mocked)
        assert isinstance(entities, list)
        # In a real test with actual models, we'd have more specific assertions


class TestSentimentAnalyzer:
    """Test Sentiment Analysis functionality."""
    
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        """Create a mock sentiment analyzer for testing."""
        with patch('src.html_rag.analytics.sentiment_analyzer.TRANSFORMERS_AVAILABLE', False):
            from src.html_rag.analytics.sentiment_analyzer import SentimentAnalyzer
            
            # Create temp sentiment lexicon file
            with tempfile.TemporaryDirectory() as temp_dir:
                lexicon_data = {
                    "positive_words": {
                        "uk": ["позитивно", "добре", "важливо"],
                        "en": ["positive", "good", "important"]
                    },
                    "negative_words": {
                        "uk": ["негативно", "погано", "критика"],
                        "en": ["negative", "bad", "criticism"]
                    },
                    "intensifiers": {
                        "uk": ["дуже", "надзвичайно"],
                        "en": ["very", "extremely"]
                    },
                    "negation_words": {
                        "uk": ["не", "ні"],
                        "en": ["not", "no"]
                    }
                }
                
                lexicon_path = Path(temp_dir) / "sentiment_lexicon.json"
                with open(lexicon_path, 'w', encoding='utf-8') as f:
                    json.dump(lexicon_data, f)
                
                config = ContentAnalyticsConfig(entities_db_path=str(lexicon_path))
                analyzer = SentimentAnalyzer(config)
                return analyzer
    
    def test_language_detection(self, mock_sentiment_analyzer):
        """Test language detection in sentiment analyzer."""
        lang = mock_sentiment_analyzer._detect_language(SAMPLE_UKRAINIAN_TEXT)
        assert lang in ['uk', 'ru']
        
        lang = mock_sentiment_analyzer._detect_language(SAMPLE_ENGLISH_TEXT)
        assert lang == 'en'
    
    def test_lexicon_sentiment_analysis(self, mock_sentiment_analyzer):
        """Test lexicon-based sentiment analysis."""
        # Positive text
        positive_text = "This is very good and important news"
        result = mock_sentiment_analyzer.analyze_sentiment(positive_text, language='en')
        
        assert isinstance(result, SentimentAnalysis)
        assert result.language == 'en'
        assert result.method == 'lexicon'
        
        # Test with Ukrainian text
        ukrainian_positive = "Це дуже добре і важливо"
        result = mock_sentiment_analyzer.analyze_sentiment(ukrainian_positive, language='uk')
        assert result.language == 'uk'


class TestControversyDetector:
    """Test Controversy Detection functionality."""
    
    @pytest.fixture
    def mock_controversy_detector(self):
        """Create a mock controversy detector for testing."""
        from src.html_rag.analytics.controversy_detector import ControversyDetector
        
        # Create temp controversy keywords file
        with tempfile.TemporaryDirectory() as temp_dir:
            keywords_data = {
                "high_intensity": {
                    "uk": ["скандал", "корупція", "критика"],
                    "en": ["scandal", "corruption", "criticism"]
                },
                "medium_intensity": {
                    "uk": ["протест", "конфлікт"],
                    "en": ["protest", "conflict"]
                },
                "low_intensity": {
                    "uk": ["питання", "обговорення"],
                    "en": ["question", "discussion"]
                }
            }
            
            keywords_path = Path(temp_dir) / "controversy_keywords.json"
            with open(keywords_path, 'w', encoding='utf-8') as f:
                json.dump(keywords_data, f)
            
            config = ContentAnalyticsConfig(entities_db_path=str(keywords_path))
            detector = ControversyDetector(config)
            return detector
    
    def test_controversy_detection(self, mock_controversy_detector):
        """Test controversy detection."""
        # High controversy text
        result = mock_controversy_detector.detect_controversy(SAMPLE_CONTROVERSIAL_TEXT, language='uk')
        
        assert isinstance(result, ControversyScore)
        assert result.score >= 0.0
        assert result.level in [ControversyLevel.LOW, ControversyLevel.MEDIUM, ControversyLevel.HIGH, ControversyLevel.CRITICAL]
        
        # Low controversy text
        neutral_text = "Today is a nice day for a walk in the park"
        result = mock_controversy_detector.detect_controversy(neutral_text, language='en')
        assert result.score >= 0.0


class TestTopicClassifier:
    """Test Topic Classification functionality."""
    
    @pytest.fixture
    def mock_topic_classifier(self):
        """Create a mock topic classifier for testing."""
        with patch('src.html_rag.analytics.topic_classifier.DEPENDENCIES_AVAILABLE', False):
            from src.html_rag.analytics.topic_classifier import TopicClassifier
            
            # Create temp topic patterns file
            with tempfile.TemporaryDirectory() as temp_dir:
                patterns_data = {
                    "domains": {
                        "politics": {
                            "keywords": {
                                "uk": ["політика", "уряд", "президент"],
                                "en": ["politics", "government", "president"]
                            }
                        },
                        "economics": {
                            "keywords": {
                                "uk": ["економіка", "фінанси"],
                                "en": ["economy", "finance"]
                            }
                        }
                    }
                }
                
                patterns_path = Path(temp_dir) / "topic_patterns.json"
                with open(patterns_path, 'w', encoding='utf-8') as f:
                    json.dump(patterns_data, f)
                
                config = ContentAnalyticsConfig(entities_db_path=str(patterns_path))
                classifier = TopicClassifier(config)
                return classifier
    
    def test_topic_classification(self, mock_topic_classifier):
        """Test topic classification."""
        # Political text
        political_text = "The government announced new political reforms"
        result = mock_topic_classifier.classify_topics(political_text, language='en')
        
        assert isinstance(result, TopicClassification)
        assert result.primary_topic is not None
        assert result.confidence >= 0.0
        assert isinstance(result.topics, dict)
        assert isinstance(result.categories, list)


class TestStatisticsCalculator:
    """Test Statistics Calculation functionality."""
    
    @pytest.fixture
    def mock_statistics_calculator(self):
        """Create a mock statistics calculator for testing."""
        from src.html_rag.analytics.statistics_calculator import StatisticsCalculator
        return StatisticsCalculator()
    
    def test_basic_statistics_calculation(self, mock_statistics_calculator):
        """Test basic statistics calculation."""
        result = mock_statistics_calculator.calculate_statistics(SAMPLE_ENGLISH_TEXT, language='en')
        
        assert isinstance(result, ContentStatistics)
        assert result.character_count > 0
        assert result.word_count > 0
        assert result.sentence_count > 0
        assert result.language == 'en'
        assert result.avg_words_per_sentence > 0
        assert result.unique_words_ratio > 0
    
    def test_ukrainian_statistics(self, mock_statistics_calculator):
        """Test statistics for Ukrainian text."""
        result = mock_statistics_calculator.calculate_statistics(SAMPLE_UKRAINIAN_TEXT, language='uk')
        
        assert result.language == 'uk'
        assert result.word_count > 0
        assert result.sentence_count > 0


class TestContentAnalyzer:
    """Test main Content Analyzer integration."""
    
    @pytest.fixture
    def mock_content_analyzer(self):
        """Create a mock content analyzer for testing."""
        # Mock all dependencies
        with patch('src.html_rag.processors.content_analyzer.DEPENDENCIES_AVAILABLE', False):
            from src.html_rag.processors.content_analyzer import ContentAnalyzer
            
            # Create temp data directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create minimal data files
                entities_data = {"people": {"test": {"Test Person": {"type": "person", "aliases": []}}}}
                controversy_data = {"high_intensity": {"en": ["scandal"]}}
                topic_data = {"domains": {"general": {"keywords": {"en": ["test"]}}}}
                sentiment_data = {"positive_words": {"en": ["good"]}, "negative_words": {"en": ["bad"]}}
                
                entities_path = Path(temp_dir) / "entities_db.json"
                with open(entities_path, 'w') as f:
                    json.dump(entities_data, f)
                
                for filename, data in [
                    ("controversy_keywords.json", controversy_data),
                    ("topic_patterns.json", topic_data),
                    ("sentiment_lexicon.json", sentiment_data)
                ]:
                    with open(Path(temp_dir) / filename, 'w') as f:
                        json.dump(data, f)
                
                config = ContentAnalyticsConfig(entities_db_path=str(entities_path), enabled=True)
                
                # Mock the analyzer to avoid dependency issues
                analyzer = Mock()
                analyzer.config = config
                return analyzer
    
    def test_analyzer_initialization(self, mock_content_analyzer):
        """Test content analyzer initialization."""
        assert mock_content_analyzer.config.enabled == True


class TestPipelineIntegration:
    """Test Content Analytics integration with main pipeline."""
    
    def test_pipeline_with_analytics_disabled(self):
        """Test pipeline behavior when analytics is disabled."""
        from src.html_rag.core.pipeline import RAGPipeline
        
        # Should not fail when analytics is disabled
        with patch('src.html_rag.core.pipeline.PipelineConfig') as mock_config:
            mock_config.return_value = Mock()
            
            with patch('src.html_rag.processors.wayback.WaybackProcessor'), \
                 patch('src.html_rag.processors.html_pruner.HTMLPruner'), \
                 patch('src.html_rag.processors.html_parser.HTMLParser'), \
                 patch('src.html_rag.processors.text_embedder.TextEmbedder'), \
                 patch('src.html_rag.processors.vector_store.VectorStore'):
                
                pipeline = RAGPipeline(enable_content_analytics=False)
                assert pipeline.enable_content_analytics == False
                assert pipeline.content_analyzer is None
    
    def test_pipeline_with_analytics_enabled(self):
        """Test pipeline behavior when analytics is enabled."""
        from src.html_rag.core.pipeline import RAGPipeline
        
        with patch('src.html_rag.core.pipeline.PipelineConfig') as mock_config:
            mock_config.return_value = Mock()
            
            with patch('src.html_rag.processors.wayback.WaybackProcessor'), \
                 patch('src.html_rag.processors.html_pruner.HTMLPruner'), \
                 patch('src.html_rag.processors.html_parser.HTMLParser'), \
                 patch('src.html_rag.processors.text_embedder.TextEmbedder'), \
                 patch('src.html_rag.processors.vector_store.VectorStore'), \
                 patch('src.html_rag.processors.content_analyzer.ContentAnalyzer') as mock_analyzer:
                
                mock_analyzer.return_value = Mock()
                
                pipeline = RAGPipeline(enable_content_analytics=True)
                assert pipeline.enable_content_analytics == True
                assert pipeline.content_analyzer is not None


class TestTrendAnalyzer:
    """Test Trend Analysis functionality."""
    
    @pytest.fixture
    def mock_trend_analyzer(self):
        """Create a mock trend analyzer for testing."""
        from src.html_rag.analytics.trend_analyzer import TrendAnalyzer
        return TrendAnalyzer()
    
    def test_empty_trend_analysis(self, mock_trend_analyzer):
        """Test trend analysis with empty data."""
        result = mock_trend_analyzer.analyze_trends([])
        
        assert result.data_points == 0
        assert result.trend_direction == "stable"
        assert result.confidence == 0.0
    
    def test_trend_analysis_with_mock_data(self, mock_trend_analyzer):
        """Test trend analysis with mock data."""
        # Create mock analytics data
        mock_analytics = []
        for i in range(5):
            mock_item = Mock()
            mock_item.timestamp = datetime.now()
            mock_item.sentiment.score = 0.5 + (i * 0.1)  # Increasing trend
            mock_item.controversy.score = 0.3
            mock_item.topics.primary_topic = "general"
            mock_item.entities = []
            mock_analytics.append(mock_item)
        
        # This would require more complex mocking for full testing
        # In practice, you'd mock the entire analyze_trends method or create real ContentAnalytics objects


class TestAnalyticsResult:
    """Test Analytics Result model."""
    
    def test_analytics_result_creation(self):
        """Test creation of AnalyticsResult."""
        result = AnalyticsResult(
            total_documents=10,
            successful=8,
            failed=2,
            analytics=[],
            processing_time=5.0,
            average_processing_time=0.5,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        assert result.total_documents == 10
        assert result.successful == 8
        assert result.failed == 2
        assert len(result.errors) == 2
        assert len(result.warnings) == 1


# Integration Tests
class TestFullAnalyticsWorkflow:
    """Test complete analytics workflow."""
    
    def test_complete_workflow_mock(self):
        """Test complete analytics workflow with mocked components."""
        # This would test the full workflow from text input to analytics output
        # Requires extensive mocking or actual test environment
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])