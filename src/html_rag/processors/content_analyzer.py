"""
Content Analytics Processor for HTML RAG Pipeline.

This processor provides comprehensive content analysis capabilities including:
- Named entity extraction and linking
- Controversy detection and scoring
- Sentiment analysis
- Topic classification
- Content statistics and readability analysis
- Trend analysis over time
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

try:
    import torch
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModel
    import spacy
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import textstat
    import networkx as nx
    from fuzzywuzzy import fuzz
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Content analytics dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

from ..core.config import ContentAnalyticsConfig
from ..analytics.models import (
    NamedEntity, ControversyScore, SentimentAnalysis, 
    TopicClassification, ContentAnalytics, ContentStatistics,
    TrendAnalysis, AnalyticsResult, EntityType, SentimentLabel, ControversyLevel
)
from ..exceptions.pipeline_exceptions import PipelineError, handle_pipeline_error
from ..utils.logging import PipelineLogger

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """
    Content analytics processor that analyzes documents from ChromaDB.
    
    Provides comprehensive analysis including entity extraction, sentiment analysis,
    controversy detection, topic classification, and trend analysis.
    """
    
    def __init__(self, config: Optional[ContentAnalyticsConfig] = None):
        """
        Initialize the content analyzer.
        
        Args:
            config: Content analytics configuration
        """
        if not DEPENDENCIES_AVAILABLE:
            raise PipelineError(
                "Content analytics dependencies not available. "
                "Install with: pip install html-rag-pipeline[analytics]"
            )
        
        self.config = config or ContentAnalyticsConfig()
        self.logger = PipelineLogger("ContentAnalyzer")
        
        # Initialize components
        self._models = {}
        self._data_cache = {}
        self._nlp_models = {}
        
        # Load data files
        self._load_data_files()
        
        # Initialize models if enabled
        if self.config.enabled:
            self._initialize_models()
    
    def _load_data_files(self) -> None:
        """Load data files for analysis."""
        try:
            # Load entities database
            entities_path = Path(self.config.entities_db_path)
            if entities_path.exists():
                with open(entities_path, 'r', encoding='utf-8') as f:
                    self._data_cache['entities'] = json.load(f)
            else:
                self.logger.warning(f"Entities database not found: {entities_path}")
                self._data_cache['entities'] = {}
            
            # Load controversy keywords
            base_path = entities_path.parent
            controversy_path = base_path / "controversy_keywords.json"
            if controversy_path.exists():
                with open(controversy_path, 'r', encoding='utf-8') as f:
                    self._data_cache['controversy_keywords'] = json.load(f)
            else:
                self._data_cache['controversy_keywords'] = {}
            
            # Load topic patterns
            patterns_path = base_path / "topic_patterns.json"
            if patterns_path.exists():
                with open(patterns_path, 'r', encoding='utf-8') as f:
                    self._data_cache['topic_patterns'] = json.load(f)
            else:
                self._data_cache['topic_patterns'] = {}
            
            # Load sentiment lexicon
            sentiment_path = base_path / "sentiment_lexicon.json"
            if sentiment_path.exists():
                with open(sentiment_path, 'r', encoding='utf-8') as f:
                    self._data_cache['sentiment_lexicon'] = json.load(f)
            else:
                self._data_cache['sentiment_lexicon'] = {}
            
            self.logger.info("Data files loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading data files: {e}")
            raise PipelineError(f"Failed to load data files: {e}")
    
    def _initialize_models(self) -> None:
        """Initialize ML models for analysis."""
        try:
            self.logger.info("Initializing content analytics models...")
            
            # Initialize sentiment analysis model
            if hasattr(transformers, 'pipeline'):
                self._models['sentiment'] = pipeline(
                    "sentiment-analysis",
                    model=self.config.sentiment_model,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Initialize zero-shot classification model
            self._models['classifier'] = pipeline(
                "zero-shot-classification",
                model=self.config.classification_model,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize spaCy models if available
            try:
                # Try to load Ukrainian model
                if spacy.util.is_package("uk_core_news_sm"):
                    self._nlp_models['uk'] = spacy.load("uk_core_news_sm")
                
                # Try to load English model
                if spacy.util.is_package("en_core_web_sm"):
                    self._nlp_models['en'] = spacy.load("en_core_web_sm")
                    
                # Fallback to blank models
                if not self._nlp_models:
                    self._nlp_models['uk'] = spacy.blank("uk")
                    self._nlp_models['en'] = spacy.blank("en")
                    self._nlp_models['ru'] = spacy.blank("ru")
                    
            except OSError:
                self.logger.warning("SpaCy models not available, using basic NLP")
                self._nlp_models['uk'] = spacy.blank("uk")
                self._nlp_models['en'] = spacy.blank("en")
                self._nlp_models['ru'] = spacy.blank("ru")
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise PipelineError(f"Failed to initialize models: {e}")
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (uk, ru, en)
        """
        try:
            # Simple Cyrillic detection
            cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
            latin_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
            
            total_chars = cyrillic_chars + latin_chars
            if total_chars == 0:
                return 'en'  # Default
            
            cyrillic_ratio = cyrillic_chars / total_chars
            
            if cyrillic_ratio > 0.3:
                # Distinguish between Ukrainian and Russian
                ukrainian_indicators = ['і', 'ї', 'є', 'ґ', 'в', 'на', 'що', 'як']
                russian_indicators = ['ы', 'э', 'ё', 'в', 'на', 'что', 'как']
                
                uk_count = sum(1 for indicator in ukrainian_indicators if indicator in text.lower())
                ru_count = sum(1 for indicator in russian_indicators if indicator in text.lower())
                
                return 'uk' if uk_count >= ru_count else 'ru'
            else:
                return 'en'
                
        except Exception:
            return 'en'  # Default fallback
    
    @handle_pipeline_error
    def extract_entities(self, text: str, language: str = 'auto') -> List[NamedEntity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            language: Language code or 'auto' for detection
            
        Returns:
            List of extracted entities
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        entities = []
        
        try:
            # Use spaCy for NER if available
            if language in self._nlp_models:
                nlp = self._nlp_models[language]
                doc = nlp(text)
                
                for ent in doc.ents:
                    entity_type = self._map_spacy_entity_type(ent.label_)
                    entities.append(NamedEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        start_position=ent.start_char,
                        end_position=ent.end_char,
                        confidence=0.8,  # Default confidence for spaCy
                        language=language
                    ))
            
            # Enhance with fuzzy matching against entities database
            entities.extend(self._fuzzy_entity_matching(text, language))
            
            # Remove duplicates and sort by position
            entities = self._deduplicate_entities(entities)
            
            return entities[:self.config.max_entities_per_document]
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def _map_spacy_entity_type(self, spacy_label: str) -> EntityType:
        """Map spaCy entity labels to our EntityType enum."""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'EVENT': EntityType.EVENT,
            'PRODUCT': EntityType.PRODUCT,
            'WORK_OF_ART': EntityType.MISC,
            'LAW': EntityType.MISC,
            'LANGUAGE': EntityType.MISC,
            'DATE': EntityType.MISC,
            'TIME': EntityType.MISC,
            'PERCENT': EntityType.MISC,
            'MONEY': EntityType.MISC,
            'QUANTITY': EntityType.MISC,
            'ORDINAL': EntityType.MISC,
            'CARDINAL': EntityType.MISC
        }
        return mapping.get(spacy_label, EntityType.MISC)
    
    def _fuzzy_entity_matching(self, text: str, language: str) -> List[NamedEntity]:
        """
        Perform fuzzy matching against entities database.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of matched entities
        """
        entities = []
        entities_db = self._data_cache.get('entities', {})
        
        text_lower = text.lower()
        
        try:
            # Check each category in entities database
            for category_name, category_data in entities_db.items():
                for subcategory_name, subcategory_data in category_data.items():
                    for entity_name, entity_info in subcategory_data.items():
                        # Check main entity name
                        self._check_entity_match(
                            text, text_lower, entity_name, entity_info,
                            entities, language
                        )
                        
                        # Check aliases
                        for alias in entity_info.get('aliases', []):
                            self._check_entity_match(
                                text, text_lower, alias, entity_info,
                                entities, language
                            )
            
        except Exception as e:
            self.logger.warning(f"Error in fuzzy entity matching: {e}")
        
        return entities
    
    def _check_entity_match(self, text: str, text_lower: str, entity_name: str,
                           entity_info: Dict, entities: List[NamedEntity], language: str) -> None:
        """Check if an entity matches in the text."""
        entity_lower = entity_name.lower()
        
        # Exact match
        if entity_lower in text_lower:
            start_pos = text_lower.find(entity_lower)
            end_pos = start_pos + len(entity_name)
            
            entities.append(NamedEntity(
                text=text[start_pos:end_pos],
                entity_type=EntityType(entity_info.get('type', 'misc')),
                start_position=start_pos,
                end_position=end_pos,
                confidence=0.9,
                normalized_form=entity_name,
                aliases=entity_info.get('aliases', []),
                description=entity_info.get('description'),
                language=language,
                external_ids=entity_info.get('external_ids', {})
            ))
        
        # Fuzzy match for longer entities
        elif len(entity_name) > 5:
            ratio = fuzz.partial_ratio(entity_lower, text_lower)
            if ratio > 85:  # High similarity threshold
                # Find best matching substring
                words = text.split()
                for i, word in enumerate(words):
                    if fuzz.ratio(word.lower(), entity_lower) > 80:
                        start_pos = text.lower().find(word.lower())
                        end_pos = start_pos + len(word)
                        
                        entities.append(NamedEntity(
                            text=word,
                            entity_type=EntityType(entity_info.get('type', 'misc')),
                            start_position=start_pos,
                            end_position=end_pos,
                            confidence=ratio / 100.0,
                            normalized_form=entity_name,
                            aliases=entity_info.get('aliases', []),
                            description=entity_info.get('description'),
                            language=language,
                            external_ids=entity_info.get('external_ids', {})
                        ))
                        break
    
    def _deduplicate_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Remove duplicate entities and resolve overlaps."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_position)
        
        deduplicated = []
        
        for entity in entities:
            # Check for overlaps with existing entities
            overlap = False
            for existing in deduplicated:
                if (entity.start_position < existing.end_position and 
                    entity.end_position > existing.start_position):
                    # Overlapping entities - keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlap = True
                    break
            
            if not overlap:
                deduplicated.append(entity)
        
        return sorted(deduplicated, key=lambda x: x.start_position)
    
    @handle_pipeline_error
    def analyze_sentiment(self, text: str, language: str = 'auto') -> SentimentAnalysis:
        """
        Analyze sentiment of the text.
        
        Args:
            text: Input text
            language: Language code or 'auto' for detection
            
        Returns:
            Sentiment analysis results
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            # Use transformer model for sentiment analysis
            if 'sentiment' in self._models:
                result = self._models['sentiment'](text[:512])  # Limit text length
                
                if result and len(result) > 0:
                    pred = result[0]
                    label_map = {
                        'POSITIVE': SentimentLabel.POSITIVE,
                        'NEGATIVE': SentimentLabel.NEGATIVE,
                        'NEUTRAL': SentimentLabel.NEUTRAL
                    }
                    
                    label = label_map.get(pred['label'], SentimentLabel.NEUTRAL)
                    score = pred['score'] if label == SentimentLabel.POSITIVE else -pred['score']
                    
                    return SentimentAnalysis(
                        label=label,
                        score=score,
                        confidence=pred['score'],
                        positive_score=pred['score'] if label == SentimentLabel.POSITIVE else 0.0,
                        negative_score=pred['score'] if label == SentimentLabel.NEGATIVE else 0.0,
                        neutral_score=pred['score'] if label == SentimentLabel.NEUTRAL else 0.0,
                        model_name=self.config.sentiment_model,
                        language=language
                    )
            
            # Fallback to lexicon-based analysis
            return self._lexicon_sentiment_analysis(text, language)
            
        except Exception as e:
            self.logger.warning(f"Error in sentiment analysis: {e}")
            return self._lexicon_sentiment_analysis(text, language)
    
    def _lexicon_sentiment_analysis(self, text: str, language: str) -> SentimentAnalysis:
        """Fallback lexicon-based sentiment analysis."""
        lexicon = self._data_cache.get('sentiment_lexicon', {})
        
        positive_words = lexicon.get('positive_words', {}).get(language, [])
        negative_words = lexicon.get('negative_words', {}).get(language, [])
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return SentimentAnalysis(
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.5,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                language=language
            )
        
        positive_ratio = positive_count / total_sentiment_words
        negative_ratio = negative_count / total_sentiment_words
        
        if positive_ratio > negative_ratio:
            label = SentimentLabel.POSITIVE
            score = positive_ratio - negative_ratio
        elif negative_ratio > positive_ratio:
            label = SentimentLabel.NEGATIVE
            score = -(negative_ratio - positive_ratio)
        else:
            label = SentimentLabel.NEUTRAL
            score = 0.0
        
        return SentimentAnalysis(
            label=label,
            score=score,
            confidence=0.7,
            positive_score=positive_ratio,
            negative_score=negative_ratio,
            neutral_score=max(0.0, 1.0 - positive_ratio - negative_ratio),
            language=language
        )
    
    @handle_pipeline_error
    def detect_controversy(self, text: str, entities: List[NamedEntity], 
                          sentiment: SentimentAnalysis, language: str = 'auto') -> ControversyScore:
        """
        Detect controversy level in the text.
        
        Args:
            text: Input text
            entities: Extracted entities
            sentiment: Sentiment analysis results
            language: Language code
            
        Returns:
            Controversy analysis results
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            keywords = self._data_cache.get('controversy_keywords', {})
            
            score = 0.0
            indicators = []
            keyword_matches = []
            evidence_snippets = []
            
            text_lower = text.lower()
            
            # Check controversy keywords
            for intensity, weight in [
                ('high_intensity', 1.0),
                ('medium_intensity', 0.6),
                ('low_intensity', 0.3)
            ]:
                words = keywords.get(intensity, {}).get(language, [])
                for word in words:
                    if word in text_lower:
                        score += weight * 0.2
                        keyword_matches.append(word)
                        indicators.append(f"{intensity}_{word}")
                        
                        # Extract evidence snippet
                        start = max(0, text_lower.find(word) - 50)
                        end = min(len(text), text_lower.find(word) + len(word) + 50)
                        evidence_snippets.append(text[start:end])
            
            # Factor in sentiment
            sentiment_factor = 0.0
            if sentiment.label == SentimentLabel.NEGATIVE:
                sentiment_factor = abs(sentiment.score) * 0.3
                score += sentiment_factor
                indicators.append("negative_sentiment")
            
            # Factor in controversial entities
            controversial_entities = []
            for entity in entities:
                if entity.entity_type in [EntityType.PERSON, EntityType.ORGANIZATION]:
                    # Check if entity is associated with controversy in our database
                    if any(keyword in entity.text.lower() for keyword in keyword_matches):
                        controversial_entities.append(entity.text)
                        score += 0.1
            
            # Normalize score
            score = min(1.0, score)
            
            # Determine controversy level
            if score >= 0.8:
                level = ControversyLevel.CRITICAL
            elif score >= 0.6:
                level = ControversyLevel.HIGH
            elif score >= 0.4:
                level = ControversyLevel.MEDIUM
            else:
                level = ControversyLevel.LOW
            
            return ControversyScore(
                score=score,
                level=level,
                confidence=min(0.9, score + 0.3),
                indicators=indicators[:10],  # Limit indicators
                keyword_matches=keyword_matches[:20],  # Limit matches
                sentiment_factor=sentiment_factor,
                evidence_snippets=evidence_snippets[:5],  # Limit evidence
                related_entities=controversial_entities[:10]
            )
            
        except Exception as e:
            self.logger.warning(f"Error in controversy detection: {e}")
            return ControversyScore(
                score=0.0,
                level=ControversyLevel.LOW,
                confidence=0.1,
                indicators=[],
                keyword_matches=[]
            )
    
    @handle_pipeline_error
    def classify_topics(self, text: str, language: str = 'auto') -> TopicClassification:
        """
        Classify topics in the text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Topic classification results
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            patterns = self._data_cache.get('topic_patterns', {})
            domains_data = patterns.get('domains', {})
            
            # Use zero-shot classification if available
            if 'classifier' in self._models:
                domain_labels = list(domains_data.keys())
                if domain_labels:
                    result = self._models['classifier'](text[:512], domain_labels)
                    
                    topics = {}
                    for label, score in zip(result['labels'], result['scores']):
                        topics[label] = float(score)
                    
                    primary_topic = result['labels'][0]
                    confidence = float(result['scores'][0])
                    
                    return TopicClassification(
                        primary_topic=primary_topic,
                        confidence=confidence,
                        topics=topics,
                        categories=self._extract_categories(primary_topic, domains_data),
                        tags=self._extract_tags(text, language, patterns),
                        domain=primary_topic,
                        subdomain=None
                    )
            
            # Fallback to keyword-based classification
            return self._keyword_topic_classification(text, language, patterns)
            
        except Exception as e:
            self.logger.warning(f"Error in topic classification: {e}")
            return TopicClassification(
                primary_topic="general",
                confidence=0.1,
                topics={"general": 1.0},
                categories=["general"],
                tags=[]
            )
    
    def _extract_categories(self, domain: str, domains_data: Dict) -> List[str]:
        """Extract broader categories for a domain."""
        categories = [domain]
        
        # Add predefined categories based on domain
        domain_categories = {
            'politics': ['government', 'public_policy'],
            'economics': ['finance', 'business'],
            'social': ['society', 'culture'],
            'technology': ['innovation', 'digital'],
            'security': ['defense', 'safety'],
            'environment': ['ecology', 'sustainability']
        }
        
        categories.extend(domain_categories.get(domain, []))
        return categories
    
    def _extract_tags(self, text: str, language: str, patterns: Dict) -> List[str]:
        """Extract content tags from text."""
        tags = []
        text_lower = text.lower()
        
        # Extract sentiment-based tags
        sentiment_indicators = patterns.get('sentiment_indicators', {})
        for sentiment_type, words_dict in sentiment_indicators.items():
            words = words_dict.get(language, [])
            for word in words:
                if word in text_lower:
                    tags.append(f"{sentiment_type}_content")
                    break
        
        return tags
    
    def _keyword_topic_classification(self, text: str, language: str, patterns: Dict) -> TopicClassification:
        """Fallback keyword-based topic classification."""
        domains_data = patterns.get('domains', {})
        text_lower = text.lower()
        
        domain_scores = {}
        
        for domain, domain_info in domains_data.items():
            score = 0.0
            keywords = domain_info.get('keywords', {}).get(language, [])
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
            
            if score > 0:
                domain_scores[domain] = score / len(keywords) if keywords else 0.0
        
        if not domain_scores:
            return TopicClassification(
                primary_topic="general",
                confidence=0.1,
                topics={"general": 1.0},
                categories=["general"],
                tags=[]
            )
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        normalized_scores = {k: v / total_score for k, v in domain_scores.items()}
        
        primary_topic = max(normalized_scores.keys(), key=lambda x: normalized_scores[x])
        confidence = normalized_scores[primary_topic]
        
        return TopicClassification(
            primary_topic=primary_topic,
            confidence=confidence,
            topics=normalized_scores,
            categories=self._extract_categories(primary_topic, domains_data),
            tags=self._extract_tags(text, language, patterns),
            domain=primary_topic
        )
    
    @handle_pipeline_error
    def calculate_statistics(self, text: str, entities: List[NamedEntity], 
                           language: str = 'auto') -> ContentStatistics:
        """
        Calculate content statistics and readability metrics.
        
        Args:
            text: Input text
            entities: Extracted entities
            language: Language code
            
        Returns:
            Content statistics
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            # Basic statistics
            character_count = len(text)
            words = text.split()
            word_count = len(words)
            sentences = text.split('.') + text.split('!') + text.split('?')
            sentence_count = len([s for s in sentences if s.strip()])
            paragraphs = text.split('\n\n')
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            # Readability scores (primarily for English)
            flesch_reading_ease = None
            flesch_kincaid_grade = None
            coleman_liau_index = None
            
            if language == 'en' and word_count > 10:
                try:
                    flesch_reading_ease = textstat.flesch_reading_ease(text)
                    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
                    coleman_liau_index = textstat.coleman_liau_index(text)
                except Exception:
                    pass  # Ignore readability calculation errors
            
            # Complexity metrics
            avg_words_per_sentence = word_count / max(1, sentence_count)
            avg_sentence_length = character_count / max(1, sentence_count)
            unique_words = len(set(word.lower() for word in words))
            unique_words_ratio = unique_words / max(1, word_count)
            
            # Entity statistics
            entity_count = len(entities)
            unique_entities = len(set(entity.normalized_form or entity.text for entity in entities))
            entity_density = (entity_count / max(1, word_count)) * 100
            
            return ContentStatistics(
                character_count=character_count,
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                flesch_reading_ease=flesch_reading_ease,
                flesch_kincaid_grade=flesch_kincaid_grade,
                coleman_liau_index=coleman_liau_index,
                language=language,
                language_confidence=0.8,  # Default confidence
                avg_words_per_sentence=avg_words_per_sentence,
                avg_sentence_length=avg_sentence_length,
                unique_words_ratio=unique_words_ratio,
                entity_count=entity_count,
                unique_entities=unique_entities,
                entity_density=entity_density
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating statistics: {e}")
            return ContentStatistics(
                character_count=len(text),
                word_count=len(text.split()),
                sentence_count=1,
                paragraph_count=1,
                language=language,
                avg_words_per_sentence=len(text.split()),
                avg_sentence_length=len(text),
                unique_words_ratio=1.0,
                entity_count=len(entities),
                unique_entities=len(entities),
                entity_density=0.0
            )
    
    @handle_pipeline_error
    def analyze_document(self, text: str, document_id: str, url: Optional[str] = None) -> ContentAnalytics:
        """
        Perform complete content analysis on a document.
        
        Args:
            text: Document text
            document_id: Document identifier
            url: Optional source URL
            
        Returns:
            Complete content analytics results
        """
        start_time = time.time()
        
        try:
            # Detect language
            language = self._detect_language(text)
            
            # Extract entities
            entities = self.extract_entities(text, language)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(text, language)
            
            # Detect controversy
            controversy = self.detect_controversy(text, entities, sentiment, language)
            
            # Classify topics
            topics = self.classify_topics(text, language)
            
            # Calculate statistics
            statistics = self.calculate_statistics(text, entities, language)
            
            processing_time = time.time() - start_time
            
            # Calculate overall confidence and quality scores
            confidence_score = np.mean([
                sentiment.confidence,
                controversy.confidence,
                topics.confidence,
                statistics.language_confidence or 0.8
            ])
            
            quality_score = self._calculate_quality_score(statistics, entities, sentiment)
            
            return ContentAnalytics(
                document_id=document_id,
                url=url,
                timestamp=datetime.now(),
                entities=entities,
                controversy=controversy,
                sentiment=sentiment,
                topics=topics,
                statistics=statistics,
                processing_time=processing_time,
                model_versions={
                    'sentiment': self.config.sentiment_model,
                    'classification': self.config.classification_model
                },
                language=language,
                confidence_score=confidence_score,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing document {document_id}: {e}")
            raise PipelineError(f"Document analysis failed: {e}")
    
    def _calculate_quality_score(self, statistics: ContentStatistics, 
                               entities: List[NamedEntity], 
                               sentiment: SentimentAnalysis) -> float:
        """Calculate content quality score based on various metrics."""
        try:
            score = 0.0
            
            # Length factor (not too short, not too long)
            if 100 <= statistics.word_count <= 2000:
                score += 0.3
            elif 50 <= statistics.word_count < 100:
                score += 0.2
            elif statistics.word_count > 2000:
                score += 0.2
            
            # Readability factor
            if statistics.flesch_reading_ease is not None:
                if 30 <= statistics.flesch_reading_ease <= 80:
                    score += 0.2
                else:
                    score += 0.1
            else:
                score += 0.1  # Default for non-English
            
            # Entity density factor
            if 1.0 <= statistics.entity_density <= 10.0:
                score += 0.2
            elif statistics.entity_density > 0:
                score += 0.1
            
            # Sentence structure factor
            if 5 <= statistics.avg_words_per_sentence <= 25:
                score += 0.1
            
            # Vocabulary diversity factor
            if statistics.unique_words_ratio >= 0.5:
                score += 0.2
            elif statistics.unique_words_ratio >= 0.3:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception:
            return 0.5  # Default quality score
    
    @handle_pipeline_error
    def analyze_batch(self, documents: List[Dict[str, Any]]) -> AnalyticsResult:
        """
        Analyze multiple documents in batch.
        
        Args:
            documents: List of documents with 'text', 'id', and optional 'url'
            
        Returns:
            Batch analytics results
        """
        start_time = time.time()
        
        analytics = []
        errors = []
        warnings = []
        
        self.logger.info(f"Starting batch analysis of {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            try:
                text = doc.get('text', '')
                doc_id = doc.get('id', f'doc_{i}')
                url = doc.get('url')
                
                if not text.strip():
                    warnings.append(f"Empty text for document {doc_id}")
                    continue
                
                result = self.analyze_document(text, doc_id, url)
                analytics.append(result)
                
            except Exception as e:
                error_msg = f"Failed to analyze document {doc.get('id', i)}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        processing_time = time.time() - start_time
        successful = len(analytics)
        failed = len(errors)
        total = len(documents)
        
        # Calculate summary metrics
        avg_sentiment = None
        avg_controversy = None
        top_entities = []
        top_topics = []
        
        if analytics:
            sentiments = [a.sentiment.score for a in analytics]
            controversies = [a.controversy.score for a in analytics]
            
            avg_sentiment = np.mean(sentiments)
            avg_controversy = np.mean(controversies)
            
            # Extract top entities and topics
            all_entities = []
            all_topics = []
            
            for result in analytics:
                all_entities.extend([e.normalized_form or e.text for e in result.entities])
                all_topics.append(result.topics.primary_topic)
            
            # Count frequencies
            entity_counts = {}
            topic_counts = {}
            
            for entity in all_entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
            
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Get top 10
            top_entities = sorted(entity_counts.keys(), 
                                key=lambda x: entity_counts[x], reverse=True)[:10]
            top_topics = sorted(topic_counts.keys(), 
                              key=lambda x: topic_counts[x], reverse=True)[:10]
        
        return AnalyticsResult(
            total_documents=total,
            successful=successful,
            failed=failed,
            analytics=analytics,
            processing_time=processing_time,
            average_processing_time=processing_time / max(1, total),
            avg_sentiment=avg_sentiment,
            avg_controversy=avg_controversy,
            top_entities=top_entities,
            top_topics=top_topics,
            errors=errors,
            warnings=warnings
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'sentiment_model': self.config.sentiment_model,
            'classification_model': self.config.classification_model,
            'nlp_models': list(self._nlp_models.keys()),
            'models_loaded': list(self._models.keys()),
            'config': self.config.dict() if hasattr(self.config, 'dict') else str(self.config)
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear model cache
            self._models.clear()
            
            # Clear NLP models
            for nlp_model in self._nlp_models.values():
                if hasattr(nlp_model, 'vocab'):
                    del nlp_model.vocab
            self._nlp_models.clear()
            
            # Clear data cache
            self._data_cache.clear()
            
            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Content analyzer cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")