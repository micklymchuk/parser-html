"""
Controversy Detection Component for Content Analytics.

This module provides controversy detection functionality including:
- Keyword-based controversy scoring
- Sentiment-based controversy amplification
- Entity-related controversy detection
- Evidence collection and explanation
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..analytics.models import (
    NamedEntity, ControversyScore, SentimentAnalysis, 
    ControversyLevel, SentimentLabel
)
from ..core.config import ContentAnalyticsConfig
from ..exceptions.pipeline_exceptions import PipelineError, handle_pipeline_error
from ..utils.logging import PipelineLogger

logger = logging.getLogger(__name__)


class ControversyDetector:
    """
    Detects and scores controversy in text content.
    
    Uses multiple signals including controversy keywords, sentiment analysis,
    and entity relationships to determine overall controversy level.
    """
    
    def __init__(self, config: Optional[ContentAnalyticsConfig] = None):
        """
        Initialize the controversy detector.
        
        Args:
            config: Content analytics configuration
        """
        self.config = config or ContentAnalyticsConfig()
        self.logger = PipelineLogger("ControversyDetector")
        self._controversy_keywords = {}
        self._load_controversy_data()
    
    def _load_controversy_data(self) -> None:
        """Load controversy keywords and patterns."""
        try:
            base_path = Path(self.config.entities_db_path).parent
            controversy_path = base_path / "controversy_keywords.json"
            
            if controversy_path.exists():
                with open(controversy_path, 'r', encoding='utf-8') as f:
                    self._controversy_keywords = json.load(f)
                self.logger.info("Controversy keywords loaded successfully")
            else:
                self.logger.warning(f"Controversy keywords file not found: {controversy_path}")
                self._controversy_keywords = {}
                
        except Exception as e:
            self.logger.error(f"Error loading controversy data: {e}")
            self._controversy_keywords = {}
    
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
    def detect_controversy(self, text: str, entities: Optional[List[NamedEntity]] = None, 
                          sentiment: Optional[SentimentAnalysis] = None, 
                          language: str = 'auto') -> ControversyScore:
        """
        Detect controversy level in text.
        
        Args:
            text: Input text to analyze
            entities: Optional list of named entities
            sentiment: Optional sentiment analysis results
            language: Language code or 'auto' for detection
            
        Returns:
            Controversy analysis results
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            score = 0.0
            indicators = []
            keyword_matches = []
            evidence_snippets = []
            confidence = 0.5
            
            text_lower = text.lower()
            
            # 1. Keyword-based controversy detection
            keyword_score = self._analyze_controversy_keywords(
                text_lower, language, keyword_matches, indicators, evidence_snippets
            )
            score += keyword_score
            
            # 2. Sentiment-based controversy amplification
            sentiment_factor = 0.0
            if sentiment:
                sentiment_factor = self._analyze_sentiment_controversy(
                    sentiment, indicators
                )
                score += sentiment_factor
            
            # 3. Entity-related controversy
            entity_factor = 0.0
            controversial_entities = []
            if entities:
                entity_factor, controversial_entities = self._analyze_entity_controversy(
                    entities, keyword_matches, indicators
                )
                score += entity_factor
            
            # 4. Contextual controversy patterns
            context_factor = self._analyze_contextual_patterns(
                text_lower, language, indicators, evidence_snippets
            )
            score += context_factor
            
            # Normalize score and calculate confidence
            score = min(1.0, score)
            confidence = self._calculate_confidence(
                keyword_score, sentiment_factor, entity_factor, context_factor
            )
            
            # Determine controversy level
            level = self._determine_controversy_level(score)
            
            return ControversyScore(
                score=score,
                level=level,
                confidence=confidence,
                indicators=indicators[:15],  # Limit indicators
                keyword_matches=keyword_matches[:25],  # Limit matches
                sentiment_factor=sentiment_factor,
                evidence_snippets=evidence_snippets[:8],  # Limit evidence
                related_entities=controversial_entities[:12]
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
    
    def _analyze_controversy_keywords(self, text_lower: str, language: str, 
                                    keyword_matches: List[str], indicators: List[str],
                                    evidence_snippets: List[str]) -> float:
        """Analyze controversy based on keywords."""
        score = 0.0
        
        # Weight multipliers from data file
        weight_multipliers = self._controversy_keywords.get('weight_multipliers', {
            'high_intensity': 1.0,
            'medium_intensity': 0.6,
            'low_intensity': 0.3,
            'political_terms': 0.4,
            'economic_terms': 0.3,
            'conflict_indicators': 0.9
        })
        
        # Check each intensity level
        intensity_levels = [
            ('high_intensity', weight_multipliers.get('high_intensity', 1.0), 0.4),
            ('medium_intensity', weight_multipliers.get('medium_intensity', 0.6), 0.25),
            ('low_intensity', weight_multipliers.get('low_intensity', 0.3), 0.15),
            ('conflict_indicators', weight_multipliers.get('conflict_indicators', 0.9), 0.35),
            ('political_terms', weight_multipliers.get('political_terms', 0.4), 0.2),
            ('economic_terms', weight_multipliers.get('economic_terms', 0.3), 0.15)
        ]
        
        for category, weight, base_score in intensity_levels:
            words = self._controversy_keywords.get(category, {}).get(language, [])
            category_matches = []
            
            for word in words:
                if word in text_lower:
                    score += base_score * weight
                    category_matches.append(word)
                    keyword_matches.append(word)
                    indicators.append(f"{category}_{word}")
                    
                    # Extract evidence snippet
                    self._extract_evidence_snippet(text_lower, word, evidence_snippets)
            
            # Bonus for multiple matches in same category
            if len(category_matches) > 1:
                score += min(0.2, len(category_matches) * 0.05)
                indicators.append(f"multiple_{category}_matches")
        
        return min(score, 0.8)  # Cap keyword score
    
    def _analyze_sentiment_controversy(self, sentiment: SentimentAnalysis, 
                                     indicators: List[str]) -> float:
        """Analyze controversy based on sentiment."""
        sentiment_factor = 0.0
        
        if sentiment.label == SentimentLabel.NEGATIVE:
            # Strong negative sentiment increases controversy
            sentiment_factor = abs(sentiment.score) * 0.3
            indicators.append("strong_negative_sentiment")
            
            # Very low confidence in sentiment might indicate controversy
            if sentiment.confidence < 0.5:
                sentiment_factor += 0.1
                indicators.append("ambiguous_sentiment")
        
        elif sentiment.label == SentimentLabel.NEUTRAL:
            # Very neutral sentiment in controversial context might be suspicious
            if sentiment.confidence < 0.6:
                sentiment_factor = 0.1
                indicators.append("forced_neutrality")
        
        return min(sentiment_factor, 0.4)  # Cap sentiment factor
    
    def _analyze_entity_controversy(self, entities: List[NamedEntity], 
                                  keyword_matches: List[str], 
                                  indicators: List[str]) -> tuple[float, List[str]]:
        """Analyze controversy based on entities."""
        entity_factor = 0.0
        controversial_entities = []
        
        for entity in entities:
            entity_text_lower = entity.text.lower()
            
            # Check if entity is mentioned with controversial keywords
            is_controversial = any(
                keyword in entity_text_lower or entity_text_lower in keyword
                for keyword in keyword_matches
            )
            
            if is_controversial:
                controversial_entities.append(entity.text)
                
                # Different entity types have different controversy weights
                if entity.entity_type.value == 'person':
                    entity_factor += 0.15
                elif entity.entity_type.value == 'organization':
                    entity_factor += 0.12
                elif entity.entity_type.value == 'location':
                    entity_factor += 0.08
                else:
                    entity_factor += 0.05
                
                indicators.append(f"controversial_{entity.entity_type.value}")
        
        # Bonus for multiple controversial entities
        if len(controversial_entities) > 2:
            entity_factor += min(0.15, len(controversial_entities) * 0.03)
            indicators.append("multiple_controversial_entities")
        
        return min(entity_factor, 0.3), controversial_entities  # Cap entity factor
    
    def _analyze_contextual_patterns(self, text_lower: str, language: str,
                                   indicators: List[str], evidence_snippets: List[str]) -> float:
        """Analyze contextual controversy patterns."""
        context_factor = 0.0
        
        # Pattern indicators for controversy
        controversy_patterns = {
            'ukrainian': [
                ('заперечує.*звинувачення', 0.2),
                ('скандальн.*заяв', 0.25),
                ('протест.*проти', 0.15),
                ('викрив.*корупці', 0.3),
                ('підозрюють.*у', 0.2)
            ],
            'russian': [
                ('отрицает.*обвинения', 0.2),
                ('скандальн.*заявлени', 0.25),
                ('протест.*против', 0.15),
                ('разоблачил.*коррупци', 0.3),
                ('подозревают.*в', 0.2)
            ],
            'english': [
                ('denies.*allegations', 0.2),
                ('controversial.*statement', 0.25),
                ('protests.*against', 0.15),
                ('exposed.*corruption', 0.3),
                ('suspected.*of', 0.2)
            ]
        }
        
        patterns = controversy_patterns.get(language, controversy_patterns['english'])
        
        import re
        for pattern, weight in patterns:
            if re.search(pattern, text_lower):
                context_factor += weight
                indicators.append(f"pattern_{pattern.split('.*')[0]}")
                
                # Extract evidence for pattern matches
                match = re.search(pattern, text_lower)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text_lower), match.end() + 30)
                    evidence_snippets.append(text_lower[start:end])
        
        return min(context_factor, 0.25)  # Cap context factor
    
    def _extract_evidence_snippet(self, text: str, word: str, 
                                evidence_snippets: List[str]) -> None:
        """Extract evidence snippet around a keyword."""
        try:
            word_pos = text.find(word)
            if word_pos != -1:
                start = max(0, word_pos - 50)
                end = min(len(text), word_pos + len(word) + 50)
                snippet = text[start:end].strip()
                if snippet and snippet not in evidence_snippets:
                    evidence_snippets.append(snippet)
        except Exception:
            pass  # Ignore extraction errors
    
    def _calculate_confidence(self, keyword_score: float, sentiment_factor: float,
                            entity_factor: float, context_factor: float) -> float:
        """Calculate confidence in controversy detection."""
        # Base confidence
        confidence = 0.3
        
        # Increase confidence based on multiple signals
        if keyword_score > 0.1:
            confidence += 0.2
        if sentiment_factor > 0.05:
            confidence += 0.15
        if entity_factor > 0.05:
            confidence += 0.15
        if context_factor > 0.05:
            confidence += 0.2
        
        # Higher scores generally mean higher confidence
        total_score = keyword_score + sentiment_factor + entity_factor + context_factor
        confidence += min(0.3, total_score * 0.5)
        
        return min(0.95, confidence)
    
    def _determine_controversy_level(self, score: float) -> ControversyLevel:
        """Determine controversy level based on score."""
        if score >= self.config.controversy_threshold:
            if score >= 0.8:
                return ControversyLevel.CRITICAL
            elif score >= 0.6:
                return ControversyLevel.HIGH
            else:
                return ControversyLevel.MEDIUM
        else:
            return ControversyLevel.LOW
    
    def get_controversy_keywords(self, language: str = 'uk') -> Dict[str, List[str]]:
        """
        Get controversy keywords for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary of keyword categories and their words
        """
        result = {}
        for category, data in self._controversy_keywords.items():
            if isinstance(data, dict) and language in data:
                result[category] = data[language]
        return result
    
    def analyze_controversy_trends(self, texts: List[str], 
                                 languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze controversy trends across multiple texts.
        
        Args:
            texts: List of text documents
            languages: Optional list of language codes for each text
            
        Returns:
            Trend analysis results
        """
        if not texts:
            return {}
        
        if languages is None:
            languages = ['auto'] * len(texts)
        
        controversy_scores = []
        keyword_frequency = {}
        level_distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for i, text in enumerate(texts):
            lang = languages[i] if i < len(languages) else 'auto'
            try:
                result = self.detect_controversy(text, language=lang)
                controversy_scores.append(result.score)
                
                # Count keyword frequencies
                for keyword in result.keyword_matches:
                    keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1
                
                # Count level distribution
                level_distribution[result.level.value] += 1
                
            except Exception as e:
                self.logger.warning(f"Error analyzing text {i}: {e}")
                controversy_scores.append(0.0)
        
        if not controversy_scores:
            return {}
        
        # Calculate statistics
        avg_score = sum(controversy_scores) / len(controversy_scores)
        max_score = max(controversy_scores)
        min_score = min(controversy_scores)
        
        # Get top keywords
        top_keywords = sorted(keyword_frequency.items(), 
                            key=lambda x: x[1], reverse=True)[:15]
        
        return {
            'total_documents': len(texts),
            'average_controversy': avg_score,
            'max_controversy': max_score,
            'min_controversy': min_score,
            'level_distribution': level_distribution,
            'top_keywords': top_keywords,
            'trend_direction': 'increasing' if controversy_scores[-1] > avg_score else 'stable'
        }