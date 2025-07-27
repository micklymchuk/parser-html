"""
Content Statistics Calculator for Content Analytics.

This module provides content statistics calculation including:
- Basic text metrics (word count, sentence count, etc.)
- Readability scores (Flesch, Coleman-Liau, etc.)
- Language complexity analysis
- Entity density calculations
- Content quality scoring
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math

try:
    import textstat
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Statistics calculation dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

from ..analytics.models import ContentStatistics, NamedEntity
from ..core.config import ContentAnalyticsConfig
from ..exceptions.pipeline_exceptions import PipelineError, handle_pipeline_error
from ..utils.logging import PipelineLogger

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """
    Calculates comprehensive content statistics and readability metrics.
    
    Provides detailed analysis of text characteristics including length metrics,
    readability scores, language complexity, and content quality indicators.
    """
    
    def __init__(self, config: Optional[ContentAnalyticsConfig] = None):
        """
        Initialize the statistics calculator.
        
        Args:
            config: Content analytics configuration
        """
        self.config = config or ContentAnalyticsConfig()
        self.logger = PipelineLogger("StatisticsCalculator")
        self._language_stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> Dict[str, List[str]]:
        """Load stopwords for different languages."""
        stopwords = {
            'en': [
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
                'his', 'its', 'our', 'their', 'a', 'an'
            ],
            'uk': [
                'в', 'на', 'з', 'за', 'до', 'від', 'під', 'над', 'через', 'про',
                'для', 'без', 'між', 'серед', 'є', 'був', 'була', 'було', 'були',
                'буде', 'будуть', 'мати', 'має', 'мав', 'мала', 'мало', 'мали',
                'це', 'той', 'та', 'те', 'ті', 'я', 'ти', 'він', 'вона', 'воно',
                'ми', 'ви', 'вони', 'мене', 'тебе', 'його', 'її', 'нас', 'вас',
                'їх', 'мій', 'твій', 'наш', 'ваш', 'і', 'а', 'але', 'або', 'чи'
            ],
            'ru': [
                'в', 'на', 'с', 'за', 'до', 'от', 'под', 'над', 'через', 'о',
                'для', 'без', 'между', 'среди', 'есть', 'был', 'была', 'было',
                'были', 'будет', 'будут', 'иметь', 'имеет', 'имел', 'имела',
                'это', 'тот', 'та', 'то', 'те', 'я', 'ты', 'он', 'она', 'оно',
                'мы', 'вы', 'они', 'меня', 'тебя', 'его', 'её', 'нас', 'вас',
                'их', 'мой', 'твой', 'наш', 'ваш', 'и', 'а', 'но', 'или', 'ли'
            ]
        }
        return stopwords
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (uk, ru, en)
        """
        try:
            # Character-based detection
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
                
                text_lower = text.lower()
                uk_count = sum(1 for indicator in ukrainian_indicators if indicator in text_lower)
                ru_count = sum(1 for indicator in russian_indicators if indicator in text_lower)
                
                return 'uk' if uk_count >= ru_count else 'ru'
            else:
                return 'en'
                
        except Exception:
            return 'en'  # Default fallback
    
    @handle_pipeline_error
    def calculate_statistics(self, text: str, entities: Optional[List[NamedEntity]] = None,
                           language: str = 'auto') -> ContentStatistics:
        """
        Calculate comprehensive content statistics.
        
        Args:
            text: Input text
            entities: Optional list of named entities
            language: Language code or 'auto' for detection
            
        Returns:
            Content statistics object
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            # Basic text metrics
            basic_stats = self._calculate_basic_metrics(text)
            
            # Readability scores
            readability_stats = self._calculate_readability_scores(text, language)
            
            # Language complexity
            complexity_stats = self._calculate_complexity_metrics(text, language)
            
            # Entity statistics
            entity_stats = self._calculate_entity_statistics(text, entities or [])
            
            # Language confidence
            language_confidence = self._calculate_language_confidence(text, language)
            
            # Combine all statistics
            return ContentStatistics(
                character_count=basic_stats['character_count'],
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                flesch_reading_ease=readability_stats.get('flesch_reading_ease'),
                flesch_kincaid_grade=readability_stats.get('flesch_kincaid_grade'),
                coleman_liau_index=readability_stats.get('coleman_liau_index'),
                language=language,
                language_confidence=language_confidence,
                avg_words_per_sentence=complexity_stats['avg_words_per_sentence'],
                avg_sentence_length=complexity_stats['avg_sentence_length'],
                unique_words_ratio=complexity_stats['unique_words_ratio'],
                entity_count=entity_stats['entity_count'],
                unique_entities=entity_stats['unique_entities'],
                entity_density=entity_stats['entity_density'],
                lexical_diversity=complexity_stats.get('lexical_diversity'),
                syllable_count=complexity_stats.get('syllable_count'),
                complex_words_ratio=complexity_stats.get('complex_words_ratio'),
                stopwords_ratio=complexity_stats.get('stopwords_ratio')
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating statistics: {e}")
            # Return minimal statistics as fallback
            return ContentStatistics(
                character_count=len(text),
                word_count=len(text.split()),
                sentence_count=1,
                paragraph_count=1,
                language=language,
                avg_words_per_sentence=len(text.split()),
                avg_sentence_length=len(text),
                unique_words_ratio=1.0,
                entity_count=len(entities) if entities else 0,
                unique_entities=len(entities) if entities else 0,
                entity_density=0.0
            )
    
    def _calculate_basic_metrics(self, text: str) -> Dict[str, int]:
        """Calculate basic text metrics."""
        # Character count (excluding whitespace)
        character_count = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        # Word count
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Paragraph count
        paragraphs = re.split(r'\n\s*\n', text)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            'character_count': character_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    def _calculate_readability_scores(self, text: str, language: str) -> Dict[str, Optional[float]]:
        """Calculate readability scores."""
        scores = {}
        
        # Only calculate English readability scores if textstat is available
        if language == 'en' and DEPENDENCIES_AVAILABLE:
            try:
                scores['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                scores['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
                scores['coleman_liau_index'] = textstat.coleman_liau_index(text)
                scores['automated_readability_index'] = textstat.automated_readability_index(text)
                scores['gunning_fog'] = textstat.gunning_fog(text)
            except Exception as e:
                self.logger.warning(f"Error calculating readability scores: {e}")
        
        # For non-English languages, calculate approximate readability
        else:
            scores.update(self._calculate_approximate_readability(text, language))
        
        return scores
    
    def _calculate_approximate_readability(self, text: str, language: str) -> Dict[str, Optional[float]]:
        """Calculate approximate readability for non-English languages."""
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            
            if not words or not sentences:
                return {}
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Approximate Flesch Reading Ease (adjusted for language)
            # Formula adapted for different languages
            language_factor = {'uk': 1.1, 'ru': 1.05, 'en': 1.0}.get(language, 1.0)
            
            approx_flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 4.7) * language_factor
            approx_flesch = max(0, min(100, approx_flesch))  # Clamp to 0-100
            
            # Approximate grade level
            approx_grade = (0.39 * avg_sentence_length) + (11.8 * avg_word_length / 4.7) - 15.59
            approx_grade = max(1, min(20, approx_grade))  # Clamp to reasonable range
            
            return {
                'flesch_reading_ease': approx_flesch,
                'flesch_kincaid_grade': approx_grade,
                'coleman_liau_index': None  # Not calculated for non-English
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating approximate readability: {e}")
            return {}
    
    def _calculate_complexity_metrics(self, text: str, language: str) -> Dict[str, float]:
        """Calculate language complexity metrics."""
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            
            if not words or not sentences:
                return {
                    'avg_words_per_sentence': 0.0,
                    'avg_sentence_length': 0.0,
                    'unique_words_ratio': 0.0
                }
            
            # Basic complexity metrics
            avg_words_per_sentence = len(words) / len(sentences)
            avg_sentence_length = len(text) / len(sentences)
            
            # Lexical diversity
            unique_words = len(set(words))
            unique_words_ratio = unique_words / len(words)
            lexical_diversity = unique_words / math.sqrt(len(words)) if len(words) > 0 else 0
            
            # Syllable-based metrics (approximate)
            syllable_count = self._estimate_syllables(words, language)
            avg_syllables_per_word = syllable_count / len(words) if words else 0
            
            # Complex words (words with 3+ syllables)
            complex_words = sum(1 for word in words if self._estimate_word_syllables(word, language) >= 3)
            complex_words_ratio = complex_words / len(words) if words else 0
            
            # Stopwords ratio
            stopwords = self._language_stopwords.get(language, [])
            stopword_count = sum(1 for word in words if word in stopwords)
            stopwords_ratio = stopword_count / len(words) if words else 0
            
            return {
                'avg_words_per_sentence': avg_words_per_sentence,
                'avg_sentence_length': avg_sentence_length,
                'unique_words_ratio': unique_words_ratio,
                'lexical_diversity': lexical_diversity,
                'syllable_count': syllable_count,
                'avg_syllables_per_word': avg_syllables_per_word,
                'complex_words_ratio': complex_words_ratio,
                'stopwords_ratio': stopwords_ratio
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating complexity metrics: {e}")
            return {
                'avg_words_per_sentence': 0.0,
                'avg_sentence_length': 0.0,
                'unique_words_ratio': 0.0
            }
    
    def _estimate_syllables(self, words: List[str], language: str) -> int:
        """Estimate total syllable count for a list of words."""
        return sum(self._estimate_word_syllables(word, language) for word in words)
    
    def _estimate_word_syllables(self, word: str, language: str) -> int:
        """Estimate syllable count for a single word."""
        try:
            word = word.lower().strip()
            if not word:
                return 0
            
            if language in ['uk', 'ru']:
                # For Cyrillic languages, count vowels
                vowels = 'аеиоуыэюяіїє'
                syllables = sum(1 for char in word if char in vowels)
                return max(1, syllables)  # At least 1 syllable
            
            else:  # English and other Latin-script languages
                vowels = 'aeiou'
                syllables = 0
                prev_was_vowel = False
                
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_was_vowel:
                        syllables += 1
                    prev_was_vowel = is_vowel
                
                # Handle silent 'e'
                if word.endswith('e') and syllables > 1:
                    syllables -= 1
                
                return max(1, syllables)  # At least 1 syllable
                
        except Exception:
            return 1  # Default to 1 syllable
    
    def _calculate_entity_statistics(self, text: str, entities: List[NamedEntity]) -> Dict[str, float]:
        """Calculate entity-related statistics."""
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            word_count = len(words)
            
            entity_count = len(entities)
            unique_entities = len(set(entity.normalized_form or entity.text for entity in entities))
            
            # Entity density (entities per 100 words)
            entity_density = (entity_count / max(1, word_count)) * 100
            
            # Entity type distribution
            entity_types = {}
            for entity in entities:
                entity_type = entity.entity_type.value
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Entity confidence statistics
            if entities:
                entity_confidences = [entity.confidence for entity in entities]
                avg_entity_confidence = sum(entity_confidences) / len(entity_confidences)
            else:
                avg_entity_confidence = 0.0
            
            return {
                'entity_count': entity_count,
                'unique_entities': unique_entities,
                'entity_density': entity_density,
                'entity_types': entity_types,
                'avg_entity_confidence': avg_entity_confidence
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating entity statistics: {e}")
            return {
                'entity_count': 0,
                'unique_entities': 0,
                'entity_density': 0.0
            }
    
    def _calculate_language_confidence(self, text: str, detected_language: str) -> float:
        """Calculate confidence in language detection."""
        try:
            # Character-based confidence
            cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
            latin_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
            total_chars = cyrillic_chars + latin_chars
            
            if total_chars == 0:
                return 0.5  # Default confidence
            
            cyrillic_ratio = cyrillic_chars / total_chars
            
            # Base confidence on character distribution
            if detected_language in ['uk', 'ru']:
                base_confidence = cyrillic_ratio
            else:  # English
                base_confidence = 1 - cyrillic_ratio
            
            # Adjust based on text length
            length_factor = min(1.0, len(text) / 100)  # More confident with longer texts
            
            # Adjust based on language-specific indicators
            language_indicators = {
                'uk': ['і', 'ї', 'є', 'ґ', 'що', 'який', 'також'],
                'ru': ['ы', 'э', 'ё', 'что', 'который', 'также'],
                'en': ['the', 'and', 'that', 'which', 'also']
            }
            
            indicators = language_indicators.get(detected_language, [])
            text_lower = text.lower()
            indicator_count = sum(1 for indicator in indicators if indicator in text_lower)
            indicator_factor = min(1.0, indicator_count / max(1, len(indicators) * 0.3))
            
            # Combine factors
            confidence = (base_confidence * 0.6 + length_factor * 0.2 + indicator_factor * 0.2)
            return max(0.1, min(0.95, confidence))
            
        except Exception:
            return 0.5  # Default confidence
    
    def calculate_quality_score(self, stats: ContentStatistics, entities: List[NamedEntity],
                              sentiment_score: Optional[float] = None) -> float:
        """
        Calculate overall content quality score.
        
        Args:
            stats: Content statistics
            entities: List of entities
            sentiment_score: Optional sentiment score
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            score = 0.0
            
            # Length factor (optimal range)
            if 100 <= stats.word_count <= 2000:
                score += 0.25
            elif 50 <= stats.word_count < 100:
                score += 0.15
            elif stats.word_count > 2000:
                score += 0.15
            
            # Readability factor
            if stats.flesch_reading_ease is not None:
                if 30 <= stats.flesch_reading_ease <= 80:
                    score += 0.2
                else:
                    score += 0.1
            else:
                score += 0.1  # Default for non-English
            
            # Entity density factor
            if 1.0 <= stats.entity_density <= 10.0:
                score += 0.2
            elif stats.entity_density > 0:
                score += 0.1
            
            # Sentence structure factor
            if 5 <= stats.avg_words_per_sentence <= 25:
                score += 0.1
            
            # Vocabulary diversity factor
            if stats.unique_words_ratio >= 0.5:
                score += 0.15
            elif stats.unique_words_ratio >= 0.3:
                score += 0.1
            
            # Language confidence factor
            if stats.language_confidence and stats.language_confidence >= 0.8:
                score += 0.1
            elif stats.language_confidence and stats.language_confidence >= 0.6:
                score += 0.05
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {e}")
            return 0.5  # Default quality score
    
    def get_content_insights(self, stats: ContentStatistics) -> Dict[str, str]:
        """
        Generate human-readable insights about content statistics.
        
        Args:
            stats: Content statistics
            
        Returns:
            Dictionary of insights
        """
        insights = {}
        
        try:
            # Length insights
            if stats.word_count < 50:
                insights['length'] = "Very short content - may lack detail"
            elif stats.word_count > 2000:
                insights['length'] = "Very long content - readers may lose interest"
            else:
                insights['length'] = "Good content length"
            
            # Readability insights
            if stats.flesch_reading_ease:
                if stats.flesch_reading_ease >= 80:
                    insights['readability'] = "Very easy to read"
                elif stats.flesch_reading_ease >= 60:
                    insights['readability'] = "Easy to read"
                elif stats.flesch_reading_ease >= 30:
                    insights['readability'] = "Moderately difficult to read"
                else:
                    insights['readability'] = "Difficult to read"
            
            # Complexity insights
            if stats.avg_words_per_sentence > 25:
                insights['complexity'] = "Long sentences may be hard to follow"
            elif stats.avg_words_per_sentence < 5:
                insights['complexity'] = "Very short sentences - may seem choppy"
            else:
                insights['complexity'] = "Good sentence length"
            
            # Vocabulary insights
            if stats.unique_words_ratio >= 0.7:
                insights['vocabulary'] = "Rich vocabulary diversity"
            elif stats.unique_words_ratio >= 0.4:
                insights['vocabulary'] = "Good vocabulary diversity"
            else:
                insights['vocabulary'] = "Limited vocabulary diversity"
            
            # Entity insights
            if stats.entity_density > 15:
                insights['entities'] = "Very high entity density - may be information-heavy"
            elif stats.entity_density > 5:
                insights['entities'] = "Good entity density - informative content"
            elif stats.entity_density > 0:
                insights['entities'] = "Some entities mentioned"
            else:
                insights['entities'] = "No entities detected - may be abstract content"
            
        except Exception as e:
            self.logger.warning(f"Error generating insights: {e}")
        
        return insights
    
    def compare_statistics(self, stats1: ContentStatistics, stats2: ContentStatistics) -> Dict[str, Any]:
        """
        Compare two sets of content statistics.
        
        Args:
            stats1: First statistics object
            stats2: Second statistics object
            
        Returns:
            Comparison results
        """
        try:
            comparison = {}
            
            # Compare basic metrics
            comparison['word_count_diff'] = stats2.word_count - stats1.word_count
            comparison['sentence_count_diff'] = stats2.sentence_count - stats1.sentence_count
            
            # Compare readability
            if stats1.flesch_reading_ease and stats2.flesch_reading_ease:
                comparison['readability_diff'] = stats2.flesch_reading_ease - stats1.flesch_reading_ease
            
            # Compare complexity
            comparison['complexity_diff'] = stats2.avg_words_per_sentence - stats1.avg_words_per_sentence
            comparison['diversity_diff'] = stats2.unique_words_ratio - stats1.unique_words_ratio
            
            # Compare entity metrics
            comparison['entity_density_diff'] = stats2.entity_density - stats1.entity_density
            
            # Generate summary
            if comparison['word_count_diff'] > 0:
                comparison['summary'] = "Second text is longer"
            elif comparison['word_count_diff'] < 0:
                comparison['summary'] = "First text is longer"
            else:
                comparison['summary'] = "Similar length"
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"Error comparing statistics: {e}")
            return {}