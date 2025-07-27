"""
Sentiment Analysis Component for Content Analytics.

This module provides sentiment analysis functionality including:
- Transformer-based sentiment analysis
- Lexicon-based sentiment analysis as fallback
- Multi-language sentiment detection
- Political and economic sentiment specialization
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re

try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModel
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

from ..analytics.models import SentimentAnalysis, SentimentLabel
from ..core.config import ContentAnalyticsConfig
from ..exceptions.pipeline_exceptions import PipelineError, handle_pipeline_error
from ..utils.logging import PipelineLogger

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment in text content using multiple approaches.
    
    Supports transformer models for high-accuracy analysis and lexicon-based
    fallback for reliability. Includes specialized handling for political
    and economic content.
    """
    
    def __init__(self, config: Optional[ContentAnalyticsConfig] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Content analytics configuration
        """
        self.config = config or ContentAnalyticsConfig()
        self.logger = PipelineLogger("SentimentAnalyzer")
        self._sentiment_model = None
        self._sentiment_lexicon = {}
        self._model_cache = {}
        
        self._load_sentiment_lexicon()
        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()
    
    def _load_sentiment_lexicon(self) -> None:
        """Load sentiment lexicon for fallback analysis."""
        try:
            base_path = Path(self.config.entities_db_path).parent
            lexicon_path = base_path / "sentiment_lexicon.json"
            
            if lexicon_path.exists():
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    self._sentiment_lexicon = json.load(f)
                self.logger.info("Sentiment lexicon loaded successfully")
            else:
                self.logger.warning(f"Sentiment lexicon not found: {lexicon_path}")
                self._sentiment_lexicon = {}
                
        except Exception as e:
            self.logger.error(f"Error loading sentiment lexicon: {e}")
            self._sentiment_lexicon = {}
    
    def _initialize_models(self) -> None:
        """Initialize transformer models for sentiment analysis."""
        try:
            self.logger.info("Initializing sentiment analysis models...")
            
            # Initialize main sentiment model
            self._sentiment_model = pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            self.logger.info("Sentiment models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Error initializing sentiment models: {e}")
            self._sentiment_model = None
    
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
    def analyze_sentiment(self, text: str, language: str = 'auto', 
                         context: str = 'general') -> SentimentAnalysis:
        """
        Analyze sentiment of the text.
        
        Args:
            text: Input text
            language: Language code or 'auto' for detection
            context: Context type ('general', 'political', 'economic')
            
        Returns:
            Sentiment analysis results
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            # Try transformer model first
            if self._sentiment_model and TRANSFORMERS_AVAILABLE:
                return self._transformer_sentiment_analysis(text, language, context)
            else:
                return self._lexicon_sentiment_analysis(text, language, context)
                
        except Exception as e:
            self.logger.warning(f"Error in sentiment analysis: {e}")
            return self._lexicon_sentiment_analysis(text, language, context)
    
    def _transformer_sentiment_analysis(self, text: str, language: str, 
                                      context: str) -> SentimentAnalysis:
        """Perform sentiment analysis using transformer models."""
        try:
            # Truncate text to model limits
            text_truncated = text[:512]
            
            # Get predictions
            results = self._sentiment_model(text_truncated)
            
            # Process results
            if results and len(results) > 0:
                # Handle different model output formats
                if isinstance(results[0], list):
                    scores = results[0]
                else:
                    scores = results
                
                # Extract scores
                sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                
                for score_dict in scores:
                    label = score_dict['label'].lower()
                    score = score_dict['score']
                    
                    # Map labels to our format
                    if 'positive' in label or 'pos' in label:
                        sentiment_scores['positive'] = score
                    elif 'negative' in label or 'neg' in label:
                        sentiment_scores['negative'] = score
                    elif 'neutral' in label or 'neu' in label:
                        sentiment_scores['neutral'] = score
                
                # Determine primary sentiment
                primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                confidence = sentiment_scores[primary_sentiment]
                
                # Calculate overall sentiment score (-1 to 1)
                sentiment_score = (sentiment_scores['positive'] - sentiment_scores['negative'])
                
                # Map to our enum
                if primary_sentiment == 'positive':
                    label = SentimentLabel.POSITIVE
                elif primary_sentiment == 'negative':
                    label = SentimentLabel.NEGATIVE
                else:
                    label = SentimentLabel.NEUTRAL
                
                # Apply context-specific adjustments
                adjusted_scores = self._apply_context_adjustments(
                    sentiment_scores, text, language, context
                )
                
                return SentimentAnalysis(
                    label=label,
                    score=sentiment_score,
                    confidence=confidence,
                    positive_score=adjusted_scores['positive'],
                    negative_score=adjusted_scores['negative'],
                    neutral_score=adjusted_scores['neutral'],
                    model_name=self.config.sentiment_model,
                    language=language,
                    context=context,
                    method='transformer'
                )
            
            # Fallback if no results
            return self._lexicon_sentiment_analysis(text, language, context)
            
        except Exception as e:
            self.logger.warning(f"Error in transformer sentiment analysis: {e}")
            return self._lexicon_sentiment_analysis(text, language, context)
    
    def _lexicon_sentiment_analysis(self, text: str, language: str, 
                                  context: str) -> SentimentAnalysis:
        """Perform lexicon-based sentiment analysis."""
        try:
            # Get lexicon data
            positive_words = self._sentiment_lexicon.get('positive_words', {}).get(language, [])
            negative_words = self._sentiment_lexicon.get('negative_words', {}).get(language, [])
            intensifiers = self._sentiment_lexicon.get('intensifiers', {}).get(language, [])
            negation_words = self._sentiment_lexicon.get('negation_words', {}).get(language, [])
            
            # Get context-specific words
            if context == 'political':
                political_sentiment = self._sentiment_lexicon.get('political_sentiment', {})
                positive_words.extend(political_sentiment.get('positive', {}).get(language, []))
                negative_words.extend(political_sentiment.get('negative', {}).get(language, []))
            elif context == 'economic':
                economic_sentiment = self._sentiment_lexicon.get('economic_sentiment', {})
                positive_words.extend(economic_sentiment.get('positive', {}).get(language, []))
                negative_words.extend(economic_sentiment.get('negative', {}).get(language, []))
            
            # Preprocess text
            words, processed_text = self._preprocess_text(text, language)
            
            # Score sentiment
            scores = self._calculate_lexicon_scores(
                words, processed_text, positive_words, negative_words, 
                intensifiers, negation_words
            )
            
            # Determine primary sentiment
            if scores['positive'] > scores['negative'] and scores['positive'] > scores['neutral']:
                label = SentimentLabel.POSITIVE
                main_score = scores['positive'] - scores['negative']
                confidence = scores['positive']
            elif scores['negative'] > scores['positive'] and scores['negative'] > scores['neutral']:
                label = SentimentLabel.NEGATIVE
                main_score = -(scores['negative'] - scores['positive'])
                confidence = scores['negative']
            else:
                label = SentimentLabel.NEUTRAL
                main_score = 0.0
                confidence = scores['neutral']
            
            return SentimentAnalysis(
                label=label,
                score=main_score,
                confidence=min(confidence, 0.85),  # Cap lexicon confidence
                positive_score=scores['positive'],
                negative_score=scores['negative'],
                neutral_score=scores['neutral'],
                language=language,
                context=context,
                method='lexicon'
            )
            
        except Exception as e:
            self.logger.warning(f"Error in lexicon sentiment analysis: {e}")
            # Return neutral sentiment as fallback
            return SentimentAnalysis(
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.1,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                language=language,
                context=context,
                method='fallback'
            )
    
    def _preprocess_text(self, text: str, language: str) -> Tuple[List[str], str]:
        """Preprocess text for lexicon analysis."""
        # Convert to lowercase
        processed_text = text.lower()
        
        # Remove punctuation but keep sentence structure
        processed_text = re.sub(r'[^\w\s]', ' ', processed_text)
        
        # Split into words
        words = processed_text.split()
        
        # Remove very short words
        words = [word for word in words if len(word) > 1]
        
        return words, processed_text
    
    def _calculate_lexicon_scores(self, words: List[str], text: str,
                                positive_words: List[str], negative_words: List[str],
                                intensifiers: List[str], negation_words: List[str]) -> Dict[str, float]:
        """Calculate sentiment scores using lexicon approach."""
        positive_score = 0.0
        negative_score = 0.0
        word_count = len(words)
        
        if word_count == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Weight multipliers from lexicon
        weights = self._sentiment_lexicon.get('weights', {
            'base_weight': 1.0,
            'political_weight': 1.2,
            'economic_weight': 1.1,
            'intensifier_multiplier': 1.5,
            'negation_multiplier': -1.0,
            'context_boost': 0.2
        })
        
        # Analyze each word with context
        for i, word in enumerate(words):
            word_sentiment = 0.0
            
            # Check if word is positive or negative
            if word in positive_words:
                word_sentiment = weights['base_weight']
            elif word in negative_words:
                word_sentiment = -weights['base_weight']
            else:
                continue
            
            # Check for intensifiers nearby
            intensifier_multiplier = 1.0
            for j in range(max(0, i-2), min(len(words), i+3)):
                if words[j] in intensifiers:
                    intensifier_multiplier = weights['intensifier_multiplier']
                    break
            
            # Check for negation nearby
            negation_multiplier = 1.0
            for j in range(max(0, i-3), i):
                if words[j] in negation_words:
                    negation_multiplier = weights['negation_multiplier']
                    break
            
            # Apply multipliers
            final_sentiment = word_sentiment * intensifier_multiplier * negation_multiplier
            
            if final_sentiment > 0:
                positive_score += final_sentiment
            else:
                negative_score += abs(final_sentiment)
        
        # Normalize scores
        total_sentiment_words = positive_score + negative_score
        if total_sentiment_words > 0:
            positive_score = positive_score / total_sentiment_words
            negative_score = negative_score / total_sentiment_words
        else:
            positive_score = 0.0
            negative_score = 0.0
        
        # Calculate neutral score
        neutral_score = max(0.0, 1.0 - positive_score - negative_score)
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }
    
    def _apply_context_adjustments(self, scores: Dict[str, float], text: str,
                                 language: str, context: str) -> Dict[str, float]:
        """Apply context-specific adjustments to sentiment scores."""
        adjusted_scores = scores.copy()
        
        if context == 'political':
            # Political content often has stronger sentiment expressions
            adjustment_factor = 1.1
            if adjusted_scores['positive'] > 0.6:
                adjusted_scores['positive'] = min(1.0, adjusted_scores['positive'] * adjustment_factor)
            if adjusted_scores['negative'] > 0.6:
                adjusted_scores['negative'] = min(1.0, adjusted_scores['negative'] * adjustment_factor)
        
        elif context == 'economic':
            # Economic content might be more neutral in tone
            if adjusted_scores['neutral'] > 0.4:
                # Slightly boost neutral if already high
                neutral_boost = min(0.1, (adjusted_scores['neutral'] - 0.4) * 0.5)
                adjusted_scores['neutral'] = min(1.0, adjusted_scores['neutral'] + neutral_boost)
                # Proportionally reduce other scores
                remaining = 1.0 - adjusted_scores['neutral']
                if remaining > 0:
                    pos_ratio = adjusted_scores['positive'] / (adjusted_scores['positive'] + adjusted_scores['negative']) if (adjusted_scores['positive'] + adjusted_scores['negative']) > 0 else 0.5
                    adjusted_scores['positive'] = remaining * pos_ratio
                    adjusted_scores['negative'] = remaining * (1 - pos_ratio)
        
        # Ensure scores sum to approximately 1.0
        total = sum(adjusted_scores.values())
        if total > 0:
            for key in adjusted_scores:
                adjusted_scores[key] = adjusted_scores[key] / total
        
        return adjusted_scores
    
    def analyze_sentiment_aspects(self, text: str, language: str = 'auto') -> Dict[str, SentimentAnalysis]:
        """
        Analyze sentiment for different aspects/topics in the text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Dictionary of aspect-based sentiment analysis
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        aspects = {}
        
        # Define aspect keywords
        aspect_keywords = {
            'government': {
                'uk': ['уряд', 'влада', 'міністр', 'президент', 'парламент'],
                'ru': ['правительство', 'власть', 'министр', 'президент', 'парламент'],
                'en': ['government', 'authority', 'minister', 'president', 'parliament']
            },
            'economy': {
                'uk': ['економіка', 'бюджет', 'фінанси', 'податки', 'інвестиції'],
                'ru': ['экономика', 'бюджет', 'финансы', 'налоги', 'инвестиции'],
                'en': ['economy', 'budget', 'finance', 'taxes', 'investment']
            },
            'society': {
                'uk': ['суспільство', 'люди', 'населення', 'громадяни'],
                'ru': ['общество', 'люди', 'население', 'граждане'],
                'en': ['society', 'people', 'population', 'citizens']
            }
        }
        
        try:
            text_lower = text.lower()
            
            for aspect, keywords_dict in aspect_keywords.items():
                keywords = keywords_dict.get(language, keywords_dict.get('en', []))
                
                # Check if aspect is mentioned in text
                if any(keyword in text_lower for keyword in keywords):
                    # Extract sentences related to this aspect
                    aspect_sentences = self._extract_aspect_sentences(text, keywords)
                    
                    if aspect_sentences:
                        aspect_text = ' '.join(aspect_sentences)
                        # Analyze sentiment for this aspect
                        context = 'political' if aspect in ['government'] else 'economic' if aspect == 'economy' else 'general'
                        aspects[aspect] = self.analyze_sentiment(aspect_text, language, context)
            
        except Exception as e:
            self.logger.warning(f"Error in aspect-based sentiment analysis: {e}")
        
        return aspects
    
    def _extract_aspect_sentences(self, text: str, keywords: List[str]) -> List[str]:
        """Extract sentences that mention specific aspect keywords."""
        sentences = re.split(r'[.!?]+', text)
        aspect_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(keyword in sentence.lower() for keyword in keywords):
                aspect_sentences.append(sentence)
        
        return aspect_sentences
    
    def get_sentiment_trends(self, texts: List[str], languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze sentiment trends across multiple texts.
        
        Args:
            texts: List of text documents
            languages: Optional list of language codes
            
        Returns:
            Trend analysis results
        """
        if not texts:
            return {}
        
        if languages is None:
            languages = ['auto'] * len(texts)
        
        sentiments = []
        scores = []
        label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for i, text in enumerate(texts):
            lang = languages[i] if i < len(languages) else 'auto'
            try:
                sentiment = self.analyze_sentiment(text, lang)
                sentiments.append(sentiment)
                scores.append(sentiment.score)
                label_counts[sentiment.label.value] += 1
                
            except Exception as e:
                self.logger.warning(f"Error analyzing sentiment for text {i}: {e}")
                continue
        
        if not scores:
            return {}
        
        # Calculate statistics
        avg_score = sum(scores) / len(scores)
        
        # Determine trend direction
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            overall_avg = sum(scores[:-3]) / max(1, len(scores) - 3)
            trend = 'improving' if recent_avg > overall_avg else 'declining' if recent_avg < overall_avg else 'stable'
        else:
            trend = 'stable'
        
        return {
            'total_documents': len(texts),
            'average_sentiment': avg_score,
            'sentiment_distribution': label_counts,
            'trend_direction': trend,
            'score_range': {'min': min(scores), 'max': max(scores)},
            'volatility': np.std(scores) if len(scores) > 1 else 0.0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded sentiment models."""
        return {
            'transformer_available': TRANSFORMERS_AVAILABLE,
            'model_loaded': self._sentiment_model is not None,
            'model_name': self.config.sentiment_model,
            'lexicon_loaded': bool(self._sentiment_lexicon),
            'supported_languages': ['uk', 'ru', 'en']
        }