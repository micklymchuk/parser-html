"""
Topic Classification Component for Content Analytics.

This module provides topic classification functionality including:
- Zero-shot topic classification using transformers
- Keyword-based topic classification as fallback
- Multi-language topic detection
- Hierarchical topic categorization
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re

try:
    import torch
    from transformers import pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Topic classification dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

from ..analytics.models import TopicClassification
from ..core.config import ContentAnalyticsConfig
from ..exceptions.pipeline_exceptions import PipelineError, handle_pipeline_error
from ..utils.logging import PipelineLogger

logger = logging.getLogger(__name__)


class TopicClassifier:
    """
    Classifies topics in text content using multiple approaches.
    
    Supports zero-shot classification with transformers and keyword-based
    fallback methods. Provides hierarchical topic organization and
    confidence scoring.
    """
    
    def __init__(self, config: Optional[ContentAnalyticsConfig] = None):
        """
        Initialize the topic classifier.
        
        Args:
            config: Content analytics configuration
        """
        self.config = config or ContentAnalyticsConfig()
        self.logger = PipelineLogger("TopicClassifier")
        self._classifier_model = None
        self._topic_patterns = {}
        self._vectorizer = None
        self._topic_vectors = {}
        
        self._load_topic_patterns()
        if DEPENDENCIES_AVAILABLE:
            self._initialize_models()
    
    def _load_topic_patterns(self) -> None:
        """Load topic patterns and keywords."""
        try:
            base_path = Path(self.config.entities_db_path).parent
            patterns_path = base_path / "topic_patterns.json"
            
            if patterns_path.exists():
                with open(patterns_path, 'r', encoding='utf-8') as f:
                    self._topic_patterns = json.load(f)
                self.logger.info("Topic patterns loaded successfully")
            else:
                self.logger.warning(f"Topic patterns file not found: {patterns_path}")
                self._topic_patterns = {}
                
        except Exception as e:
            self.logger.error(f"Error loading topic patterns: {e}")
            self._topic_patterns = {}
    
    def _initialize_models(self) -> None:
        """Initialize ML models for topic classification."""
        try:
            self.logger.info("Initializing topic classification models...")
            
            # Initialize zero-shot classification model
            self._classifier_model = pipeline(
                "zero-shot-classification",
                model=self.config.classification_model,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize TF-IDF vectorizer for keyword-based classification
            self._vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            self.logger.info("Topic classification models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Error initializing topic models: {e}")
            self._classifier_model = None
            self._vectorizer = None
    
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
    def classify_topics(self, text: str, language: str = 'auto') -> TopicClassification:
        """
        Classify topics in the text.
        
        Args:
            text: Input text
            language: Language code or 'auto' for detection
            
        Returns:
            Topic classification results
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            # Try transformer-based classification first
            if self._classifier_model and DEPENDENCIES_AVAILABLE:
                return self._transformer_topic_classification(text, language)
            else:
                return self._keyword_topic_classification(text, language)
                
        except Exception as e:
            self.logger.warning(f"Error in topic classification: {e}")
            return self._fallback_classification(text, language)
    
    def _transformer_topic_classification(self, text: str, language: str) -> TopicClassification:
        """Perform topic classification using transformer models."""
        try:
            # Get domain labels from patterns
            domains_data = self._topic_patterns.get('domains', {})
            domain_labels = list(domains_data.keys())
            
            if not domain_labels:
                # Use default domains
                domain_labels = ['politics', 'economics', 'social', 'technology', 'security', 'environment']
            
            # Truncate text for model
            text_truncated = text[:512]
            
            # Perform zero-shot classification
            result = self._classifier_model(text_truncated, domain_labels)
            
            # Process results
            topics = {}
            for label, score in zip(result['labels'], result['scores']):
                topics[label] = float(score)
            
            primary_topic = result['labels'][0]
            confidence = float(result['scores'][0])
            
            # Extract additional information
            categories = self._extract_categories(primary_topic, domains_data)
            tags = self._extract_tags(text, language)
            subdomain = self._determine_subdomain(text, primary_topic, language)
            
            return TopicClassification(
                primary_topic=primary_topic,
                confidence=confidence,
                topics=topics,
                categories=categories,
                tags=tags,
                domain=primary_topic,
                subdomain=subdomain,
                method='transformer',
                model_name=self.config.classification_model
            )
            
        except Exception as e:
            self.logger.warning(f"Error in transformer topic classification: {e}")
            return self._keyword_topic_classification(text, language)
    
    def _keyword_topic_classification(self, text: str, language: str) -> TopicClassification:
        """Perform keyword-based topic classification."""
        try:
            domains_data = self._topic_patterns.get('domains', {})
            text_lower = text.lower()
            
            domain_scores = {}
            keyword_matches = {}
            
            # Score each domain based on keyword matches
            for domain, domain_info in domains_data.items():
                score = 0.0
                matches = []
                
                # Check keywords
                keywords = domain_info.get('keywords', {}).get(language, [])
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        score += 1.0
                        matches.append(keyword)
                
                # Check regex patterns
                patterns = domain_info.get('patterns', [])
                for pattern in patterns:
                    try:
                        if re.search(pattern, text_lower, re.IGNORECASE):
                            score += 0.5
                            matches.append(f"pattern:{pattern}")
                    except re.error:
                        continue  # Skip invalid patterns
                
                if score > 0:
                    # Normalize score by number of possible keywords/patterns
                    total_possible = len(keywords) + len(patterns)
                    normalized_score = score / max(1, total_possible)
                    domain_scores[domain] = normalized_score
                    keyword_matches[domain] = matches
            
            # Apply TF-IDF similarity if available
            if self._vectorizer and domain_scores:
                tfidf_scores = self._calculate_tfidf_similarity(text, domain_scores.keys())
                # Combine keyword and TF-IDF scores
                for domain in domain_scores:
                    if domain in tfidf_scores:
                        domain_scores[domain] = 0.7 * domain_scores[domain] + 0.3 * tfidf_scores[domain]
            
            if not domain_scores:
                return self._fallback_classification(text, language)
            
            # Normalize scores to sum to 1
            total_score = sum(domain_scores.values())
            if total_score > 0:
                topics = {domain: score / total_score for domain, score in domain_scores.items()}
            else:
                topics = {domain: 1.0 / len(domain_scores) for domain in domain_scores}
            
            # Get primary topic
            primary_topic = max(topics.keys(), key=lambda x: topics[x])
            confidence = topics[primary_topic]
            
            # Extract additional information
            categories = self._extract_categories(primary_topic, domains_data)
            tags = self._extract_tags(text, language)
            subdomain = self._determine_subdomain(text, primary_topic, language)
            
            return TopicClassification(
                primary_topic=primary_topic,
                confidence=confidence,
                topics=topics,
                categories=categories,
                tags=tags,
                domain=primary_topic,
                subdomain=subdomain,
                method='keyword',
                keyword_matches=keyword_matches.get(primary_topic, [])
            )
            
        except Exception as e:
            self.logger.warning(f"Error in keyword topic classification: {e}")
            return self._fallback_classification(text, language)
    
    def _calculate_tfidf_similarity(self, text: str, domains: List[str]) -> Dict[str, float]:
        """Calculate TF-IDF similarity between text and domain descriptions."""
        try:
            # Create domain descriptions from keywords
            domain_descriptions = {}
            domains_data = self._topic_patterns.get('domains', {})
            
            for domain in domains:
                if domain in domains_data:
                    keywords = domains_data[domain].get('keywords', {})
                    # Combine keywords from all languages
                    all_keywords = []
                    for lang_keywords in keywords.values():
                        all_keywords.extend(lang_keywords)
                    domain_descriptions[domain] = ' '.join(all_keywords)
            
            if not domain_descriptions:
                return {}
            
            # Create corpus
            corpus = [text] + list(domain_descriptions.values())
            
            # Fit vectorizer if not already fitted
            if not hasattr(self._vectorizer, 'vocabulary_') or self._vectorizer.vocabulary_ is None:
                try:
                    tfidf_matrix = self._vectorizer.fit_transform(corpus)
                except Exception:
                    return {}
            else:
                try:
                    tfidf_matrix = self._vectorizer.transform(corpus)
                except Exception:
                    return {}
            
            # Calculate similarity
            text_vector = tfidf_matrix[0:1]
            domain_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(text_vector, domain_vectors)[0]
            
            # Map similarities back to domains
            similarity_scores = {}
            for i, domain in enumerate(domain_descriptions.keys()):
                similarity_scores[domain] = float(similarities[i])
            
            return similarity_scores
            
        except Exception as e:
            self.logger.warning(f"Error calculating TF-IDF similarity: {e}")
            return {}
    
    def _extract_categories(self, domain: str, domains_data: Dict) -> List[str]:
        """Extract broader categories for a domain."""
        categories = [domain]
        
        # Add predefined hierarchical categories
        category_hierarchy = {
            'politics': ['government', 'public_policy', 'governance'],
            'economics': ['finance', 'business', 'trade'],
            'social': ['society', 'culture', 'community'],
            'technology': ['innovation', 'digital', 'science'],
            'security': ['defense', 'safety', 'protection'],
            'environment': ['ecology', 'sustainability', 'climate']
        }
        
        if domain in category_hierarchy:
            categories.extend(category_hierarchy[domain])
        
        # Add categories from domain data if available
        if domain in domains_data:
            domain_categories = domains_data[domain].get('categories', [])
            categories.extend(domain_categories)
        
        return list(set(categories))  # Remove duplicates
    
    def _extract_tags(self, text: str, language: str) -> List[str]:
        """Extract content tags from text."""
        tags = []
        text_lower = text.lower()
        
        # Get sentiment indicators for tagging
        sentiment_indicators = self._topic_patterns.get('sentiment_indicators', {})
        
        for sentiment_type, words_dict in sentiment_indicators.items():
            words = words_dict.get(language, [])
            for word in words:
                if word.lower() in text_lower:
                    tags.append(f"{sentiment_type}_content")
                    break  # Only add tag once per sentiment type
        
        # Add topic-specific tags
        topics_data = self._topic_patterns.get('topics', {})
        for topic_category, topic_list in topics_data.items():
            for topic in topic_list:
                topic_keywords = topic.replace('_', ' ').split()
                if any(keyword in text_lower for keyword in topic_keywords):
                    tags.append(topic)
        
        # Add general content type tags
        if any(word in text_lower for word in ['новини', 'news', 'новости']):
            tags.append('news')
        if any(word in text_lower for word in ['аналіз', 'analysis', 'анализ']):
            tags.append('analysis')
        if any(word in text_lower for word in ['коментар', 'comment', 'комментарий']):
            tags.append('commentary')
        
        return list(set(tags))  # Remove duplicates
    
    def _determine_subdomain(self, text: str, primary_domain: str, language: str) -> Optional[str]:
        """Determine subdomain within the primary domain."""
        try:
            topics_data = self._topic_patterns.get('topics', {})
            text_lower = text.lower()
            
            # Map domains to their topic categories
            domain_topic_map = {
                'politics': 'political_topics',
                'economics': 'economic_topics',
                'social': 'social_topics',
                'security': 'security_topics'
            }
            
            topic_category = domain_topic_map.get(primary_domain)
            if not topic_category or topic_category not in topics_data:
                return None
            
            # Score subdomains
            subdomain_scores = {}
            for subdomain in topics_data[topic_category]:
                score = 0.0
                subdomain_keywords = subdomain.replace('_', ' ').split()
                
                for keyword in subdomain_keywords:
                    if keyword in text_lower:
                        score += 1.0
                
                if score > 0:
                    subdomain_scores[subdomain] = score
            
            if subdomain_scores:
                best_subdomain = max(subdomain_scores.keys(), key=lambda x: subdomain_scores[x])
                return best_subdomain
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error determining subdomain: {e}")
            return None
    
    def _fallback_classification(self, text: str, language: str) -> TopicClassification:
        """Fallback classification when other methods fail."""
        return TopicClassification(
            primary_topic="general",
            confidence=0.1,
            topics={"general": 1.0},
            categories=["general"],
            tags=self._extract_tags(text, language),
            domain="general",
            method="fallback"
        )
    
    def classify_hierarchical(self, text: str, language: str = 'auto') -> Dict[str, TopicClassification]:
        """
        Perform hierarchical topic classification.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Dictionary with classification at different hierarchy levels
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        results = {}
        
        try:
            # Level 1: Primary domain classification
            primary_classification = self.classify_topics(text, language)
            results['primary'] = primary_classification
            
            # Level 2: Subdomain classification within primary domain
            if primary_classification.primary_topic != 'general':
                subdomain_text = self._extract_domain_specific_text(text, primary_classification.primary_topic, language)
                if subdomain_text:
                    subdomain_classification = self._classify_subdomain(subdomain_text, primary_classification.primary_topic, language)
                    results['subdomain'] = subdomain_classification
            
            # Level 3: Specific topic classification
            specific_topics = self._classify_specific_topics(text, language)
            if specific_topics:
                results['specific'] = specific_topics
            
        except Exception as e:
            self.logger.warning(f"Error in hierarchical classification: {e}")
        
        return results
    
    def _extract_domain_specific_text(self, text: str, domain: str, language: str) -> str:
        """Extract text segments relevant to a specific domain."""
        domains_data = self._topic_patterns.get('domains', {})
        if domain not in domains_data:
            return text
        
        keywords = domains_data[domain].get('keywords', {}).get(language, [])
        if not keywords:
            return text
        
        # Extract sentences that contain domain keywords
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(keyword.lower() in sentence.lower() for keyword in keywords):
                relevant_sentences.append(sentence)
        
        return ' '.join(relevant_sentences) if relevant_sentences else text
    
    def _classify_subdomain(self, text: str, primary_domain: str, language: str) -> TopicClassification:
        """Classify subdomain within a primary domain."""
        # This is a simplified subdomain classification
        # In practice, you might want separate models for each domain
        return self.classify_topics(text, language)
    
    def _classify_specific_topics(self, text: str, language: str) -> Optional[TopicClassification]:
        """Classify specific topics using fine-grained categories."""
        # This would use more specific topic models or keyword sets
        # For now, return None to indicate this level is not implemented
        return None
    
    def get_topic_trends(self, texts: List[str], languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze topic trends across multiple texts.
        
        Args:
            texts: List of text documents
            languages: Optional list of language codes
            
        Returns:
            Topic trend analysis
        """
        if not texts:
            return {}
        
        if languages is None:
            languages = ['auto'] * len(texts)
        
        topic_counts = {}
        domain_evolution = []
        confidence_scores = []
        
        for i, text in enumerate(texts):
            lang = languages[i] if i < len(languages) else 'auto'
            try:
                classification = self.classify_topics(text, lang)
                
                # Count primary topics
                primary_topic = classification.primary_topic
                topic_counts[primary_topic] = topic_counts.get(primary_topic, 0) + 1
                
                # Track domain evolution
                domain_evolution.append({
                    'index': i,
                    'domain': primary_topic,
                    'confidence': classification.confidence
                })
                
                confidence_scores.append(classification.confidence)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing text {i}: {e}")
                continue
        
        if not topic_counts:
            return {}
        
        # Calculate statistics
        most_common_topic = max(topic_counts.keys(), key=lambda x: topic_counts[x])
        topic_diversity = len(topic_counts) / len(texts) if texts else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'total_documents': len(texts),
            'topic_distribution': topic_counts,
            'most_common_topic': most_common_topic,
            'topic_diversity': topic_diversity,
            'average_confidence': avg_confidence,
            'domain_evolution': domain_evolution[-10:],  # Last 10 for trend
            'unique_topics': len(topic_counts)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded topic classification models."""
        return {
            'transformer_available': DEPENDENCIES_AVAILABLE,
            'classifier_loaded': self._classifier_model is not None,
            'model_name': self.config.classification_model,
            'vectorizer_loaded': self._vectorizer is not None,
            'patterns_loaded': bool(self._topic_patterns),
            'supported_languages': ['uk', 'ru', 'en'],
            'available_domains': list(self._topic_patterns.get('domains', {}).keys())
        }