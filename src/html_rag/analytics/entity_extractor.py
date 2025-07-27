"""
Named Entity Extraction Component for Content Analytics.

This module provides entity extraction functionality including:
- Multi-language named entity recognition
- Fuzzy matching against custom entity databases
- Entity linking and normalization
- Confidence scoring and validation
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import json
import re

try:
    import spacy
    from fuzzywuzzy import fuzz
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Entity extraction dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

from ..analytics.models import NamedEntity, EntityType
from ..core.config import ContentAnalyticsConfig
from ..exceptions.pipeline_exceptions import PipelineError, handle_pipeline_error
from ..utils.logging import PipelineLogger

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts and links named entities from text content.
    
    Supports multi-language entity recognition using spaCy models and 
    custom entity databases with fuzzy matching capabilities.
    """
    
    def __init__(self, config: Optional[ContentAnalyticsConfig] = None):
        """
        Initialize the entity extractor.
        
        Args:
            config: Content analytics configuration
        """
        if not DEPENDENCIES_AVAILABLE:
            raise PipelineError(
                "Entity extraction dependencies not available. "
                "Install with: pip install html-rag-pipeline[analytics]"
            )
        
        self.config = config or ContentAnalyticsConfig()
        self.logger = PipelineLogger("EntityExtractor")
        self._nlp_models = {}
        self._entities_db = {}
        self._entity_cache = {}
        
        self._load_entity_database()
        self._initialize_nlp_models()
    
    def _load_entity_database(self) -> None:
        """Load custom entity database."""
        try:
            entities_path = Path(self.config.entities_db_path)
            if entities_path.exists():
                with open(entities_path, 'r', encoding='utf-8') as f:
                    self._entities_db = json.load(f)
                self.logger.info("Entity database loaded successfully")
            else:
                self.logger.warning(f"Entity database not found: {entities_path}")
                self._entities_db = {}
                
        except Exception as e:
            self.logger.error(f"Error loading entity database: {e}")
            self._entities_db = {}
    
    def _initialize_nlp_models(self) -> None:
        """Initialize spaCy NLP models for different languages."""
        try:
            # Model configurations
            model_configs = [
                ('uk', 'uk_core_news_sm'),
                ('en', 'en_core_web_sm'),
                ('ru', 'ru_core_news_sm')
            ]
            
            for lang_code, model_name in model_configs:
                try:
                    if spacy.util.is_package(model_name):
                        self._nlp_models[lang_code] = spacy.load(model_name)
                        self.logger.info(f"Loaded spaCy model for {lang_code}: {model_name}")
                    else:
                        # Fallback to blank model
                        self._nlp_models[lang_code] = spacy.blank(lang_code)
                        self.logger.warning(f"Using blank model for {lang_code}")
                except OSError:
                    self._nlp_models[lang_code] = spacy.blank(lang_code)
                    self.logger.warning(f"Model {model_name} not available, using blank model")
            
            # Ensure we have at least basic models
            if not self._nlp_models:
                for lang in ['uk', 'en', 'ru']:
                    self._nlp_models[lang] = spacy.blank(lang)
                    
        except Exception as e:
            self.logger.error(f"Error initializing NLP models: {e}")
            # Fallback to basic models
            for lang in ['uk', 'en', 'ru']:
                self._nlp_models[lang] = spacy.blank(lang)
    
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
                ukrainian_indicators = ['і', 'ї', 'є', 'ґ', 'в', 'на', 'що', 'як', 'також']
                russian_indicators = ['ы', 'э', 'ё', 'в', 'на', 'что', 'как', 'также']
                
                text_lower = text.lower()
                uk_count = sum(1 for indicator in ukrainian_indicators if indicator in text_lower)
                ru_count = sum(1 for indicator in russian_indicators if indicator in text_lower)
                
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
            List of extracted entities with confidence scores
        """
        if language == 'auto':
            language = self._detect_language(text)
        
        try:
            entities = []
            
            # 1. Extract using spaCy NER
            spacy_entities = self._extract_spacy_entities(text, language)
            entities.extend(spacy_entities)
            
            # 2. Extract using custom database fuzzy matching
            db_entities = self._extract_database_entities(text, language)
            entities.extend(db_entities)
            
            # 3. Extract using regex patterns
            pattern_entities = self._extract_pattern_entities(text, language)
            entities.extend(pattern_entities)
            
            # 4. Deduplicate and resolve conflicts
            entities = self._deduplicate_entities(entities)
            
            # 5. Validate and score entities
            entities = self._validate_entities(entities, text)
            
            # 6. Sort by position and limit results
            entities.sort(key=lambda x: x.start_position)
            return entities[:self.config.max_entities_per_document]
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_spacy_entities(self, text: str, language: str) -> List[NamedEntity]:
        """Extract entities using spaCy NLP models."""
        entities = []
        
        if language not in self._nlp_models:
            return entities
        
        try:
            nlp = self._nlp_models[language]
            doc = nlp(text)
            
            for ent in doc.ents:
                entity_type = self._map_spacy_entity_type(ent.label_)
                
                # Calculate confidence based on entity characteristics
                confidence = self._calculate_spacy_confidence(ent)
                
                entity = NamedEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_position=ent.start_char,
                    end_position=ent.end_char,
                    confidence=confidence,
                    language=language,
                    extraction_method='spacy_ner',
                    spacy_label=ent.label_
                )
                entities.append(entity)
                
        except Exception as e:
            self.logger.warning(f"Error in spaCy entity extraction: {e}")
        
        return entities
    
    def _extract_database_entities(self, text: str, language: str) -> List[NamedEntity]:
        """Extract entities using fuzzy matching against custom database."""
        entities = []
        text_lower = text.lower()
        
        try:
            # Iterate through entity database categories
            for category_name, category_data in self._entities_db.items():
                for subcategory_name, subcategory_data in category_data.items():
                    for entity_name, entity_info in subcategory_data.items():
                        
                        # Check main entity name
                        matches = self._find_entity_matches(
                            text, text_lower, entity_name, entity_info, language
                        )
                        entities.extend(matches)
                        
                        # Check aliases
                        for alias in entity_info.get('aliases', []):
                            alias_matches = self._find_entity_matches(
                                text, text_lower, alias, entity_info, language
                            )
                            entities.extend(alias_matches)
                            
        except Exception as e:
            self.logger.warning(f"Error in database entity extraction: {e}")
        
        return entities
    
    def _extract_pattern_entities(self, text: str, language: str) -> List[NamedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Common patterns for different entity types
        patterns = {
            'email': [
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', EntityType.MISC),
            ],
            'phone': [
                (r'\+?[\d\s\-\(\)]{10,}', EntityType.MISC),
            ],
            'url': [
                (r'https?://[^\s]+', EntityType.MISC),
            ],
            'date': [
                (r'\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4}', EntityType.MISC),
                (r'\d{2,4}[\./\-]\d{1,2}[\./\-]\d{1,2}', EntityType.MISC),
            ]
        }
        
        # Language-specific patterns
        if language == 'uk':
            patterns.update({
                'ukrainian_names': [
                    (r'[А-ЯІЇЄҐ][а-яіїєґ]+\s+[А-ЯІЇЄҐ][а-яіїєґ]+(?:івич|ович|евич|їч)?', EntityType.PERSON),
                ]
            })
        elif language == 'ru':
            patterns.update({
                'russian_names': [
                    (r'[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:ович|евич|ич)?', EntityType.PERSON),
                ]
            })
        
        try:
            for pattern_type, pattern_list in patterns.items():
                for pattern, entity_type in pattern_list:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        confidence = self._calculate_pattern_confidence(match.group(), pattern_type)
                        
                        entity = NamedEntity(
                            text=match.group(),
                            entity_type=entity_type,
                            start_position=match.start(),
                            end_position=match.end(),
                            confidence=confidence,
                            language=language,
                            extraction_method='regex_pattern',
                            pattern_type=pattern_type
                        )
                        entities.append(entity)
                        
        except Exception as e:
            self.logger.warning(f"Error in pattern entity extraction: {e}")
        
        return entities
    
    def _find_entity_matches(self, text: str, text_lower: str, entity_name: str,
                           entity_info: Dict, language: str) -> List[NamedEntity]:
        """Find matches for a specific entity in text."""
        matches = []
        entity_lower = entity_name.lower()
        
        try:
            # Exact match
            if entity_lower in text_lower:
                positions = self._find_all_positions(text_lower, entity_lower)
                
                for start_pos in positions:
                    end_pos = start_pos + len(entity_name)
                    
                    entity = NamedEntity(
                        text=text[start_pos:end_pos],
                        entity_type=EntityType(entity_info.get('type', 'misc')),
                        start_position=start_pos,
                        end_position=end_pos,
                        confidence=0.95,  # High confidence for exact matches
                        normalized_form=entity_name,
                        aliases=entity_info.get('aliases', []),
                        description=entity_info.get('description'),
                        language=language,
                        external_ids=entity_info.get('external_ids', {}),
                        extraction_method='database_exact'
                    )
                    matches.append(entity)
            
            # Fuzzy match for longer entities
            elif len(entity_name) > 4:
                fuzzy_matches = self._find_fuzzy_matches(
                    text, text_lower, entity_name, entity_info, language
                )
                matches.extend(fuzzy_matches)
                
        except Exception as e:
            self.logger.warning(f"Error finding matches for {entity_name}: {e}")
        
        return matches
    
    def _find_fuzzy_matches(self, text: str, text_lower: str, entity_name: str,
                          entity_info: Dict, language: str) -> List[NamedEntity]:
        """Find fuzzy matches for an entity."""
        matches = []
        entity_lower = entity_name.lower()
        
        try:
            # Split text into words for fuzzy matching
            words = re.findall(r'\b\w+\b', text_lower)
            
            for i, word in enumerate(words):
                # Single word fuzzy match
                ratio = fuzz.ratio(word, entity_lower)
                if ratio > 85:  # High similarity threshold
                    start_pos = text_lower.find(word)
                    if start_pos != -1:
                        end_pos = start_pos + len(word)
                        confidence = ratio / 100.0
                        
                        entity = NamedEntity(
                            text=text[start_pos:end_pos],
                            entity_type=EntityType(entity_info.get('type', 'misc')),
                            start_position=start_pos,
                            end_position=end_pos,
                            confidence=confidence * 0.9,  # Reduce confidence for fuzzy matches
                            normalized_form=entity_name,
                            aliases=entity_info.get('aliases', []),
                            description=entity_info.get('description'),
                            language=language,
                            external_ids=entity_info.get('external_ids', {}),
                            extraction_method='database_fuzzy',
                            fuzzy_score=ratio
                        )
                        matches.append(entity)
                
                # Multi-word fuzzy match
                if i < len(words) - 1:
                    two_words = f"{words[i]} {words[i+1]}"
                    ratio = fuzz.ratio(two_words, entity_lower)
                    if ratio > 80:
                        # Find position of two-word phrase
                        phrase_pos = text_lower.find(two_words)
                        if phrase_pos != -1:
                            end_pos = phrase_pos + len(two_words)
                            confidence = ratio / 100.0
                            
                            entity = NamedEntity(
                                text=text[phrase_pos:end_pos],
                                entity_type=EntityType(entity_info.get('type', 'misc')),
                                start_position=phrase_pos,
                                end_position=end_pos,
                                confidence=confidence * 0.85,
                                normalized_form=entity_name,
                                aliases=entity_info.get('aliases', []),
                                description=entity_info.get('description'),
                                language=language,
                                external_ids=entity_info.get('external_ids', {}),
                                extraction_method='database_fuzzy',
                                fuzzy_score=ratio
                            )
                            matches.append(entity)
                            
        except Exception as e:
            self.logger.warning(f"Error in fuzzy matching for {entity_name}: {e}")
        
        return matches
    
    def _find_all_positions(self, text: str, substring: str) -> List[int]:
        """Find all positions of a substring in text."""
        positions = []
        start = 0
        while True:
            pos = text.find(substring, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions
    
    def _map_spacy_entity_type(self, spacy_label: str) -> EntityType:
        """Map spaCy entity labels to our EntityType enum."""
        mapping = {
            'PERSON': EntityType.PERSON,
            'PER': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'GEOP': EntityType.LOCATION,
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
    
    def _calculate_spacy_confidence(self, ent) -> float:
        """Calculate confidence for spaCy entities."""
        # Base confidence
        confidence = 0.7
        
        # Longer entities typically more reliable
        if len(ent.text) > 10:
            confidence += 0.1
        elif len(ent.text) < 3:
            confidence -= 0.2
        
        # Capitalized entities more likely to be proper nouns
        if ent.text[0].isupper():
            confidence += 0.1
        
        # Check if entity has punctuation (might be noise)
        if any(char in ent.text for char in '.,;!?'):
            confidence -= 0.2
        
        return max(0.1, min(0.9, confidence))
    
    def _calculate_pattern_confidence(self, text: str, pattern_type: str) -> float:
        """Calculate confidence for pattern-based entities."""
        base_confidence = {
            'email': 0.9,
            'phone': 0.8,
            'url': 0.95,
            'date': 0.7,
            'ukrainian_names': 0.6,
            'russian_names': 0.6
        }
        
        confidence = base_confidence.get(pattern_type, 0.5)
        
        # Adjust based on text characteristics
        if len(text) > 20:
            confidence += 0.1
        elif len(text) < 5:
            confidence -= 0.1
        
        return max(0.1, min(0.95, confidence))
    
    def _deduplicate_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Remove duplicate and overlapping entities."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_position)
        
        deduplicated = []
        
        for entity in entities:
            # Check for overlaps with existing entities
            should_add = True
            
            for i, existing in enumerate(deduplicated):
                overlap = self._calculate_overlap(entity, existing)
                
                if overlap > 0.5:  # Significant overlap
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated[i] = entity
                    should_add = False
                    break
                elif overlap > 0:  # Partial overlap
                    # Keep the longer entity if confidence is similar
                    if (abs(entity.confidence - existing.confidence) < 0.1 and
                        len(entity.text) > len(existing.text)):
                        deduplicated[i] = entity
                        should_add = False
                        break
                    elif entity.confidence <= existing.confidence:
                        should_add = False
                        break
            
            if should_add:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _calculate_overlap(self, entity1: NamedEntity, entity2: NamedEntity) -> float:
        """Calculate overlap ratio between two entities."""
        start1, end1 = entity1.start_position, entity1.end_position
        start2, end2 = entity2.start_position, entity2.end_position
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        min_length = min(end1 - start1, end2 - start2)
        
        return overlap_length / min_length if min_length > 0 else 0.0
    
    def _validate_entities(self, entities: List[NamedEntity], text: str) -> List[NamedEntity]:
        """Validate and filter entities."""
        validated = []
        
        for entity in entities:
            # Skip very short entities unless they're high confidence
            if len(entity.text) < 2 and entity.confidence < 0.8:
                continue
            
            # Skip entities that are mostly punctuation
            if sum(1 for c in entity.text if c.isalnum()) / len(entity.text) < 0.5:
                continue
            
            # Skip entities with very low confidence
            if entity.confidence < 0.1:
                continue
            
            # Validate position bounds
            if (entity.start_position < 0 or 
                entity.end_position > len(text) or 
                entity.start_position >= entity.end_position):
                continue
            
            validated.append(entity)
        
        return validated
    
    def get_entity_statistics(self, entities: List[NamedEntity]) -> Dict[str, Any]:
        """Get statistics about extracted entities."""
        if not entities:
            return {}
        
        type_counts = {}
        method_counts = {}
        total_confidence = 0
        
        for entity in entities:
            # Count by type
            entity_type = entity.entity_type.value
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            # Count by extraction method
            method = getattr(entity, 'extraction_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            
            total_confidence += entity.confidence
        
        return {
            'total_entities': len(entities),
            'average_confidence': total_confidence / len(entities),
            'type_distribution': type_counts,
            'extraction_methods': method_counts,
            'unique_entities': len(set(e.normalized_form or e.text for e in entities))
        }
    
    def link_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Link entities to external knowledge bases."""
        linked_entities = []
        
        for entity in entities:
            linked_entity = entity
            
            # Try to find additional links in the database
            if entity.normalized_form:
                entity_info = self._find_entity_in_database(entity.normalized_form)
                if entity_info:
                    # Update external IDs
                    if 'external_ids' in entity_info:
                        linked_entity.external_ids.update(entity_info['external_ids'])
                    
                    # Update description if not present
                    if not linked_entity.description and 'description' in entity_info:
                        linked_entity.description = entity_info['description']
            
            linked_entities.append(linked_entity)
        
        return linked_entities
    
    def _find_entity_in_database(self, entity_name: str) -> Optional[Dict]:
        """Find entity information in the database."""
        entity_lower = entity_name.lower()
        
        for category_data in self._entities_db.values():
            for subcategory_data in category_data.values():
                for name, info in subcategory_data.items():
                    if name.lower() == entity_lower:
                        return info
                    
                    # Check aliases
                    for alias in info.get('aliases', []):
                        if alias.lower() == entity_lower:
                            return info
        
        return None