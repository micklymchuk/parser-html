"""
Stage 2: HTML Parsing using BeautifulSoup to extract structured data
"""

import logging
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup, Tag, NavigableString
import re
from .semantic_chunker import SemanticChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTMLParser:
    """HTML Parser that extracts text blocks with metadata from cleaned HTML."""
    
    def __init__(self, config=None):
        """
        Initialize HTML Parser with semantic chunking capabilities.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        
        # Initialize semantic chunker if enabled
        if config and getattr(config, 'use_semantic_chunking', True):
            self.semantic_chunker = SemanticChunker(
                model_name=getattr(config, 'semantic_chunking_model', 'all-mpnet-base-v2'),
                similarity_threshold=getattr(config, 'semantic_similarity_threshold', 0.5),
                max_chunk_size=getattr(config, 'max_semantic_chunk_size', 2000),
                min_chunk_size=getattr(config, 'min_semantic_chunk_size', 50)
            )
            logger.info("HTMLParser initialized with semantic chunking enabled")
        else:
            self.semantic_chunker = None
            logger.info("HTMLParser initialized with semantic chunking disabled")
    
    def parse_html(self, cleaned_html: str, url: str = "") -> List[Dict[str, Any]]:
        """
        Parse cleaned HTML into structured data with metadata.
        
        Args:
            cleaned_html: Cleaned HTML content from Stage 1
            url: Source URL of the HTML content
            
        Returns:
            List of dictionaries with structured data containing:
            - text: clean text content
            - element_type: "heading", "paragraph", "list_item", etc.
            - hierarchy_level: for headings (h1=1, h2=2, etc.)
            - position: order on page
            - url: source URL
        """
        try:
            logger.info("Starting HTML parsing")
            self.position_counter = 0
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(cleaned_html, 'html.parser')
            
            # Extract structured data
            structured_data = []
            
            # Process all relevant elements in document order
            elements_found = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th', 'div', 'span', 'blockquote'])
            logger.debug(f"Found {len(elements_found)} HTML elements to process")
            
            for element in elements_found:
                text_blocks = self._extract_text_blocks(element, url)
                if text_blocks:
                    logger.debug(f"Extracted {len(text_blocks)} text blocks from {element.name} element")
                structured_data.extend(text_blocks)
            
            logger.info(f"HTML parsing completed. Extracted {len(structured_data)} text blocks")
            return structured_data
            
        except Exception as e:
            logger.error(f"Error during HTML parsing: {e}")
            return []
    
    def _extract_text_blocks(self, element: Tag, url: str) -> List[Dict[str, Any]]:
        """
        Extract text blocks from a single HTML element.
        
        Args:
            element: BeautifulSoup Tag element
            url: Source URL
            
        Returns:
            List of text block dictionaries
        """
        text_blocks = []
        
        # Skip if element has no text content
        text_content = self._get_clean_text(element)
        # Further reduced minimum length to capture short Ukrainian words like "ми", "він", "він", etc.
        if not text_content or len(text_content.strip()) < 2:  # Reduced to 2 characters for Ukrainian
            return text_blocks
        
        # Determine element type and hierarchy level
        element_type, hierarchy_level = self._get_element_info(element)
        
        # Handle list items specially to capture individual items
        if element.name == 'ul' or element.name == 'ol':
            for li in element.find_all('li', recursive=False):
                li_text = self._get_clean_text(li)
                if li_text and len(li_text.strip()) >= 2:  # Reduced for Ukrainian words
                    self.position_counter += 1
                    text_blocks.append({
                        'text': li_text.strip(),
                        'element_type': 'list_item',
                        'hierarchy_level': None,
                        'position': self.position_counter,
                        'url': url
                    })
        else:
            # Regular element processing
            self.position_counter += 1
            text_blocks.append({
                'text': text_content.strip(),
                'element_type': element_type,
                'hierarchy_level': hierarchy_level,
                'position': self.position_counter,
                'url': url
            })
        
        return text_blocks
    
    def _get_clean_text(self, element: Tag) -> str:
        """
        Extract clean text from an element, handling nested elements.
        
        Args:
            element: BeautifulSoup Tag element
            
        Returns:
            Clean text content
        """
        # Get text and clean it
        text = element.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _get_element_info(self, element: Tag) -> tuple[str, Optional[int]]:
        """
        Determine element type and hierarchy level.
        
        Args:
            element: BeautifulSoup Tag element
            
        Returns:
            Tuple of (element_type, hierarchy_level)
        """
        tag_name = element.name.lower()
        
        # Handle headings
        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            hierarchy_level = int(tag_name[1])
            return 'heading', hierarchy_level
        
        # Handle other elements
        element_type_mapping = {
            'p': 'paragraph',
            'li': 'list_item',
            'td': 'table_cell',
            'th': 'table_header',
            'blockquote': 'quote',
            'div': 'division',
            'span': 'span'
        }
        
        element_type = element_type_mapping.get(tag_name, 'text')
        return element_type, None
    
    def chunk_long_text(self, text_blocks: List[Dict[str, Any]], max_chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Apply semantic chunking to text blocks.
        
        Args:
            text_blocks: List of text block dictionaries
            max_chunk_size: Legacy parameter (ignored in semantic chunking)
            
        Returns:
            List of semantically chunked text blocks
        """
        if self.semantic_chunker:
            # Use semantic chunking
            chunked_blocks = self.semantic_chunker.chunk_text_blocks(text_blocks)
            logger.info(f"Semantic chunking completed. {len(text_blocks)} blocks became {len(chunked_blocks)} chunks")
            return chunked_blocks
        else:
            # Fallback: return original blocks without chunking
            logger.warning("Semantic chunking disabled. Returning original text blocks.")
            for i, block in enumerate(text_blocks):
                block.update({
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'semantic_similarity': 1.0,
                    'topic_boundary': True,
                    'chunk_method': 'none'
                })
            return text_blocks