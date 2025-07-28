"""
Semantic Chunker for HTML RAG Pipeline.
Uses sentence embeddings to determine semantic boundaries for better chunking.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Semantic chunking implementation that uses embeddings to detect topic boundaries.
    """
    
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 50
    ):
        """
        Initialize the Semantic Chunker.
        
        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Threshold for semantic similarity (0.0-1.0)
            max_chunk_size: Maximum characters per chunk (fallback limit)
            min_chunk_size: Minimum characters per chunk
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.model: Optional[SentenceTransformer] = None
        
        logger.info(f"Initializing SemanticChunker with model: {model_name}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
    
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading semantic chunking model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading semantic chunking model: {e}")
            raise
    
    def chunk_text_blocks(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply semantic chunking to text blocks.
        
        Args:
            text_blocks: List of text block dictionaries from HTML parser
            
        Returns:
            List of semantically chunked text blocks with enhanced metadata
        """
        if not self.model:
            self.load_model()
        
        try:
            logger.info(f"Starting semantic chunking for {len(text_blocks)} text blocks")
            
            all_chunked_blocks = []
            
            for block in text_blocks:
                text = block['text']
                
                # Skip if text is too short
                if len(text) <= self.min_chunk_size:
                    enhanced_block = block.copy()
                    enhanced_block.update({
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'semantic_similarity': 1.0,
                        'topic_boundary': True,
                        'chunk_method': 'semantic'
                    })
                    all_chunked_blocks.append(enhanced_block)
                    continue
                
                # Apply semantic chunking to this block
                chunks = self._chunk_single_text(text)
                
                # Create enhanced blocks for each chunk
                for i, chunk_data in enumerate(chunks):
                    chunked_block = block.copy()
                    chunked_block.update({
                        'text': chunk_data['text'],
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'semantic_similarity': chunk_data['similarity'],
                        'topic_boundary': chunk_data['topic_boundary'],
                        'chunk_method': 'semantic'
                    })
                    all_chunked_blocks.append(chunked_block)
            
            logger.info(f"Semantic chunking completed. {len(text_blocks)} blocks became {len(all_chunked_blocks)} chunks")
            return all_chunked_blocks
            
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            raise
    
    def _chunk_single_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Apply semantic chunking to a single text.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Step 1: Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [{
                'text': text,
                'similarity': 1.0,
                'topic_boundary': True
            }]
        
        # Step 2: Generate embeddings for sentences
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        # Step 3: Calculate similarity scores between consecutive sentences
        similarities = self._calculate_consecutive_similarities(embeddings)
        
        # Step 4: Find semantic boundaries (topic changes)
        boundaries = self._find_semantic_boundaries(similarities)
        
        # Step 5: Create chunks based on boundaries
        chunks = self._create_chunks_from_boundaries(sentences, similarities, boundaries)
        
        # Step 6: Apply size constraints and merge if necessary
        final_chunks = self._apply_size_constraints(chunks)
        
        return final_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s+(?=[А-Я])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) >= 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences if cleaned_sentences else [text]
    
    def _calculate_consecutive_similarities(self, embeddings: np.ndarray) -> List[float]:
        """
        Calculate cosine similarity between consecutive sentence embeddings.
        
        Args:
            embeddings: Array of sentence embeddings
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for i in range(len(embeddings) - 1):
            # Calculate cosine similarity between consecutive sentences
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        
        return similarities
    
    def _find_semantic_boundaries(self, similarities: List[float]) -> List[int]:
        """
        Find semantic boundaries based on similarity threshold.
        
        Args:
            similarities: List of consecutive similarities
            
        Returns:
            List of boundary indices
        """
        boundaries = [0]  # Always start with first sentence
        
        for i, similarity in enumerate(similarities):
            # If similarity drops below threshold, it's a topic boundary
            if similarity < self.similarity_threshold:
                boundaries.append(i + 1)
        
        return boundaries
    
    def _create_chunks_from_boundaries(
        self, 
        sentences: List[str], 
        similarities: List[float], 
        boundaries: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Create chunks based on identified boundaries.
        
        Args:
            sentences: List of sentences
            similarities: List of similarity scores
            boundaries: List of boundary indices
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        for i, start_idx in enumerate(boundaries):
            # Determine end index for this chunk
            end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
            
            # Combine sentences in this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            # Calculate average similarity within chunk
            if start_idx < len(similarities):
                chunk_similarities = similarities[start_idx:min(end_idx-1, len(similarities))]
                avg_similarity = np.mean(chunk_similarities) if chunk_similarities else 1.0
            else:
                avg_similarity = 1.0
            
            # Determine if this is a topic boundary
            is_topic_boundary = (i == 0) or (start_idx > 0 and similarities[start_idx-1] < self.similarity_threshold)
            
            chunks.append({
                'text': chunk_text,
                'similarity': float(avg_similarity),
                'topic_boundary': is_topic_boundary
            })
        
        return chunks
    
    def _apply_size_constraints(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply size constraints and merge/split chunks if necessary.
        
        Args:
            chunks: List of initial chunks
            
        Returns:
            List of size-constrained chunks
        """
        final_chunks = []
        
        for chunk in chunks:
            text = chunk['text']
            
            # If chunk is too large, split it
            if len(text) > self.max_chunk_size:
                split_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(split_chunks)
            
            # If chunk is too small, try to merge with previous
            elif len(text) < self.min_chunk_size and final_chunks:
                last_chunk = final_chunks[-1]
                
                # Check if merging would exceed max size
                if len(last_chunk['text']) + len(text) <= self.max_chunk_size:
                    # Merge with previous chunk
                    last_chunk['text'] += ' ' + text
                    last_chunk['similarity'] = (last_chunk['similarity'] + chunk['similarity']) / 2
                    # Keep topic_boundary from the earlier chunk
                else:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a chunk that exceeds maximum size.
        
        Args:
            chunk: Chunk to split
            
        Returns:
            List of smaller chunks
        """
        text = chunk['text']
        sentences = self._split_into_sentences(text)
        
        split_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    split_chunks.append({
                        'text': current_chunk.strip(),
                        'similarity': chunk['similarity'],
                        'topic_boundary': len(split_chunks) == 0 and chunk['topic_boundary']
                    })
                current_chunk = sentence + " "
        
        # Add final chunk
        if current_chunk.strip():
            split_chunks.append({
                'text': current_chunk.strip(),
                'similarity': chunk['similarity'],
                'topic_boundary': len(split_chunks) == 0 and chunk['topic_boundary']
            })
        
        return split_chunks if split_chunks else [chunk]