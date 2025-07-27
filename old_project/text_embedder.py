"""
Stage 3: Text Embedding using sentence-transformers with paraphrase-multilingual-mpnet-base-v2
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """Text Embedder using sentence-transformers for converting text to vectors."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize the Text Embedder.
        
        Args:
            model_name: The name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension = 768  # Default for mpnet-base-v2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """Load the sentence-transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=str(self.device))
            
            # Get the actual embedding dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dimension = len(test_embedding)
            
            logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def embed_text_blocks(self, text_blocks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Convert text fragments to embeddings.
        
        Args:
            text_blocks: List of text block dictionaries from Stage 2
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of text block dictionaries with added 'embedding' field containing 768-dimensional vectors
        """
        if not self.model:
            self.load_model()
        
        try:
            logger.info(f"Starting text embedding for {len(text_blocks)} text blocks")
            
            # Extract texts for embedding
            texts = [block['text'] for block in text_blocks]
            
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Normalize for better similarity search
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings)
            
            # Add embeddings to text blocks
            embedded_blocks = []
            for i, block in enumerate(text_blocks):
                embedded_block = block.copy()
                embedded_block['embedding'] = all_embeddings[i]
                embedded_blocks.append(embedded_block)
            
            logger.info(f"Text embedding completed. Generated {len(all_embeddings)} embeddings")
            return embedded_blocks
            
        except Exception as e:
            logger.error(f"Error during text embedding: {e}")
            raise
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            768-dimensional numpy array
        """
        if not self.model:
            self.load_model()
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding single text: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query for similarity matching.
        
        Args:
            query: Search query text
            
        Returns:
            768-dimensional numpy array
        """
        return self.embed_single_text(query)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Ensure embeddings are normalized
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def batch_similarity(self, query_embedding: np.ndarray, embeddings: List[np.ndarray]) -> List[float]:
        """
        Calculate similarity between a query embedding and multiple embeddings.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: List of embedding vectors
            
        Returns:
            List of similarity scores
        """
        try:
            # Convert to numpy array for efficient computation
            embeddings_array = np.array(embeddings)
            
            # Normalize query embedding
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            # Normalize all embeddings
            embeddings_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            
            # Calculate cosine similarities
            similarities = np.dot(embeddings_norm, query_norm)
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error in batch similarity calculation: {e}")
            return [0.0] * len(embeddings)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'device': str(self.device),
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown') if self.model else 'Not loaded'
        }
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Text Embedder resources cleaned up")