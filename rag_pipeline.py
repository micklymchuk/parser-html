"""
Main RAG Pipeline that orchestrates all 4 stages:
1. HTML Pruning
2. HTML Parsing  
3. Text Embedding
4. ChromaDB Storage
"""

import logging
from typing import List, Dict, Any, Optional, Union
import time
from pathlib import Path

from html_pruner import HTMLPruner
from html_parser import HTMLParser
from text_embedder import TextEmbedder
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for processing HTML content into a searchable vector database."""
    
    def __init__(
        self,
        html_pruner_model: str = "zstanjj/HTML-Pruner-Phi-3.8B",
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        collection_name: str = "html_documents",
        persist_directory: str = "./chroma_db",
        max_chunk_size: int = 512
    ):
        """
        Initialize the RAG pipeline with all components.
        
        Args:
            html_pruner_model: Model name for HTML pruning
            embedding_model: Model name for text embedding
            collection_name: ChromaDB collection name
            persist_directory: Directory to persist ChromaDB
            max_chunk_size: Maximum characters per text chunk
        """
        self.max_chunk_size = max_chunk_size
        
        # Initialize all pipeline components
        logger.info("Initializing RAG pipeline components...")
        
        # Stage 1: HTML Pruner
        self.html_pruner = HTMLPruner(model_name=html_pruner_model)
        
        # Stage 2: HTML Parser
        self.html_parser = HTMLParser()
        
        # Stage 3: Text Embedder
        self.text_embedder = TextEmbedder(model_name=embedding_model)
        
        # Stage 4: Vector Store
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        logger.info("RAG pipeline initialized successfully")
    
    def process_html(self, raw_html: str, url: str = "") -> Dict[str, Any]:
        """
        Process raw HTML through the complete pipeline.
        
        Args:
            raw_html: Raw HTML content
            url: Source URL of the HTML content
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        logger.info(f"Starting HTML processing pipeline for URL: {url}")
        
        try:
            # Stage 1: HTML Pruning
            logger.info("Stage 1: HTML Pruning...")
            stage1_start = time.time()
            cleaned_html = self.html_pruner.prune_html(raw_html)
            stage1_time = time.time() - stage1_start
            
            # Stage 2: HTML Parsing
            logger.info("Stage 2: HTML Parsing...")
            stage2_start = time.time()
            text_blocks = self.html_parser.parse_html(cleaned_html, url)
            
            # Chunk long texts if needed
            chunked_blocks = self.html_parser.chunk_long_text(text_blocks, self.max_chunk_size)
            stage2_time = time.time() - stage2_start
            
            # Stage 3: Text Embedding
            logger.info("Stage 3: Text Embedding...")
            stage3_start = time.time()
            embedded_blocks = self.text_embedder.embed_text_blocks(chunked_blocks)
            stage3_time = time.time() - stage3_start
            
            # Stage 4: ChromaDB Storage
            logger.info("Stage 4: ChromaDB Storage...")
            stage4_start = time.time()
            self.vector_store.add_documents(embedded_blocks)
            stage4_time = time.time() - stage4_start
            
            total_time = time.time() - start_time
            
            # Compile results
            results = {
                'success': True,
                'url': url,
                'original_html_length': len(raw_html),
                'cleaned_html_length': len(cleaned_html),
                'text_blocks_count': len(text_blocks),
                'chunked_blocks_count': len(chunked_blocks),
                'embedded_blocks_count': len(embedded_blocks),
                'processing_times': {
                    'stage1_pruning': stage1_time,
                    'stage2_parsing': stage2_time,
                    'stage3_embedding': stage3_time,
                    'stage4_storage': stage4_time,
                    'total': total_time
                },
                'embedding_dimension': self.text_embedder.embedding_dimension
            }
            
            logger.info(f"HTML processing completed successfully in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error in HTML processing pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def process_multiple_html(self, html_documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple HTML documents.
        
        Args:
            html_documents: List of dictionaries with 'html' and 'url' keys
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing {len(html_documents)} HTML documents")
        results = []
        
        for i, doc in enumerate(html_documents):
            logger.info(f"Processing document {i+1}/{len(html_documents)}")
            result = self.process_html(doc['html'], doc.get('url', f'document_{i}'))
            results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        return results
    
    def search(self, query: str, n_results: int = 10, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the vector database for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List of relevant documents with similarity scores
        """
        try:
            logger.info(f"Searching for: '{query}' (returning {n_results} results)")
            
            # Embed the query
            query_embedding = self.text_embedder.embed_query(query)
            
            # Search the vector store
            results = self.vector_store.search_by_similarity(
                query_embedding=query_embedding,
                n_results=n_results,
                metadata_filter=metadata_filter
            )
            
            logger.info(f"Search completed: found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search documents by metadata criteria only.
        
        Args:
            metadata_filter: Metadata filter criteria
            limit: Maximum number of results
            
        Returns:
            List of documents matching the criteria
        """
        return self.vector_store.filter_by_metadata(metadata_filter, limit)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline and database.
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            stats = {
                'vector_store': self.vector_store.get_collection_stats(),
                'embedding_model': self.text_embedder.get_model_info(),
                'html_pruner_model': self.html_pruner.model_name,
                'max_chunk_size': self.max_chunk_size
            }
            return stats
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {}
    
    def reset_database(self) -> None:
        """Reset the vector database (delete all documents)."""
        logger.info("Resetting vector database...")
        self.vector_store.reset_collection()
        logger.info("Vector database reset completed")
    
    def load_models(self) -> None:
        """Pre-load all models to avoid delays during processing."""
        logger.info("Pre-loading all models...")
        
        # Load HTML pruner model
        self.html_pruner.load_model()
        
        # Load embedding model
        self.text_embedder.load_model()
        
        logger.info("All models loaded successfully")
    
    def cleanup(self) -> None:
        """Clean up all pipeline resources."""
        logger.info("Cleaning up RAG pipeline resources...")
        
        self.html_pruner.cleanup()
        self.text_embedder.cleanup()
        self.vector_store.cleanup()
        
        logger.info("RAG pipeline cleanup completed")
    
    def export_documents(self, output_file: str, format: str = 'json') -> None:
        """
        Export all documents to a file.
        
        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        try:
            import json
            import csv
            
            logger.info(f"Exporting documents to {output_file} in {format} format")
            
            # Get all documents
            all_docs = self.vector_store.filter_by_metadata({})
            
            if format.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_docs, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                if all_docs:
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=['text'] + list(all_docs[0]['metadata'].keys()))
                        writer.writeheader()
                        
                        for doc in all_docs:
                            row = {'text': doc['text']}
                            row.update(doc['metadata'])
                            writer.writerow(row)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Successfully exported {len(all_docs)} documents to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting documents: {e}")
            raise