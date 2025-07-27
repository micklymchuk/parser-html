"""
Main RAG Pipeline that orchestrates all 5 stages:
0. Wayback Snapshot Processing (optional)
1. HTML Pruning
2. HTML Parsing  
3. Text Embedding
4. ChromaDB Storage

Enhanced with modern Python practices, comprehensive error handling, and metrics.
"""
import inspect
from typing import List, Dict, Any, Optional, Union
import time
from pathlib import Path

from ..processors.wayback import WaybackProcessor
from ..processors.html_pruner import HTMLPruner
from ..processors.html_parser import HTMLParser
from ..processors.text_embedder import TextEmbedder
from ..processors.vector_store import VectorStore
from ..processors.content_analyzer import ContentAnalyzer
from ..core.config import PipelineConfig, WaybackConfig, SearchConfig, ContentAnalyticsConfig
from ..utils.logging import PipelineLogger
from ..utils.metrics import MetricsCollector, track_stage, track_processing
from ..utils.validators import PipelineValidator, validate_search_query
from ..exceptions.pipeline_exceptions import (
    PipelineError, HTMLProcessingError, EmbeddingError, 
    VectorStoreError, ValidationError, handle_pipeline_error
)
from ..analytics.models import ContentAnalytics, AnalyticsResult


class RAGPipeline:
    """Complete RAG pipeline for processing HTML content into a searchable vector database."""
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        html_pruner_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        max_chunk_size: Optional[int] = None,
        prefer_basic_cleaning: Optional[bool] = None,
        cyrillic_detection_threshold: Optional[float] = None,
        enable_metrics: bool = True,
        enable_content_analytics: bool = False,
        analytics_config: Optional[ContentAnalyticsConfig] = None
    ):
        """
        Initialize the RAG pipeline with all components.
        
        Args:
            config: Optional PipelineConfig instance
            html_pruner_model: Model name for HTML pruning (overrides config)
            embedding_model: Model name for text embedding (overrides config)
            collection_name: ChromaDB collection name (overrides config)
            persist_directory: Directory to persist ChromaDB (overrides config)
            max_chunk_size: Maximum characters per text chunk (overrides config)
            prefer_basic_cleaning: If True, default to basic cleaning over AI model (overrides config)
            cyrillic_detection_threshold: Threshold for Cyrillic content detection (overrides config)
            enable_metrics: Enable performance metrics collection
            enable_content_analytics: Enable content analytics processing
            analytics_config: Optional content analytics configuration
        """
        # Load configuration
        if config is None:
            config = PipelineConfig()
        
        # Override config with explicit parameters
        if html_pruner_model is not None:
            config.html_pruner_model = html_pruner_model
        if embedding_model is not None:
            config.embedding_model = embedding_model
        if collection_name is not None:
            config.collection_name = collection_name
        if persist_directory is not None:
            config.persist_directory = persist_directory
        if max_chunk_size is not None:
            config.max_chunk_size = max_chunk_size
        if prefer_basic_cleaning is not None:
            config.prefer_basic_cleaning = prefer_basic_cleaning
        if cyrillic_detection_threshold is not None:
            config.cyrillic_detection_threshold = cyrillic_detection_threshold
        
        self.config = config
        self.enable_metrics = enable_metrics
        self.enable_content_analytics = enable_content_analytics
        
        # Initialize logger
        self.logger = PipelineLogger("RAGPipeline")
        
        # Initialize metrics collector
        if self.enable_metrics:
            self.metrics_collector = MetricsCollector(
                enable_resource_monitoring=config.enable_metrics
            )
        else:
            self.metrics_collector = None
        
        # Initialize all pipeline components
        self.logger.info("Initializing RAG pipeline components...")
        
        try:
            # Stage 0: Wayback Processor
            self.wayback_processor = WaybackProcessor()
            
            # Stage 1: HTML Pruner
            self.html_pruner = HTMLPruner(model_name=config.html_pruner_model)
            
            # Stage 2: HTML Parser
            self.html_parser = HTMLParser()
            
            # Stage 3: Text Embedder
            self.text_embedder = TextEmbedder(model_name=config.embedding_model)
            
            # Stage 4: Vector Store
            self.vector_store = VectorStore(
                collection_name=config.collection_name,
                persist_directory=config.persist_directory
            )
            
            # Stage 5: Content Analytics (optional)
            self.content_analyzer = None
            if self.enable_content_analytics:
                try:
                    if analytics_config is None:
                        analytics_config = ContentAnalyticsConfig(enabled=True)
                    self.content_analyzer = ContentAnalyzer(analytics_config)
                    self.logger.info("Content analytics enabled")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize content analytics: {e}")
                    self.enable_content_analytics = False
            
            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error_with_context(e, {"component": "initialization"})
            raise PipelineError(f"Failed to initialize RAG pipeline: {str(e)}") from e
    
    def _detect_cyrillic_content(self, html_content: str) -> bool:
        """
        Detect if HTML content contains significant Cyrillic/Ukrainian text.
        
        Args:
            html_content: Raw HTML content to analyze
            
        Returns:
            True if significant Cyrillic content is detected
        """
        try:
            # Count Cyrillic characters (Ukrainian, Russian, etc.)
            cyrillic_count = 0
            latin_count = 0
            
            # Sample first 2000 characters for performance
            sample_text = html_content[:2000]
            
            for char in sample_text:
                if '\u0400' <= char <= '\u04FF':  # Cyrillic Unicode range
                    cyrillic_count += 1
                elif 'a' <= char.lower() <= 'z':  # Latin characters
                    latin_count += 1
            
            total_letters = cyrillic_count + latin_count
            if total_letters == 0:
                return False
            
            # Use configurable threshold for Cyrillic content detection
            cyrillic_ratio = cyrillic_count / total_letters
            self.logger.debug(f"Cyrillic detection: {cyrillic_count}/{total_letters} letters ({cyrillic_ratio:.2%})")
            
            return cyrillic_ratio > self.config.cyrillic_detection_threshold
            
        except Exception as e:
            self.logger.warning(f"Error in Cyrillic detection: {e}")
            return False
    
    @handle_pipeline_error
    def process_html(
        self, 
        raw_html: str, 
        url: str = "", 
        wayback_metadata: Optional[Dict[str, Any]] = None, 
        force_basic_cleaning: bool = False,
        analyze_content: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process raw HTML through the complete pipeline.
        
        Args:
            raw_html: Raw HTML content
            url: Source URL of the HTML content
            wayback_metadata: Optional Wayback Machine metadata
            force_basic_cleaning: If True, forces basic HTML cleaning instead of AI model
            analyze_content: If True, performs content analytics (overrides pipeline setting)
            
        Returns:
            Dictionary with processing results and statistics
            
        Raises:
            ValidationError: If input validation fails
            HTMLProcessingError: If HTML processing fails
            EmbeddingError: If embedding generation fails
            VectorStoreError: If vector storage fails
        """
        # Validate input
        try:
            validated_input = PipelineValidator.validate_pipeline_input(
                html=raw_html, url=url, wayback_metadata=wayback_metadata
            )
            raw_html = validated_input['html']
            url = validated_input['url']
        except Exception as e:
            raise ValidationError(f"Input validation failed: {str(e)}")
        
        # Start metrics collection
        if self.metrics_collector:
            self.metrics_collector.start_collection()
        
        self.logger.log_stage_start("HTML Processing", {"url": url})
        
        try:
            # Stage 1: HTML Pruning
            with track_stage(self.metrics_collector, "stage1_pruning") if self.metrics_collector else nullcontext():
                self.logger.log_stage_start("Stage 1: HTML Pruning")
                
                # SMART DECISION LOGIC: Determine whether to use AI model or basic cleaning
                # Priority order:
                # 1. force_basic_cleaning parameter overrides everything
                # 2. prefer_basic_cleaning configuration setting
                # 3. Wayback content → basic cleaning (preserve Ukrainian/non-English)
                # 4. Cyrillic content detection → basic cleaning
                # 5. Default → AI model for English content
                
                if force_basic_cleaning:
                    use_ai_model = False
                    self.logger.info("Forced basic cleaning: Using basic HTML cleaning (force_basic_cleaning=True)")
                elif self.config.prefer_basic_cleaning:
                    use_ai_model = False
                    self.logger.info("Pipeline configured for basic cleaning: Using basic HTML cleaning (prefer_basic_cleaning=True)")
                elif wayback_metadata is not None:
                    # Wayback content detected → Use basic cleaning to preserve Ukrainian text
                    use_ai_model = False
                    self.logger.info("Wayback metadata detected: Using basic HTML cleaning to preserve Ukrainian/non-English content")
                elif self._detect_cyrillic_content(raw_html):
                    # Cyrillic content detected → Use basic cleaning
                    use_ai_model = False
                    self.logger.info("Cyrillic/Ukrainian content detected: Using basic HTML cleaning to preserve non-English text")
                else:
                    # Regular English content → Can use AI model
                    use_ai_model = True
                    self.logger.info("English content detected: Using AI model for HTML pruning")
                
                try:
                    cleaned_html = self.html_pruner.prune_html(raw_html, use_ai_model=use_ai_model)
                except Exception as e:
                    raise HTMLProcessingError(f"HTML pruning failed: {str(e)}")
                
                self.logger.log_stage_end("Stage 1: HTML Pruning", {
                    "original_length": len(raw_html),
                    "cleaned_length": len(cleaned_html),
                    "use_ai_model": use_ai_model
                })

            # Stage 2: HTML Parsing
            with track_stage(self.metrics_collector, "stage2_parsing") if self.metrics_collector else nullcontext():
                self.logger.log_stage_start("Stage 2: HTML Parsing")
                
                try:
                    text_blocks = self.html_parser.parse_html(cleaned_html, url)
                except Exception as e:
                    raise HTMLProcessingError(f"HTML parsing failed: {str(e)}")
                
                self.logger.info(f"Stage 2: Extracted {len(text_blocks)} text blocks")
                
                # Check if we have any text blocks
                if not text_blocks:
                    self._record_processing_failure(
                        "No text blocks extracted from HTML content",
                        len(raw_html), len(cleaned_html), 0, 0
                    )
                    return {
                        'success': False,
                        'error': 'No text blocks extracted from HTML content',
                        'url': url,
                        'original_html_length': len(raw_html),
                        'cleaned_html_length': len(cleaned_html)
                    }
                
                # Add wayback metadata to each text block if provided
                if wayback_metadata:
                    for block in text_blocks:
                        block.update({
                            'wayback_timestamp': wayback_metadata.get('timestamp'),
                            'wayback_archive_url': wayback_metadata.get('archive_url'),
                            'wayback_original_url': wayback_metadata.get('original_url'),
                            'wayback_domain': wayback_metadata.get('domain'),
                            'wayback_title': wayback_metadata.get('title')
                        })
                
                # Chunk long texts if needed
                chunked_blocks = self.html_parser.chunk_long_text(text_blocks, self.config.max_chunk_size)
                
                self.logger.log_stage_end("Stage 2: HTML Parsing", {
                    "text_blocks": len(text_blocks),
                    "chunked_blocks": len(chunked_blocks)
                })

            # Stage 3: Text Embedding
            with track_stage(self.metrics_collector, "stage3_embedding") if self.metrics_collector else nullcontext():
                self.logger.log_stage_start("Stage 3: Text Embedding")
                
                if not chunked_blocks:
                    self._record_processing_failure(
                        "No text blocks available for embedding",
                        len(raw_html), len(cleaned_html), len(text_blocks), 0
                    )
                    return {
                        'success': False,
                        'error': 'No text blocks available for embedding',
                        'url': url,
                        'text_blocks_count': len(text_blocks),
                        'chunked_blocks_count': len(chunked_blocks)
                    }
                
                try:
                    embedded_blocks = self.text_embedder.embed_text_blocks(chunked_blocks)
                except Exception as e:
                    raise EmbeddingError(f"Text embedding failed: {str(e)}")
                
                self.logger.log_stage_end("Stage 3: Text Embedding", {
                    "embedded_blocks": len(embedded_blocks),
                    "embedding_dimension": self.text_embedder.embedding_dimension
                })

            # Stage 4: ChromaDB Storage
            with track_stage(self.metrics_collector, "stage4_storage") if self.metrics_collector else nullcontext():
                self.logger.log_stage_start("Stage 4: ChromaDB Storage")
                
                try:
                    self.vector_store.add_documents(embedded_blocks)
                except Exception as e:
                    raise VectorStoreError(f"Vector storage failed: {str(e)}")
                
                self.logger.log_stage_end("Stage 4: ChromaDB Storage", {
                    "documents_stored": len(embedded_blocks)
                })

            # Stage 5: Content Analytics (optional)
            content_analytics_result = None
            should_analyze = analyze_content if analyze_content is not None else self.enable_content_analytics
            
            if should_analyze and self.content_analyzer:
                with track_stage(self.metrics_collector, "stage5_analytics") if self.metrics_collector else nullcontext():
                    self.logger.log_stage_start("Stage 5: Content Analytics")
                    
                    try:
                        # Combine all text blocks for analysis
                        full_text = ' '.join([block['text'] for block in text_blocks if block.get('text')])
                        
                        if full_text.strip():
                            # Generate unique document ID
                            doc_id = f"{url}_{int(time.time())}" if url else f"doc_{int(time.time())}"
                            
                            # Perform content analysis
                            content_analytics_result = self.content_analyzer.analyze_document(
                                text=full_text,
                                document_id=doc_id,
                                url=url
                            )
                            
                            # Store analytics results in vector store metadata
                            analytics_metadata = {
                                'analytics_processed': True,
                                'sentiment_score': content_analytics_result.sentiment.score,
                                'sentiment_label': content_analytics_result.sentiment.label.value,
                                'controversy_score': content_analytics_result.controversy.score,
                                'controversy_level': content_analytics_result.controversy.level.value,
                                'primary_topic': content_analytics_result.topics.primary_topic,
                                'topic_confidence': content_analytics_result.topics.confidence,
                                'entity_count': len(content_analytics_result.entities),
                                'language': content_analytics_result.language,
                                'quality_score': content_analytics_result.quality_score,
                                'confidence_score': content_analytics_result.confidence_score
                            }
                            
                            # Update document metadata with analytics
                            for block in embedded_blocks:
                                block['metadata'].update(analytics_metadata)
                            
                            # Re-store documents with updated metadata
                            self.vector_store.add_documents(embedded_blocks)
                            
                            self.logger.info(f"Content analytics completed for {doc_id}")
                        else:
                            self.logger.warning("No text content available for analytics")
                            
                    except Exception as e:
                        self.logger.warning(f"Content analytics failed: {e}")
                        # Continue processing even if analytics fails
                    
                    self.logger.log_stage_end("Stage 5: Content Analytics", {
                        "analytics_performed": content_analytics_result is not None
                    })

            # Record success metrics
            if self.metrics_collector:
                self.metrics_collector.record_document_processed(
                    success=True,
                    html_length=len(raw_html),
                    cleaned_length=len(cleaned_html),
                    text_blocks=len(text_blocks),
                    embedded_blocks=len(embedded_blocks)
                )
                self.metrics_collector.end_collection()
            
            # Compile results
            results = {
                'success': True,
                'url': url,
                'original_html_length': len(raw_html),
                'cleaned_html_length': len(cleaned_html),
                'text_blocks_count': len(text_blocks),
                'chunked_blocks_count': len(chunked_blocks),
                'embedded_blocks_count': len(embedded_blocks),
                'embedding_dimension': self.text_embedder.embedding_dimension,
                'content_analytics': content_analytics_result.dict() if content_analytics_result else None
            }
            
            # Add metrics if available
            if self.metrics_collector:
                results['metrics'] = self.metrics_collector.get_metrics_dict()
            
            self.logger.log_stage_end("HTML Processing", results)
            return results
            
        except PipelineError:
            # Re-raise pipeline errors as-is
            if self.metrics_collector:
                self.metrics_collector.end_collection()
            raise
        except Exception as e:
            # Handle unexpected errors
            self._record_processing_failure(str(e), len(raw_html), 0, 0, 0, e)
            raise PipelineError(f"Unexpected error in HTML processing: {str(e)}") from e
    
    def _record_processing_failure(
        self, 
        error_message: str, 
        html_length: int, 
        cleaned_length: int, 
        text_blocks: int, 
        embedded_blocks: int, 
        error: Optional[Exception] = None
    ) -> None:
        """Record processing failure in metrics."""
        if self.metrics_collector:
            self.metrics_collector.record_document_processed(
                success=False,
                html_length=html_length,
                cleaned_length=cleaned_length,
                text_blocks=text_blocks,
                embedded_blocks=embedded_blocks,
                error=error
            )
            self.metrics_collector.end_collection()
        
        self.logger.warning(f"Processing failed: {error_message}")
    
    @handle_pipeline_error
    def process_multiple_html(
        self, 
        html_documents: List[Dict[str, str]], 
        force_basic_cleaning: bool = False,
        analyze_content: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple HTML documents.
        
        Args:
            html_documents: List of dictionaries with 'html' and 'url' keys
            force_basic_cleaning: If True, forces basic cleaning for all documents
            analyze_content: If True, performs content analytics on all documents
            
        Returns:
            List of processing results
            
        Raises:
            ValidationError: If input validation fails
        """
        # Validate batch input
        try:
            validated_documents = PipelineValidator.validate_batch_input(html_documents)
        except Exception as e:
            raise ValidationError(f"Batch input validation failed: {str(e)}")
        
        self.logger.info(f"Processing {len(validated_documents)} HTML documents")
        
        with track_processing(self.config.enable_metrics) as batch_metrics:
            results = []
            
            for i, doc in enumerate(validated_documents):
                self.logger.info(f"Processing document {i+1}/{len(validated_documents)}")
                
                wayback_metadata = doc.get('wayback_metadata')
                try:
                    result = self.process_html(
                        doc['html'], 
                        doc.get('url', f'document_{i}'),
                        wayback_metadata,
                        force_basic_cleaning,
                        analyze_content
                    )
                    results.append(result)
                except Exception as e:
                    error_result = {
                        'success': False,
                        'error': str(e),
                        'url': doc.get('url', f'document_{i}'),
                        'document_index': i
                    }
                    results.append(error_result)
                    self.logger.error(f"Failed to process document {i}: {str(e)}")
            
            # Summary statistics
            successful = sum(1 for r in results if r.get('success', False))
            failed = len(results) - successful
            
            summary = {
                'total_documents': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(results) if results else 0.0
            }
            
            if batch_metrics:
                summary['batch_metrics'] = batch_metrics.get_metrics_dict()
            
            self.logger.log_processing_stats(summary)
        
        return results
    
    @handle_pipeline_error
    def search(
        self, 
        query: str, 
        n_results: int = 10, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search the vector database for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            metadata_filter: Optional metadata filter
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of relevant documents with similarity scores
            
        Raises:
            ValidationError: If search parameters are invalid
            EmbeddingError: If query embedding fails
            VectorStoreError: If search fails
        """
        try:
            # Validate search query
            validated_params = validate_search_query(
                query=query,
                n_results=n_results,
                metadata_filter=metadata_filter,
                similarity_threshold=similarity_threshold
            )
            
            self.logger.info(f"Searching for: '{query}' (returning {n_results} results)")
            
            # Embed the query
            try:
                query_embedding = self.text_embedder.embed_query(query)
            except Exception as e:
                raise EmbeddingError(f"Query embedding failed: {str(e)}")
            
            # Search the vector store
            try:
                results = self.vector_store.search_by_similarity(
                    query_embedding=query_embedding,
                    n_results=n_results,
                    metadata_filter=metadata_filter
                )
            except Exception as e:
                raise VectorStoreError(f"Vector search failed: {str(e)}")
            
            # Filter by similarity threshold
            if similarity_threshold > 0:
                results = [r for r in results if r.get('similarity_score', 0) >= similarity_threshold]
            
            self.logger.info(f"Search completed: found {len(results)} results")
            return results
            
        except ValidationError:
            raise
        except (EmbeddingError, VectorStoreError):
            raise
        except Exception as e:
            raise PipelineError(f"Search operation failed: {str(e)}") from e
    
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
    
    @handle_pipeline_error
    def process_wayback_snapshots(
        self, 
        snapshots_directory: str, 
        wayback_config: Optional[WaybackConfig] = None,
        **filter_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process Wayback Machine snapshots from a directory (Stage 0 + Stages 1-4).
        
        Args:
            snapshots_directory: Path to directory containing wayback snapshots
            wayback_config: Optional Wayback configuration
            **filter_kwargs: Optional filtering criteria (domain_filter, year_filter, min_content_length)
            
        Returns:
            List of processing results
            
        Raises:
            ValidationError: If directory validation fails
            PipelineError: If processing fails
        """
        if wayback_config is None:
            wayback_config = WaybackConfig()
        
        try:
            self.logger.info(f"Processing Wayback snapshots from: {snapshots_directory}")
            
            # Stage 0: Process Wayback snapshots
            self.logger.log_stage_start("Stage 0: Wayback Snapshot Processing")
            
            # Validate directory first
            validation_result = self.wayback_processor.validate_snapshot_directory(snapshots_directory)
            if not validation_result['is_valid']:
                raise ValidationError(f"Invalid snapshot directory: {validation_result['errors']}")
            
            # Process snapshots
            snapshots = self.wayback_processor.process_snapshots_directory(
                directory_path=snapshots_directory,
                require_metadata=wayback_config.require_metadata
            )
            
            if not snapshots:
                self.logger.warning("No snapshots found to process")
                return []
            
            # Apply filters if provided
            combined_filters = {**wayback_config.dict(), **filter_kwargs}
            self.logger.info(combined_filters)
            if any(v for v in combined_filters.values() if v):
                snapshots = self.wayback_processor.filter_snapshots_by_criteria(snapshots, **filter_kwargs)
                self.logger.info(f"After filtering: {len(snapshots)} snapshots remaining")
            
            self.logger.log_stage_end("Stage 0: Wayback Snapshot Processing", {
                "snapshots_found": len(snapshots)
            })
            
            # Process through remaining stages (1-4)
            results = self.process_multiple_html(
                snapshots, 
                force_basic_cleaning=wayback_config.force_basic_cleaning
            )
            
            # Add wayback processing metadata to results
            for result in results:
                if result.get('success'):
                    result['wayback_processed'] = True
            
            # Add summary statistics
            wayback_stats = self.wayback_processor.get_processing_stats()
            self.logger.log_processing_stats(wayback_stats)
            
            return results
            
        except ValidationError:
            raise
        except Exception as e:
            raise PipelineError(f"Wayback processing failed: {str(e)}") from e
    
    def validate_wayback_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Validate a Wayback snapshots directory.
        
        Args:
            directory_path: Path to the snapshots directory
            
        Returns:
            Dictionary with validation results
        """
        return self.wayback_processor.validate_snapshot_directory(directory_path)
    
    def search_wayback_snapshots(
        self, 
        query: str, 
        n_results: int = 10,
        timestamp_filter: Optional[str] = None,
        domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for Wayback snapshot content.
        
        Args:
            query: Search query
            n_results: Number of results to return
            timestamp_filter: Filter by specific timestamp (YYYYMMDDHHMMSS)
            domain_filter: Filter by domain
            
        Returns:
            List of relevant Wayback snapshot documents
        """
        # Build metadata filter for wayback content
        metadata_filter = {}
        
        if timestamp_filter:
            metadata_filter['wayback_timestamp'] = timestamp_filter
        
        if domain_filter:
            metadata_filter['wayback_domain'] = domain_filter
        
        # Search with metadata filter
        try:
            return self.search(query, n_results, metadata_filter)
        except Exception as e:
            self.logger.warning(f"Wayback-specific search failed, falling back to normal search: {e}")
            return self.search(query, n_results)
    
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
                'wayback_processor': 'Available',
                'content_analytics': {
                    'enabled': self.enable_content_analytics,
                    'available': self.content_analyzer is not None,
                    'model_info': self.content_analyzer.get_model_info() if self.content_analyzer else None
                },
                'config': self.config.dict(),
                'pipeline_version': '2.0.0'
            }
            
            if self.metrics_collector:
                stats['metrics'] = self.metrics_collector.get_metrics_dict()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline stats: {e}")
            return {}
    
    def reset_database(self) -> None:
        """Reset the vector database (delete all documents)."""
        self.logger.info("Resetting vector database...")
        self.vector_store.reset_collection()
        self.logger.info("Vector database reset completed")
    
    def load_models(self) -> None:
        """Pre-load all models to avoid delays during processing."""
        self.logger.info("Pre-loading all models...")
        
        # Load HTML pruner model
        self.html_pruner.load_model()
        
        # Load embedding model
        self.text_embedder.load_model()
        
        self.logger.info("All models loaded successfully")
    
    def cleanup(self) -> None:
        """Clean up all pipeline resources."""
        self.logger.info("Cleaning up RAG pipeline resources...")
        
        try:
            self.html_pruner.cleanup()
            self.text_embedder.cleanup()
            self.vector_store.cleanup()
            
            if self.content_analyzer:
                self.content_analyzer.cleanup()
            
            self.logger.info("RAG pipeline cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def export_documents(self, output_file: str, format: str = 'json') -> None:
        """
        Export all documents to a file.
        
        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
            
        Raises:
            ValueError: If format is unsupported
            PipelineError: If export fails
        """
        try:
            import json
            import csv
            
            self.logger.info(f"Exporting documents to {output_file} in {format} format")
            
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
            
            self.logger.info(f"Successfully exported {len(all_docs)} documents to {output_file}")
            
        except Exception as e:
            raise PipelineError(f"Export failed: {str(e)}") from e
    
    # Content Analytics Methods
    
    @handle_pipeline_error
    def analyze_existing_documents(self, limit: Optional[int] = None) -> AnalyticsResult:
        """
        Perform content analytics on existing documents in the vector store.
        
        Args:
            limit: Maximum number of documents to analyze
            
        Returns:
            Analytics results for existing documents
            
        Raises:
            PipelineError: If content analytics is not enabled or analysis fails
        """
        if not self.enable_content_analytics or not self.content_analyzer:
            raise PipelineError("Content analytics is not enabled")
        
        try:
            self.logger.info("Analyzing existing documents in vector store...")
            
            # Get all documents from vector store
            all_docs = self.vector_store.filter_by_metadata({}, limit=limit)
            
            if not all_docs:
                self.logger.warning("No documents found in vector store")
                return AnalyticsResult(
                    total_documents=0,
                    successful=0,
                    failed=0,
                    analytics=[],
                    processing_time=0.0,
                    average_processing_time=0.0,
                    errors=[],
                    warnings=[]
                )
            
            # Prepare documents for analysis
            documents_for_analysis = []
            for doc in all_docs:
                documents_for_analysis.append({
                    'text': doc['text'],
                    'id': doc['metadata'].get('chunk_id', 'unknown'),
                    'url': doc['metadata'].get('url', '')
                })
            
            # Perform batch analysis
            result = self.content_analyzer.analyze_batch(documents_for_analysis)
            
            self.logger.info(f"Analyzed {result.successful} documents successfully")
            return result
            
        except Exception as e:
            raise PipelineError(f"Failed to analyze existing documents: {str(e)}") from e
    
    def search_by_sentiment(self, sentiment_label: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by sentiment label.
        
        Args:
            sentiment_label: Sentiment label ('positive', 'negative', 'neutral')
            n_results: Number of results to return
            
        Returns:
            List of documents with the specified sentiment
        """
        metadata_filter = {'sentiment_label': sentiment_label}
        return self.search_by_metadata(metadata_filter, limit=n_results)
    
    def search_by_controversy_level(self, controversy_level: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by controversy level.
        
        Args:
            controversy_level: Controversy level ('low', 'medium', 'high', 'critical')
            n_results: Number of results to return
            
        Returns:
            List of documents with the specified controversy level
        """
        metadata_filter = {'controversy_level': controversy_level}
        return self.search_by_metadata(metadata_filter, limit=n_results)
    
    def search_by_topic(self, topic: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by primary topic.
        
        Args:
            topic: Primary topic to search for
            n_results: Number of results to return
            
        Returns:
            List of documents with the specified topic
        """
        metadata_filter = {'primary_topic': topic}
        return self.search_by_metadata(metadata_filter, limit=n_results)
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of analytics data in the vector store.
        
        Returns:
            Dictionary with analytics summary statistics
        """
        try:
            # Get all documents with analytics metadata
            all_docs = self.vector_store.filter_by_metadata({'analytics_processed': True})
            
            if not all_docs:
                return {'message': 'No documents with analytics data found'}
            
            # Calculate summary statistics
            sentiment_counts = {}
            controversy_counts = {}
            topic_counts = {}
            sentiment_scores = []
            controversy_scores = []
            quality_scores = []
            
            for doc in all_docs:
                metadata = doc['metadata']
                
                # Sentiment statistics
                sentiment_label = metadata.get('sentiment_label')
                if sentiment_label:
                    sentiment_counts[sentiment_label] = sentiment_counts.get(sentiment_label, 0) + 1
                
                sentiment_score = metadata.get('sentiment_score')
                if sentiment_score is not None:
                    sentiment_scores.append(sentiment_score)
                
                # Controversy statistics
                controversy_level = metadata.get('controversy_level')
                if controversy_level:
                    controversy_counts[controversy_level] = controversy_counts.get(controversy_level, 0) + 1
                
                controversy_score = metadata.get('controversy_score')
                if controversy_score is not None:
                    controversy_scores.append(controversy_score)
                
                # Topic statistics
                topic = metadata.get('primary_topic')
                if topic:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                # Quality statistics
                quality_score = metadata.get('quality_score')
                if quality_score is not None:
                    quality_scores.append(quality_score)
            
            # Calculate averages
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            avg_controversy = sum(controversy_scores) / len(controversy_scores) if controversy_scores else 0
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            return {
                'total_documents_analyzed': len(all_docs),
                'sentiment_distribution': sentiment_counts,
                'controversy_distribution': controversy_counts,
                'topic_distribution': dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'average_sentiment_score': avg_sentiment,
                'average_controversy_score': avg_controversy,
                'average_quality_score': avg_quality,
                'sentiment_range': {'min': min(sentiment_scores), 'max': max(sentiment_scores)} if sentiment_scores else None,
                'controversy_range': {'min': min(controversy_scores), 'max': max(controversy_scores)} if controversy_scores else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting analytics summary: {e}")
            return {'error': str(e)}


# Context manager for nullcontext (Python 3.7+ compatibility)
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager
    
    @contextmanager
    def nullcontext():
        yield


# Factory function for easy pipeline creation
def create_pipeline(
    preset: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with predefined configurations.
    
    Args:
        preset: Configuration preset name ("ukrainian", "english", "wayback", "performance")
        config_path: Path to configuration file
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured RAGPipeline instance
    """
    from .config import get_config_preset, load_config
    
    if preset:
        config = get_config_preset(preset)
    elif config_path:
        config = load_config(config_path)
    else:
        config = PipelineConfig()
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return RAGPipeline(config=config)