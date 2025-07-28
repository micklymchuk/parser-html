"""
Main RAG Pipeline that orchestrates all 6 stages:
0. Wayback Snapshot Processing (optional)
1. HTML Pruning
2. HTML Parsing
3. Topic Analysis
4. Text Embedding
5. ChromaDB Storage

Enhanced with modern Python practices, comprehensive error handling, and metrics.
"""

from typing import List, Dict, Any, Optional, Union
import time
from pathlib import Path

from ..processors.wayback import WaybackProcessor
from ..processors.html_pruner import HTMLPruner
from ..processors.html_parser import HTMLParser
from ..processors.topic_analyzer import TopicAnalyzer
from ..processors.text_embedder import TextEmbedder
from ..processors.vector_store import VectorStore
from ..search.topic_aware_search import SimplifiedTopicSearcher, ContradictionAnalyzer
from ..core.config import PipelineConfig, WaybackConfig, SearchConfig
from ..utils.logging import PipelineLogger
from ..utils.metrics import MetricsCollector, track_stage, track_processing
from ..utils.validators import PipelineValidator, validate_search_query
from ..exceptions.pipeline_exceptions import (
    PipelineError, HTMLProcessingError, EmbeddingError,
    VectorStoreError, ValidationError, handle_pipeline_error
)


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
        enable_metrics: bool = True
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

            # Stage 3: Topic Analyzer
            self.topic_analyzer = TopicAnalyzer()

            # Stage 4: Text Embedder
            self.text_embedder = TextEmbedder(model_name=config.embedding_model)

            # Stage 5: Vector Store
            self.vector_store = VectorStore(
                collection_name=config.collection_name,
                persist_directory=config.persist_directory
            )

            # Topic-Aware Search System
            self.topic_aware_searcher = SimplifiedTopicSearcher(  # ИЗМЕНЕНО
                vector_store=self.vector_store,
                text_embedder=self.text_embedder
            )

            self.contradiction_analyzer = ContradictionAnalyzer()

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
        force_basic_cleaning: bool = False
    ) -> Dict[str, Any]:
        """
        Process raw HTML through the complete pipeline.

        Args:
            raw_html: Raw HTML content
            url: Source URL of the HTML content
            wayback_metadata: Optional Wayback Machine metadata
            force_basic_cleaning: If True, forces basic HTML cleaning instead of AI model

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

            # Stage 3: Topic Analysis
            with track_stage(self.metrics_collector, "stage3_topic_analysis") if self.metrics_collector else nullcontext():
                self.logger.log_stage_start("Stage 3: Topic Analysis")

                try:
                    # Analyze topics for each chunk
                    topic_analyzed_blocks = self.topic_analyzer.analyze_batch(chunked_blocks)
                    topics_found = sum(1 for block in topic_analyzed_blocks
                                     if block.get('topic_analysis', {}).get('topics'))

                    self.logger.log_stage_end("Stage 3: Topic Analysis", {
                        "analyzed_blocks": len(topic_analyzed_blocks),
                        "blocks_with_topics": topics_found
                    })

                    # Use topic-analyzed blocks for next stage
                    chunked_blocks = topic_analyzed_blocks

                except Exception as e:
                    # If topic analysis fails, continue without topics
                    self.logger.warning(f"Topic analysis failed, continuing without topics: {str(e)}")
                    # chunked_blocks remains unchanged

            # Stage 4: Text Embedding
            with track_stage(self.metrics_collector, "stage4_embedding") if self.metrics_collector else nullcontext():
                self.logger.log_stage_start("Stage 4: Text Embedding")

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

                self.logger.log_stage_end("Stage 4: Text Embedding", {
                    "embedded_blocks": len(embedded_blocks),
                    "embedding_dimension": self.text_embedder.embedding_dimension
                })

            # Stage 5: ChromaDB Storage
            with track_stage(self.metrics_collector, "stage5_storage") if self.metrics_collector else nullcontext():
                self.logger.log_stage_start("Stage 5: ChromaDB Storage")

                try:
                    self.vector_store.add_documents(embedded_blocks)
                except Exception as e:
                    raise VectorStoreError(f"Vector storage failed: {str(e)}")

                self.logger.log_stage_end("Stage 5: ChromaDB Storage", {
                    "documents_stored": len(embedded_blocks)
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
                'embedding_dimension': self.text_embedder.embedding_dimension
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
        force_basic_cleaning: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple HTML documents.

        Args:
            html_documents: List of dictionaries with 'html' and 'url' keys
            force_basic_cleaning: If True, forces basic cleaning for all documents

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
                        force_basic_cleaning
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
    def topic_aware_search(
        self,
        query: str,
        n_results: int = 10,
        find_contradictions: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform topic-aware search using Llama for intelligent query analysis.

        This method uses Llama to understand the user's intent and search strategy,
        then applies appropriate filtering and ranking based on topic analysis.

        Args:
            query: Natural language search query in Ukrainian
            n_results: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of relevant documents with enhanced context and relevance scoring

        Examples:
            >>> pipeline.topic_aware_search("знайди суперечності Тимошенко про приватизацію")
            >>> pipeline.topic_aware_search("покажи зміни позиції щодо освіти")
            >>> pipeline.topic_aware_search("що говорив про економіку в 2024 році")

        Raises:
            ValidationError: If query parameters are invalid
            PipelineError: If search operation fails
        """
        try:
            # Validate query
            if not query or not isinstance(query, str):
                raise ValidationError("Query must be a non-empty string")

            if not isinstance(n_results, int) or n_results < 1:
                raise ValidationError("n_results must be a positive integer")

            self.logger.info(f"Starting topic-aware search for: '{query[:100]}{'...' if len(query) > 100 else ''}'")

            # Perform topic-aware search
            results = self.topic_aware_searcher.topic_aware_search(
                query=query,
                n_results=n_results,
                **kwargs
            )

            self.logger.info(f"Topic-aware search completed: found {len(results)} results")

            # НОВОЕ: Добавить анализ противоречий если запрошено
            if find_contradictions and results:
                # Извлечь темы из анализа запроса для фокуса
                query_analysis = self.topic_aware_searcher.analyze_query(query)
                focus_topics = query_analysis.get('topics', [])

                contradiction_analysis = self.contradiction_analyzer.find_contradictions_in_results(
                    results, topics=focus_topics
                )

                # Добавить анализ противоречий к первому результату как метаданные
                if results:
                    results[0]['contradiction_analysis'] = contradiction_analysis

                self.logger.info(f"Contradiction analysis: found {contradiction_analysis.get('contradictions_found', 0)} pairs")

            return results

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Topic-aware search failed: {str(e)}")
            # Fallback to regular search
            try:
                self.logger.info("Falling back to regular semantic search")
                return self.search(query, n_results)
            except Exception as fallback_error:
                raise PipelineError(f"Both topic-aware and fallback search failed: {str(e)}, {str(fallback_error)}") from e

    @handle_pipeline_error
    def process_wayback_snapshots(
        self,
        snapshots_directory: str,
        wayback_config: Optional[WaybackConfig] = None,
        **filter_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process Wayback Machine snapshots from a directory (Stage 0 + Stages 1-5).

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
                snapshots_directory,
                require_metadata=wayback_config.require_metadata
            )

            if not snapshots:
                self.logger.warning("No snapshots found to process")
                return []

            # Apply filters if provided
            combined_filters = {**wayback_config.dict(), **filter_kwargs}
            if any(v for v in combined_filters.values() if v):
                snapshots = self.wayback_processor.filter_snapshots_by_criteria(snapshots, **combined_filters)
                self.logger.info(f"After filtering: {len(snapshots)} snapshots remaining")

            self.logger.log_stage_end("Stage 0: Wayback Snapshot Processing", {
                "snapshots_found": len(snapshots)
            })

            # Process through remaining stages (1-5)
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
                'topic_analyzer': self.topic_analyzer.get_service_info(),
                'simplified_topic_search': self.topic_aware_searcher.get_search_stats(),  # ИЗМЕНЕНО
                'contradiction_analyzer': 'Available',  # НОВОЕ
                'wayback_processor': 'Available',
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