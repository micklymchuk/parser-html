"""
Unit tests for the main RAG Pipeline.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from html_rag.core.pipeline import RAGPipeline, create_pipeline
from html_rag.core.config import PipelineConfig
from html_rag.exceptions.pipeline_exceptions import (
    PipelineError, ValidationError, HTMLProcessingError
)
from tests.conftest import assert_processing_success, assert_search_results_valid


class TestRAGPipelineInitialization:
    """Test RAG Pipeline initialization."""
    
    def test_default_initialization(self):
        """Test pipeline initialization with default config."""
        pipeline = create_pipeline()
        assert pipeline is not None
        assert isinstance(pipeline.config, PipelineConfig)
        pipeline.cleanup()
    
    def test_custom_config_initialization(self):
        """Test pipeline initialization with custom config."""
        config = PipelineConfig(
            collection_name="test_custom",
            max_chunk_size=256,
            prefer_basic_cleaning=True
        )
        pipeline = create_pipeline(config=config)
        
        assert pipeline.config.collection_name == "test_custom"
        assert pipeline.config.max_chunk_size == 256
        assert pipeline.config.prefer_basic_cleaning is True
        pipeline.cleanup()
    
    def test_preset_initialization(self):
        """Test pipeline initialization with presets."""
        pipeline = create_pipeline(preset="ukrainian")
        assert pipeline.config.prefer_basic_cleaning is True
        pipeline.cleanup()
    
    def test_parameter_override(self):
        """Test parameter override during initialization."""
        pipeline = create_pipeline(
            collection_name="override_test",
            max_chunk_size=128
        )
        
        assert pipeline.config.collection_name == "override_test"
        assert pipeline.config.max_chunk_size == 128
        pipeline.cleanup()


class TestHTMLProcessing:
    """Test HTML processing functionality."""
    
    def test_basic_html_processing(self, pipeline, sample_html):
        """Test basic HTML processing."""
        result = pipeline.process_html(sample_html, url="https://test.com")
        assert_processing_success(result)
        assert result['url'] == "https://test.com"
        assert result['text_blocks_count'] > 0
    
    def test_ukrainian_html_processing(self, pipeline, ukrainian_html):
        """Test Ukrainian HTML processing."""
        result = pipeline.process_html(ukrainian_html, url="https://ukrainian-test.com")
        assert_processing_success(result)
        
        # Verify Ukrainian content is preserved
        assert result['text_blocks_count'] > 0
        assert result['embedded_blocks_count'] > 0
    
    def test_force_basic_cleaning(self, pipeline, sample_html):
        """Test forced basic cleaning."""
        result = pipeline.process_html(
            sample_html, 
            url="https://test.com",
            force_basic_cleaning=True
        )
        assert_processing_success(result)
    
    def test_wayback_metadata_processing(self, pipeline, sample_html, wayback_metadata):
        """Test processing with Wayback metadata."""
        result = pipeline.process_html(
            sample_html,
            url="https://test.com",
            wayback_metadata=wayback_metadata
        )
        assert_processing_success(result)
    
    def test_empty_html_processing(self, pipeline, empty_html):
        """Test processing of empty HTML."""
        result = pipeline.process_html(empty_html, url="https://test.com")
        assert result['success'] is False
        assert 'error' in result
    
    def test_malformed_html_processing(self, pipeline, malformed_html):
        """Test processing of malformed HTML."""
        # Should still process successfully due to BeautifulSoup's robustness
        result = pipeline.process_html(malformed_html, url="https://test.com")
        # Malformed HTML might still be processable
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_large_html_processing(self, pipeline, large_html):
        """Test processing of large HTML documents."""
        result = pipeline.process_html(large_html, url="https://test.com")
        assert_processing_success(result)
        assert result['text_blocks_count'] > 10  # Should have many blocks
    
    def test_invalid_input_validation(self, pipeline):
        """Test input validation."""
        with pytest.raises(ValidationError):
            pipeline.process_html("", url="https://test.com")
        
        with pytest.raises(ValidationError):
            pipeline.process_html("<html></html>", url="")


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def test_batch_processing_success(self, pipeline, sample_documents):
        """Test successful batch processing."""
        results = pipeline.process_multiple_html(sample_documents)
        
        assert len(results) == len(sample_documents)
        successful = sum(1 for r in results if r.get('success', False))
        assert successful > 0
    
    def test_batch_processing_with_errors(self, pipeline):
        """Test batch processing with some errors."""
        documents = [
            {'html': '<html><body><h1>Good</h1></body></html>', 'url': 'https://good.com'},
            {'html': '', 'url': 'https://empty.com'},  # This should fail
            {'html': '<html><body><h1>Also Good</h1></body></html>', 'url': 'https://good2.com'}
        ]
        
        results = pipeline.process_multiple_html(documents)
        assert len(results) == 3
        
        # Check that we have both successes and failures
        successes = [r for r in results if r.get('success', False)]
        failures = [r for r in results if not r.get('success', False)]
        
        assert len(successes) >= 1  # At least one should succeed
        assert len(failures) >= 1   # At least one should fail
    
    def test_empty_batch_processing(self, pipeline):
        """Test processing empty batch."""
        with pytest.raises(ValidationError):
            pipeline.process_multiple_html([])


class TestSearchFunctionality:
    """Test search functionality."""
    
    def test_basic_search(self, pipeline, sample_html):
        """Test basic search functionality."""
        # First process a document
        result = pipeline.process_html(sample_html, url="https://test.com")
        assert_processing_success(result)
        
        # Then search for content
        search_results = pipeline.search("test content", n_results=5)
        assert_search_results_valid(search_results)
    
    def test_search_with_threshold(self, pipeline, sample_html):
        """Test search with similarity threshold."""
        # Process document
        pipeline.process_html(sample_html, url="https://test.com")
        
        # Search with threshold
        results = pipeline.search(
            "test content", 
            n_results=5, 
            similarity_threshold=0.5
        )
        
        # All results should meet threshold
        for result in results:
            assert result['similarity_score'] >= 0.5
    
    def test_search_with_metadata_filter(self, pipeline, sample_html, wayback_metadata):
        """Test search with metadata filtering."""
        # Process document with metadata
        pipeline.process_html(
            sample_html, 
            url="https://test.com",
            wayback_metadata=wayback_metadata
        )
        
        # Search with metadata filter
        results = pipeline.search(
            "test",
            n_results=5,
            metadata_filter={'wayback_domain': 'example.com'}
        )
        
        assert_search_results_valid(results)
    
    def test_search_no_results(self, pipeline, sample_html):
        """Test search that returns no results."""
        # Process document
        pipeline.process_html(sample_html, url="https://test.com")
        
        # Search for something that shouldn't exist
        results = pipeline.search("nonexistent quantum flux capacitor")
        assert isinstance(results, list)
        # Results might be empty or have very low scores
    
    def test_search_validation(self, pipeline):
        """Test search input validation."""
        with pytest.raises(ValidationError):
            pipeline.search("", n_results=5)
        
        with pytest.raises(ValidationError):
            pipeline.search("test", n_results=0)
        
        with pytest.raises(ValidationError):
            pipeline.search("test", n_results=101)


class TestCyrillicDetection:
    """Test Cyrillic content detection."""
    
    def test_cyrillic_detection_positive(self, pipeline, ukrainian_html):
        """Test detection of Cyrillic content."""
        # This is a private method, so we test it through processing
        result = pipeline.process_html(ukrainian_html, url="https://test.com")
        assert_processing_success(result)
        
        # The pipeline should have used basic cleaning for Ukrainian content
        # We can verify this by checking that it processed successfully
    
    def test_cyrillic_detection_negative(self, pipeline, sample_html):
        """Test no false positive for Latin content."""
        result = pipeline.process_html(sample_html, url="https://test.com")
        assert_processing_success(result)
    
    def test_mixed_content_detection(self, pipeline):
        """Test detection with mixed Latin/Cyrillic content."""
        mixed_html = """<!DOCTYPE html>
        <html>
        <head><title>Mixed Content</title></head>
        <body>
            <h1>English Title</h1>
            <p>This is English content.</p>
            <h2>Українська частина</h2>
            <p>Це український контент для тестування.</p>
        </body>
        </html>"""
        
        result = pipeline.process_html(mixed_html, url="https://test.com")
        assert_processing_success(result)


class TestWaybackProcessing:
    """Test Wayback Machine processing."""
    
    @pytest.mark.wayback
    def test_wayback_directory_validation(self, pipeline, wayback_directory):
        """Test Wayback directory validation."""
        validation_result = pipeline.validate_wayback_directory(str(wayback_directory))
        assert validation_result['is_valid'] is True
        assert 'errors' in validation_result
    
    @pytest.mark.wayback
    def test_wayback_snapshots_processing(self, pipeline, wayback_directory):
        """Test processing Wayback snapshots."""
        results = pipeline.process_wayback_snapshots(str(wayback_directory))
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that at least some processing was successful
        successful = sum(1 for r in results if r.get('success', False))
        assert successful > 0
    
    @pytest.mark.wayback
    def test_wayback_search(self, pipeline, wayback_directory):
        """Test Wayback-specific search."""
        # First process wayback snapshots
        pipeline.process_wayback_snapshots(str(wayback_directory))
        
        # Then search with wayback filters
        results = pipeline.search_wayback_snapshots(
            query="snapshot",
            n_results=5,
            domain_filter="test.com"
        )
        
        assert_search_results_valid(results)
    
    def test_invalid_wayback_directory(self, pipeline, temp_dir):
        """Test processing invalid Wayback directory."""
        invalid_dir = temp_dir / "nonexistent"
        
        validation_result = pipeline.validate_wayback_directory(str(invalid_dir))
        assert validation_result['is_valid'] is False


class TestPipelineStats:
    """Test pipeline statistics functionality."""
    
    def test_get_pipeline_stats(self, pipeline):
        """Test getting pipeline statistics."""
        stats = pipeline.get_pipeline_stats()
        
        assert isinstance(stats, dict)
        assert 'vector_store' in stats
        assert 'embedding_model' in stats
        assert 'html_pruner_model' in stats
        assert 'config' in stats
    
    def test_stats_after_processing(self, pipeline, sample_html):
        """Test statistics after processing documents."""
        # Process a document
        pipeline.process_html(sample_html, url="https://test.com")
        
        # Get stats
        stats = pipeline.get_pipeline_stats()
        
        assert stats['vector_store']['document_count'] > 0


class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_pipeline_error_handling(self, pipeline):
        """Test pipeline error handling."""
        # Test with invalid HTML that should cause processing errors
        with pytest.raises(ValidationError):
            pipeline.process_html("", url="https://test.com")
    
    def test_cleanup_after_error(self, pipeline):
        """Test cleanup after errors."""
        try:
            pipeline.process_html("", url="https://test.com")
        except ValidationError:
            pass
        
        # Pipeline should still be functional after error
        result = pipeline.process_html(
            "<html><body><h1>Test</h1></body></html>", 
            url="https://test.com"
        )
        assert_processing_success(result)


class TestPipelineConfiguration:
    """Test pipeline configuration options."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = PipelineConfig(max_chunk_size=512)
        assert config.max_chunk_size == 512
        
        # Invalid config should raise validation error
        with pytest.raises(ValueError):
            PipelineConfig(max_chunk_size=10)  # Too small
    
    def test_config_presets(self):
        """Test configuration presets."""
        ukrainian_pipeline = create_pipeline(preset="ukrainian")
        assert ukrainian_pipeline.config.prefer_basic_cleaning is True
        ukrainian_pipeline.cleanup()
        
        english_pipeline = create_pipeline(preset="english")
        assert english_pipeline.config.prefer_basic_cleaning is False
        english_pipeline.cleanup()
    
    def test_config_environment_variables(self):
        """Test configuration from environment variables."""
        import os
        
        # Set environment variable
        os.environ['RAG_MAX_CHUNK_SIZE'] = '256'
        
        try:
            config = PipelineConfig()
            assert config.max_chunk_size == 256
        finally:
            # Cleanup
            del os.environ['RAG_MAX_CHUNK_SIZE']


class TestMemoryManagement:
    """Test memory management and cleanup."""
    
    def test_pipeline_cleanup(self, test_config):
        """Test pipeline cleanup."""
        pipeline = create_pipeline(config=test_config)
        
        # Process some content
        pipeline.process_html(
            "<html><body><h1>Test</h1></body></html>",
            url="https://test.com"
        )
        
        # Cleanup should not raise errors
        pipeline.cleanup()
    
    def test_multiple_pipeline_instances(self, test_config):
        """Test multiple pipeline instances."""
        pipelines = []
        
        try:
            # Create multiple pipelines
            for i in range(3):
                config = PipelineConfig(
                    collection_name=f"test_multi_{i}",
                    persist_directory=f"./test_multi_{i}_db"
                )
                pipeline = create_pipeline(config=config)
                pipelines.append(pipeline)
            
            # All should work independently
            for i, pipeline in enumerate(pipelines):
                result = pipeline.process_html(
                    f"<html><body><h1>Test {i}</h1></body></html>",
                    url=f"https://test{i}.com"
                )
                assert_processing_success(result)
        
        finally:
            # Cleanup all pipelines
            for pipeline in pipelines:
                try:
                    pipeline.cleanup()
                except Exception:
                    pass