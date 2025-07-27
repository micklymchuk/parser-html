"""
Performance metrics and monitoring utilities for HTML RAG Pipeline.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path


@dataclass
class ProcessingMetrics:
    """Data class for processing metrics."""
    
    # Time metrics
    total_duration: float = 0.0
    stage_durations: Dict[str, float] = field(default_factory=dict)
    
    # Processing metrics
    documents_processed: int = 0
    documents_successful: int = 0
    documents_failed: int = 0
    
    # Content metrics
    total_html_length: int = 0
    total_cleaned_length: int = 0
    total_text_blocks: int = 0
    total_embedded_blocks: int = 0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Error metrics
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.documents_processed == 0:
            return 0.0
        return self.documents_successful / self.documents_processed
    
    def avg_processing_time(self) -> float:
        """Calculate average processing time per document."""
        if self.documents_processed == 0:
            return 0.0
        return self.total_duration / self.documents_processed
    
    def compression_ratio(self) -> float:
        """Calculate HTML compression ratio."""
        if self.total_html_length == 0:
            return 0.0
        return self.total_cleaned_length / self.total_html_length
    
    def text_extraction_ratio(self) -> float:
        """Calculate text extraction efficiency."""
        if self.total_cleaned_length == 0:
            return 0.0
        # Estimate text length from text blocks (assuming avg 100 chars per block)
        estimated_text_length = self.total_text_blocks * 100
        return estimated_text_length / self.total_cleaned_length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "processing": {
                "total_duration": self.total_duration,
                "stage_durations": self.stage_durations,
                "avg_processing_time": self.avg_processing_time(),
            },
            "documents": {
                "processed": self.documents_processed,
                "successful": self.documents_successful,
                "failed": self.documents_failed,
                "success_rate": self.success_rate(),
            },
            "content": {
                "total_html_length": self.total_html_length,
                "total_cleaned_length": self.total_cleaned_length,
                "total_text_blocks": self.total_text_blocks,
                "total_embedded_blocks": self.total_embedded_blocks,
                "compression_ratio": self.compression_ratio(),
                "text_extraction_ratio": self.text_extraction_ratio(),
            },
            "resources": {
                "peak_memory_mb": self.peak_memory_mb,
                "avg_cpu_percent": self.avg_cpu_percent,
            },
            "errors": {
                "count": len(self.errors),
                "details": self.errors,
            }
        }


class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, enable_resource_monitoring: bool = True):
        self.metrics = ProcessingMetrics()
        self.enable_resource_monitoring = enable_resource_monitoring
        self._start_time: Optional[float] = None
        self._stage_start_times: Dict[str, float] = {}
        self._resource_monitor: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._resource_samples: List[Dict[str, float]] = []
    
    def start_collection(self) -> None:
        """Start metrics collection."""
        self._start_time = time.time()
        
        if self.enable_resource_monitoring:
            self._start_resource_monitoring()
    
    def end_collection(self) -> None:
        """End metrics collection."""
        if self._start_time:
            self.metrics.total_duration = time.time() - self._start_time
        
        if self.enable_resource_monitoring:
            self._stop_resource_monitoring()
    
    def start_stage(self, stage_name: str) -> None:
        """Start timing a pipeline stage."""
        self._stage_start_times[stage_name] = time.time()
    
    def end_stage(self, stage_name: str) -> None:
        """End timing a pipeline stage."""
        if stage_name in self._stage_start_times:
            duration = time.time() - self._stage_start_times[stage_name]
            self.metrics.stage_durations[stage_name] = duration
            del self._stage_start_times[stage_name]
    
    def record_document_processed(
        self,
        success: bool,
        html_length: int = 0,
        cleaned_length: int = 0,
        text_blocks: int = 0,
        embedded_blocks: int = 0,
        error: Optional[Exception] = None
    ) -> None:
        """Record document processing metrics."""
        self.metrics.documents_processed += 1
        
        if success:
            self.metrics.documents_successful += 1
            self.metrics.total_html_length += html_length
            self.metrics.total_cleaned_length += cleaned_length
            self.metrics.total_text_blocks += text_blocks
            self.metrics.total_embedded_blocks += embedded_blocks
        else:
            self.metrics.documents_failed += 1
            if error:
                self.metrics.errors.append({
                    "timestamp": time.time(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "html_length": html_length
                })
    
    def _start_resource_monitoring(self) -> None:
        """Start resource monitoring in background thread."""
        self._monitoring_active = True
        self._resource_monitor = threading.Thread(target=self._monitor_resources)
        self._resource_monitor.daemon = True
        self._resource_monitor.start()
    
    def _stop_resource_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring_active = False
        if self._resource_monitor:
            self._resource_monitor.join(timeout=1.0)
        
        # Calculate average metrics
        if self._resource_samples:
            self.metrics.peak_memory_mb = max(s["memory_mb"] for s in self._resource_samples)
            self.metrics.avg_cpu_percent = sum(s["cpu_percent"] for s in self._resource_samples) / len(self._resource_samples)
    
    def _monitor_resources(self) -> None:
        """Monitor system resources."""
        process = psutil.Process()
        
        while self._monitoring_active:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                self._resource_samples.append({
                    "timestamp": time.time(),
                    "memory_mb": memory_mb,
                    "cpu_percent": cpu_percent
                })
                
                time.sleep(1.0)  # Sample every second
            except Exception:
                break  # Exit if monitoring fails
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current metrics."""
        return self.metrics
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        return self.metrics.to_dict()
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.get_metrics_dict(), f, indent=2)
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct metrics from saved data
        self.metrics.total_duration = data["processing"]["total_duration"]
        self.metrics.stage_durations = data["processing"]["stage_durations"]
        self.metrics.documents_processed = data["documents"]["processed"]
        self.metrics.documents_successful = data["documents"]["successful"]
        self.metrics.documents_failed = data["documents"]["failed"]
        self.metrics.total_html_length = data["content"]["total_html_length"]
        self.metrics.total_cleaned_length = data["content"]["total_cleaned_length"]
        self.metrics.total_text_blocks = data["content"]["total_text_blocks"]
        self.metrics.total_embedded_blocks = data["content"]["total_embedded_blocks"]
        self.metrics.peak_memory_mb = data["resources"]["peak_memory_mb"]
        self.metrics.avg_cpu_percent = data["resources"]["avg_cpu_percent"]
        self.metrics.errors = data["errors"]["details"]


@contextmanager
def track_stage(metrics_collector: MetricsCollector, stage_name: str):
    """Context manager for tracking pipeline stages."""
    metrics_collector.start_stage(stage_name)
    try:
        yield
    finally:
        metrics_collector.end_stage(stage_name)


@contextmanager
def track_processing(enable_resource_monitoring: bool = True):
    """Context manager for tracking entire processing session."""
    collector = MetricsCollector(enable_resource_monitoring)
    collector.start_collection()
    try:
        yield collector
    finally:
        collector.end_collection()


def benchmark_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function call.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with benchmark results
    """
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    return {
        "function_name": func.__name__,
        "duration": end_time - start_time,
        "memory_start_mb": start_memory,
        "memory_end_mb": end_memory,
        "memory_delta_mb": end_memory - start_memory,
        "success": success,
        "error": error,
        "result": result
    }


class PerformanceProfiler:
    """Advanced performance profiler for the pipeline."""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
    
    def profile_function(self, function_name: str):
        """Decorator to profile function calls."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                benchmark_result = benchmark_function(func, *args, **kwargs)
                
                if function_name not in self.profiles:
                    self.profiles[function_name] = []
                
                self.profiles[function_name].append(benchmark_result)
                
                if not benchmark_result["success"]:
                    raise Exception(benchmark_result["error"])
                
                return benchmark_result["result"]
            return wrapper
        return decorator
    
    def get_profile_summary(self, function_name: str) -> Dict[str, Any]:
        """Get performance summary for a function."""
        if function_name not in self.profiles:
            return {}
        
        profiles = self.profiles[function_name]
        successful_profiles = [p for p in profiles if p["success"]]
        
        if not successful_profiles:
            return {"error": "No successful executions"}
        
        durations = [p["duration"] for p in successful_profiles]
        memory_deltas = [p["memory_delta_mb"] for p in successful_profiles]
        
        return {
            "function_name": function_name,
            "total_calls": len(profiles),
            "successful_calls": len(successful_profiles),
            "failed_calls": len(profiles) - len(successful_profiles),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "total_duration": sum(durations)
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summaries for all profiled functions."""
        return {
            func_name: self.get_profile_summary(func_name)
            for func_name in self.profiles.keys()
        }


# Global profiler instance
profiler = PerformanceProfiler()