"""
Logging utilities for HTML RAG Pipeline using loguru.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import json
import time


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    format_string: Optional[str] = None,
    enable_console: bool = True,
    enable_json_logs: bool = False
) -> None:
    """
    Setup logging configuration using loguru.
    
    Args:
        level: Logging level
        log_file: Log file path
        rotation: Log rotation size
        retention: Log retention period
        format_string: Custom format string
        enable_console: Enable console logging
        enable_json_logs: Enable JSON formatted logs
    """
    # Remove default logger
    logger.remove()
    
    # Default format
    if format_string is None:
        if enable_json_logs:
            format_string = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        else:
            format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    
    # Console handler
    if enable_console:
        logger.add(
            sys.stderr,
            level=level,
            format=format_string,
            colorize=not enable_json_logs
        )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if enable_json_logs:
            logger.add(
                log_file,
                level=level,
                format=format_string,
                rotation=rotation,
                retention=retention,
                serialize=True
            )
        else:
            logger.add(
                log_file,
                level=level,
                format=format_string,
                rotation=rotation,
                retention=retention
            )


def get_logger(name: str) -> 'logger':
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class PipelineLogger:
    """Enhanced logger for pipeline operations with metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(name=name)
        self._start_times: Dict[str, float] = {}
        self._metrics: Dict[str, Any] = {}
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional context."""
        self.logger.error(message, **kwargs)
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self._start_times[operation] = time.time()
        self.debug(f"Started {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self._start_times:
            self.warning(f"Timer for {operation} was never started")
            return 0.0
        
        duration = time.time() - self._start_times[operation]
        del self._start_times[operation]
        
        self.info(f"Completed {operation} in {duration:.2f} seconds")
        return duration
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self._metrics.update(metrics)
        self.info("Performance metrics", metrics=metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self._metrics.copy()
    
    def log_stage_start(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log the start of a pipeline stage."""
        self.start_timer(stage)
        if details:
            self.info(f"Starting {stage}", stage=stage, **details)
        else:
            self.info(f"Starting {stage}", stage=stage)
    
    def log_stage_end(self, stage: str, results: Optional[Dict[str, Any]] = None) -> float:
        """Log the end of a pipeline stage."""
        duration = self.end_timer(stage)
        if results:
            self.info(f"Completed {stage}", stage=stage, duration=duration, **results)
        else:
            self.info(f"Completed {stage}", stage=stage, duration=duration)
        return duration
    
    def log_processing_stats(self, stats: Dict[str, Any]) -> None:
        """Log processing statistics."""
        self.info("Processing statistics", **stats)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with additional context."""
        self.error(
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context
        )


def create_context_filter(context_key: str, context_value: str):
    """
    Create a filter for logging based on context.
    
    Args:
        context_key: Context key to filter on
        context_value: Context value to match
        
    Returns:
        Filter function
    """
    def filter_func(record):
        return record["extra"].get(context_key) == context_value
    return filter_func


def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        function_logger = PipelineLogger(func.__module__)
        
        # Log function start
        function_logger.debug(
            f"Calling {func.__name__}",
            function=func.__name__,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys())
        )
        
        # Time execution
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log successful completion
            function_logger.debug(
                f"Completed {func.__name__}",
                function=func.__name__,
                duration=duration,
                success=True
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            function_logger.error(
                f"Error in {func.__name__}: {str(e)}",
                function=func.__name__,
                duration=duration,
                error_type=type(e).__name__,
                success=False
            )
            raise
    
    return wrapper


# Module-level logger instance
pipeline_logger = PipelineLogger(__name__)