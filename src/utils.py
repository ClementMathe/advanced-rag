"""
Utility functions for the Advanced RAG system.

This module provides:
- Professional logging configuration
- GPU detection and management
- Performance tracking utilities
- Common helper functions
"""

import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger


class LoggerConfig:
    """
    Centralized logging configuration for the RAG system.

    Uses loguru for structured, colored logging with rotation.
    """

    @staticmethod
    def setup(
        log_dir: str = "logs",
        level: str = "INFO",
        rotation: str = "500 MB",
        retention: str = "10 days",
    ) -> None:
        """
        Configure application-wide logging.

        Args:
            log_dir: Directory to store log files
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            rotation: When to rotate log files
            retention: How long to keep old log files
        """
        # Remove default logger
        logger.remove()

        # Console output with colors
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True,
        )

        # File output with rotation
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path / "rag_{time}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

        logger.info(f"Logger initialized - Level: {level}, Log dir: {log_dir}")


class GPUManager:
    """
    GPU detection and memory management utilities.

    Provides methods to check CUDA availability, select devices,
    and monitor GPU memory usage.
    """

    @staticmethod
    def get_device(prefer_gpu: bool = True) -> torch.device:
        """
        Get the best available device for computation.

        Args:
            prefer_gpu: Whether to prefer GPU over CPU if available

        Returns:
            torch.device: Selected device (cuda or cpu)
        """
        if prefer_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Memory: {GPUManager.get_gpu_memory_info()}")
        else:
            device = torch.device("cpu")
            logger.warning("GPU not available or not preferred. Using CPU.")

        return device

    @staticmethod
    def get_gpu_memory_info() -> str:
        """
        Get current GPU memory usage information.

        Returns:
            str: Formatted string with memory statistics
        """
        if not torch.cuda.is_available():
            return "N/A"

        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return f"{allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total"

    @staticmethod
    def clear_cache() -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")


class Timer:
    """
    Context manager and decorator for timing operations.

    Usage as context manager:
        with Timer("operation_name"):
            # code to time

    Usage as decorator:
        @Timer.timeit
        def my_function():
            # code to time
    """

    def __init__(self, name: str = "Operation", logger_level: str = "INFO"):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
            logger_level: Log level for timing information
        """
        self.name = name
        self.logger_level = logger_level
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = time.perf_counter()
        logger.log(self.logger_level, f"Starting: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration when exiting context."""
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        logger.log(
            self.logger_level, f"Completed: {self.name} - Duration: {duration:.4f}s"
        )

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time

    @staticmethod
    def timeit(func):
        """
        Decorator to time function execution.

        Args:
            func: Function to time

        Returns:
            Wrapped function that logs execution time
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(func.__name__):
                return func(*args, **kwargs)

        return wrapper


class MetricsTracker:
    """
    Track and aggregate metrics across multiple runs.

    Useful for tracking retrieval performance, latency, etc.
    """

    def __init__(self):
        """Initialize empty metrics storage."""
        self.metrics: Dict[str, List[float]] = {}

    def record(self, metric_name: str, value: float) -> None:
        """
        Record a single metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value to record
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with mean, min, max, std statistics
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = self.metrics[metric_name]
        import statistics

        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "count": len(values),
        }

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all tracked metrics.

        Returns:
            Dictionary mapping metric names to their statistics
        """
        return {name: self.get_stats(name) for name in self.metrics.keys()}

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self.metrics.clear()
        logger.info("Metrics tracker reset")


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object pointing to the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration parameters
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration
    """
    import yaml

    ensure_dir(Path(output_path).parent)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Configuration saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    LoggerConfig.setup(level="DEBUG")

    # Test GPU detection
    device = GPUManager.get_device()
    logger.info(f"Selected device: {device}")

    # Test timer
    with Timer("Test operation"):
        time.sleep(1)

    # Test metrics tracker
    tracker = MetricsTracker()
    for i in range(10):
        tracker.record("test_metric", i * 0.1)

    logger.info(f"Metrics summary: {tracker.summary()}")
