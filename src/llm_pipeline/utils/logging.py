# filename: src/llm_pipeline/utils/logging.py
"""
Logging utilities.

This module provides a comprehensive set of functions and a class for managing
logging within the LLM pipeline. It covers basic logger setup, structured
logging of metrics, configuration logging, and persistent storage of metrics
to a file.

Purpose:
    To provide consistent and configurable logging across the entire LLM pipeline.
    Effective logging is crucial for monitoring training progress, debugging issues,
    and understanding the behavior of models and data pipelines.

    Logging is fundamental for any complex software system, especially in machine
    learning where long-running processes are common. These utilities centralize
    logging configuration, ensure important information (metrics, configurations)
    is recorded systematically, and enable easy analysis of historical runs.

LLM Pipeline Fit:
    This module is part of the `llm_pipeline.utils` package. It is widely used
    by various components of the LLM pipeline, including the `Trainer`, data
    processing scripts, and evaluation modules, to report status, metrics,
    and errors. The `setup_logger` function is typically called once at the
    application's entry point to configure global logging.
"""

import logging                          # Python's standard logging library.
import sys                              # Provides access to system-specific parameters and functions (used for stdout).
from pathlib import Path                # For object-oriented filesystem paths, which is robust and platform-independent.
from typing import Any, Optional, Union # Type hinting for better code readability and static analysis.
import json                             # For serializing and deserializing data to/from JSON format.
from datetime import datetime           # For timestamps in metrics logging.


def setup_logger(
    name: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    log_level: Union[str, int] = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.

    Purpose:
        To provide a standardized way to set up logging for different parts
        of the application, ensuring messages are displayed to the console
        and optionally saved to a file, with customizable detail and format.

    Args:
        name: The name of the logger. If `None`, the root logger is configured.
              Using distinct names for different modules (e.g., `__name__`)
              allows for more granular control and filtering.
        log_file: Optional. The path to a file where log messages should also be written.
                  If provided, the parent directory will be created if it doesn't exist.
        log_level: The minimum logging level to capture (e.g., `logging.INFO`,
                   `logging.DEBUG`, `logging.WARNING`). Can be a string ("INFO")
                   or an integer (20 for INFO).
        format_string: Optional. A custom format string for log messages.
                       If `None`, a default format `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
                       is used, which includes timestamp, logger name, level, and message.

    Returns:
        The configured `logging.Logger` instance.
    """
    # Get the logger instance by name.
    logger = logging.getLogger(name)
    logger.setLevel(log_level) # Set the overall logging level for this logger.
    
    # Clear existing handlers to prevent duplicate messages if called multiple times.
    logger.handlers = []
    
    # Define the log message formatter.
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    
    # Setup Console Handler: messages go to standard output.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler) # Add the console handler to the logger.
    
    # Setup File Handler: messages go to a specified file.
    if log_file is not None:
        log_file = Path(log_file)                          # Ensure log_file is a Path object.
        log_file.parent.mkdir(parents=True, exist_ok=True) # Create parent directories if they don't exist.
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler) # Add the file handler to the logger.
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve an existing logger instance by name.

    Purpose:
        A convenience function to get a logger that has already been configured
        (e.g., by `setup_logger` or implicitly by calling `logging.getLogger`).

    Args:
        name: The name of the logger to retrieve. If `None`, returns the root logger.

    Returns:
        The `logging.Logger` instance.
    """
    return logging.getLogger(name)


def log_metrics(
    metrics: dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log a dictionary of metrics in a formatted string.

    Purpose:
        To provide a standardized way to log key performance indicators (metrics)
        during training or evaluation, making them easily readable in the logs.

    Args:
        metrics: A dictionary where keys are metric names (strings) and values
                 are the metric values (e.g., float, int).
        step: Optional. The current step number (e.g., global training step, epoch).
              If provided, it will be included in the log message.
        prefix: Optional. A string prefix to add to each metric name (e.g., "train_", "eval_").
        logger: Optional. The `logging.Logger` instance to use. If `None`, the
                root logger is used.
    """
    if logger is None:
        logger = logging.getLogger() # Use the root logger if none is provided.
    
    parts = []
    if step is not None:
        parts.append(f"Step {step}") # Add step information if available.
    
    # Iterate through metrics, sort by key for consistent logging order.
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            # Format floats to 4 decimal places for readability.
            parts.append(f"{prefix}{key}: {value:.4f}")
        else:
            # Log other types directly.
            parts.append(f"{prefix}{key}: {value}")
    
    # Join all parts with " | " and log as an INFO message.
    logger.info(" | ".join(parts))


def log_config(
    config: Union[dict[str, Any], Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log a configuration object or dictionary.

    Purpose:
        To record the exact configuration used for a particular run, which is vital
        for reproducibility and debugging. It can handle both dictionaries and
        objects with `to_dict` or `__dict__` methods.

    Args:
        config: The configuration to log. Can be a dictionary or an object
                (e.g., a custom config class, `argparse.Namespace`).
        logger: Optional. The `logging.Logger` instance to use. If `None`, the
                root logger is used.
    """
    if logger is None:
        logger = logging.getLogger() # Use the root logger if none is provided.
    
    # Convert configuration object to a dictionary if it's not already one.
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dict__"):
        config_dict = config.__dict__
    else:
        config_dict = config # Assume it's already a dictionary.
    
    # Log the configuration, iterating through its key-value pairs.
    logger.info("Configuration:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}") # Indent for readability.


class MetricsLogger:
    """
    A dedicated logger class for tracking and persisting metrics over time to a file.

    Purpose:
        To provide a structured way to save training/evaluation metrics persistently,
        typically in a line-delimited JSON (JSONL) format, allowing for later
        analysis, plotting, and comparison across runs.

    Attributes:
        log_dir (Path): The directory where the metrics file will be stored.
        log_file (Path): The full path to the metrics file (e.g., `metrics.jsonl`).
    """
    
    def __init__(self, log_dir: Union[str, Path], filename: str = "metrics.jsonl"):
        """
        Initialize the MetricsLogger.

        Args:
            log_dir: The directory where the metrics file should be saved.
                     It will be created if it does not exist.
            filename: The name of the file to save the metrics to. Defaults to "metrics.jsonl".
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True) # Ensure the log directory exists.
        self.log_file = self.log_dir / filename         # Construct the full path to the metrics file.
    
    def log(self, metrics: dict[str, Any], step: int, timestamp: Optional[datetime] = None):
        """
        Log a dictionary of metrics for a given step, appending it to the metrics file.

        Purpose:
            To write a single entry of metrics (e.g., at a specific training step)
            to the persistent log file. Each entry is a JSON object on a new line.

        Args:
            metrics: The dictionary of metrics to log for the current step.
            step: The current step number associated with these metrics.
            timestamp: Optional. The `datetime` object for when these metrics were logged.
                       If `None`, the current time is used.
        """
        if timestamp is None:
            timestamp = datetime.now() # Use current time if no timestamp is provided.
        
        # Create a dictionary entry for the log file, including step, timestamp, and metrics.
        entry = {
            "step": step,
            "timestamp": timestamp.isoformat(), # Convert datetime to ISO format string.
            "metrics": metrics,
        }
        
        # Open the file in append mode and write the JSON entry, followed by a newline.
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def read_metrics(self) -> list[dict[str, Any]]:
        """
        Read all logged metrics from the metrics file.

        Purpose:
            To load previously logged metrics into memory, enabling post-hoc analysis,
            plotting, or resumption of metric tracking.

        Returns:
            A list of dictionaries, where each dictionary represents a logged entry
            (containing "step", "timestamp", and "metrics").
        """
        metrics = []
        
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                for line in f:
                    metrics.append(json.loads(line)) # Parse each line as a JSON object.
        
        return metrics