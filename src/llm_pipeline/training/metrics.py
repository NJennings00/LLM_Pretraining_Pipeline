# filename: src/llm_pipeline/training/metrics.py
"""
Training metrics tracking and computation.

This module provides classes and utility functions for tracking, storing, and
computing various metrics during the training of an LLM. It includes mechanisms
for storing historical metric values, computing smoothed averages, and calculating
common norms (gradient and parameter norms).

Purpose:
    To centralize and standardize the handling of training and evaluation metrics.
    Accurate and insightful metrics are essential for understanding model
    performance, detecting issues like overfitting or vanishing/exploding gradients,
    and making informed decisions about hyperparameter tuning and early stopping.

    Metrics are the primary way to monitor the progress and health of a training run.
    The `TrainingMetrics` and `MetricsTracker` classes provide structured ways to
    collect and aggregate these numbers, making them readily available for
    logging callbacks, experiment tracking systems, and direct analysis.
    The norm computation functions are crucial for diagnosing training stability
    and applying gradient clipping.

LLM Pipeline Fit:
    During the training loop in `src/llm_pipeline/training/trainer.py`,
    `MetricsTracker` instances are used to record loss values, evaluation metrics,
    and other relevant statistics at each step or epoch. Callbacks (e.g.,
    `LoggingCallback`, `WandbCallback`, `TensorBoardCallback`) then query
    these `MetricsTracker` objects to retrieve the latest or smoothed values
    for reporting. The `compute_gradient_norm` and `compute_parameter_norm`
    functions are called at specific points in the training loop (e.g., before
    or after optimizer step) to get insights into the model's training dynamics.
"""

import time                                 # Used for timing operations (e.g., step duration, total training time).
from typing import Optional, Any, Union     # Type hinting for better code readability and error checking.
from collections import defaultdict, deque  # `defaultdict` for easy initialization of dicts with default values, `deque` for efficient sliding window.
import numpy as np                          # Numerical computing library, used for mean, std, etc.
import torch                                # PyTorch library, for tensor operations and model parameters.
import torch.nn as nn                       # Required for type hinting `torch.nn.Module`.
import logging                              # Standard Python logging library.

logger = logging.getLogger(__name__) # Initializes a logger for this module.


class TrainingMetrics:
    """
    Container for training metrics, providing historical storage and basic aggregation.

    Purpose:
        To store all reported metrics (e.g., loss, evaluation scores) along with
        their corresponding global step, allowing for retrieval of specific values,
        last values, or overall averages.

        Provides a foundational structure for accumulating all metrics throughout
        the training process. It's useful for saving a complete history of metrics
        with checkpoints and for calculating overall statistics at the end of training.

    LLM Pipeline Fit:
        While `MetricsTracker` handles per-step/smoothed metrics, `TrainingMetrics`
        could be used in scenarios where a complete, non-windowed history of
        all reported metrics is desired, or for simpler training setups.
        The `Trainer` might consolidate metrics into such a container for
        final reporting or checkpointing.
    """

    def __init__(self):
        self.metrics = defaultdict(list)    # Stores metrics as {metric_name: [(global_step, value), ...]}
        self.current_epoch = 0              # Tracks the current epoch number.
        self.global_step = 0                # Tracks the current global step number.

    def update(self, metrics: dict[str, Union[float, torch.Tensor]], step: Optional[int] = None): # Changed Dict to dict
        """
        Update metrics with new values.

        Args:
            metrics: A dictionary where keys are metric names and values are
                     float or torch.Tensor.
            step: The global step at which these metrics are recorded. If None,
                  uses the internally tracked `self.global_step`.
        """
        if step is not None:        # If a specific step is provided.
            self.global_step = step # Updates the internal global step.

        for key, value in metrics.items():                      # Iterates through the provided metrics.
            if isinstance(value, torch.Tensor):                 # If the value is a PyTorch tensor.
                value = value.item()                            # Converts tensor to a Python scalar.
            self.metrics[key].append((self.global_step, value)) # Appends (step, value) to the list for the given metric.

    def get_last(self, key: str, default: float = 0.0) -> float:
        """
        Get the last reported value for a specific metric.

        Args:
            key: The name of the metric.
            default: The default value to return if the metric is not found.

        Returns:
            The last value of the metric, or the default value.
        """
        if key in self.metrics and self.metrics[key]:   # Checks if the metric exists and has values.
            return self.metrics[key][-1][1]             # Returns the value of the last entry.
        return default                                  # Returns the default if not found.

    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """
        Get the average value for a specific metric.

        Args:
            key: The name of the metric.
            last_n: If specified, computes the average over the last `n` values.
                    Otherwise, computes over all recorded values.

        Returns:
            The average value of the metric. Returns 0.0 if no values are recorded.
        """
        if key not in self.metrics or not self.metrics[key]: # Checks if the metric exists and has values.
            return 0.0 # Returns 0.0 if no values.

        values = [v for _, v in self.metrics[key]]  # Extracts only the values from the (step, value) pairs.
        if last_n is not None:                      # If `last_n` is specified.
            values = values[-last_n:]               # Takes only the last `n` values.

        return float(np.mean(values)) # Ensure float return # Computes and returns the mean as a float.

    def get_metrics(self) -> dict[str, Any]: 
        """
        Get all stored metrics with their full history, last value, mean, and standard deviation.

        Returns:
            A dictionary where each key is a metric name, and its value is another
            dictionary containing "values" (list of (step, value) tuples),
            "last", "mean", and "std".
        """
        result = {}                                                 # Initializes an empty dictionary for results.
        for key, values in self.metrics.items():                    # Iterates through all stored metrics.
            if values:                                              # If there are values for the metric.
                result[key] = {                                     # Adds a dictionary of statistics for the metric.
                    "values": values,                               # The full list of (step, value) pairs.
                    "last": values[-1][1],                          # The last reported value.
                    "mean": float(np.mean([v for _, v in values])), # Ensure float # Mean of all values.
                    "std": float(np.std([v for _, v in values])),   # Ensure float # Standard deviation of all values.
                }
        return result # Returns the aggregated metrics.

    def to_dict(self) -> dict[str, Any]: 
        """
        Convert the entire `TrainingMetrics` state to a dictionary for serialization.

        Returns:
            A dictionary containing the `metrics` (converted from defaultdict to dict),
            `current_epoch`, and `global_step`.
        """
        return {
            "metrics": dict(self.metrics),          # Converts defaultdict to a regular dict.
            "current_epoch": self.current_epoch,    # Stores current epoch.
            "global_step": self.global_step,        # Stores current global step.
        }

    def from_dict(self, data: dict[str, Any]): 
        """
        Load the state of `TrainingMetrics` from a dictionary (e.g., from a checkpoint).

        Args:
            data: A dictionary containing the metrics state, typically loaded from `to_dict`.
        """
        self.metrics = defaultdict(list, data["metrics"])   # Reconstructs defaultdict from the loaded dict.
        self.current_epoch = data["current_epoch"]          # Loads current epoch.
        self.global_step = data["global_step"]              # Loads current global step.


class MetricsTracker:
    """
    Track and compute various training metrics, including smoothed averages and timing.

    Purpose:
        Provides a dynamic way to track metrics during training,
        especially useful for calculating moving averages (smoothed metrics)
        and timing statistics. It's designed to be updated frequently (e.g., per step).

    Why Tested/Used:
        This class is critical for real-time monitoring of training. Smoothed
        metrics provide a clearer trend by reducing noise from individual steps.
        Timing metrics help assess the throughput of the training process.
        It's often used by logging callbacks.

    LLM Pipeline Fit:
        The `Trainer` class will likely instantiate `MetricsTracker` to
        collect and provide metrics to callbacks for logging and reporting.
        It's designed for active tracking during the training loop.

    Inputs to `__init__`:
        - `window_size` (int): The number of recent steps to consider for
          calculating smoothed (moving average) metrics.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of sliding window for smoothed metrics
        """
        self.window_size = window_size                                  # Size of the sliding window for smoothed metrics.
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))   # `metrics` stores `deque` for smoothed averages (last `window_size` values).
        self.all_metrics = defaultdict(list)                            # `all_metrics` stores all historical (step, value) pairs without windowing.
        self.step = 0                                                   # Current global step.
        self.start_time = time.time()                                   # Start time of the tracker (or training run).
        self.step_times = deque(maxlen=window_size)                     # Stores time taken for last `window_size` steps.
        self.last_step_time = time.time()                               # Time of the last update.

    def update(self, metrics: dict[str, Union[float, torch.Tensor]], prefix: str = ""): 
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metrics (e.g., {"loss": 0.5}).
            prefix: Optional prefix for metric names (e.g., "train", "eval").
        """
        current_time = time.time()                                  # Records current time.
        self.step_times.append(current_time - self.last_step_time)  # Calculates and appends time taken since last step.
        self.last_step_time = current_time                          # Updates last step time.

        for key, value in metrics.items():      # Iterates through provided metrics.
            if isinstance(value, torch.Tensor): # If value is a tensor.
                value = value.item()            # Converts to scalar.

            full_key = f"{prefix}/{key}" if prefix else key         # Creates a full key with prefix if provided.
            self.metrics[full_key].append(value)                    # Appends value to the deque for smoothed metrics.
            self.all_metrics[full_key].append((self.step, value))   # Appends (step, value) to the full history.

        self.step += 1 # Increments the global step counter.

    def get_smoothed_metrics(self) -> dict[str, float]: # Changed Dict to dict
        """
        Get smoothed metrics using the sliding window average.

        Returns:
            A dictionary where keys are metric names (with "_smoothed" suffix)
            and values are their current smoothed averages.
        """
        result = {}                                                 # Initializes an empty dictionary.
        for key, values in self.metrics.items():                    # Iterates through metrics stored in deques.
            if values:                                              # If deque is not empty.
                result[f"{key}_smoothed"] = float(np.mean(values))  # Ensure float # Computes mean of values in deque.
        return result                                               # Returns smoothed metrics.

    def get_current_metrics(self) -> dict[str, float]: # Changed Dict to dict
        """
        Get current (latest) metrics.

        Returns:
            A dictionary where keys are metric names and values are their latest reported values.
        """
        result = {}                                 # Initializes an empty dictionary.
        for key, values in self.metrics.items():    # Iterates through metrics stored in deques.
            if values:                              # If deque is not empty.
                result[key] = values[-1]            # Gets the last (most recent) value.
        return result                               # Returns current metrics.

    def get_metrics(self) -> dict[str, Any]: 
        """
        Get all available metrics, including current, smoothed, timing, and overall statistics.

        Returns:
            A comprehensive dictionary of metrics.
        """
        result = {} # Initializes an empty dictionary.

        # Add current and smoothed metrics
        result.update(self.get_current_metrics())  # Adds current metrics.
        result.update(self.get_smoothed_metrics()) # Adds smoothed metrics.

        # Add timing metrics
        if self.step_times: # If there are recorded step times.
            result["steps_per_second"] = 1.0 / np.mean(self.step_times) # Calculates steps per second.
            result["avg_step_time"] = float(np.mean(self.step_times))   # Ensure float # Calculates average time per step.

        result["total_time"] = time.time() - self.start_time # Calculates total time elapsed.
        result["total_steps"] = self.step # Reports total steps processed.

        # Add statistics for all metrics
        for key, values in self.all_metrics.items(): # Iterates through all historical metrics.
            if values: # If there are values.
                metric_values = [v for _, v in values]              # Extracts values only.
                result[f"{key}_min"] = float(np.min(metric_values)) # Ensure float # Minimum value.
                result[f"{key}_max"] = float(np.max(metric_values)) # Ensure float # Maximum value.
                result[f"{key}_std"] = float(np.std(metric_values)) # Ensure float # Standard deviation.

        return result # Returns the comprehensive metrics dictionary.

    def reset(self):
        """Reset all tracked metrics to their initial state."""
        self.metrics.clear()                # Clears smoothed metrics.
        self.all_metrics.clear()            # Clears all historical metrics.
        self.step = 0                       # Resets step count.
        self.start_time = time.time()       # Resets start time.
        self.step_times.clear()             # Clears step times.
        self.last_step_time = time.time()   # Resets last step time.

    def state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for checkpointing.

        Returns:
            A dictionary representing the current state of the `MetricsTracker`,
            suitable for saving and loading.
        """
        # Convert deque to list for serialization
        metrics_for_dict = {k: list(v) for k, v in self.metrics.items()} # Converts deques to lists for serialization.
        return {
            "metrics": metrics_for_dict,           # Smoothed metrics.
            "all_metrics": dict(self.all_metrics), # All historical metrics.
            "step": self.step,                     # Current step.
            "start_time": self.start_time,         # Start time.
            "window_size": self.window_size,       # Window size for reconstruction.
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """
        Load state dictionary from checkpoint.

        Args:
            state_dict: A dictionary containing the saved state of the `MetricsTracker`.
        """

        # Handle backward compatibility for window_size
        self.window_size = state_dict.get("window_size", 100)  # Default to 100 if missing # Loads window size, with a default for compatibility.

        self.metrics = defaultdict( # Reconstructs defaultdict for smoothed metrics.
            lambda: deque(maxlen=self.window_size),
            {k: deque(v, maxlen=self.window_size) for k, v in state_dict["metrics"].items()},
        )
        self.all_metrics = defaultdict(list, state_dict["all_metrics"]) # Reconstructs defaultdict for all metrics.
        self.step = state_dict["step"]             # Loads step.
        self.start_time = state_dict["start_time"] # Loads start time.


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the total L2 norm of gradients for all parameters in a model.

    Purpose:
        To assess the magnitude of gradients during training. Very large gradients
        can indicate instability (exploding gradients), while very small gradients
        can indicate a lack of learning (vanishing gradients).

        This is a common diagnostic metric in deep learning. It's often used
        before or after the optimizer step, especially when gradient clipping
        is employed, to monitor its effectiveness.

    LLM Pipeline Fit:
        Typically called by a `Callback` (e.g., `LoggingCallback`, `WandbCallback`)
        during the `on_log` hook to report gradient norms alongside other metrics.

    Args:
        model: The `torch.nn.Module` for which to compute the gradient norm.

    Returns:
        The total L2 norm of the model's gradients.
    """
    total_norm = 0.0                             # Initializes total norm.
    for p in model.parameters():                 # Iterates through all parameters in the model.
        if p.grad is not None:                   # If the parameter has a gradient.
            param_norm = p.grad.data.norm(2)     # Computes the L2 norm of the parameter's gradient.
            total_norm += param_norm.item() ** 2 # Squares the norm and adds to total_norm.
    return total_norm ** 0.5                     # Returns the square root of the sum of squared norms.


def compute_parameter_norm(model: torch.nn.Module) -> float:
    """
    Compute the total L2 norm of model parameters.

    Purpose:
        To monitor the magnitude of model weights. Large parameter norms can sometimes
        indicate overfitting or unstable training, especially without regularization.

        Another diagnostic metric that provides insight into the scale of model weights.
        It can be useful in conjunction with gradient norms to understand training dynamics.

    LLM Pipeline Fit:
        Similar to `compute_gradient_norm`, this function can be called by a
        `Callback` to report the parameter norm during training logs.

    Args:
        model: The `torch.nn.Module` for which to compute the parameter norm.

    Returns:
        The total L2 norm of the model's parameters.
    """
    total_norm = 0.0                         # Initializes total norm.
    for p in model.parameters():             # Iterates through all parameters in the model.
        param_norm = p.data.norm(2)          # Computes the L2 norm of the parameter's data (weights).
        total_norm += param_norm.item() ** 2 # Squares the norm and adds to total_norm.
    return total_norm ** 0.5                 # Returns the square root of the sum of squared norms.