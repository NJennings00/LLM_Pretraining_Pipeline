# filename: src/llm_pipeline/utils/profiling.py
"""
Profiling utilities for performance analysis.

This module provides tools for analyzing the performance characteristics of
PyTorch models and the underlying system resources. It includes functionalities
for timing code blocks, profiling model architecture and parameter counts,
measuring forward pass speed, and monitoring CPU/GPU memory usage.

Purpose:
    To help developers and researchers understand and optimize the performance
    of large language models during training and inference. Profiling is critical
    for identifying bottlenecks, managing memory, and ensuring efficient resource utilization.

    Performance is a key concern for LLMs due to their scale. These utilities are
    essential for:
    1. **Optimization:** Pinpointing slow operations or memory hogs.
    2. **Resource Planning:** Estimating hardware requirements.
    3. **Debugging:** Understanding unexpected performance drops.
    They provide quantitative data to inform design decisions.

LLM Pipeline Fit:
    This module is part of the `llm_pipeline.utils` package. Its functions can
    be integrated into training scripts, evaluation routines, or used in standalone
    analysis scripts to gather performance insights. The `PerformanceMonitor` class
    is particularly useful for real-time monitoring within the training loop.
"""

import time                                            # For time-related operations (e.g., `time.time()`).
import logging                                         # For logging profiling results.
from typing import Dict, Any, Optional, Callable, List # Type hinting for better code readability.
from contextlib import contextmanager                  # For creating context managers (`with` statements).
from functools import wraps                            # For preserving metadata of wrapped functions.
import torch                                           # The primary PyTorch library.
import torch.nn as nn                                  # Neural network module from PyTorch.
import psutil                                          # A cross-platform library for retrieving information 
                                                       # on running processes and system utilization (CPU, memory).
try:
    import GPUtil # For querying GPU utilization and memory information.
except ImportError:
    GPUtil = None # Handle case where GPUtil might not be installed.


logger = logging.getLogger(__name__) # Initializes a logger specifically for this module.


@contextmanager
def profile_time(name: str = "Operation"):
    """
    A context manager for timing the execution duration of a code block.

    Purpose:
        To easily measure how long specific sections of code take to execute,
        providing quick performance insights.

    Usage:
        ```python
        with profile_time("Data Loading"):
            # ... data loading code ...
        ```

    Args:
        name: A descriptive string for the operation being timed, used in the log message.
    """
    start_time = time.time()                               # Record the start time.
    yield                                                  # Execute the code block within the `with` statement.
    elapsed_time = time.time() - start_time                # Calculate elapsed time.
    logger.info(f"{name} took {elapsed_time:.4f} seconds") # Log the result.


def profile_model(model: nn.Module) -> Dict[str, Any]:
    """
    Profile a PyTorch model's architecture, parameter counts, and estimated memory footprint.

    Purpose:
        To provide a high-level overview of a model's complexity and resource requirements
        before actual execution, aiding in model design and resource planning.

    Args:
        model: The PyTorch `nn.Module` to be profiled.

    Returns:
        A dictionary containing:
        - `total_params`: Total number of parameters in the model.
        - `trainable_params`: Number of parameters that will be updated during training.
        - `layers`: A dictionary detailing parameters for each leaf module (layer).
        - `memory_mb`: Estimated memory usage in MB, assuming `float32` parameters.
    """
    profile = {
        "total_params": sum(p.numel() for p in model.parameters()),                        # Sum of elements in all parameters.
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), # Sum of elements in trainable parameters.
        "layers": {},                                                                      # Placeholder for detailed layer information.
    }
    
    # Iterate through all named modules (layers) in the model.
    for name, module in model.named_modules():
        # Check if it's a "leaf" module (i.e., it doesn't contain other `nn.Module`s).
        # This prevents double-counting parameters of sub-modules.
        if len(list(module.children())) == 0:
            layer_params = sum(p.numel() for p in module.parameters()) # Parameters specific to this leaf module.
            if layer_params > 0: # Only include layers with parameters.
                profile["layers"][name] = {
                    "type": type(module).__name__, # Class name of the module (e.g., 'Linear', 'Conv2d').
                    "params": layer_params,
                    "trainable": all(p.requires_grad for p in module.parameters()) if layer_params > 0 else False, # Check if all params in layer are trainable.
                }
    
    # Estimate memory footprint: assuming each float32 parameter takes 4 bytes.
    profile["memory_mb"] = profile["total_params"] * 4 / (1024 * 1024)
    
    return profile


def profile_forward_pass(
    model: nn.Module,
    input_shape: tuple,
    batch_size: int = 1,
    num_runs: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Profile the performance of a model's forward pass.

    Purpose:
        To measure the average time and throughput for inference, providing
        insights into the model's speed on a given device.

    Args:
        model: The PyTorch model to profile.
        input_shape: A tuple representing the shape of a single input sample
                     (excluding the batch dimension, e.g., `(3, 224, 224)` for an image).
        batch_size: The batch size to use for the dummy input.
        num_runs: The number of forward passes to perform for averaging the time.
                  A higher number reduces noise.
        device: The `torch.device` to run the profiling on (e.g., `torch.device("cuda")`).
                If `None`, it defaults to the device of the model's first parameter.

    Returns:
        A dictionary with performance metrics:
        - `avg_forward_time_ms`: Average time per forward pass in milliseconds.
        - `throughput_samples_per_sec`: Number of samples processed per second.
        - `total_time_seconds`: Total time taken for all profiling runs.
    """
    if device is None:
        # Infer device from model's parameters if not specified.
        device = next(model.parameters()).device
    
    model.eval() # Set model to evaluation mode (disables dropout, BatchNorm updates, etc.).
    
    # Create a dummy input tensor with random data, moved to the specified device.
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Warmup runs: Perform a few initial runs to allow GPU/CPU to allocate resources
    # and settle into a stable performance state.
    for _ in range(10):
        with torch.no_grad(): # Disable gradient calculation for inference.
            _ = model(dummy_input)
    
    # Synchronize CUDA operations before starting the timer to ensure all previous
    # GPU computations are complete.
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    # Main timing loop.
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Synchronize CUDA again after the timing loop.
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.time() - start_time
    
    return {
        "avg_forward_time_ms": (total_time / num_runs) * 1000,              # Convert to milliseconds.
        "throughput_samples_per_sec": (batch_size * num_runs) / total_time, # Samples processed per second.
        "total_time_seconds": total_time,
    }


def profile_memory_usage() -> Dict[str, float]:
    """
    Profile current system (CPU) and GPU memory usage.

    Purpose:
        To provide a snapshot of memory consumption, helping to monitor
        resource utilization during long-running processes like training.

    Returns:
        A dictionary with various memory metrics (in GB for CPU, MB for GPU).
        Includes total, used, available percentages for CPU, and allocated/reserved
        for each detected GPU. Also attempts to get GPU utilization if `GPUtil` is installed.
    """
    memory_info = {}
    
    # CPU memory usage using `psutil`.
    cpu_memory = psutil.virtual_memory()
    memory_info["cpu_memory_used_gb"] = cpu_memory.used / (1024**3) # Convert bytes to GB.
    memory_info["cpu_memory_available_gb"] = cpu_memory.available / (1024**3)
    memory_info["cpu_memory_percent"] = cpu_memory.percent
    
    # GPU memory usage using `torch.cuda` and optionally `GPUtil`.
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info[f"gpu_{i}_memory_allocated_mb"] = torch.cuda.memory_allocated(i) / (1024**2) # Convert bytes to MB.
            memory_info[f"gpu_{i}_memory_reserved_mb"] = torch.cuda.memory_reserved(i) / (1024**2)
            
            # Attempt to get GPU utilization and total memory using GPUtil.
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        memory_info[f"gpu_{i}_utilization_percent"] = gpus[i].load * 100
                        memory_info[f"gpu_{i}_memory_total_mb"] = gpus[i].memoryTotal
                except Exception: # Catch any exceptions if GPUtil fails to get info.
                    pass
    
    return memory_info


def get_gpu_memory_info(device: Optional[int] = None) -> Dict[str, float]:
    """
    Get detailed PyTorch-specific GPU memory information for a specified device.

    Purpose:
        To provide granular details about PyTorch's allocation and reservation
        of GPU memory, useful for debugging out-of-memory errors.

    Args:
        device: The index of the GPU device to query. If `None`, the currently
                selected CUDA device is used.

    Returns:
        A dictionary with GPU memory information (in MB):
        - `allocated_mb`: Memory currently allocated by PyTorch tensors.
        - `reserved_mb`: Total memory reserved by PyTorch's caching allocator.
        - `free_mb`: Free memory within the reserved pool (`reserved - allocated`).
        - `max_allocated_mb`: Peak memory ever allocated by PyTorch.
        - `max_reserved_mb`: Peak memory ever reserved by PyTorch.
        Returns an empty dictionary if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return {}
    
    if device is None:
        device = torch.cuda.current_device() # Get the current default CUDA device.
    
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "free_mb": (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


class PerformanceMonitor:
    """
    A class to monitor and log performance metrics (step time, memory usage)
    during training or other iterative processes.

    Purpose:
        To track real-time performance indicators and periodically log aggregated
        statistics, giving continuous feedback on resource usage and throughput.

    Attributes:
        log_interval (int): How often (in steps) to log accumulated metrics.
        metrics (Dict): Stores lists of raw metric values for averaging (e.g., 'step_times', 'memory_usage').
        step_count (int): Total number of `step()` calls since initialization.
        last_log_step (int): The `step_count` at which metrics were last logged.
        last_step_time (float): `time.time()` value at the beginning of the last step.
    """
    
    def __init__(self, log_interval: int = 100):
        """
        Initialize the PerformanceMonitor.

        Args:
            log_interval: The number of steps after which to log the averaged performance metrics.
        """
        self.log_interval = log_interval
        self.metrics = {
            "step_times": [],
            "memory_usage": [],
            "gpu_utilization": [], # Although not currently used in `step()`, good to keep for future expansion.
        }
        self.step_count = 0
        self.last_log_step = 0
        self.last_step_time = time.time() # Initialize timer for the first step.
    
    def step(self):
        """
        Record performance metrics for the current step.
        This method should be called at the beginning or end of each training/processing step.
        """
        current_time = time.time()
        step_time = current_time - self.last_step_time # Calculate time taken for the previous step.
        self.last_step_time = current_time             # Update for the next step.
        
        self.metrics["step_times"].append(step_time) # Store individual step time.
        
        # Record GPU memory allocated by PyTorch if CUDA is available.
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024**2)
            self.metrics["memory_usage"].append(memory_mb)
        
        self.step_count += 1 # Increment total step count.
        
        # Check if it's time to log based on the `log_interval`.
        if self.step_count - self.last_log_step >= self.log_interval:
            self.log()                           # Call the logging method.
            self.last_log_step = self.step_count # Update last logged step.
    
    def log(self):
        """
        Log the aggregated performance metrics over the last `log_interval` steps.
        """
        if not self.metrics["step_times"]:
            return # Do nothing if no steps have been recorded yet.

        # Calculate average step time from the most recent `log_interval` steps.
        avg_step_time = sum(self.metrics["step_times"][-self.log_interval:]) / min(self.log_interval, len(self.metrics["step_times"])) # Handle cases where fewer than `log_interval` steps have passed.
        
        log_msg = f"Performance - Step {self.step_count}: "
        log_msg += f"avg_step_time={avg_step_time:.4f}s"
        
        if self.metrics["memory_usage"]:
            # Calculate average GPU memory usage from the most recent `log_interval` steps.
            avg_memory = sum(self.metrics["memory_usage"][-self.log_interval:]) / min(self.log_interval, len(self.metrics["memory_usage"]))
            log_msg += f", avg_gpu_memory={avg_memory:.1f}MB"
        
        logger.info(log_msg) # Log the formatted performance message.
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get a summary of all recorded performance metrics.

        Purpose:
            To provide overall performance statistics after a training run or analysis.

        Returns:
            A dictionary summarizing:
            - `avg_step_time`: Average time per step over all recorded steps.
            - `total_time`: Total cumulative time of all recorded steps.
            - `avg_memory_mb`: Average GPU memory usage over all recorded steps.
            - `peak_memory_mb`: Maximum GPU memory usage observed.
        """
        summary = {}
        
        if self.metrics["step_times"]:
            summary["avg_step_time"] = sum(self.metrics["step_times"]) / len(self.metrics["step_times"])
            summary["total_time"] = sum(self.metrics["step_times"])
        
        if self.metrics["memory_usage"]:
            summary["avg_memory_mb"] = sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"])
            summary["peak_memory_mb"] = max(self.metrics["memory_usage"])
        
        return summary