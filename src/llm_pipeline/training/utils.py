# filename: src/llm_pipeline/training/utils.py
"""
Training utilities and helper functions.

This module provides a collection of utility functions essential for managing
the training lifecycle of large language models. This includes functionalities
for saving and loading model checkpoints, managing checkpoint retention,
estimating training progress, and querying model size.

Purpose:
    To encapsulate common, reusable operations related to training management,
    reducing redundancy and making the main `Trainer` class cleaner and more focused
    on the training loop logic itself.

    These utilities are crucial for the robustness and practicality of the
    training pipeline. Checkpointing ensures training progress is not lost,
    and allows for resuming interrupted training or continuing fine-tuning.
    Monitoring and cleanup functions help in managing resources efficiently.

LLM Pipeline Fit:
    This module is part of the `llm_pipeline.training` package. Its functions
    are directly called by the `Trainer` class and potentially by other scripts
    that need to interact with saved models or training states (e.g., for inference,
    evaluation, or resuming training from a specific point).
"""

import os                                      # For interacting with the operating system, though `pathlib` is preferred for paths.
import logging                                 # For logging information, warnings, and errors.
from pathlib import Path                       # For object-oriented filesystem paths, which is robust and platform-independent.
from typing import Dict, Optional, Any, Union  # Type hinting for better code readability and static analysis.
import torch                                   # The primary PyTorch library for deep learning operations.
import torch.nn as nn                          # Neural network module from PyTorch.
import json                                    # For serializing and deserializing data to/from JSON format (e.g., training state, model config).
import shutil                                  # For high-level file operations, such as deleting directories.


logger = logging.getLogger(__name__) # Initializes a logger specifically for this module.


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    checkpoint_dir: Union[str, Path],
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save the current state of the training to a checkpoint directory.

    Purpose:
        To persistently store the critical components of the training session
        (model weights, optimizer state, LR scheduler state, and training progress)
        at a given point, enabling resumption of training from this state.

    Args:
        model: The PyTorch model (`nn.Module`) whose state dictionary will be saved.
        optimizer: The PyTorch optimizer (`torch.optim.Optimizer`) whose state
                   dictionary will be saved.
        lr_scheduler: The PyTorch learning rate scheduler (`_LRScheduler`) whose
                      state dictionary will be saved.
        epoch: The current training epoch number.
        global_step: The current global optimization step number. This is often
                     the primary indicator of training progress.
        checkpoint_dir: The directory where the checkpoint files will be saved.
                        It will be created if it does not exist.
        metrics: Optional. A dictionary of metrics (`Dict[str, Any]`) associated
                 with the current training state (e.g., validation loss, accuracy).
                 This allows metrics to be restored upon resuming training.
    """
    checkpoint_dir = Path(checkpoint_dir)             # Ensure checkpoint_dir is a Path object.
    checkpoint_dir.mkdir(parents=True, exist_ok=True) # Create the directory and any necessary parent directories.
    
    # Save model's state dictionary.
    model_path = checkpoint_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    
    # Save optimizer's state dictionary.
    optimizer_path = checkpoint_dir / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save learning rate scheduler's state dictionary.
    scheduler_path = checkpoint_dir / "scheduler.pt"
    torch.save(lr_scheduler.state_dict(), scheduler_path)
    
    # Save core training state (epoch, global step, and metrics) as a JSON file.
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics or {}, # Store provided metrics, or an empty dict if none.
    }
    
    state_path = checkpoint_dir / "training_state.json"
    with open(state_path, "w") as f:
        json.dump(training_state, f, indent=2) # Write JSON with pretty-printing.
    
    # If the model has a `config` attribute (common in custom models), save it.
    if hasattr(model, "config"):
        config_path = checkpoint_dir / "config.json"
        config_dict = model.config.__dict__ # Convert config object to dictionary.
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}") # Log the successful save.


def load_checkpoint(
    checkpoint_dir: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Load the state of a training checkpoint into the provided objects.

    Purpose:
        To restore a previous training state by loading saved model weights,
        optimizer, and scheduler states. This is crucial for resuming interrupted
        training or continuing training from a specific point.

    Args:
        checkpoint_dir: The directory from which to load the checkpoint files.
        model: Optional. The PyTorch model (`nn.Module`) into which the saved
               model state dictionary will be loaded.
        optimizer: Optional. The PyTorch optimizer (`torch.optim.Optimizer`)
                   into which the saved optimizer state dictionary will be loaded.
        lr_scheduler: Optional. The PyTorch learning rate scheduler (`_LRScheduler`)
                      into which the saved scheduler state dictionary will be loaded.
        map_location: Optional. A string or `torch.device` specifying how to remap
                      storage locations (e.g., 'cpu' to load a GPU-trained model
                      onto CPU).

    Returns:
        A dictionary containing the loaded training state (epoch, global_step, metrics).

    Raises:
        FileNotFoundError: If the specified `checkpoint_dir` does not exist.
    """
    checkpoint_dir = Path(checkpoint_dir) # Ensure checkpoint_dir is a Path object.
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Load model state if a model object is provided.
    if model is not None:
        model_path = checkpoint_dir / "pytorch_model.bin"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=map_location)
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {model_path}")
    
    # Load optimizer state if an optimizer object is provided.
    if optimizer is not None:
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            state_dict = torch.load(optimizer_path, map_location=map_location)
            optimizer.load_state_dict(state_dict)
            logger.info(f"Optimizer loaded from {optimizer_path}")
    
    # Load scheduler state if a scheduler object is provided.
    if lr_scheduler is not None:
        scheduler_path = checkpoint_dir / "scheduler.pt"
        if scheduler_path.exists():
            state_dict = torch.load(scheduler_path, map_location=map_location)
            lr_scheduler.load_state_dict(state_dict)
            logger.info(f"Scheduler loaded from {scheduler_path}")
    
    # Load the core training state (epoch, global_step, metrics).
    state_path = checkpoint_dir / "training_state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            training_state = json.load(f)
    else:
        training_state = {} # Return empty dict if state file not found.
    
    return training_state # Return the loaded training state.


def get_last_checkpoint(output_dir: Union[str, Path]) -> Optional[Path]:
    """
    Identify and return the path to the most recent checkpoint in a given directory.

    Purpose:
        Automates the process of finding the latest checkpoint, which is useful
        for resuming training without explicitly specifying the checkpoint path.

    Args:
        output_dir: The base directory where checkpoints are saved (e.g., parent
                    directory containing "checkpoint-1000", "checkpoint-2000").

    Returns:
        A `Path` object pointing to the directory of the latest checkpoint,
        or `None` if no checkpoints are found.
    """
    output_dir = Path(output_dir) # Ensure output_dir is a Path object.
    
    if not output_dir.exists():
        return None # Return None if the output directory doesn't exist.
    
    checkpoints = []
    # Iterate through items in the output directory to find checkpoint folders.
    for path in output_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                # Extract the step number from the checkpoint folder name.
                step = int(path.name.split("-")[1])
                checkpoints.append((step, path)) # Store as (step, path) tuple.
            except (IndexError, ValueError):
                # Skip directories that don't follow the 'checkpoint-STEP' naming convention.
                continue
    
    if not checkpoints:
        return None # Return None if no checkpoints were found.
    
    # Sort checkpoints by step number in ascending order.
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1] # Return the path of the checkpoint with the highest step number.


def should_save_checkpoint(
    global_step: int,
    save_strategy: str,
    save_steps: int,
    epoch_end: bool = False,
) -> bool:
    """
    Determine whether a checkpoint should be saved based on the configured strategy.

    Purpose:
        Provides a logical check to the `Trainer` to decide when to trigger
        a checkpoint save operation, decoupling this logic from the main loop.

    Args:
        global_step: The current global optimization step.
        save_strategy: The strategy for saving checkpoints ("steps", "epoch", or "no").
                       - "steps": Save every `save_steps`.
                       - "epoch": Save at the end of each epoch.
                       - "no": Never save.
        save_steps: The interval (in steps) at which to save checkpoints if
                    `save_strategy` is "steps".
        epoch_end: A boolean flag indicating if the current call is at the
                   very end of an epoch. Relevant for the "epoch" strategy.

    Returns:
        `True` if a checkpoint should be saved, `False` otherwise.

    Raises:
        ValueError: If an unknown `save_strategy` is provided.
    """
    if save_strategy == "no":
        return False
    elif save_strategy == "steps":
        return global_step % save_steps == 0 # Save if current step is a multiple of `save_steps`.
    elif save_strategy == "epoch":
        return epoch_end # Save only if it's the end of an epoch.
    else:
        raise ValueError(f"Unknown save strategy: {save_strategy}")


def estimate_training_time(
    num_steps: int,
    step_time: float,
    completed_steps: int = 0,
) -> Dict[str, float]:
    """
    Estimate the remaining training time.

    Purpose:
        Provides real-time feedback on how much time is left for training,
        useful for monitoring and resource planning.

    Args:
        num_steps: The total number of steps planned for the entire training run.
        step_time: The average time (in seconds) taken to complete one training step.
        completed_steps: The number of training steps that have already been completed.

    Returns:
        A dictionary containing various time estimates:
        - `remaining_steps`: Number of steps yet to complete.
        - `remaining_seconds`: Estimated time remaining in seconds.
        - `remaining_hours`: Estimated time remaining in hours.
        - `eta_hours`: Estimated time of arrival (ETA) in hours (same as remaining_hours).
        - `progress_percent`: Training progress as a percentage.
    """
    remaining_steps = num_steps - completed_steps   # Calculate steps left.
    remaining_seconds = remaining_steps * step_time # Calculate remaining time in seconds.
    
    return {
        "remaining_steps": float(remaining_steps), # Cast to float for consistency in dict values.
        "remaining_seconds": remaining_seconds,
        "remaining_hours": remaining_seconds / 3600, # Convert seconds to hours.
        "eta_hours": remaining_seconds / 3600, # ETA is synonymous with remaining_hours in this context.
        "progress_percent": (completed_steps / num_steps) * 100 if num_steps > 0 else 0.0, # Calculate percentage, handle division by zero.
    }


def cleanup_checkpoints(
    output_dir: Union[str, Path],
    save_total_limit: int,
) -> None:
    """
    Clean up old checkpoint directories, retaining only the most recent ones.

    Purpose:
        To manage disk space by automatically deleting older checkpoints,
        preventing the accumulation of excessive files while ensuring that
        a sufficient number of recent checkpoints are kept.

    Args:
        output_dir: The base directory where checkpoint directories are located.
        save_total_limit: The maximum number of recent checkpoints to retain.
                          Older checkpoints beyond this limit will be removed.
    """
    output_dir = Path(output_dir) # Ensure output_dir is a Path object.
    
    checkpoints = []
    # Find all checkpoint directories within the output directory.
    for path in output_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                # Extract the step number from the directory name.
                step = int(path.name.split("-")[1])
                checkpoints.append((step, path))
            except (IndexError, ValueError):
                # Skip invalidly named directories.
                continue
    
    # Sort checkpoints by their step number to identify the oldest ones.
    checkpoints.sort(key=lambda x: x[0])
    
    # If the number of checkpoints exceeds the limit, remove the oldest ones.
    if len(checkpoints) > save_total_limit:
        for _, checkpoint_path in checkpoints[:-save_total_limit]: # Iterate through all but the 'save_total_limit' most recent.
            logger.info(f"Removing old checkpoint: {checkpoint_path}")
            shutil.rmtree(checkpoint_path) # Recursively delete the checkpoint directory.


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """
    Calculate and return the size of a PyTorch model in bytes and megabytes.

    Purpose:
        Provides a utility to inspect the memory footprint of a model, which
        is important for resource management and debugging.

    Args:
        model: The PyTorch model (`nn.Module`) to analyze.

    Returns:
        A dictionary containing:
        - `param_size_bytes`: Total size of model parameters in bytes.
        - `buffer_size_bytes`: Total size of model buffers in bytes.
        - `total_size_bytes`: Combined size of parameters and buffers in bytes.
        - `total_size_mb`: Combined size of parameters and buffers in megabytes.
    """
    param_size = 0 # Initialize parameter size counter.
    buffer_size = 0 # Initialize buffer size counter.
    
    # Sum the memory occupied by each parameter.
    for param in model.parameters():
        param_size += param.nelement() * param.element_size() # `nelement()` is total number of elements, `element_size()` is size of one element in bytes.
    
    # Sum the memory occupied by each buffer (e.g., BatchNorm running stats).
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024 # Convert total bytes to megabytes.
    
    return {
        "param_size_bytes": param_size,
        "buffer_size_bytes": buffer_size,
        "total_size_bytes": param_size + buffer_size,
        "total_size_mb": size_mb,
    }