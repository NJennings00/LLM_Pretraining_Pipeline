# filename: src/llm_pipeline/training/__init__.py
"""
Training modules for language model pretraining.

This `__init__.py` file serves as the package initializer for the
`llm_pipeline.training` module. It defines the public API for the training
components, making key classes, functions, and utilities directly accessible
when `llm_pipeline.training` is imported.

Purpose:
    To provide a clean and organized interface to the training functionalities
    of the LLM pipeline. Instead of requiring users to import from specific
    sub-modules (e.g., `llm_pipeline.training.trainer.Trainer`), they can
    import directly from the `training` package (e.g., `from llm_pipeline.training import Trainer`).
    It aggregates the most commonly used components from its sub-modules.

    This file is crucial for the modularity and usability of the training
    sub-package. It simplifies imports for downstream users and ensures that
    only intended components are exposed as part of the public API, following
    Python's package conventions.

LLM Pipeline Fit:
    This package is a core part of the LLM pipeline, specifically focusing
    on the pretraining phase. It brings together the `Trainer` (the orchestrator),
    optimizer/scheduler creation, metric tracking, and a flexible callback system
    to manage the entire training process.
"""

# Import the main Trainer class and its associated arguments.
# These are central to initiating and configuring a training run.
from llm_pipeline.training.trainer import Trainer, TrainingArguments

# Import functions related to optimizer and learning rate scheduler creation.
# These abstract away the details of setting up the optimization strategy.
from llm_pipeline.training.optimizer import (
    create_optimizer,  # Function to create a PyTorch optimizer.
    create_scheduler,  # Function to create a PyTorch learning rate scheduler.
    get_optimizer_cls, # Utility to get optimizer class by name.
    get_scheduler_cls, # Utility to get scheduler class by name.
)

# Import classes for tracking and managing training metrics.
# Essential for monitoring model performance during training and evaluation.
from llm_pipeline.training.metrics import TrainingMetrics, MetricsTracker

# Import various callback classes and their handler.
# Callbacks provide hooks into the training loop for custom logic
# like logging, checkpointing, early stopping, and integration with external tools.
from llm_pipeline.training.callbacks import (
    Callback,              # Base class for all callbacks.
    CallbackHandler,       # Manages the execution of registered callbacks.
    LoggingCallback,       # Handles logging of training progress to console/files.
    CheckpointCallback,    # Manages saving model checkpoints.
    EarlyStoppingCallback, # Implements early stopping logic based on metrics.
    WandbCallback,         # Integrates with Weights & Biases for experiment tracking.
    TensorBoardCallback,   # Integrates with TensorBoard for visualization.
)

# Import utility functions for common training operations.
# These functions provide reusable logic for checkpoint management and time estimation.
from llm_pipeline.training.utils import (
    save_checkpoint,        # Saves the current training state.
    load_checkpoint,        # Loads a previously saved training state.
    get_last_checkpoint,    # Finds the most recent checkpoint in a directory.
    should_save_checkpoint, # Determines if a checkpoint should be saved based on strategy.
    estimate_training_time, # Estimates remaining training time.
)

# Define __all__ to explicitly specify which names are exported when
# `from llm_pipeline.training import *` is used. This is good practice for
# controlling the public API of a package.
__all__ = [
    "Trainer",
    "TrainingArguments",
    "create_optimizer",
    "create_scheduler",
    "get_optimizer_cls",
    "get_scheduler_cls",
    "TrainingMetrics",
    "MetricsTracker",
    "Callback",
    "CallbackHandler",
    "LoggingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "WandbCallback",
    "TensorBoardCallback",
    "save_checkpoint",
    "load_checkpoint",
    "get_last_checkpoint",
    "should_save_checkpoint",
    "estimate_training_time",
]