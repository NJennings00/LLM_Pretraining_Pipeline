# filename: src/llm_pipeline/training/callbacks.py
"""
Training callbacks for monitoring and control.

This module defines a system of "callbacks" for the LLM training pipeline.
Callbacks are hooks that can be inserted into the training loop to perform
actions at specific stages (e.g., at the beginning/end of an epoch, after a step).
This modular design allows for flexible and extensible monitoring, logging,
checkpointing, and early stopping functionalities without modifying the
core training loop logic.

Purpose:
    To provide a plug-and-play mechanism for adding custom behaviors to the
    training process, such as logging metrics, saving model checkpoints,
    implementing early stopping, or integrating with experiment tracking tools
    like Weights & Biases or TensorBoard.

    Callbacks are a best practice in deep learning frameworks for managing side
    effects during training. They promote code reusability, modularity, and
    separation of concerns. This system ensures that the `Trainer` class
    remains clean and focused solely on the training loop itself, delegating
    monitoring and control tasks to specialized callback classes.

LLM Pipeline Fit:
    During the `Trainer`'s execution, it iterates through a list of registered
    callbacks and calls the appropriate method (e.g., `on_step_end`, `on_log`)
    at predefined points. This enables:
    - **Logging**: `LoggingCallback`, `WandbCallback`, `TensorBoardCallback`
      collect and report training progress.
    - **Checkpointing**: `CheckpointCallback` saves the model state periodically.
    - **Early Stopping**: `EarlyStoppingCallback` monitors validation metrics
      to prevent overfitting and save computational resources.

The `CallbackHandler` orchestrates the execution of multiple callbacks, ensuring
that all registered callbacks receive the relevant training state and arguments
at each hook point.
"""

import logging                           # Imports the logging module for emitting messages.
import time                              # Imports the time module for timing operations.
from typing import Optional, Any, Union  # Imports type hints for better readability and type checking.
from pathlib import Path                 # Imports Path for object-oriented filesystem paths.
from abc import ABC, abstractmethod      # Imports Abstract Base Classes for defining interfaces.
import torch                             # Imports PyTorch library.
import torch.nn as nn                    # Imports neural network module from PyTorch.
from torch.utils.data import DataLoader  # Imports DataLoader for handling datasets.

from llm_pipeline.training.metrics import compute_gradient_norm, compute_parameter_norm # Imports utility functions for computing norms.
from llm_pipeline.training.utils import save_checkpoint # Imports utility function for saving model checkpoints.


logger = logging.getLogger(__name__) # Initializes a logger for this module.


class Callback(ABC):
    """
    Base callback class.

    Purpose:
        Defines the interface for all callback classes in the training pipeline.
        It specifies a set of methods (hooks) that the `Trainer` will call at
        different stages of the training process. Subclasses must implement
        these methods to perform specific actions.

        This abstract base class enforces a consistent structure for all callbacks,
        making them interchangeable and easy to manage by the `CallbackHandler`.
        It ensures that any custom callback adheres to the expected API.

    LLM Pipeline Fit:
        Individual callback implementations (e.g., `LoggingCallback`, `CheckpointCallback`)
        inherit from this base class. The `Trainer` iterates through a list of
        `Callback` objects and calls their methods as appropriate.
    """

    def on_init(self, args, state):
        """Called at initialization."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_train_begin(self, args, state):
        """Called at the beginning of training."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_train_end(self, args, state):
        """Called at the end of training."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_epoch_begin(self, args, state):
        """Called at the beginning of an epoch."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_epoch_end(self, args, state):
        """Called at the end of an epoch."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_step_begin(self, args, state):
        """Called at the beginning of a training step."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_step_end(self, args, state):
        """Called at the end of a training step."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_evaluate(self, args, state, metrics):
        """Called after evaluation."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_save(self, args, state):
        """Called when saving a checkpoint."""
        pass # Placeholder, intended to be overridden by subclasses.

    def on_log(self, args, state):
        """Called when logging."""
        pass # Placeholder, intended to be overridden by subclasses.


class CallbackHandler:
    """
    Handles multiple callbacks.

    Purpose:
        Manages a collection of `Callback` instances and dispatches events to them.
        It acts as an intermediary between the `Trainer` (or training loop) and
        the individual callbacks, ensuring that each callback's appropriate method
        is called at the right time.

        This class centralizes the callback management logic, preventing the
        `Trainer` from being cluttered with direct calls to individual callbacks.
        It also facilitates passing shared objects (model, optimizer, etc.)
        to all callbacks. The `should_stop_training` flag provides a global
        mechanism for callbacks (like `EarlyStoppingCallback`) to signal
        the trainer to halt.

    LLM Pipeline Fit:
        The `Trainer` initializes a `CallbackHandler` with a list of desired
        callbacks. During the training loop, the `Trainer` calls methods on
        this `CallbackHandler` (e.g., `self.callback_handler.on_step_end(...)`),
        and the handler then propagates that call to all its registered callbacks.

    Inputs to `__init__`:
        - `callbacks` (list[Callback]): A list of `Callback` instances to manage.
        - `model` (nn.Module): The training model.
        - `optimizer` (torch.optim.Optimizer): The optimizer used for training.
        - `lr_scheduler` (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        - `train_dataloader` (DataLoader): The data loader for the training set.
        - `eval_dataloader` (Optional[DataLoader]): The data loader for the evaluation set (optional).

    Attributes:
        - `should_stop_training` (bool): A flag that callbacks can set to `True`
          to signal the training loop to terminate prematurely.
    """

    def __init__(
        self,
        callbacks: list[Callback],                           # List of callback instances to handle.
        model: nn.Module,                                    # The model being trained.
        optimizer: torch.optim.Optimizer,                    # The optimizer instance.
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler, # The learning rate scheduler instance.
        train_dataloader: DataLoader,                        # DataLoader for training data.
        eval_dataloader: Optional[DataLoader] = None,        # Optional DataLoader for evaluation data.
    ):
        self.callbacks = callbacks                # Stores the list of callbacks.
        self.model = model                        # Stores the model.
        self.optimizer = optimizer                # Stores the optimizer.
        self.lr_scheduler = lr_scheduler          # Stores the LR scheduler.
        self.train_dataloader = train_dataloader  # Stores the train DataLoader.
        self.eval_dataloader = eval_dataloader    # Stores the eval DataLoader.

        # State
        self.should_stop_training = False # Flag to signal early stopping.

        # Assign common objects to callbacks (this is done in Trainer's __init__ for CallbackHandler)
        for callback in self.callbacks:                   # Iterates through registered callbacks.
            callback.model = model                        # Assigns the model to each callback.
            callback.optimizer = optimizer                # Assigns the optimizer to each callback.
            callback.lr_scheduler = lr_scheduler          # Assigns the LR scheduler to each callback.
            callback.train_dataloader = train_dataloader  # Assigns the train DataLoader to each callback.
            callback.eval_dataloader = eval_dataloader    # Assigns the eval DataLoader to each callback.
            # The on_init call will now be done by the Trainer via self.callback_handler.on_init()

    def on_init(self, args, state):
        """Dispatches `on_init` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_init(args, state)

    def on_train_begin(self, args, state):
        """Dispatches `on_train_begin` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(args, state)

    def on_train_end(self, args, state):
        """Dispatches `on_train_end` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(args, state)

    def on_epoch_begin(self, args, state):
        """Dispatches `on_epoch_begin` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(args, state)

    def on_epoch_end(self, args, state):
        """Dispatches `on_epoch_end` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(args, state)

    def on_step_begin(self, args, state):
        """Dispatches `on_step_begin` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_step_begin(args, state)

    def on_step_end(self, args, state):
        """Dispatches `on_step_end` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_step_end(args, state)

    def on_evaluate(self, args, state, metrics):
        """Dispatches `on_evaluate` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_evaluate(args, state, metrics)

    def on_save(self, args, state):
        """Dispatches `on_save` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_save(args, state)

    def on_log(self, args, state):
        """Dispatches `on_log` call to all registered callbacks."""
        for callback in self.callbacks:
            callback.on_log(args, state)


class LoggingCallback(Callback):
    """
    Callback for logging training progress to the console/logger.

    Purpose:
        Provides basic logging of training metrics (loss, learning rate,
        gradient norm) at specified intervals. It prints progress information
        to the console or a configured logger.

        Essential for real-time monitoring of training progress. It helps
        developers understand if the model is learning, if the loss is
        decreasing, and if gradients are behaving as expected.

    LLM Pipeline Fit:
        An instance of `LoggingCallback` is typically included in the list
        of callbacks passed to the `CallbackHandler` in the `Trainer`.
        Its `on_log` method is called periodically during training to report
        the current state.
    """

    def __init__(self, log_interval: int = 10): # Constructor for LoggingCallback.
        self.log_interval = log_interval        # Frequency of logging (in steps).
        self.start_time = None                  # To track total training time.

    def on_init(self, args, state):
        """Initialisation hook."""
        pass # No specific action needed on initialization.

    def on_train_begin(self, args, state):
        """Sets the start time and logs a message at the beginning of training."""
        self.start_time = time.time()   # Records start time of training.
        logger.info("Training started") # Logs training start message.

    def on_train_end(self, args, state):
        """Logs the total training time at the end of training."""
        total_time = time.time() - self.start_time                     # Calculates total training time.
        logger.info(f"Training completed in {total_time:.2f} seconds") # Logs training completion message.

    def on_log(self, args, state):
        """
        Logs training metrics at each logging interval.

        It constructs a log message including step, epoch, loss, eval loss (if available),
        learning rate, and gradient norm.
        """
        metrics = state["metrics"] # Retrieves current metrics from the state.

        # Build log message
        log_parts = [ # Initializes a list to build parts of the log message.
            f"Step {state['global_step']}", # Adds current global step.
            f"Epoch {state['epoch']}",      # Adds current epoch.
        ]

                                                             # Add key metrics
        if "loss" in metrics:                                # If training loss is available.
            log_parts.append(f"Loss: {metrics['loss']:.4f}") # Adds training loss.

        if "eval/loss" in metrics:                                     # If evaluation loss is available.
            log_parts.append(f"Eval Loss: {metrics['eval/loss']:.4f}") # Adds evaluation loss.

        if "learning_rate" in metrics:                              # If learning rate is available.
            log_parts.append(f"LR: {metrics['learning_rate']:.2e}") # Adds learning rate.

                                                                       # Use gradient norm from metrics instead of computing it
        if "grad_norm" in metrics:                                     # If gradient norm is available in metrics.
            log_parts.append(f"Grad Norm: {metrics['grad_norm']:.4f}") # Adds gradient norm from metrics.

        logger.info(" | ".join(log_parts)) # Joins parts and logs the message.


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints.

    Purpose:
        Periodically saves the current state of the model, optimizer,
        and learning rate scheduler, along with training progress (epoch, step, metrics).
        This allows for resuming training from a specific point or deploying
        a trained model.

        Crucial for long-running training jobs, preventing data loss in case
        of interruptions, and enabling hyperparameter tuning by allowing
        different runs to start from a common checkpoint.

    LLM Pipeline Fit:
        An instance of `CheckpointCallback` is registered with the `CallbackHandler`.
        Its `on_save` method is triggered by the `Trainer` (e.g., after a certain
        number of steps or epochs) to persist the training state to disk.
    """

    def on_init(self, args, state):
        """Initialisation hook."""
        pass # No specific action needed on initialization.

    def on_save(self, args, state):
        """
        Saves a checkpoint to a directory named after the global step.

        It uses the `save_checkpoint` utility function to handle the actual saving.
        """
        logger.info(f"Saving checkpoint at step {state['global_step']}") # Logs checkpoint saving message.

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state['global_step']}" # Defines the checkpoint directory path.

        save_checkpoint(                        # Calls the utility function to save the checkpoint.
            model=self.model,                   # Passes the model.
            optimizer=self.optimizer,           # Passes the optimizer.
            lr_scheduler=self.lr_scheduler,     # Passes the LR scheduler.
            epoch=state["epoch"],               # Passes the current epoch.
            global_step=state["global_step"],   # Passes the current global step.
            checkpoint_dir=checkpoint_dir,      # Passes the checkpoint directory.
            metrics=state.get("metrics", {}),   # Passes current metrics (if available).
        )


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on validation metric.

    Purpose:
        Monitors a specified metric (e.g., validation loss) and stops training
        if the metric does not improve for a certain number of evaluation steps
        (patience). This helps prevent overfitting and saves computational resources.

        An essential tool for robust model training, ensuring that training
        stops when the model's generalization performance on unseen data
        starts to degrade.

    LLM Pipeline Fit:
        An `EarlyStoppingCallback` instance is added to the `CallbackHandler`.
        Its `on_evaluate` method is called after each evaluation phase. It
        maintains a counter of epochs without improvement and, if the patience
        limit is reached, sets the `should_stop_training` flag on the
        `CallbackHandler` to `True`, which the `Trainer` checks to terminate
        the training loop.

    Inputs to `__init__`:
        - `metric` (str): The name of the metric to monitor (e.g., "eval/loss").
        - `patience` (int): Number of evaluations to wait for improvement before stopping.
        - `min_delta` (float): Minimum change in the monitored metric to qualify as an improvement.
        - `mode` (str): "min" for metrics where lower is better (e.g., loss), "max" for higher is better (e.g., accuracy).
    """

    def __init__(
        self,
        metric: str = "eval/loss",  # Name of the metric to monitor.
        patience: int = 3,          # Number of evaluation cycles to wait for improvement.
        min_delta: float = 0.0,     # Minimum change to be considered an improvement.
        mode: str = "min",          # Mode for metric monitoring ("min" or "max").
    ):
        self.metric = metric        # Stores the metric name.
        self.patience = patience    # Stores patience value.
        self.min_delta = min_delta  # Stores minimum delta value.
        self.mode = mode            # Stores the mode.

        self.best_value = float("inf") if mode == "min" else float("-inf") # Initializes best value based on mode.
        self.counter = 0 # Counter for unimproved evaluations.

    def on_init(self, args, state):
        """Initialisation hook."""
        pass # No specific action needed on initialization.

    def on_evaluate(self, args, state, metrics):
        """
        Monitors the specified metric and updates the early stopping counter.

        If the metric does not improve within `patience` evaluations, it sets
        the `should_stop_training` flag on the `CallbackHandler`.
        """
        if self.metric not in metrics: # Checks if the monitored metric is present in current metrics.
            return # If not, do nothing.

        current_value = metrics[self.metric] # Gets the current value of the monitored metric.

        if self.mode == "min":                                          # If monitoring for minimum value.
            improved = current_value < self.best_value - self.min_delta # Checks if current value is an improvement.
        else:                                                           # If monitoring for maximum value.
            improved = current_value > self.best_value + self.min_delta # Checks if current value is an improvement.

        if improved:                        # If metric improved.
            self.best_value = current_value # Updates best value.
            self.counter = 0                # Resets counter.
            logger.info(                    # Logs improvement message.
                f"Metric {self.metric} improved to {current_value:.4f}"
            )
        else:                   # If metric did not improve.
            self.counter += 1   # Increments counter.
            logger.info(        # Logs no improvement message.
                f"Metric {self.metric} did not improve "
                f"(best: {self.best_value:.4f}, current: {current_value:.4f}). "
                f"Patience: {self.counter}/{self.patience}"
            )

        if self.counter >= self.patience:                                   # If patience limit is reached.
            logger.info("Early stopping triggered")                         # Logs early stopping message.
            if hasattr(self, 'callback_handler') and self.callback_handler: # If callback handler is available.
                self.callback_handler.should_stop_training = True           # Signals the trainer to stop training.


class WandbCallback(Callback):
    """
    Callback for Weights & Biases logging.

    Purpose:
        Integrates the training pipeline with Weights & Biases (W&B), an
        experiment tracking platform. It initializes a W&B run, logs
        hyperparameters, tracks metrics, and can optionally watch the model
        (gradients, parameters).

        W&B provides powerful visualization, comparison, and collaboration
        features for machine learning experiments. This callback makes it
        seamless to integrate with the `llm_pipeline`.

    LLM Pipeline Fit:
        An instance of `WandbCallback` is included in the callbacks.
        - `on_init`: Attempts to import `wandb`.
        - `on_train_begin`: Initializes a W&B run and optionally starts watching the model.
        - `on_train_end`: Ends the W&B run.
        - `on_log`: Logs current metrics to W&B at each logging interval.

    Inputs to `__init__`:
        - `project` (str): The W&B project name.
        - `config` (dict[str, Any]): A dictionary of hyperparameters to log to W&B.
        - `**kwargs`: Additional keyword arguments passed to `wandb.init()`.
    """

    def __init__(self, project: str, config: dict[str, Any], **kwargs): # Constructor for WandbCallback.
        self.project = project                                          # W&B project name.
        self.config = config                                            # Configuration dictionary to log.
        self.kwargs = kwargs                                            # Additional kwargs for wandb.init.
        self.run = None                                                 # Stores the wandb run object.
        self.wandb = None                                               # Stores the wandb module reference.

    def on_init(self, args, state):
        """
        Attempts to import `wandb` and warns if it's not installed.

        This ensures the callback doesn't crash if W&B is not available in the environment.
        """
        try: # Tries to import wandb.
            import wandb
            self.wandb = wandb # Stores wandb module.
        except ImportError: # If ImportError occurs.
            logger.warning("wandb not installed, WandbCallback will be disabled") # Logs a warning.
            self.wandb = None # Sets wandb to None, effectively disabling the callback.

    def on_train_begin(self, args, state):
        """
        Initializes the Weights & Biases run and starts watching the model.
        """
        if self.wandb is not None:      # If wandb was successfully imported.
            self.run = self.wandb.init( # Initializes a new wandb run.
                project=self.project,   # Sets the project name.
                config=self.config,     # Logs the configuration.
                **self.kwargs,          # Passes additional kwargs.
            )

            # Watch model
            if hasattr(self, 'model') and self.model: # If the model is available.
                self.wandb.watch(self.model, log="all", log_freq=100) # Starts watching the model for gradients and parameters.

    def on_train_end(self, args, state):
        """Ends the Weights & Biases run."""
        if self.run is not None: # If a wandb run was initialized.
            self.run.finish() # Ends the wandb run.

    def on_log(self, args, state):
        """Logs metrics to Weights & Biases."""
        if self.run is not None: # If a wandb run is active.
            metrics = state["metrics"] # Retrieves current metrics.
            self.wandb.log(metrics, step=state["global_step"]) # Logs metrics with the current global step.


class TensorBoardCallback(Callback):
    """
    Callback for TensorBoard logging.

    Purpose:
        Integrates the training pipeline with TensorBoard, enabling visualization
        of scalar metrics, histograms of gradients and parameters, and other
        training insights.

        TensorBoard is a widely used visualization tool for machine learning.
        This callback provides a native way to export training data for
        TensorBoard's dashboards.

    LLM Pipeline Fit:
        An instance of `TensorBoardCallback` is added to the callbacks list.
        - `on_init`: Attempts to import `SummaryWriter`.
        - `on_train_begin`: Initializes the `SummaryWriter` and creates the log directory.
        - `on_train_end`: Closes the `SummaryWriter`.
        - `on_log`: Logs scalar metrics, learning rates, and optional histograms
          of gradients and parameters to TensorBoard.

    Inputs to `__init__`:
        - `log_dir` (Union[str, Path]): The directory where TensorBoard logs will be saved.
    """

    def __init__(self, log_dir: Union[str, Path]):  # Constructor for TensorBoardCallback.
        self.log_dir = Path(log_dir)                # Stores the log directory path.
        self.writer = None                          # Initialize writer as None # Stores the SummaryWriter instance.

        # Try to import SummaryWriter class in __init__
        try: # Tries to import SummaryWriter.
            from torch.utils.tensorboard import SummaryWriter
            self._SummaryWriter_class = SummaryWriter # Store the class # Stores the SummaryWriter class.
        except ImportError: # If ImportError occurs.
            logger.warning( # Logs a warning.
                "tensorboard not installed, TensorBoardCallback will be disabled"
            )
            self._SummaryWriter_class = None # Set to None if not available # Sets to None, effectively disabling the callback.

    def on_init(self, args, state):
        """Initialisation hook."""
        # This method is now explicitly called by CallbackHandler's on_init.
        # But for TensorBoard, actual writer instantiation can still happen in on_train_begin
        # because log_dir needs to be created, and we typically want to start logging
        # when training actually begins.
        pass # No specific action needed on initialization, deferring writer creation.
        # TODO, make decision on whether to remove this throughout

    def on_train_begin(self, args, state):
        """Initializes the `SummaryWriter` at the beginning of training."""
        if self._SummaryWriter_class is not None:                   # If SummaryWriter class is available.
            self.log_dir.mkdir(parents=True, exist_ok=True)         # Creates the log directory.
            self.writer = self._SummaryWriter_class(self.log_dir)   # Instantiate here # Initializes the SummaryWriter.

    def on_train_end(self, args, state):
        """Closes the `SummaryWriter` at the end of training."""
        if self.writer is not None: # If writer is active.
            self.writer.close() # Closes the writer.

    def on_log(self, args, state):
        """
        Logs scalar metrics, learning rates, and optionally histograms of
        gradients and parameters to TensorBoard.
        """
        if self.writer is not None: # If writer is active.
            metrics = state["metrics"] # Retrieves current metrics.
            step = state["global_step"] # Retrieves current global step.

            # Log scalars
            for key, value in metrics.items(): # Iterates through metrics.
                if isinstance(value, (int, float)): # If metric value is a scalar.
                    self.writer.add_scalar(key, value, step) # Logs scalar to TensorBoard.

            # Log learning rate
            if hasattr(self, "lr_scheduler"): # If LR scheduler is available.
                lr = self.lr_scheduler.get_last_lr()[0] # Gets current learning rate.
                self.writer.add_scalar("learning_rate", lr, step) # Logs learning rate.

            # Log gradients and parameters (less frequently)
            if hasattr(self, "model") and step % 100 == 0: # If model is available and at logging interval.
                for name, param in self.model.named_parameters(): # Iterates through model parameters.
                    if param.grad is not None: # If parameter has gradients.
                        self.writer.add_histogram( # Logs gradient histogram.
                            f"gradients/{name}", param.grad, step
                        )
                        self.writer.add_histogram( # Logs parameter histogram.
                            f"parameters/{name}", param, step
                        )