# filename: src/llm_pipeline/training/trainer.py
"""
Main trainer class for language model pretraining.

This module defines the core `Trainer` class responsible for orchestrating
the pretraining process of large language models. It integrates various
components like the model, optimizer, learning rate scheduler, data loaders,
and callbacks to manage the training loop, evaluation, checkpointing, and logging.

Purpose:
    To provide a comprehensive and extensible framework for training
    Transformer-based language models. It encapsulates the complexities of
    the training process, including mixed-precision training, gradient accumulation,
    and interaction with various training utilities.

    The `Trainer` is the central execution unit for the entire training pipeline.
    It ensures that training progresses correctly, hyperparameters are applied,
    and progress is monitored and saved. Its robust design is crucial for
    reliable and reproducible LLM training.

LLM Pipeline Fit:
    This module is the heart of the `llm_pipeline.training` package.
    It utilizes components from:
    - `llm_pipeline.config`: For `TrainingConfig` to set up training arguments.
    - `llm_pipeline.models`: Takes a `TransformerLM` instance for training.
    - `llm_pipeline.training.optimizer`: Calls `create_optimizer` and `create_scheduler`
      to set up the training dynamics.
    - `llm_pipeline.training.metrics`: Uses `MetricsTracker` to manage and
      report training and evaluation metrics.
    - `llm_pipeline.training.callbacks`: Integrates a `CallbackHandler` to
      allow for extensible behavior during training (e.g., logging, checkpointing,
      early stopping, integration with TensorBoard/Wandb).
    - `llm_pipeline.training.utils`: Leverages utility functions for saving
      and loading checkpoints.
    - `llm_pipeline.evaluation`: Uses an `Evaluator` for performing periodic
      model evaluations.
    - `llm_pipeline.data.tokenizer`: The `TokenizerWrapper` is now passed to the
      Trainer to facilitate proper evaluation metrics that might require tokenization.
    - `llm_pipeline.utils`: Employs general utilities like `set_seed` and `setup_logger`.
"""

import os                                      # For interacting with the operating system, e.g., file paths.
import time                                    # For time-related functions, potentially for logging elapsed time.
import logging                                 # For logging events and information during training.
from pathlib import Path                       # For object-oriented filesystem paths.
from typing import Optional, Any, Union, Tuple # Type hinting for improved code readability and maintainability.
from dataclasses import dataclass, field       # For creating data classes for training arguments.
import torch                                   # PyTorch library, the foundation for model and tensor operations.
import torch.nn as nn                          # Neural network module from PyTorch.
from torch.utils.data import DataLoader        # For efficient batching and loading of data.
from torch.cuda.amp import GradScaler          # For mixed-precision training (FP16 scaling).
from torch.amp import autocast                 # Context manager for automatic mixed precision.
from tqdm import tqdm                          # For creating progress bars in loops.
import numpy as np                             # For numerical operations, potentially for metrics.

from llm_pipeline.config import Config, TrainingConfig                                           # Imports configuration classes.
from llm_pipeline.models import TransformerLM                                                    # Imports the specific Transformer Language Model.
from llm_pipeline.training.optimizer import create_optimizer, create_scheduler                   # Imports functions to create optimizer and LR scheduler.
from llm_pipeline.training.metrics import MetricsTracker, TrainingMetrics, compute_gradient_norm # Imports metrics tracking utilities.
from llm_pipeline.training.callbacks import (                                                    # Imports various callback classes and their handler.
    Callback,
    CallbackHandler,
    LoggingCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    WandbCallback,
)
from llm_pipeline.training.utils import ( # Imports utility functions for checkpointing.
    save_checkpoint,
    load_checkpoint,
    get_last_checkpoint,
    should_save_checkpoint,
)
from llm_pipeline.evaluation import Evaluator                                   # Imports the Evaluator class for model assessment.
from llm_pipeline.data.tokenizer import TokenizerWrapper                        # Import TokenizerWrapper for evaluation needs.
from llm_pipeline.utils import set_seed, setup_logger, get_rank, get_world_size # Imports general utility functions.


logger = logging.getLogger(__name__) # Initializes a logger for this module.


@dataclass
class TrainingArguments:
    """
    Arguments for configuring the training process.

    Purpose:
        To consolidate all configurable hyperparameters and settings related to
        training into a single, structured object. This makes configuration
        management easier and less prone to errors compared to passing
        individual arguments.

        Provides a clear and organized way to define the training setup.
        Its `from_training_config` class method allows for easy conversion
        from a broader configuration object, streamlining the setup.

    LLM Pipeline Fit:
        An instance of `TrainingArguments` is passed to the `Trainer`
        during its initialization, dictating how the training loop should
        behave.
    """
    
    output_dir: Path                     # Directory to save checkpoints, logs, and final model.
    num_train_epochs: int = 3            # Total number of training epochs to perform.
    per_device_train_batch_size: int = 8 # Batch size per GPU/CPU for training.
    per_device_eval_batch_size: int = 16 # Batch size per GPU/CPU for evaluation.
    gradient_accumulation_steps: int = 1 # Number of updates steps to accumulate gradients before performing a backward/optimizer step.
    learning_rate: float = 5e-4          # The initial learning rate for the optimizer.
    weight_decay: float = 0.01           # The weight decay to apply (if not zero).
    max_grad_norm: float = 1.0           # Maximum gradient norm (for gradient clipping).
    
    # Optimizer specific arguments
    adam_beta1: float = 0.9    # Beta1 parameter for Adam/AdamW optimizer.
    adam_beta2: float = 0.999  # Beta2 parameter for Adam/AdamW optimizer.
    adam_epsilon: float = 1e-8 # Epsilon parameter for Adam/AdamW optimizer.
    
    # Learning rate scheduler arguments
    lr_scheduler_type: str = "cosine" # The scheduler type to use (e.g., "linear", "cosine").
    warmup_steps: int = 500           # Number of warmup steps for the learning rate scheduler.
    warmup_ratio: float = 0.0         # Ratio of total training steps used for warmup. Overrides `warmup_steps` if > 0.
    
    # Mixed precision training
    fp16: bool = False # Whether to use float16 (mixed precision) training.
    bf16: bool = False # Whether to use bfloat16 (mixed precision) training.
    
    # Evaluation strategy
    eval_strategy: str = "steps" # The evaluation strategy: "no", "steps", or "epoch".
    eval_steps: int = 500        # Evaluate every N update steps when `eval_strategy` is "steps".
    
    # Checkpointing strategy
    save_strategy: str = "steps" # The checkpoint saving strategy: "no", "steps", or "epoch".
    save_steps: int = 1000       # Save checkpoint every N update steps when `save_strategy` is "steps".
    save_total_limit: int = 3    # Maximum number of checkpoints to keep.
    
    # Logging strategy
    logging_steps: int = 50         # Log every N update steps.
    logging_first_step: bool = True # Whether to log the first step.
    
    # Other arguments
    seed: int = 42                                             # Random seed for reproducibility.
    dataloader_num_workers: int = 4                            # Number of subprocesses to use for data loading.
    remove_unused_columns: bool = True                         # Whether to remove unused columns from the dataset.
    resume_from_checkpoint: Optional[Union[Path, bool]] = None # Path to a checkpoint directory to resume from, or `True` to auto-find last.
    
    @classmethod
    def from_training_config(cls, config: TrainingConfig) -> "TrainingArguments":
        """
        Create `TrainingArguments` from a `TrainingConfig` object.

        Purpose:
            Facilitates easy conversion from a configuration object (potentially
            loaded from a file) to the specific arguments required by the `Trainer`.

        Args:
            config: An instance of `TrainingConfig` containing training parameters.

        Returns:
            An initialized `TrainingArguments` instance.
        """
        return cls(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            # Direct access, as these are now top-level in TrainingConfig
            per_device_train_batch_size=config.train_batch_size, 
            per_device_eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            adam_epsilon=config.adam_epsilon,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_steps=config.warmup_steps,
            warmup_ratio=config.warmup_ratio,
            fp16=config.fp16,
            bf16=config.bf16,
            eval_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            # Direct access for logging steps
            logging_steps=config.logging_steps, 
            logging_first_step=config.logging_first_step, 
            # Direct access for seed and resume_from_checkpoint
            seed=config.seed, 
            resume_from_checkpoint=config.resume_from_checkpoint,
        )


class Trainer:
    """
    Trainer for language model pretraining.

    Purpose:
        To encapsulate the entire training and evaluation loop for a
        Transformer-based language model. It handles the low-level details
        of optimizer steps, gradient accumulation, mixed precision, and
        integrates with a callback system for extensible functionality.

        This class is the orchestrator of the training process. Its
        correctness and robustness are paramount for successfully training
        LLMs. It provides a standardized training workflow, making it
        easier to experiment with different models, datasets, and hyperparameters.

    LLM Pipeline Fit:
        The `Trainer` is instantiated in the main training script. It takes
        the model, data loaders, and training arguments, then executes the
        `train` method to begin the training process.
    """
    
    def __init__(
        self,
        model: TransformerLM,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[list] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]] = (None, None),
        tokenizer: Optional[TokenizerWrapper] = None, # ADDED: tokenizer argument
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: The `TransformerLM` model to be trained.
            args: An instance of `TrainingArguments` defining training configuration.
            train_dataloader: DataLoader for the training dataset.
            eval_dataloader: Optional DataLoader for the evaluation dataset.
            callbacks: Optional list of custom `Callback` instances to extend trainer behavior.
            optimizers: Optional tuple of `(optimizer, lr_scheduler)`. If `None`, they are created.
            tokenizer: Optional `TokenizerWrapper` for use in evaluation (e.g., perplexity calculation).
        """
        self.model = model                       # The model to be trained.
        self.args = args                         # Training arguments.
        self.train_dataloader = train_dataloader # Data loader for training.
        self.eval_dataloader = eval_dataloader   # Data loader for evaluation.
        self.tokenizer = tokenizer               # Stores the tokenizer for evaluation.
        
        # Device setup: Automatically uses CUDA if available, otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # Moves the model to the selected device.
        
        # Optimizer and scheduler initialization. If not provided, they are created.
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer = create_optimizer(self.model, self.args) # Creates optimizer using `optimizer.py`.
        if self.lr_scheduler is None:
            self.lr_scheduler = create_scheduler( # Creates scheduler using `optimizer.py`.
                self.optimizer,
                scheduler_type=self.args.lr_scheduler_type,
                num_training_steps=self.get_num_training_steps(),
                warmup_steps=self.get_warmup_steps(),
            )
        
        # Mixed precision scaler (enabled if fp16 is true).
        self.scaler = GradScaler() if self.args.fp16 else None
        
        # Metrics tracker to record and manage training/evaluation metrics.
        self.metrics_tracker = MetricsTracker()
        
        # Callbacks setup: Default callbacks are always included, then custom ones.
        default_callbacks = [
            LoggingCallback(),    # Handles logging of metrics.
            CheckpointCallback(), # Manages saving and loading checkpoints.
        ]
        if self.args.eval_strategy != "no" and self.eval_dataloader is not None:
            default_callbacks.append(EarlyStoppingCallback(patience=3)) # Adds early stopping if evaluation is enabled.
        
        # The CallbackHandler manages the execution of all registered callbacks.
        self.callback_handler = CallbackHandler(
            callbacks=default_callbacks + (callbacks or []), # Combines default and user-provided callbacks.
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        
        # Training state variables.
        self.global_step = 0         # Tracks the total number of optimization steps.
        self.epoch = 0               # Tracks the current epoch number.
        self.total_steps_trained = 0 # Total steps trained so far (useful for resuming).
        
        # Performs initial setup and calls `on_init` for callbacks.
        self._setup()
    
    def _setup(self):
        """
        Perform initial setup tasks for training.

        Purpose:
            Configures the environment, sets up logging, creates output directories,
            and logs initial training information. It also handles resuming from
            a checkpoint if specified.

            Ensures that the training environment is correctly prepared before
            the main loop begins.
        """
        # Set random seed for reproducibility.
        set_seed(self.args.seed)
        
        # Create the output directory if it doesn't exist.
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure the logger to save logs to a file.
        setup_logger(
            log_file=self.args.output_dir / "training.log",
            log_level=logging.INFO,
        )
        
        # Log key training configuration details.
        logger.info(f"Training arguments: {self.args}")
        logger.info(f"Model architecture: {self.model}")
        logger.info(f"Number of parameters: {self.model.num_parameters():,}")
        logger.info(f"Number of training steps: {self.get_num_training_steps()}")

        # Call the `on_init` callback method for all registered callbacks.
        self.callback_handler.on_init(self.args, self.get_train_state())

        # Resume from checkpoint if specified in arguments, handling different input types for `resume_from_checkpoint`.
        if self.args.resume_from_checkpoint:
            self._resume_from_checkpoint()
    
    def get_num_training_steps(self) -> int:
        """
        Calculate the total number of effective training steps.

        Purpose:
            Determines the total number of optimizer updates that will occur
            over the entire training duration, considering batch size and gradient
            accumulation.

        Returns:
            The total number of training steps.
        """
        # Steps per epoch = total batches per epoch / gradient accumulation steps
        steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
        return steps_per_epoch * self.args.num_train_epochs # Total steps = steps per epoch * number of epochs.
    
    def get_warmup_steps(self) -> int:
        """
        Calculate the number of warmup steps based on `warmup_steps` or `warmup_ratio`.

        Purpose:
            Provides a flexible way to define the warmup phase for the learning
            rate scheduler.

        Returns:
            The calculated number of warmup steps.
        """
        if self.args.warmup_steps > 0: # If `warmup_steps` is explicitly set.
            return self.args.warmup_steps
        elif self.args.warmup_ratio > 0: # If `warmup_ratio` is set, calculate based on total steps.
            return int(self.get_num_training_steps() * self.args.warmup_ratio)
        return 0 # Default to no warmup.
    
    def train(self) -> TrainingMetrics:
        """
        Execute the main training loop.

        Purpose:
            Orchestrates the entire training process, iterating through epochs,
            handling batch processing, gradient accumulation, optimization steps,
            logging, evaluation, and checkpointing.

        Returns:
            A `TrainingMetrics` object containing the final training statistics.
        """
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Batch size = {self.args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.get_num_training_steps()}")
        
        self.model.train() # Sets the model to training mode.
        self.callback_handler.on_train_begin(self.args, self.get_train_state()) # Notifies callbacks that training has begun.
        
        for epoch in range(self.epoch, self.args.num_train_epochs): # Loop through epochs, starting from current epoch for resume.
            self.epoch = epoch # Update current epoch.
            self.callback_handler.on_epoch_begin(self.args, self.get_train_state()) # Notifies callbacks that an epoch has begun.
            
            # Train for one epoch.
            epoch_metrics = self._train_epoch()
            
            # Perform evaluation if specified.
            if self._should_evaluate():
                eval_metrics = self.evaluate()
                self.metrics_tracker.update(eval_metrics, prefix="eval") # Update metrics with evaluation results.
            
            self.callback_handler.on_epoch_end(self.args, self.get_train_state()) # Notifies callbacks that an epoch has ended.
            
            # Check if any callback (e.g., EarlyStoppingCallback) has requested to stop training.
            if self.callback_handler.should_stop_training:
                logger.info("Early stopping triggered")
                break # Exit the training loop.
        
        self.callback_handler.on_train_end(self.args, self.get_train_state()) # Notifies callbacks that training has ended.
        
        # Perform a final evaluation after training completes.
        if self.eval_dataloader is not None:
            final_metrics = self.evaluate()
            self.metrics_tracker.update(final_metrics, prefix="final_eval")
        
        return self.metrics_tracker.get_metrics() # Return the accumulated training metrics.
    
    def _train_epoch(self) -> dict[str, float]:
        """
        Train the model for a single epoch.

        Purpose:
            Handles the iteration over the training DataLoader, forward and
            backward passes, gradient accumulation, optimization steps,
            and intermediate logging and evaluation within an epoch.

        Returns:
            A dictionary containing the average training loss and number of steps
            for the epoch.
        """
        epoch_loss = 0.0       # Accumulator for epoch loss.
        epoch_steps = 0        # Counter for effective optimization steps in the epoch.
        accumulation_steps = 0 # Counter for gradient accumulation steps.
        
        # Progress bar for visual feedback during training.
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            disable=get_rank() != 0, # Disable for non-main processes in distributed training.
        )
        
        for step, batch in enumerate(progress_bar): # Iterate over batches in the training data loader.
            # Move batch tensors to the appropriate device (CUDA or CPU).
            if isinstance(batch, dict):
                # Standard dictionary format (e.g., from a DataCollator)
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                # Raw tensor format (e.g., from TensorDataset) - attempt to convert to dict
                moved = [item.to(self.device) if hasattr(item, 'to') else item for item in batch]
                if len(moved) == 3: # Common case: input_ids, attention_mask, labels
                    batch = {'input_ids': moved[0], 'attention_mask': moved[1], 'labels': moved[2]}
                elif len(moved) == 2: # Common case: input_ids, labels
                    batch = {'input_ids': moved[0], 'labels': moved[1]}
                else: # Generic handling for other tuple/list formats
                    batch = {f'tensor_{i}': tensor for i, tensor in enumerate(moved)}
            else:
                # Single tensor - move to device
                batch = batch.to(self.device) if hasattr(batch, 'to') else batch

            # Forward pass with mixed precision (if enabled).
            with autocast(device_type=self.device.type, enabled=self.args.fp16 or self.args.bf16):
                outputs = self.model(**batch) # Pass batch to model.
                loss = outputs["loss"] # Extract loss from model outputs.
                
                # Scale loss for gradient accumulation: Loss is averaged over current micro-batch,
                # so divide by accumulation steps to ensure gradient sum over effective batch is correct.
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass (gradient calculation).
            if self.args.fp16: # If using FP16, use GradScaler for backward pass.
                self.scaler.scale(loss).backward()
            else: # Otherwise, standard backward pass.
                loss.backward()
            
            accumulation_steps += 1 # Increment accumulation step counter.
            epoch_loss += loss.item() # Accumulate loss for epoch average.
            
            # Perform optimizer step only after `gradient_accumulation_steps` micro-batches.
            if accumulation_steps % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients.
                if self.args.max_grad_norm > 0:
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping for FP16.
                    torch.nn.utils.clip_grad_norm_( # Clips gradients of model parameters.
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )
                
                # Compute gradient norm BEFORE zero_grad()
                grad_norm = compute_gradient_norm(self.model)
                
                # Optimizer step (applies accumulated gradients).
                if self.args.fp16:
                    self.scaler.step(self.optimizer) # Optimizer step with scaler.
                    self.scaler.update()             # Updates the GradScaler for next iteration.
                else:
                    self.optimizer.step() # Standard optimizer step.
                
                self.lr_scheduler.step()   # Updates the learning rate according to the schedule.
                self.optimizer.zero_grad() # Clears gradients for the next micro-batch accumulation.
                
                self.global_step += 1 # Increment global training step counter.
                epoch_steps += 1      # Increment effective epoch step counter.
                
                # Update and log metrics.
                current_loss = epoch_loss / epoch_steps # Current average loss for the epoch.
                current_lr = self.lr_scheduler.get_last_lr()[0] # Current learning rate.
                
                self.metrics_tracker.update({ # Update metrics tracker with current values.
                    "loss": current_loss,
                    "learning_rate": current_lr,
                    "epoch": self.epoch,
                    "grad_norm": grad_norm,  #  Store gradient norm in metrics
                })
                
                # Logging to console and via callbacks.
                if self._should_log():
                    progress_bar.set_postfix({ # Update tqdm progress bar with current loss and LR.
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "grad_norm": f"{grad_norm:.4f}",  # Add grad norm to progress bar
                    })
                    self.callback_handler.on_log(self.args, self.get_train_state()) # Trigger logging callback.
                
                # Evaluation at specified intervals.
                if self._should_evaluate():
                    eval_metrics = self.evaluate() # Perform evaluation.
                    self.metrics_tracker.update(eval_metrics, prefix="eval") # Update metrics with evaluation results.
                    self.model.train() # Switch model back to training mode after evaluation.
                
                # Checkpointing at specified intervals.
                if self._should_save():
                    self._save_checkpoint() # Save current training state.
                
                # Check if any callback has triggered an early stop.
                if self.callback_handler.should_stop_training:
                    break # Exit the epoch loop.
        
        # Return epoch-level aggregated metrics.
        return {
            "train_loss": epoch_loss / epoch_steps,
            "train_steps": epoch_steps,
        }
    
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Purpose:
            Runs the model in inference mode over the evaluation dataset to
            compute performance metrics.

        Returns:
            A dictionary of evaluation metrics (e.g., perplexity, loss).
        """
        if self.eval_dataloader is None: # If no evaluation data is provided.
            return {} # Return empty metrics.
        
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(self.eval_dataloader.dataset)}")
        logger.info(f"  Batch size = {self.args.per_device_eval_batch_size}")
        
        self.model.eval() # Sets the model to evaluation mode (e.g., disables dropout).
        
        # Initialize the Evaluator with model, data, tokenizer, and device.
        evaluator = Evaluator(
            model=self.model,
            dataloader=self.eval_dataloader,
            tokenizer=self.tokenizer, # Pass the tokenizer here for potential token-based metrics.
            device=self.device,
        )
        
        metrics = evaluator.evaluate() # Perform evaluation using the Evaluator.
        
        # Log the computed evaluation metrics.
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics # Return the evaluation results.
    
    def _should_log(self) -> bool:
        """
        Determine if current global step warrants logging.

        Purpose:
            Controls when training progress (loss, LR) should be logged,
            based on `logging_steps` and `logging_first_step` arguments.

        Returns:
            `True` if logging should occur, `False` otherwise.
        """
        return (
            self.global_step % self.args.logging_steps == 0 or # Log every `logging_steps`.
            (self.args.logging_first_step and self.global_step == 1) # Or if it's the very first step and configured to log it.
        )
    
    def _should_evaluate(self) -> bool:
        """
        Determine if current global step warrants evaluation.

        Purpose:
            Controls when evaluation should be performed during training,
            based on `eval_strategy` and `eval_steps` arguments.

        Returns:
            `True` if evaluation should occur, `False` otherwise.
        """
        if self.args.eval_strategy == "no" or self.eval_dataloader is None: # If evaluation is disabled or no data.
            return False
        
        if self.args.eval_strategy == "steps": # Evaluate every N steps.
            return self.global_step % self.args.eval_steps == 0
        elif self.args.eval_strategy == "epoch": # Evaluate at the end of each epoch.
            return True  # This check is called at the end of the epoch.
        
        return False
    
    def _should_save(self) -> bool:
        """
        Determine if current global step warrants saving a checkpoint.

        Purpose:
            Controls when training checkpoints should be saved, based on
            `save_strategy` and `save_steps` arguments.

        Returns:
            `True` if a checkpoint should be saved, `False` otherwise.
        """
        if self.args.save_strategy == "no": # If saving is disabled.
            return False
        
        if self.args.save_strategy == "steps": # Save every N steps.
            return self.global_step % self.args.save_steps == 0
        elif self.args.save_strategy == "epoch": # Save at the end of each epoch.
            return True  # This check is called at the end of the epoch.
        
        return False
    
    def _save_checkpoint(self):
        """
        Save the current training state as a checkpoint.

        Purpose:
            Preserves the model, optimizer, scheduler state, and training progress
            at a specific step, allowing for resuming training later. Also manages
            cleanup of old checkpoints.
        """
        checkpoint_dir = self.args.output_dir / f"checkpoint-{self.global_step}" # Create a unique directory for the checkpoint.
        
        save_checkpoint( # Calls utility function to save components.
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            epoch=self.epoch,
            global_step=self.global_step,
            checkpoint_dir=checkpoint_dir,
            metrics=self.metrics_tracker.state_dict(), # Save metrics tracker state for resumption.
        )
        
        # Clean up old checkpoints if a limit is set.
        if self.args.save_total_limit is not None:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """
        Remove older checkpoints to adhere to `save_total_limit`.

        Purpose:
            Manages disk space by deleting checkpoints beyond the specified limit,
            keeping only the most recent ones.
        """
        checkpoints = sorted( # Get list of existing checkpoint directories, sorted by step number.
            self.args.output_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1]),
        )
        
        if len(checkpoints) > self.args.save_total_limit: # If number of checkpoints exceeds the limit.
            for checkpoint in checkpoints[:-self.args.save_total_limit]: # Iterate and remove the oldest ones.
                logger.info(f"Removing old checkpoint: {checkpoint}")
                import shutil
                shutil.rmtree(checkpoint) # Deletes the checkpoint directory.
    
    def _resume_from_checkpoint(self):
        """
        Resume training from a specified or automatically found checkpoint.

        Purpose:
            Loads the model's state, optimizer's state, scheduler's state,
            and the training progress (epoch, global_step) from a saved checkpoint,
            allowing training to continue from where it left off.
        """
        checkpoint_path = self.args.resume_from_checkpoint # Get the desired checkpoint path from arguments.

        # Handle different `resume_from_checkpoint` values:
        # True: auto-find the last checkpoint in output_dir
        if checkpoint_path is True:
            checkpoint_path = get_last_checkpoint(self.args.output_dir)
        # None: auto-find the last checkpoint (same behavior as True if no path given)
        elif checkpoint_path is None:
            checkpoint_path = get_last_checkpoint(self.args.output_dir)
        # False: explicitly do not resume
        elif checkpoint_path is False:
            return # Exit if explicitly told not to resume.
        
        if checkpoint_path is None: # If no checkpoint is found or specified.
            logger.warning("No checkpoint found to resume from")
            return

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = load_checkpoint( # Load checkpoint data.
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )
        
        self.epoch = checkpoint["epoch"] # Restore epoch from checkpoint.
        self.global_step = checkpoint["global_step"] # Restore global step from checkpoint.
        
        if "metrics" in checkpoint: # Attempt to load metrics tracker state.
            try:
                self.metrics_tracker.load_state_dict(checkpoint["metrics"]) # Load if in state_dict format.
            except (KeyError, TypeError):
                # Fallback for older checkpoint formats that might not have a metrics state_dict.
                logger.warning("Skipping metrics loading due to old checkpoint format or corrupted data.")
    
    def get_train_state(self) -> dict[str, Any]:
        """
        Retrieve the current training state.

        Purpose:
            Provides a snapshot of the current training progress, including
            epoch, global step, metrics, and whether training should stop.
            This is primarily used by callbacks.

        Returns:
            A dictionary containing current training state information.
        """
        return {
            "epoch": self.epoch, # Current epoch number.
            "global_step": self.global_step, # Current global step number.
            "metrics": self.metrics_tracker.get_metrics(), # Current metrics collected.
            "should_stop": self.callback_handler.should_stop_training, # Flag indicating if training should stop.
        }
    
    def save_model(self, output_dir: Optional[Path] = None):
        """
        Save the final trained model.

        Purpose:
            Stores the trained model's state dictionary and configuration
            to a specified directory, allowing it to be reloaded and used
            for inference.

        Args:
            output_dir: Optional. The directory where the model should be saved.
                        If `None`, it defaults to a 'final_model' subdirectory
                        within the main output directory.
        """
        output_dir = output_dir or self.args.output_dir / "final_model" # Determine the save directory.
        output_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist.
        
        # Save the model's state dictionary.
        torch.save(self.model.state_dict(), output_dir / "pytorch_model.bin")
        
        # Save the model's configuration as a JSON file.
        import json
        config_dict = self.model.config.__dict__
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2) # Writes the config dict as pretty-printed JSON.
        
        logger.info(f"Model saved to {output_dir}") # Logs the save location.