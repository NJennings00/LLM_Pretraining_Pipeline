# filename: src/llm_pipeline/training/optimizer.py
"""
Optimizer and learning rate scheduler utilities.

This module provides functions to create and configure PyTorch optimizers and
custom learning rate schedulers for the LLM training pipeline. It abstracts
away the specifics of optimizer and scheduler initialization, including
handling of weight decay and different learning rate decay strategies.

Purpose:
    To centralize the logic for setting up the optimization strategy for training
    large language models. This includes choosing the right optimizer (e.g., AdamW),
    applying weight decay appropriately, and defining learning rate schedules
    (e.g., linear decay, cosine annealing with warmup).

    The choice and configuration of optimizers and learning rate schedulers
    are critical for successful and stable training of deep learning models,
    especially LLMs. This module ensures that these components are correctly
    initialized and integrated into the training loop, providing common
    strategies used in practice. It also handles the common practice of
    applying weight decay only to certain parameters.

LLM Pipeline Fit:
    The `Trainer` class in `src/llm_pipeline/training/trainer.py` will use
    the `create_optimizer` and `create_scheduler` functions from this module
    to set up the optimization components before starting the training loop.
    The custom scheduler classes (`LinearScheduler`, `CosineScheduler`, etc.)
    define how the learning rate changes over the course of training, which
    is crucial for convergence and preventing overfitting. The `get_parameter_names`
    function helps in fine-grained control over weight decay application.
"""

import logging                                      # Imports the logging module for emitting messages.
from typing import Optional, Tuple, Union, Any      # Type hinting for better readability and type checking.
import torch                                        # Imports PyTorch library.
import torch.nn as nn                               # Imports neural network module from PyTorch for nn.Module.
from torch.optim import Optimizer                   # Imports the base Optimizer class.
from torch.optim.lr_scheduler import _LRScheduler   # Imports the base Learning Rate Scheduler class.
import math                                         # Imports the math module for mathematical operations (e.g., cosine for schedulers).

logger = logging.getLogger(__name__) # Initializes a logger for this module.


def get_optimizer_cls(name: str) -> type:
    """
    Get an optimizer class from `torch.optim` by its string name.

    Purpose:
        Provides a convenient way to select an optimizer class based on a
        configuration string, making the optimizer choice configurable
        without directly importing `torch.optim` classes everywhere.

    Args:
        name: The string name of the optimizer (e.g., "adamw", "sgd").

    Returns:
        The corresponding PyTorch optimizer class.

    Raises:
        ValueError: If an unknown optimizer name is provided.
    """
    optimizers = { # Dictionary mapping string names to PyTorch optimizer classes.
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "rmsprop": torch.optim.RMSprop,
    }

    if name.lower() not in optimizers:                  # Checks if the provided name is in the supported list.
        raise ValueError(f"Unknown optimizer: {name}")  # Raises error if not found.

    return optimizers[name.lower()] # Returns the corresponding optimizer class.


def get_scheduler_cls(name: str) -> type:
    """
    Get a custom learning rate scheduler class by its string name.

    Purpose:
        Similar to `get_optimizer_cls`, this function allows for flexible
        selection of learning rate scheduler types based on a configuration.

    Args:
        name: The string name of the scheduler (e.g., "linear", "cosine").

    Returns:
        The corresponding custom learning rate scheduler class defined in this module.

    Raises:
        ValueError: If an unknown scheduler name is provided.
    """
    schedulers = { # Dictionary mapping string names to custom scheduler classes.
        "linear": LinearScheduler,
        "cosine": CosineScheduler,
        "cosine_with_restarts": CosineWithRestartsScheduler,
        "polynomial": PolynomialScheduler,
        "constant": ConstantScheduler,
        "constant_with_warmup": ConstantWithWarmupScheduler,
    }

    if name.lower() not in schedulers: # Checks if the provided name is in the supported list.
        raise ValueError(f"Unknown scheduler: {name}") # Raises error if not found.

    return schedulers[name.lower()] # Returns the corresponding scheduler class.


def create_optimizer(
    model: nn.Module,
    args: "TrainingArguments", # Keep as string literal for forward reference to avoid circular import.
    optimizer_cls: Optional[type] = None,
) -> Optimizer:
    """
    Create an optimizer for the given model, applying weight decay selectively.

    Purpose:
        Initializes the chosen optimizer (defaulting to AdamW) and configures
        it with the learning rate and specific weight decay rules. It typically
        applies weight decay to most parameters but excludes biases and LayerNorm
        parameters, a common practice in NLP.

        Correct optimizer setup is fundamental for effective training.
        Selective weight decay is a crucial regularization technique for
        transformers that can improve performance and stability.

    LLM Pipeline Fit:
        Called by the `Trainer` to construct the optimizer before starting
        the training loop. It takes training arguments (`args`) to configure
        optimizer-specific parameters.

    Args:
        model: The `torch.nn.Module` (LLM) to be optimized.
        args: An instance of `TrainingArguments` containing optimization
              hyperparameters like `learning_rate`, `weight_decay`, `adam_beta1`, etc.
        optimizer_cls: Optional. The specific optimizer class to use. If `None`,
                       `torch.optim.AdamW` is used by default.

    Returns:
        An initialized `torch.optim.Optimizer` instance.

    Raises:
        TypeError: If `args` is not an instance of `TrainingArguments` (after deferred import).
    """
    # Deferring the import here to prevent circular dependency
    from llm_pipeline.training.trainer import TrainingArguments
    
    # TODO, maybe add a type check if 'args' could be anything else
    if not isinstance(args, TrainingArguments): # Runtime check for the type of args.
        # This check confirms that 'args' is indeed the expected type at runtime
        # TODO, maybe adjust this error message or logging based on your needs
        logger.error(f"Expected TrainingArguments instance, but received {type(args)}")
        raise TypeError("Argument 'args' must be an instance of TrainingArguments.")


    if optimizer_cls is None: # If no specific optimizer class is provided.
        optimizer_cls = torch.optim.AdamW # Defaults to AdamW.

    # Get parameters with weight decay
    # Parameters in LayerNorm and biases are typically excluded from weight decay.
    decay_parameters = get_parameter_names(model, [nn.LayerNorm]) # Gets names of parameters for weight decay.
    decay_parameters = [name for name in decay_parameters if "bias" not in name] # Excludes bias parameters.

    optimizer_grouped_parameters = [ # Defines parameter groups for the optimizer.
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters], # Parameters to apply weight decay.
            "weight_decay": args.weight_decay, # Applies specified weight decay.
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters], # Parameters NOT to apply weight decay.
            "weight_decay": 0.0, # Sets weight decay to 0.
        },
    ]

    # Create optimizer
    optimizer_kwargs = { # Base kwargs for any optimizer.
        "lr": args.learning_rate, # Sets the initial learning rate.
    }

    if optimizer_cls in [torch.optim.Adam, torch.optim.AdamW]: # Adds specific kwargs for Adam/AdamW optimizers.
        optimizer_kwargs.update({
            "betas": (args.adam_beta1, args.adam_beta2), # Sets beta parameters.
            "eps": args.adam_epsilon, # Sets epsilon parameter.
        })

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs) # Instantiates the optimizer.

    logger.info(f"Created {optimizer_cls.__name__} optimizer with lr={args.learning_rate}") # Logs optimizer creation.

    return optimizer # Returns the created optimizer.


def get_parameter_names(model: nn.Module, forbidden_layer_types: list[type]) -> list[str]:
    """
    Recursively gets parameter names that should have weight decay.

    Purpose:
        Identifies which parameters in a PyTorch model should (or should not)
        have weight decay applied to them. This is often based on the type of
        layer they belong to (e.g., typically excluding LayerNorm and bias terms).

        Implements a common best practice in training deep learning models,
        where weight decay is selectively applied for better regularization.

    LLM Pipeline Fit:
        Used by `create_optimizer` to correctly group parameters for the optimizer,
        ensuring proper application of weight decay.

    Args:
        model: The `torch.nn.Module` to inspect.
        forbidden_layer_types: A list of `torch.nn.Module` types for which
                               parameters should generally *not* have weight decay.

    Returns:
        A list of string names of parameters that are eligible for weight decay.
    """
    result = [] # Initializes an empty list to store parameter names.
    for name, child in model.named_children(): # Recursively iterates through child modules.
        result += [
            f"{name}.{n}" # Constructs the full name for nested parameters.
            for n in get_parameter_names(child, forbidden_layer_types) # Recursive call for child modules.
            if not isinstance(child, tuple(forbidden_layer_types)) # Excludes parameters if the child module's type is forbidden.
        ]

    # Add model specific parameters (defined with nn.Parameter) that are directly attributes of the current module
    # and not part of a named_children submodule.
    result += list(model._parameters.keys()) # Adds parameters directly defined within this module.

    return result # Returns the list of parameter names.


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int = 0,
    **kwargs,
) -> _LRScheduler:
    """
    Create a learning rate scheduler based on the specified type.

    Purpose:
        Initializes a learning rate scheduler, which dictates how the learning
        rate changes over the course of training. This is crucial for optimizing
        convergence and performance.

        Learning rate schedules are a critical hyperparameter. This function
        provides a unified interface to create different common schedules
        (e.g., linear decay, cosine annealing) with warmup.

    LLM Pipeline Fit:
        Called by the `Trainer` to construct the learning rate scheduler.
        The scheduler's `step()` method is called periodically in the training
        loop to update the learning rate of the optimizer.

    Args:
        optimizer: The `torch.optim.Optimizer` instance for which the
                   learning rate will be scheduled.
        scheduler_type: A string indicating the type of scheduler to create
                        (e.g., "linear", "cosine", "constant_with_warmup").
        num_training_steps: The total number of training steps (or epochs if
                            scheduler steps per epoch).
        warmup_steps: The number of initial steps during which the learning
                      rate linearly increases from 0 to the initial learning rate.
        **kwargs: Additional arguments specific to certain scheduler types.

    Returns:
        An initialized `torch.optim.lr_scheduler._LRScheduler` instance.
    """
    scheduler_cls = get_scheduler_cls(scheduler_type) # Gets the appropriate scheduler class.

    scheduler = scheduler_cls( # Instantiates the scheduler.
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        warmup_steps=warmup_steps,
        **kwargs,
    )

    logger.info( # Logs scheduler creation details.
        f"Created {scheduler_type} scheduler with "
        f"num_training_steps={num_training_steps}, warmup_steps={warmup_steps}"
    )

    return scheduler # Returns the created scheduler.


class LinearScheduler(_LRScheduler):
    """
    Linear learning rate scheduler with optional warmup.

    The learning rate increases linearly from 0 during warmup_steps,
    then decreases linearly to 0 over the remaining steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps # Total number of training steps.
        self.warmup_steps = warmup_steps # Number of warmup steps.
        super().__init__(optimizer, last_epoch) # Calls the base class constructor.

    def get_lr(self) -> list[float]:
        """Calculates the learning rate for the current epoch/step."""
        if self.last_epoch < self.warmup_steps: # During warmup phase.
            # Warmup: LR increases linearly from 0 to base_lr.
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else: # After warmup, linear decay.
            # Linear decay: LR decreases linearly from base_lr to 0.
            progress = (self.last_epoch - self.warmup_steps) / ( # Progress after warmup.
                self.num_training_steps - self.warmup_steps
            )
            return [base_lr * (1 - progress) for base_lr in self.base_lrs] # Applies linear decay.


class CosineScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with optional warmup.

    The learning rate increases linearly during warmup_steps, then follows
    a cosine annealing schedule down to 0.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        num_cycles: float = 0.5, # Number of cosine cycles. 0.5 for half a cycle (decay to 0).
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps # Total number of training steps.
        self.warmup_steps = warmup_steps # Number of warmup steps.
        self.num_cycles = num_cycles # Number of cosine cycles.
        super().__init__(optimizer, last_epoch) # Calls the base class constructor.

    def get_lr(self) -> list[float]:
        """Calculates the learning rate for the current epoch/step."""
        if self.last_epoch < self.warmup_steps: # During warmup phase.
            # Warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else: # After warmup, cosine decay.
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / ( # Progress after warmup.
                self.num_training_steps - self.warmup_steps
            )
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * self.num_cycles * 2 * progress)) # Cosine annealing formula.
                for base_lr in self.base_lrs
            ]


class CosineWithRestartsScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with restarts and optional warmup.

    The learning rate increases linearly during warmup_steps, then follows
    a cosine annealing schedule with specified number of restarts.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        num_cycles: int = 1, # Number of restarts (full cycles).
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps    # Total number of training steps.
        self.warmup_steps = warmup_steps                # Number of warmup steps.
        self.num_cycles = num_cycles                    # Number of full cosine cycles (restarts).
        super().__init__(optimizer, last_epoch)         # Calls the base class constructor.

    def get_lr(self) -> list[float]:
        """Calculates the learning rate for the current epoch/step."""
        if self.last_epoch < self.warmup_steps: # During warmup phase.
            # Warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else: # After warmup, cosine with restarts.
            # Cosine with restarts
            progress = (self.last_epoch - self.warmup_steps) / ( # Progress after warmup.
                self.num_training_steps - self.warmup_steps
            )
            cycle_progress = progress * self.num_cycles % 1.0 # Current position within the current cycle.
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * cycle_progress)) # Cosine annealing formula for restarts.
                for base_lr in self.base_lrs
            ]


class PolynomialScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler with optional warmup.

    The learning rate increases linearly during warmup_steps, then decays
    polynomially to 0 with a specified power.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        power: float = 1.0, # The power of the polynomial decay.
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps    # Total number of training steps.
        self.warmup_steps = warmup_steps                # Number of warmup steps.
        self.power = power                              # The power for polynomial decay.
        super().__init__(optimizer, last_epoch)         # Calls the base class constructor.

    def get_lr(self) -> list[float]:
        """Calculates the learning rate for the current epoch/step."""
        if self.last_epoch < self.warmup_steps: # During warmup phase.
            # Warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else: # After warmup, polynomial decay.
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_steps) / ( # Progress after warmup.
                self.num_training_steps - self.warmup_steps
            )
            return [
                base_lr * (1 - progress) ** self.power # Polynomial decay formula.
                for base_lr in self.base_lrs
            ]


class ConstantScheduler(_LRScheduler):
    """
    Constant learning rate scheduler.

    The learning rate remains constant throughout training. This scheduler does
    not incorporate a warmup phase; for warmup, use `ConstantWithWarmupScheduler`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int = None, # Not used, but kept for compatibility with scheduler signature.
        warmup_steps: int = None,   # Not used, but kept for compatibility with scheduler signature.
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch) # Calls the base class constructor.

    def get_lr(self) -> list[float]:
        """Returns the base learning rate(s) constantly."""
        return self.base_lrs # Returns the initial learning rate(s) unchanged.


class ConstantWithWarmupScheduler(_LRScheduler):
    """
    Constant learning rate scheduler with warmup.

    The learning rate increases linearly from 0 during warmup_steps,
    then remains constant at the initial learning rate for the rest of training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int = None, # Not used, but kept for compatibility with scheduler signature.
        warmup_steps: int = 0,          # Number of warmup steps.
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps # Stores the number of warmup steps.
        super().__init__(optimizer, last_epoch) # Calls the base class constructor.

    def get_lr(self) -> list[float]:
        """Calculates the learning rate for the current epoch/step."""
        if self.last_epoch < self.warmup_steps: # During warmup phase.
            # Warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps # Linearly increases LR during warmup.
                for base_lr in self.base_lrs
            ]
        else: # After warmup, constant learning rate.
            # Constant
            return self.base_lrs # Returns the initial learning rate(s) unchanged.