# filename: src/llm_pipeline/config/__init__.py
"""
Configuration modules for the LLM pipeline.

This `__init__.py` file serves as the public API for the `llm_pipeline.config`
package. It re-exports key configuration dataclasses and utility functions,
making them directly accessible when importing from `llm_pipeline.config`.

Purpose:
    To provide a clean and convenient interface for accessing configuration
    definitions and related utilities. Instead of requiring users to import
    from nested modules like `llm_pipeline.config.config` or `llm_pipeline.config.utils`,
    they can simply import from `llm_pipeline.config`.

    This file is fundamental for the maintainability and usability of the
    configuration system. It's used because it:
    1. **Simplifies Imports:** Reduces verbosity for users importing config classes.
    2. **Defines Public API:** Clearly outlines which configuration components
       and utilities are intended for external use within the `llm_pipeline`
       package and by external scripts.
    3. **Encourages Modularity:** Allows the actual implementation details
       (e.g., specific dataclass definitions or utility functions) to reside
       in separate files while presenting a unified interface.

LLM Pipeline Fit:
    The `llm_pipeline` relies heavily on structured configuration. This `__init__.py`
    file makes these structures readily available to other parts of the pipeline,
    such as the CLI for loading configurations, or the training and evaluation
    modules for type-hinting and accessing specific parameters. It acts as the
    gateway to the entire configuration system.
"""

# Import and re-export all individual configuration dataclasses from `config.py`.
# These classes define the structure and default values for various aspects of the LLM pipeline.
from llm_pipeline.config.config import (
    Config,             # The main, top-level configuration class that composes all others.
    DataConfig,         # Configuration for data loading and preprocessing.
    ModelConfig,        # Configuration for the transformer model's architecture.
    TrainingConfig,     # Configuration for the training process hyperparameters.
    TokenizerConfig,    # Configuration for tokenizer creation and behavior.
    EvaluationConfig,   # Configuration for the model evaluation process and metrics.
    LoggingConfig,      # Configuration for logging and experiment monitoring.
)

# Import and re-export selected utility functions from `config.utils`.
# These functions provide common operations like loading, saving, merging, and validating configurations.
from llm_pipeline.config.utils import (
    save_config,     # Function to save a Config object to a file.
    merge_configs,   # Function to deeply merge two configuration dictionaries/objects.
    validate_config, # Function to perform cross-component validation on a Config object.
    load_config,     # Function to load a Config object from a file (YAML or JSON).
)

# Define `__all__` to explicitly list the public API of the `llm_pipeline.config` package.
# This specifies what symbols are imported when a user does `from llm_pipeline.config import *`.
# It's good practice for clarity and preventing unintended imports.
__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "TokenizerConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "load_config",     
    "save_config",     
    "merge_configs",   
    "validate_config", 
]