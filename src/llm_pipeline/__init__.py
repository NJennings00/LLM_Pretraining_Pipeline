# filename: src/llm_pipeline/__init__.py
"""
LLM Pretraining Pipeline

A complete end-to-end pipeline for pretraining transformer-based language models.

This `__init__.py` file serves as the top-level package initializer for the
`llm_pipeline`. Its primary role is to define the public interface of the
`llm_pipeline` package by re-exporting key components from its sub-modules.

Purpose:
    To provide a clean and direct way to import the most essential classes and
    functions required to build and run an LLM pretraining job, without needing
    to specify the full sub-module path (e.g., `from llm_pipeline import Config`
    instead of `from llm_pipeline.config import Config`).

    Like other `__init__.py` files that re-export symbols, this file primarily
    ensures that the intended components are correctly exposed. It's used as the
    entry point for users and other parts of the system that need to access the
    core functionalities of the LLM pipeline, such as:
    1. **Configuration Management:** Loading and handling experiment configurations.
    2. **Data Handling:** Preparing datasets and data loaders.
    3. **Model Definition:** Accessing the core language model architecture.

LLM Pipeline Fit:
    This file is at the root of the `llm_pipeline` package, making it central
    to how the entire pipeline is consumed. It brings together the major building
    blocks of the LLM pretraining process, allowing for a more streamlined and
    user-friendly experience when interacting with the library.
"""

# Re-export configuration classes and functions from the `config` sub-package.
# `Config` is likely the main configuration class, and `load_config` is for
# loading configurations from files (e.g., YAML, OmegaConf).
from llm_pipeline.config import Config, load_config

# Re-export data-related classes from the `data` sub-package.
# `WikiTextDataset` likely handles loading and preparing the WikiText dataset,
# and `DataCollator` is probably responsible for collating (batching and padding)
# data samples for model input.
from llm_pipeline.data import WikiTextDataset, DataCollator

# Re-export the core model class from the `models` sub-package.
# `TransformerLM` represents the transformer-based language model architecture.
from llm_pipeline.models import TransformerLM

# Define `__all__` to explicitly list the public API of the `llm_pipeline` package.
# This makes it clear which components are intended for direct import and use
# by users of this library.
__all__ = [
    "Config",          # Core configuration class.
    "load_config",     # Function to load configuration settings.
    "WikiTextDataset", # Dataset class for WikiText.
    "DataCollator",    # Utility for batching and preparing data.
    "TransformerLM",   # The main transformer language model class.
]