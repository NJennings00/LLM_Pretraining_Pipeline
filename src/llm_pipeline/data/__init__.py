# filename: src/llm_pipeline/data/__init__.py
"""
This `__init__.py` file serves as the package initializer for the `llm_pipeline.data` module.
Its primary purpose is to expose key classes and functions from sub-modules (`dataset`, `preprocessing`,
`tokenizer`, `collator`, `utils`) directly under the `llm_pipeline.data` namespace.

This design pattern simplifies imports for other parts of the LLM pretraining pipeline. Instead of
having to import specific classes from deeply nested paths (e.g., `from llm_pipeline.data.dataset import WikiTextDataset`),
users can import them directly from the top-level data package (e.g., `from llm_pipeline.data import WikiTextDataset`).

This file is crucial for organizing the data-related components of the pipeline, making them
easily discoverable and accessible. It defines the public interface of the `data` package,
which is a fundamental part of the LLM pretraining process as it handles all aspects
of data loading, preprocessing, tokenization, and batching.
"""

# Import specific classes and functions from their respective sub-modules.
# These imports make them directly accessible when 'llm_pipeline.data' is imported.

# Imports dataset classes responsible for loading and managing text data.
from llm_pipeline.data.dataset import WikiTextDataset, TextDataset
# Imports preprocessing utilities for cleaning and normalizing raw text.
from llm_pipeline.data.preprocessing import TextPreprocessor, PreprocessingPipeline
# Imports tokenizer utilities for converting text to token IDs and vice-versa, and for building tokenizers.
from llm_pipeline.data.tokenizer import TokenizerWrapper, build_tokenizer
# Imports data collator classes that handle batching and preparing data for model input.
from llm_pipeline.data.collator import DataCollator, DataCollatorForLanguageModeling
# Imports general data utility functions for creating dataloaders, splitting datasets, and computing statistics.
from llm_pipeline.data.utils import (
    create_dataloaders,         # Function to create PyTorch DataLoaders.
    split_dataset,              # Function to split datasets into train/validation/test sets.
    compute_dataset_statistics, # Function to compute statistics on a dataset.
)

# The __all__ variable defines the public API of the package.
# When a user does `from llm_pipeline.data import *`, only the names listed here will be imported.
# This explicitly declares which components are intended for external use.
# It ensures clarity about the public interface of the `data` package,
# which represents all core functionalities related to data handling in the LLM pipeline.
__all__ = [
    "WikiTextDataset",                 # Publicly exposes WikiTextDataset for loading large text corpora.
    "TextDataset",                     # Publicly exposes TextDataset for handling generic text lists.
    "TextPreprocessor",                # Publicly exposes TextPreprocessor for direct access to individual cleaning functions.
    "PreprocessingPipeline",           # Publicly exposes PreprocessingPipeline for composing text cleaning steps.
    "TokenizerWrapper",                # Publicly exposes TokenizerWrapper for interacting with tokenizers.
    "build_tokenizer",                 # Publicly exposes build_tokenizer for creating and training tokenizers.
    "DataCollator",                    # Publicly exposes the base DataCollator class.
    "DataCollatorForLanguageModeling", # Publicly exposes the specific language modeling data collator.
    "create_dataloaders",              # Publicly exposes the function to create data loaders.
    "split_dataset",                   # Publicly exposes the function to split datasets.
    "compute_dataset_statistics",      # Publicly exposes the function to compute dataset statistics.
]
