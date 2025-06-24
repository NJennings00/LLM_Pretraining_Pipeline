# filename: src/llm_pipeline/cli/preprocess.py
"""
Data preprocessing command-line interface.

This module provides a command-line interface (CLI) for downloading, cleaning,
tokenizing, and saving text datasets for the LLM pipeline. It leverages `click`
for CLI argument parsing and `datasets` for data handling.

Purpose:
    To automate and standardize the process of preparing raw text data for
    language model training. This includes:
    - Loading datasets (e.g., WikiText).
    - Applying a series of text cleaning steps.
    - Training and saving a custom tokenizer.
    - Saving the preprocessed datasets and tokenizer artifacts to disk.
    - Generating statistics about the preprocessed data.

    Data preprocessing is often the most time-consuming and error-prone part
    of an NLP project. This CLI is crucial because it:
    1. **Ensures Consistency:** Applies a uniform preprocessing pipeline.
    2. **Handles Tokenization:** Manages the creation of a domain-specific tokenizer.
    3. **Provides Reproducibility:** By defining steps and parameters via CLI,
       the exact preprocessing can be repeated.
    4. **Facilitates Data Readiness:** Outputs data in a format suitable for
       downstream training processes.

LLM Pipeline Fit:
    This module is part of the `llm_pipeline.cli` package. It's the first step
    in the LLM pipeline's data preparation phase. It directly interacts with
    `llm_pipeline.config` for data and tokenizer settings, and `llm_pipeline.data`
    for the core preprocessing and tokenizer building logic.
"""

import logging                    # For logging progress and important messages.
from pathlib import Path          # For object-oriented filesystem path manipulation.
from typing import Optional, List # For type hinting (e.g., optional arguments, lists of strings).
import click                      # The library used to build the command-line interface.
from datasets import load_dataset # Hugging Face Datasets library for loading and managing datasets.
import json                       # For saving configuration and statistics in JSON format.
import torch                      # Imported for `torch.randperm` for efficient dataset sampling.

# Import specific components from other sub-packages of llm_pipeline.
from llm_pipeline.config import DataConfig, TokenizerConfig # Configuration classes for data and tokenizers.
from llm_pipeline.data import (                             # Functions and classes for data preprocessing and tokenization.
    build_tokenizer,
    TextPreprocessor,
    PreprocessingPipeline,
    compute_dataset_statistics, 
)
from llm_pipeline.utils import setup_logger # Utility for setting up the logging system.


logger = logging.getLogger(__name__) # Initialize a logger for this module.


@click.command() # Decorator from click to define the command-line command.
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["wikitext-2", "wikitext-103"]), # Restricts dataset choices.
    default="wikitext-2",
    help="The dataset to preprocess. Currently supports 'wikitext-2' and 'wikitext-103'.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True, # This option is mandatory.
    help="The output directory where preprocessed data, tokenizer, and configuration will be saved.",
)
@click.option(
    "--tokenizer-type",
    "-t",
    type=click.Choice(["bpe", "wordpiece", "unigram"]), # Restricts tokenizer type choices.
    default="bpe",
    help="The type of tokenizer to build (e.g., 'bpe' for Byte-Pair Encoding).",
)
@click.option(
    "--vocab-size",
    "-v",
    type=int,
    default=32000,
    help="The desired vocabulary size for the tokenizer.",
)
@click.option(
    "--min-frequency",
    type=int,
    default=2,
    help="Minimum frequency a token must have to be included in the vocabulary.",
)
@click.option(
    "--max-length",
    type=int,
    default=512,
    help="Maximum sequence length after tokenization. Texts will be truncated or padded to this length.",
)
@click.option(
    "--preprocessing-steps",
    "-p",
    multiple=True, # Allows this option to be specified multiple times, collecting values into a list.
    type=click.Choice([ # Defines allowed text preprocessing steps.
        "normalize_whitespace",
        "remove_control_characters",
        "normalize_unicode",
        "lowercase",
        "remove_urls",
        "remove_emails",
        "remove_extra_punctuation",
        "fix_encoding_errors",
    ]),
    default=[ # Default set of preprocessing steps.
        "fix_encoding_errors",
        "normalize_unicode",
        "remove_control_characters",
        "normalize_whitespace",
    ],
    help="List of text preprocessing steps to apply in order. Can be specified multiple times.",
)
@click.option(
    "--sample-size",
    type=int,
    default=10000,
    help="Number of text samples to use for training the tokenizer. Larger datasets will be sampled.",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of worker processes to use for parallelizing preprocessing tasks (e.g., dataset mapping).",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    help="Optional directory to cache downloaded datasets. If not provided, Hugging Face Datasets default cache will be used.",
)
def preprocess_command(
    dataset: str,
    output: Path,
    tokenizer_type: str,
    vocab_size: int,
    min_frequency: int,
    max_length: int,
    preprocessing_steps: List[str],
    sample_size: int,
    num_workers: int,
    cache_dir: Optional[Path],
):
    """
    Preprocess dataset and build tokenizer.

    This command downloads a specified dataset (e.g., WikiText), applies a series
    of text cleaning transformations, trains a new tokenizer (BPE, WordPiece, or Unigram)
    based on the preprocessed data, and saves all artifacts to the specified
    output directory. It also computes and saves basic statistics about the
    preprocessed dataset.
    """
    # Setup logging for the CLI script.
    setup_logger(log_level=logging.INFO)
    
    # Ensure the output directory exists, creating parent directories if necessary.
    output.mkdir(parents=True, exist_ok=True)
    
    # --- Load Raw Dataset ---
    logger.info(f"Loading {dataset} dataset...")
    
    # Map CLI dataset choice to Hugging Face `datasets` library's dataset name and config.
    if dataset == "wikitext-2":
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"
    else: # Defaulting to wikitext-103 if not wikitext-2
        dataset_name = "wikitext"
        dataset_config = "wikitext-103-raw-v1"
    
    # Load the raw dataset splits (train, validation, test).
    raw_datasets = load_dataset(
        dataset_name,
        dataset_config,
        cache_dir=cache_dir, # Use the specified cache directory.
    )
    
    logger.info(f"Dataset loaded: {raw_datasets}")
    
    # --- Create Preprocessing Pipeline ---
    logger.info("Creating preprocessing pipeline...")
    preprocessing_funcs = []
    
    # Dynamically build the list of preprocessing functions based on user selection.
    for step in preprocessing_steps:
        if hasattr(TextPreprocessor, step): # Check if the method exists in TextPreprocessor.
            preprocessing_funcs.append(getattr(TextPreprocessor, step)) # Add the method to the list.
        else:
            logger.warning(f"Unknown preprocessing step: {step}. Skipping this step.")
    
    # Initialize the PreprocessingPipeline with the selected functions.
    preprocessing_pipeline = PreprocessingPipeline(preprocessing_funcs)
    
    # --- Apply Preprocessing to Dataset ---
    logger.info("Preprocessing texts...")
    
    # Define a function to apply the pipeline to a batch of examples.
    def preprocess_function(examples):
        return {
            "text": [preprocessing_pipeline(text) for text in examples["text"]]
        }
    
    # Apply the preprocessing function to all splits of the dataset using `map`.
    preprocessed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,                   # Process in batches for efficiency.
        num_proc=num_workers,           # Use multiple processes for faster execution.
        desc="Preprocessing raw texts", # Description for progress bar.
    )
    
    # --- Filter Empty Texts ---
    # Define a filter function to remove examples where text becomes empty after preprocessing.
    def filter_empty(example):
        return len(example["text"].strip()) > 0
    
    # Apply the filter to remove empty strings.
    preprocessed_datasets = preprocessed_datasets.filter(
        filter_empty,
        num_proc=num_workers,
        desc="Filtering empty texts",
    )
    
    # --- Save Preprocessed Datasets ---
    preprocessed_path = output / "preprocessed"
    logger.info(f"Saving preprocessed datasets to {preprocessed_path}")
    # Save the processed datasets to disk in the Hugging Face Datasets format.
    preprocessed_datasets.save_to_disk(str(preprocessed_path))
    
    # --- Prepare Sample for Tokenizer Training ---
    logger.info("Sampling texts for tokenizer training...")
    train_dataset = preprocessed_datasets["train"]
    
    # Take a sample of the training data for tokenizer training if the dataset is large.
    if len(train_dataset) > sample_size:
        # Use torch.randperm for random sampling of indices.
        indices = torch.randperm(len(train_dataset))[:sample_size].tolist()
        sample_texts = [train_dataset[i]["text"] for i in indices]
    else:
        sample_texts = train_dataset["text"] # Use all texts if dataset is smaller than sample_size.
    
    # --- Build and Save Tokenizer ---
    # Create a TokenizerConfig object from CLI arguments.
    tokenizer_config = TokenizerConfig(
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    
    logger.info("Building tokenizer...")
    tokenizer = build_tokenizer(tokenizer_config, sample_texts) # Train the tokenizer.
    
    # Save the trained tokenizer to the specified output directory.
    tokenizer_path = output / "tokenizer"
    logger.info(f"Saving tokenizer to {tokenizer_path}")
    tokenizer.save(tokenizer_path)
    
    # --- Test Tokenizer ---
    logger.info("Testing tokenizer...")
    test_text = "This is a test sentence for the tokenizer."
    encoded = tokenizer.encode(test_text) # Encode a test sentence.
    decoded = tokenizer.decode(encoded)   # Decode it back.
    
    logger.info(f"Original: {test_text}")
    logger.info(f"Encoded: {encoded}")
    logger.info(f"Decoded: {decoded}")
    
    # --- Compute Dataset Statistics ---
    logger.info("Computing dataset statistics...")
    
    stats = {}
    for split_name, split_dataset in preprocessed_datasets.items():
        logger.info(f"Processing {split_name} split...")
        
        # Calculate statistics based on word count (for a sample of the dataset).
        # This gives a quick estimate of text lengths.
        # Limiting to 10000 examples for potentially very large datasets to prevent excessive memory/time usage.
        text_lengths = [len(text.split()) for text in split_dataset["text"][:10000]]
        
        stats[split_name] = {
            "num_examples": len(split_dataset),
            "avg_words": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "min_words": min(text_lengths) if text_lengths else 0,
            "max_words": max(text_lengths) if text_lengths else 0,
        }
        
        # Calculate statistics based on token count (for a smaller sample due to tokenization overhead).
        # Limiting to 1000 examples for token statistics.
        token_lengths = []
        for i in range(min(1000, len(split_dataset))):
            tokens = tokenizer.encode(split_dataset[i]["text"])
            token_lengths.append(len(tokens))
        
        stats[split_name].update({
            "avg_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
            "min_tokens": min(token_lengths) if token_lengths else 0,
            "max_tokens": max(token_lengths) if token_lengths else 0,
        })
    
    # Save the computed statistics to a JSON file.
    stats_path = output / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_path}")
    
    # --- Save Preprocessing Configuration ---
    # Store all relevant parameters used for preprocessing and tokenizer building.
    config = {
        "dataset": dataset,
        "tokenizer": tokenizer_config.__dict__, # Convert tokenizer config object to dict for saving.
        "preprocessing_steps": preprocessing_steps,
        "max_length": max_length,
        "statistics": stats, # Include computed statistics in the config.
    }
    
    config_path = output / "preprocessing_config.json"
    with open(config_path, "w") as f:
        # `default=str` is used to handle Path objects or other non-serializable types if any.
        json.dump(config, f, indent=2, default=str) 
    
    logger.info(f"Configuration saved to {config_path}")
    
    # --- Print Summary ---
    logger.info("\nPreprocessing Summary:")
    logger.info("-" * 50)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Tokenizer: {tokenizer_type} (vocab size: {vocab_size})")
    logger.info(f"Preprocessing steps: {', '.join(preprocessing_steps)}")
    
    # Print formatted statistics for each split.
    for split_name, split_stats in stats.items():
        logger.info(f"\n{split_name} split:")
        for key, value in split_stats.items():
            logger.info(f"  {key}: {value:,.2f}" if isinstance(value, float) else f"  {key}: {value:,}")
    
    logger.info(f"\nOutput directory: {output}")
    logger.info("Preprocessing completed successfully!")


def main():
    """Main entry point for the preprocessing CLI."""
    preprocess_command()


if __name__ == "__main__":
    main() # Execute the command when the script is run directly.