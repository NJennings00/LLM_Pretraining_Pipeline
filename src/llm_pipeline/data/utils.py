# filename: src/llm_pipeline/data/utils.py
"""
This module provides essential data utilities and helper functions for the LLM pretraining pipeline.
It handles common data management tasks such as:
1.  **Creating PyTorch DataLoaders:** Facilitating batching, shuffling, and multi-process data loading
    for efficient training and evaluation.
2.  **Splitting Datasets:** Dividing a single dataset into training, validation, and test subsets
    to ensure proper model evaluation and prevent data leakage.
3.  **Computing Dataset Statistics:** Analyzing characteristics of the processed data, such as
    sequence lengths and token frequencies, which are valuable for understanding the corpus
    and debugging.
4.  **Caching Datasets:** Saving and loading preprocessed datasets to disk to avoid redundant
    computation during multiple training runs or pipeline restarts.

In the LLM pretraining pipeline, these utilities are critical for the **efficient and robust flow of data**
from raw text to the model's input layer. They ensure that data is presented to the training
framework (PyTorch) in an optimized format, that the training and evaluation splits are
handled reproducibly, and that repetitive, time-consuming data processing steps can be
cached. This module directly supports the `training` components by preparing data for the
`Trainer` and helps in `evaluation` by providing structured data for metrics computation.
"""

import logging                                                 # Imports the logging library for structured logging.
from typing import Any, Dict, List, Optional, Tuple, Union     # Imports typing hints for better code readability and validation.
from pathlib import Path                                       # Imports Path for object-oriented filesystem paths.
import torch                                                   # Imports PyTorch for tensor operations.
from torch.utils.data import DataLoader, Dataset, random_split # Imports PyTorch's DataLoader, Dataset base class, and dataset splitting utility.
import numpy as np                                             # Imports NumPy for numerical operations, particularly for statistics.
from collections import Counter                                # Imports Counter for counting hashable objects (used for token frequencies).

from llm_pipeline.config import DataConfig, TrainingConfig     # Imports the DataConfig dataclass for data-related configurations.
from llm_pipeline.data.collator import DataCollator # Imports the DataCollator abstract base class for type hinting.


logger = logging.getLogger(__name__) # Initializes a logger for this module.         

def create_dataloaders(
    train_dataset: Dataset,                     # The PyTorch Dataset object for training data.
    eval_dataset: Optional[Dataset],            # Optional PyTorch Dataset object for evaluation (validation/test data).
    collator: DataCollator,                     # An instance of a DataCollator (e.g., DataCollatorForLanguageModeling) to batch and process features.
    data_config: DataConfig,                    # A DataConfig object containing batch sizes and worker configurations.
    training_config: TrainingConfig, 
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Creates PyTorch `DataLoader` instances for the training and (optionally)
    evaluation datasets. DataLoaders are crucial for efficient batching,
    shuffling (for training), and loading data from datasets in a multi-process manner.

    Why it's needed: DataLoaders are the standard way to feed data to PyTorch models
    during training. They abstract away the complexities of data fetching and batching,
    allowing the training loop to simply iterate over batches. Their correct
    configuration (batch size, shuffling, number of workers) is vital for
    training efficiency and model performance.

    How it fits into the LLM pipeline: This function prepares the final input
    streams for the model. The `Trainer` component will consume these DataLoaders
    to get batches of processed `input_ids`, `attention_mask`, and `labels` for
    each training step. It confirms the data is ready for the model's training loop.

    Inputs:
    - train_dataset (Dataset): The dataset containing training examples.
    - eval_dataset (Optional[Dataset]): The dataset containing evaluation examples; can be None.
    - collator (DataCollator): The collator responsible for transforming individual dataset items into batches.
    - data_config (DataConfig): Configuration for batch sizes, number of workers, etc.

    Outputs:
    - Tuple[DataLoader, Optional[DataLoader]]: A tuple containing the training DataLoader
                                                and the optional evaluation DataLoader.
    """
    logger.info("Creating data loaders...")

    # Create the DataLoader for the training dataset.
    train_dataloader = DataLoader(                      # Instantiates a PyTorch DataLoader.
        train_dataset,                                  # The dataset to load data from.
        batch_size=training_config.train_batch_size,    # Sets the number of samples per batch for training.
        shuffle=True,                                   # Shuffles the data at each epoch for better generalization during training.
        collate_fn=collator,                            # Specifies the function to use to form a batch from a list of samples.
        num_workers=data_config.num_workers,            # Sets the number of subprocesses to use for data loading.
        pin_memory=data_config.pin_memory,              # If True, copies Tensors to CUDA pinned memory before returning.
        drop_last=data_config.drop_last,                # If True, drops the last incomplete batch if its size is less than `batch_size`.
    )

    # Create the DataLoader for the evaluation dataset if one is provided.
    eval_dataloader = None                                  # Initializes the evaluation dataloader variable.
                                                            # Create eval dataloader only if an evaluation dataset is provided.
    if eval_dataset is not None:                            # Checks if an evaluation dataset is available.
        eval_dataloader = DataLoader(                       # Instantiates a PyTorch DataLoader for evaluation.
            eval_dataset,                                   # The dataset to load data from for evaluation.
            batch_size=training_config.eval_batch_size,     # Sets the number of samples per batch for evaluation.
            shuffle=False,                                  # Does not shuffle evaluation data for consistent results across epochs.
            collate_fn=collator,                            # Specifies the collate function.
            num_workers=data_config.num_workers,            # Sets the number of subprocesses for loading.
            pin_memory=data_config.pin_memory,              # Enables pinned memory if configured.
            drop_last=False,                                # Does not drop the last batch for evaluation to ensure all data is covered.
        )

    return train_dataloader, eval_dataloader

def split_dataset(
    dataset: Dataset,          # The input PyTorch Dataset to be split.
    train_ratio: float = 0.9,  # The proportion of data to allocate to the training set.
    val_ratio: float = 0.05,   # The proportion of data to allocate to the validation set.
    test_ratio: float = 0.05,  # The proportion of data to allocate to the test set.
    seed: int = 42,            # The random seed for reproducibility of the split.
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits a given PyTorch `Dataset` into three subsets: training, validation, and test.
    Ensures that the sum of ratios is approximately 1.0. The splitting is reproducible
    due to the use of a fixed random seed.

    Why it's needed: A standard practice in machine learning is to split data into
    distinct sets to:
    - Train (train set): For model parameter optimization.
    - Validate (validation set): For hyperparameter tuning and early stopping, preventing overfitting to the train set.
    - Test (test set): For a final, unbiased evaluation of the model's generalization performance on unseen data.
    Reproducible splits are crucial for comparing different model configurations or experiments.

    How it fits into the LLM pipeline: This function prepares the distinct datasets
    that will then be used by `create_dataloaders` to form the training, validation,
    and testing data streams. It confirms the integrity and reproducibility of the data splits.

    Inputs:
    - dataset (Dataset): The full dataset to be divided.
    - train_ratio (float): Desired proportion for the training set.
    - val_ratio (float): Desired proportion for the validation set.
    - test_ratio (float): Desired proportion for the test set.
    - seed (int): Random seed for reproducibility.

    Outputs:
    - Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and test subsets.
    """
    logger.info("Splitting dataset into train, validation, and test sets...") # Logs the start of dataset splitting.

    # Assert that the sum of ratios is approximately 1.0 to ensure correct partitioning.
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0" # Raises an AssertionError if ratios do not sum to 1.

    total_size = len(dataset)                       # Gets the total number of examples in the dataset.
    train_size = int(train_ratio * total_size)      # Calculates the size of the training set.
    val_size = int(val_ratio * total_size)          # Calculates the size of the validation set.
    test_size = total_size - train_size - val_size  # Calculates the remaining size for the test set to account for rounding.
    
    # Use random_split with a PyTorch generator for reproducibility.
    # A `torch.Generator` initialized with a fixed seed ensures the same split every time.
    generator = torch.Generator().manual_seed(seed)           # Creates a PyTorch random number generator with a fixed seed.
    train_dataset, val_dataset, test_dataset = random_split(  # Performs the random split.
        dataset,                                              # The dataset to split.
        [train_size, val_size, test_size],                    # A list specifying the size of each split.
        generator=generator                                   # The random number generator to use for reproducibility.
    )
    
    logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}") # Logs the sizes of the created splits.
    
    return train_dataset, val_dataset, test_dataset # Returns the three split datasets.


def compute_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """
    Computes and returns various statistics about a given dataset, focusing on
    sequence lengths and token frequencies. For token frequencies, it samples
    a subset of the dataset to manage computation time for very large datasets.

    Why it's needed: Understanding the characteristics of the data (e.g., average
    sequence length, distribution of token IDs, most common tokens) is vital for:
    - **Debugging:** Identifying potential issues in data preprocessing or tokenization.
    - **Hyperparameter Tuning:** Informing decisions about `max_seq_length`, vocabulary size, etc.
    - **Reporting:** Providing insights into the dataset used for pretraining.

    How it fits into the LLM pipeline: This utility is typically run after datasets
    have been loaded and preprocessed, providing a summary of the data that will
    actually be fed into the model. It confirms the quality and properties of the
    data being used for training.

    Inputs:
    - dataset (Dataset): The PyTorch Dataset object to analyze (expected to return
                         dictionaries with "input_ids" when indexed).

    Outputs:
    - Dict[str, Any]: A dictionary containing various statistics including:
        - "num_examples": Total number of examples in the dataset.
        - "sample_size": Number of examples sampled for token statistics.
        - "sequence_lengths": Dictionary with mean, std, min, max, and median of sequence lengths.
        - "token_statistics": Dictionary with unique token count, total token count, and most common tokens.
    """
    logger.info("Computing dataset statistics...") # Logs the start of statistics computation.
    
    lengths = []             # List to store the lengths of individual input_ids sequences.
    token_counts = Counter() # A Counter object to count occurrences of each token ID.
    
    # Determine a sample size for token statistics to avoid excessive computation
    # for very large datasets, limiting it to a maximum of 10,000 examples.
    sample_size = min(len(dataset), 10000) # Sets the sample size, capped at 10,000.
    # Randomly choose indices to sample without replacement.
    indices = np.random.choice(len(dataset), sample_size, replace=False) # Selects `sample_size` unique random indices.
    
    # Iterate through the sampled indices to collect lengths and token counts.
    for idx in indices:               # Loops through each sampled index.
        item = dataset[idx]           # Retrieves the dataset item at the current index.
        input_ids = item["input_ids"] # Extracts the input_ids tensor from the item.
        
        if isinstance(input_ids, torch.Tensor): # Checks if input_ids is a PyTorch tensor.
            input_ids = input_ids.tolist()      # Converts the tensor to a Python list of integers for easier length calculation and counting.
        
        lengths.append(len(input_ids)) # Appends the length of the current sequence to the lengths list.
        token_counts.update(input_ids) # Updates the token_counts counter with tokens from the current sequence.
    
    # Compute and organize the statistics into a dictionary.
    stats = {                                    # Dictionary to hold all computed statistics.
        "num_examples": len(dataset),            # Total number of examples in the entire dataset.
        "sample_size": sample_size,              # Number of examples actually sampled for detailed token statistics.
        "sequence_lengths": {                    # Statistics related to sequence lengths.
            "mean": float(np.mean(lengths)),     # Mean (average) sequence length.
            "std": float(np.std(lengths)),       # Standard deviation of sequence lengths.
            "min": int(np.min(lengths)),         # Minimum sequence length.
            "max": int(np.max(lengths)),         # Maximum sequence length.
            "median": float(np.median(lengths)), # Median sequence length.
        },
        "token_statistics": {                            # Statistics related to token distribution.
            "unique_tokens": len(token_counts),          # Number of unique token IDs encountered in the sample.
            "total_tokens": sum(token_counts.values()),  # Total number of tokens in the sample.
            "most_common": token_counts.most_common(20), # 20 most frequently occurring tokens and their counts.
        },
    }
    
    logger.info("Dataset statistics computed.") # Logs completion of statistics computation.
    return stats                                # Returns the dictionary of statistics.


def save_dataset_cache(dataset: Dataset, cache_path: Union[str, Path]) -> None:
    """
    Saves a processed PyTorch `Dataset` object to disk using `torch.save()`.
    This allows for rapid reloading of preprocessed data, avoiding time-consuming
    recomputation (e.g., tokenization) in subsequent runs.

    Why it's needed: Preprocessing large datasets can be computationally expensive
    and time-consuming. Caching saves time and resources, especially during
    iterative development or resuming training.

    How it fits into the LLM pipeline: This utility is typically called after
    a dataset (like `WikiTextDataset` after its `_preprocess_and_tokenize` step)
    has been fully prepared. It confirms that the processed data can be persisted.

    Inputs:
    - dataset (Dataset): The PyTorch Dataset object to be saved.
    - cache_path (Union[str, Path]): The full file path (including filename) where the dataset will be saved.
    Outputs:
    - None.
    """
    cache_path = Path(cache_path)                        # Converts the cache_path to a Path object for easier manipulation.
    cache_path.parent.mkdir(parents=True, exist_ok=True) # Ensures the parent directories for the cache file exist.
    
    logger.info(f"Saving dataset cache to {cache_path}") # Logs the path where the cache is being saved.
    torch.save(dataset, cache_path)                      # Saves the entire dataset object to the specified file using PyTorch's serialization.
    logger.info("Dataset cache saved.")                  # Logs successful saving.


def load_dataset_cache(cache_path: Union[str, Path]) -> Optional[Dataset]:
    """
    Loads a cached PyTorch `Dataset` object from disk using `torch.load()`.
    If the cache file does not exist, it returns `None`.

    Why it's needed: Complements `save_dataset_cache` by allowing the pipeline
    to quickly load previously prepared data, significantly reducing startup
    time and resource consumption for iterative training runs.

    How it fits into the LLM pipeline: This utility is called at the beginning
    of the data preparation phase to check if preprocessed data is already available.
    It confirms that previously saved processed data can be reliably loaded.

    Inputs:
    - cache_path (Union[str, Path]): The full file path (including filename) from which to load the dataset.
    Outputs:
    - Optional[Dataset]: The loaded PyTorch Dataset object if the cache file exists, otherwise None.
    """
    cache_path = Path(cache_path) # Converts the cache_path to a Path object.
    
    if not cache_path.exists(): # Checks if the specified cache file actually exists on disk.
        logger.info(f"No dataset cache found at {cache_path}. Proceeding with fresh processing.") # Logs if no cache is found.
        return None # Returns None if the cache file does not exist.
    
    logger.info(f"Loading dataset cache from {cache_path}") # Logs the path from which the cache is being loaded.
    loaded_dataset = torch.load(cache_path)                 # Loads the dataset object from the specified file.
    logger.info("Dataset cache loaded.")                    # Logs successful loading.
    return loaded_dataset                                   # Returns the loaded dataset.
