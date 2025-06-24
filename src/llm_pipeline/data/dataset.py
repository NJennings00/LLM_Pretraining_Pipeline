# filename: src/llm_pipeline/data/dataset.py
"""
This module provides dataset implementations for preparing text data for language model pretraining.
It includes a basic `TextDataset` for in-memory lists of texts and a `WikiTextDataset` for handling
larger, Hugging Face-style datasets, including features like batch processing, chunking of long
documents, and filtering based on sequence length.

These datasets serve as the crucial input layer for the entire LLM pretraining pipeline. They
are responsible for transforming raw text into tokenized numerical inputs (`input_ids`) and
their corresponding `attention_mask` tensors, which are directly consumed by the transformer model.
Correct implementation of these datasets ensures that data is prepared in the expected format
and dimensions, preventing downstream errors in the model, training loop, and evaluation.
"""

import logging                                                        # Imports the logging library for structured logging.
from typing import Optional, Union, Any, Callable                     # Imports typing hints for better code readability and validation.
from pathlib import Path                                              # Imports Path for object-oriented filesystem paths.
import torch                                                          # Imports PyTorch for tensor operations.
from torch.utils.data import Dataset                                  # Imports PyTorch's base Dataset class.
from datasets import load_dataset, DatasetDict, Dataset as HFDataset  # Imports Hugging Face datasets library components.
import numpy as np                                                    # Imports NumPy for numerical operations, particularly for statistics.
from tqdm import tqdm                                                 # Imports tqdm for progress bars during data processing.

from llm_pipeline.config import DataConfig                            # Imports the DataConfig dataclass for dataset configuration.
from llm_pipeline.data.preprocessing import PreprocessingPipeline     # Imports the PreprocessingPipeline for text cleaning.
from llm_pipeline.data.tokenizer import TokenizerWrapper              # Imports the TokenizerWrapper for tokenization.


logger = logging.getLogger(__name__) # Initializes a logger for this module.


class TextDataset(Dataset):
    """
    A foundational dataset class for handling lists of text strings.
    It prepares individual text examples by tokenizing them into numerical
    input IDs and attention masks, ensuring they are ready for a PyTorch model.

    This class provides the most direct way to feed arbitrary text
    data into the tokenization process and prepares it as
    PyTorch tensors. Its correct functioning is essential for basic data
    handling, and it can serve as a base for custom, smaller datasets or
    testing scenarios. It confirms that individual text strings can be
    correctly transformed into the numerical input format required by the model.

    Inputs:
    - texts (list[str]): A list of raw text strings to be processed.
    - tokenizer (TokenizerWrapper): An instance of the tokenizer responsible for converting text to token IDs.
    - max_length (int): The maximum sequence length for tokenized inputs; texts will be truncated or padded to this length.
    - preprocessing_fn (Optional[Callable[[str], str]]): An optional function to apply text preprocessing before tokenization.

    Outputs:
    - __len__ (int): The number of text samples in the dataset.
    - __getitem__ (dict[str, torch.Tensor]): A dictionary containing "input_ids" and "attention_mask" as PyTorch tensors for a given index.
    """
    
    def __init__(
        self,
        texts: list[str],                                        # List of text strings that the dataset will contain.
        tokenizer: TokenizerWrapper,                             # The tokenizer to use for converting texts to token IDs.
        max_length: int = 512,                                   # The maximum length to which sequences will be padded/truncated.
        preprocessing_fn: Optional[Callable[[str], str]] = None, # An optional function for text cleaning/normalization.
    ):
        """
        Initializes the TextDataset with provided texts, tokenizer, maximum length, and an optional preprocessing function.
        """
        self.texts = texts                        # Stores the input texts.
        self.tokenizer = tokenizer                # Stores the tokenizer instance.
        self.max_length = max_length              # Stores the maximum sequence length.
        self.preprocessing_fn = preprocessing_fn  # Stores the preprocessing function.
        
    def __len__(self) -> int:
        """
        Returns the total number of text examples in the dataset.
        This method is required by PyTorch's `Dataset` class.
        """
        return len(self.texts) # Returns the number of texts provided at initialization.
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieves a single processed item (tokenized text and attention mask) from the dataset.
        This method is required by PyTorch's `Dataset` class and is called by `DataLoader`.

        Inputs:
        - idx (int): The index of the item to retrieve.

        Outputs:
        - dict[str, torch.Tensor]: A dictionary containing:
            - "input_ids" (torch.Tensor): A 1D tensor of token IDs.
            - "attention_mask" (torch.Tensor): A 1D tensor indicating actual tokens (1) and padding (0).
        """
        text = self.texts[idx] # Retrieves the raw text string at the given index.
        
        if self.preprocessing_fn is not None:  # Checks if a preprocessing function is provided.
            text = self.preprocessing_fn(text) # Applies the preprocessing function to the text.
        
        # Tokenize the text, applying truncation and padding as specified.
        # `padding="max_length"` ensures the output is always `self.max_length` long.
        # `return_tensors="pt"` ensures the output is PyTorch tensors.
        encoded = self.tokenizer(        # Calls the tokenizer on the text.
            text,                        # The text string to tokenize.
            max_length=self.max_length,  # Truncates/pads to this length.
            truncation=True,             # Enables truncation if the text is longer than max_length.
            padding="max_length",        # Ensures padding to max_length if the text is shorter.
            return_tensors="pt",         # Specifies that the output should be PyTorch tensors.
        )
        
        # Squeeze the batch dimension (if present) from the tokenized outputs,
        # as __getitem__ is expected to return single examples, not batches.
        input_ids = encoded["input_ids"].squeeze(0) if encoded["input_ids"].ndim > 1 else encoded["input_ids"] # Removes batch dimension from input_ids.
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))                             # Gets attention_mask or creates one if missing.
        attention_mask = attention_mask.squeeze(0) if attention_mask.ndim > 1 else attention_mask              # Removes batch dimension from attention_mask.

        return {                              # Returns the prepared dictionary.
            "input_ids": input_ids,           # The token IDs tensor.
            "attention_mask": attention_mask, # The attention mask tensor.
        }


class WikiTextDataset(Dataset):
    """
    A specialized dataset class for loading and preparing the WikiText corpus
    (or other Hugging Face datasets) for language model pretraining. It handles
    loading from disk, optional raw-only loading, batch processing, text chunking,
    tokenization, and filtering based on sequence length requirements.

    This is the primary component for handling large-scale
    real-world datasets. Its tests verify robust data ingestion, efficient
    preprocessing, and adherence to complex configuration parameters like
    `min_seq_length` and `max_seq_length` across large text volumes. It confirms
    the pipeline's ability to prepare production-scale data.

    How it fits into the LLM pipeline: This dataset orchestrates the loading of
    massive text corpora, applies necessary preprocessing (like chunking and
    filtering), and then tokenizes the data. It's the critical link between
    the raw text source and the numerical inputs expected by the training loop.
    Its efficiency and correctness directly impact the overall pretraining speed and quality.

    Inputs:
    - config (DataConfig): Configuration object specifying dataset name, split, cache directory, and sequence lengths.
    - tokenizer (Optional[TokenizerWrapper]): The tokenizer to use for converting text to token IDs; can be None for raw-only loading.
    - split (str): The specific split of the dataset to load (e.g., "train", "validation", "test").
    - preprocessing_pipeline (Optional[PreprocessingPipeline]): An optional pipeline of preprocessing steps to apply.
    - load_raw_only (bool): If True, only the raw Hugging Face dataset is loaded, skipping tokenization and internal preprocessing.

    Outputs:
    - __len__ (int): The number of processed examples (or raw examples if `load_raw_only` is True).
    - __getitem__ (dict[str, torch.Tensor] or dict[str, str]): A dictionary containing "input_ids" and "attention_mask" (if tokenized) or "text" (if raw-only).
    - get_statistics (dict[str, Any]): A dictionary containing various statistics about the dataset.
    """
    
    def __init__(
        self,
        config: DataConfig,                                              # Data configuration object.
        tokenizer: Optional[TokenizerWrapper] = None,                    # Optional tokenizer. If None and not raw_only, it will error.
        split: str = "train",                                            # The specific data split to load (e.g., "train", "validation").
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,  # Optional pipeline for text preprocessing.
        load_raw_only: bool = False,                                     # Flag to load only raw text, bypassing tokenization.
    ):
        """
        Initializes the WikiTextDataset, loading the raw data and then optionally preprocessing and tokenizing it.
        """
        self.config = config                                  # Stores the data configuration.
        self.tokenizer = tokenizer                            # Stores the tokenizer instance.
        self.split = split                                    # Stores the dataset split.
        self.preprocessing_pipeline = preprocessing_pipeline  # Stores the preprocessing pipeline.
        self.load_raw_only = load_raw_only                    # Stores the raw-only loading flag.

        # Load raw dataset first from Hugging Face datasets.
        logger.info(f"Loading WikiText dataset: {config.dataset_config}, split: {split}") # Logs the loading process.
        self.dataset = self._load_dataset() # Calls a private method to load the raw Hugging Face dataset.

        self.examples = [] # Initializes an empty list to store processed (tokenized) examples.
        # Checks if full processing is required and a tokenizer is available.
        if not self.load_raw_only and self.tokenizer is not None: 
            logger.info("Preprocessing and tokenizing dataset...")       # Logs the start of preprocessing.
            self.examples = self._preprocess_and_tokenize()              # Calls a private method to preprocess and tokenize.
            logger.info(f"Dataset size: {len(self.examples)} examples")  # Logs the final number of processed examples.
        # Checks if only raw data loading is requested.
        elif self.load_raw_only: 
            logger.info("WikiTextDataset initialized for raw data loading only. Tokenization skipped.")   # Logs skipping tokenization.
        # If tokenizer is None and not raw_only, it means configuration is incomplete for full processing.
        else: 
            logger.warning("WikiTextDataset initialized without tokenizer and 'load_raw_only' is False. " # Warns about missing tokenizer.
                            "No examples will be preprocessed/tokenized.")                                # Informs that no examples will be processed.

    def _load_dataset(self) -> HFDataset:
        """
        Private method to load the specified Hugging Face dataset split.
        Inputs:
        - None (uses self.config and self.split).
        Outputs:
        - HFDataset: The raw Hugging Face Dataset object.
        """
        dataset = load_dataset(              # Calls the Hugging Face `load_dataset` function.
            self.config.dataset_name,        # The name of the dataset (e.g., "wikitext").
            self.config.dataset_config,      # The specific configuration of the dataset (e.g., "wikitext-2-raw-v1").
            split=self.split,                # The dataset split to load.
            cache_dir=self.config.cache_dir, # Directory to cache downloaded data.
        )
        return dataset # Returns the loaded Hugging Face Dataset.
    
    def _preprocess_and_tokenize(self) -> list[dict[str, list[int]]]:
        """
        Private method to apply preprocessing, split long texts into chunks,
        and tokenize the dataset, filtering by `min_seq_length`.
        Inputs:
        - None (uses internal self.dataset, self.config, self.tokenizer, self.preprocessing_pipeline).
        Outputs:
        - list[dict[str, list[int]]]: A list of dictionaries, each containing "input_ids" and "attention_mask" (as lists of integers).
        """
        examples = [] # Initializes an empty list to collect the tokenized examples.
        
        # Process in batches for efficiency to avoid loading all raw texts into memory at once.
        batch_size = self.config.preprocessing_batch_size                   # Retrieves batch size from configuration.
        total_batches = (len(self.dataset) + batch_size - 1) // batch_size  # Calculates total number of batches.
        
        # Iterate over the raw Hugging Face dataset in batches.
        # Loops through dataset with a progress bar.
        for i in tqdm(range(0, len(self.dataset), batch_size), total=total_batches, desc="Processing text batches"): 
            batch = self.dataset[i:i + batch_size]  # Retrieves a batch of raw data from the HF dataset.
            batch_texts = batch["text"]             # Extracts the raw text strings from the batch.
            
            # Apply preprocessing to each text in the batch.
            # Checks if a preprocessing pipeline is provided.
            if self.preprocessing_pipeline is not None: 
                batch_texts = [self.preprocessing_pipeline(text) for text in batch_texts] # Applies the pipeline.
            
            # Filter out empty or too short texts based on a simple word count heuristic.
            # This is an initial filter before detailed token-based filtering.
            batch_texts = [                                                 # Filters the texts.
                text for text in batch_texts                                # Keeps text if:
                if text and len(text.split()) >= self.config.min_seq_length # Text is not empty and has enough words.
            ]
            
            # Tokenize each text in the filtered batch.
            # Iterates over each text in the processed batch.
            for text in batch_texts: 
                # Split long texts into chunks before tokenization if necessary.
                # This prevents the tokenizer from receiving excessively long strings which might
                # lead to memory issues or inefficiency, even with truncation.
                chunks = self._split_into_chunks(text) # Calls the private method to split text into chunks.
                
                # Processes each chunk.
                for chunk in chunks: 
                    # self.tokenizer should not be None at this point if _preprocess_and_tokenize is called
                    # Tokenize the chunk. `padding=False` here is correct because padding to `max_length`
                    # is typically handled by the `DataLoader`'s collate_fn for batching, not individual items.
                    # This `padding=False` ensures that `encoded["input_ids"]` length is the actual token count
                    # after truncation, before final batch padding.
                    encoded = self.tokenizer(                  # Calls the tokenizer on the current chunk.
                        chunk,                                 # The text chunk to tokenize.
                        max_length=self.config.max_seq_length, # Truncates to this length.
                        truncation=True,                       # Enables truncation.
                        padding=False,                         # IMPORTANT: No padding at this stage, padding happens in TextDataset or collator.
                        return_tensors=None,                   # Returns Python lists of integers, not PyTorch tensors yet.
                    )
                    
                    # Only keep sequences that are not too short after tokenization and truncation.
                    # This is the definitive filter based on token count.
                    # Checks if the tokenized length meets minimum.
                    if len(encoded["input_ids"]) >= self.config.min_seq_length: 
                        examples.append({                       # Appends the processed example.
                            "input_ids": encoded["input_ids"],  # List of token IDs.
                            "attention_mask": encoded.get(      # Gets attention mask or creates a default.
                                "attention_mask", 
                                [1] * len(encoded["input_ids"]) # Default mask if none provided (all ones).
                            ),
                        })
        
        return examples # Returns the list of fully processed examples.
    
    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Private method to split a long text string into smaller, more manageable
        chunks based on a heuristic word limit derived from `max_seq_length`.
        This is a pre-tokenization step to avoid feeding extremely long strings
        to the tokenizer.

        Inputs:
        - text (str): The raw text string to split.
        Outputs:
        - list[str]: A list of text chunks.
        """
        if not text.strip():  # Checks if the text is empty or only whitespace.
            return []         # Returns an empty list if so.

        words = text.split()  # Splits the text into a list of words.
        
        if len(words) <= self.config.max_seq_length: # If text is already shorter than max_seq_length, no chunking needed.
            return [text]                            # Returns the original text as a single chunk.
        
        chunks = []        # Initializes a list to store the generated chunks.
        current_chunk = [] # Initializes a list to build the current chunk's words.
        
        # Heuristic factor to estimate words per max_seq_length (adjust as needed)
        # Assuming ~5 characters per word + space, max_seq_length tokens might be
        # max_seq_length * 5 words for English. This is a rough estimation.
        # It ensures that even if max_seq_length is small, we don't end up with
        # ridiculously large string chunks before tokenization.
        # This factor is a heuristic to prevent passing extremely long strings
        # to the tokenizer, which could be slow or memory intensive.
        heuristic_word_limit = self.config.max_seq_length * 5 # Calculates a heuristic word limit for chunks.

        # Iterates through each word in the text.
        for word in words: 
            # Check if adding the next word would exceed the heuristic limit.
            # Checks estimated length of next chunk.
            if len(" ".join(current_chunk + [word])) > heuristic_word_limit: 
                # If the current chunk is not empty.
                if current_chunk: 
                    chunks.append(" ".join(current_chunk))  # Adds the current chunk to the list of chunks.
                current_chunk = [word]                      # Starts a new chunk with the current word.
            # If adding the word doesn't exceed the limit.
            else: 
                current_chunk.append(word) # Adds the word to the current chunk.
        
        if current_chunk: # After the loop, if there's any remaining part in current_chunk.
            chunks.append(" ".join(current_chunk)) # Adds the last chunk.
            
        return chunks # Returns the list of text chunks.
    
    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset.
        If `load_raw_only` is True, returns the length of the raw dataset.
        Otherwise, returns the number of processed (tokenized) examples.
        """
        # If in raw-only mode, return the length of the original raw dataset.
        if self.load_raw_only: 
            return len(self.dataset)
        # If not in raw-only mode, the `examples` list holds the actual processed data.
        # Its length reflects how many examples passed all preprocessing and filtering steps.
        return len(self.examples) # Always return the length of self.examples if not in raw-only mode.
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieves a single item from the dataset. Returns raw text if in `load_raw_only` mode,
        otherwise returns a tokenized example as PyTorch tensors.

        Inputs:
        - idx (int): The index of the item to retrieve.
        Outputs:
        - dict[str, torch.Tensor] or dict[str, str]: The requested data item.
        """
        # Checks if the dataset is in raw-only mode.
        if self.load_raw_only: 
            # Return raw text if in raw-only mode
            return {"text": self.dataset[idx]["text"]} # Returns a dictionary with the raw text.
        # If not in raw-only mode, return tokenized examples.
        else: 
            # Return tokenized example, converting lists of ints to PyTorch tensors.
            example = self.examples[idx]                                                     # Retrieves the processed example (lists of ints).
            return {                                                                         # Returns a dictionary with PyTorch tensors.
                "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),           # Converts input_ids list to LongTensor.
                "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long), # Converts attention_mask list to LongTensor.
            }
    
    def get_statistics(self) -> dict[str, Any]:
        """
        Computes and returns various statistics about the dataset.
        If processed examples are available, it returns statistics based on tokenized lengths.
        Otherwise (e.g., in raw-only mode), it returns statistics based on raw word counts.

        Inputs:
        - None.
        Outputs:
        - dict[str, Any]: A dictionary of computed statistics.
        """
        # Only compute tokenized statistics if examples are populated
        # Checks if processed (tokenized) examples exist.
        if self.examples: 
            lengths = [len(ex["input_ids"]) for ex in self.examples] # Gathers the lengths of all tokenized input_ids.
            
            return {                                   # Returns statistics for tokenized data.
                "num_examples": len(self.examples),    # Total number of tokenized examples.
                "avg_length": float(np.mean(lengths)), # Average tokenized sequence length.
                "min_length": int(np.min(lengths)),    # Minimum tokenized sequence length.
                "max_length": int(np.max(lengths)),    # Maximum tokenized sequence length.
                "std_length": float(np.std(lengths)),  # Standard deviation of tokenized sequence lengths.
                "total_tokens": int(sum(lengths)),     # Total number of tokens across all examples.
            }
        # If no processed examples (e.g., in raw-only mode).
        else: 
            # If no examples (raw-only mode), provide statistics about raw data.
            raw_texts = self.dataset["text"]                        # Retrieves all raw texts.
            raw_lengths = [len(text.split()) for text in raw_texts] # Calculates word counts for raw texts.
            return {                                          # Returns statistics for raw data.
                "num_raw_examples": len(self.dataset),        # Total number of raw text examples.
                "avg_raw_words": float(np.mean(raw_lengths)), # Average raw word count.
                "min_raw_words": int(np.min(raw_lengths)),    # Minimum raw word count.
                "max_raw_words": int(np.max(raw_lengths)),    # Maximum raw word count.
                "std_raw_words": float(np.std(raw_lengths)),  # Standard deviation of raw word counts.
                "total_raw_words": int(sum(raw_lengths)),     # Total number of words across all raw examples.
                "note": "Statistics are based on raw words, not tokens, as no tokenizer was provided." # Explanatory note.
            }
