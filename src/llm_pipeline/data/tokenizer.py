# filename: src/llm_pipeline/data/tokenizer.py
"""
This module provides a `TokenizerWrapper` class and utility functions for building
and managing tokenizers within the LLM pretraining pipeline. It leverages the
`tokenizers` and `transformers` libraries to create fast, custom tokenizers
(like BPE, WordPiece) and provide a consistent interface for encoding and decoding text.

Tokenizers are a fundamental component in the LLM pretraining pipeline, responsible
for converting raw text into numerical representations (`token IDs`) that a neural
network can understand, and vice-versa. The `TokenizerWrapper` ensures that the
tokenization process is robust, configurable (e.g., handling special tokens,
vocabulary size), and compatible with PyTorch's tensor requirements.
It confirms that text can be accurately transformed to and from numerical sequences,
a critical step for all subsequent model training and inference.
"""

import logging                           # Imports the logging library for structured logging.
from typing import Optional, Union, Any  # Imports typing hints for better code readability and validation.
from pathlib import Path                 # Imports Path for object-oriented filesystem paths.
import json                              # Imports json for saving/loading tokenizer configuration.
from tokenizers import (                 # Imports specific tokenizer components from the `tokenizers` library.
    Tokenizer, 
    models, 
    pre_tokenizers, 
    decoders, 
    trainers, 
    processors
    )
from tokenizers.models import BPE, WordPiece, Unigram # Imports specific tokenizer models.
from transformers import PreTrainedTokenizerFast      # Imports Hugging Face Transformers' fast tokenizer wrapper.
import torch                                          # Imports PyTorch for tensor operations.

from llm_pipeline.config import TokenizerConfig       # Imports the TokenizerConfig dataclass for tokenizer configuration.


logger = logging.getLogger(__name__) # Initializes a logger for this module.


class TokenizerWrapper:
    """
    A wrapper class that provides a consistent interface for various tokenizers
    (from `tokenizers` library, wrapped in `PreTrainedTokenizerFast`)
    and integrates them with the LLM pipeline's configuration. It handles
    tokenization, encoding, decoding, and managing special token IDs.

    This wrapper abstracts away the complexities of different tokenizer 
    implementations and ensures they conform to a unified API. 
    It's critical for reliably converting raw text into the numerical
    `input_ids` and `attention_mask` tensors that the model expects, and for
    converting model outputs back into human-readable text. It confirms
    that tokenization, padding, truncation, and special token handling work
    as expected, which are core preprocessing steps.

    Inputs:
    - tokenizer (Union[Tokenizer, PreTrainedTokenizerFast]): The underlying tokenizer object (from `tokenizers` or `transformers`).
    - config (TokenizerConfig): Configuration object specifying tokenizer type, special tokens, etc.

    Outputs:
    - __call__ (dict[str, Any]): Tokenizes text and returns a dictionary of encoded inputs.
    - encode (Union[list[int], list[list[int]]]): Encodes text to lists of token IDs.
    - decode (str): Decodes token IDs back to text.
    - batch_decode (list[str]): Decodes batches of token IDs back to lists of text.
    - vocab_size (int): Property returning the size of the tokenizer's vocabulary.
    - save (None): Saves the tokenizer to disk.
    - load (TokenizerWrapper): Class method to load a tokenizer from disk.
    """
    
    def __init__(self, tokenizer: Union[Tokenizer, PreTrainedTokenizerFast], config: TokenizerConfig):
        """
        Initializes the TokenizerWrapper, ensuring the underlying tokenizer is a `PreTrainedTokenizerFast`
        for compatibility with Hugging Face Transformers API.
        """
        self.config = config # Stores the tokenizer configuration.
        
        # Checks if the provided tokenizer is a `tokenizers` library `Tokenizer` object.
        if isinstance(tokenizer, Tokenizer): 
            # Wrap in HuggingFace tokenizer for compatibility with Transformers API.
            self._tokenizer = PreTrainedTokenizerFast( # Wraps the raw tokenizer.
                tokenizer_object=tokenizer,            # The core tokenizer from `tokenizers` library.
                unk_token=config.unk_token,            # Sets the unknown token.
                pad_token=config.pad_token,            # Sets the padding token.
                bos_token=config.bos_token,            # Sets the beginning-of-sentence token.
                eos_token=config.eos_token,            # Sets the end-of-sentence token.
                mask_token=config.mask_token,          # Sets the mask token.
            )
        # If the provided tokenizer is already a `PreTrainedTokenizerFast`.
        else: 
            self._tokenizer = tokenizer # Uses it directly.
        
        # Cache special token IDs for quick access and consistent behavior.
        self.pad_token_id = self._tokenizer.pad_token_id   # Stores the ID for padding token.
        self.unk_token_id = self._tokenizer.unk_token_id   # Stores the ID for unknown token.
        self.bos_token_id = self._tokenizer.bos_token_id   # Stores the ID for beginning-of-sentence token.
        self.eos_token_id = self._tokenizer.eos_token_id   # Stores the ID for end-of-sentence token.
        self.mask_token_id = self._tokenizer.mask_token_id # Stores the ID for mask token.
        
    def __call__(
        self,
        text: Union[str, list[str]],          # The text or list of texts to tokenize.
        max_length: Optional[int] = None,     # Maximum length for truncation/padding.
        padding: Union[bool, str] = False,    # Whether to pad, and how (e.g., `True`, `"max_length"`).
        truncation: bool = False,             # Whether to truncate if longer than `max_length`.
        return_tensors: Optional[str] = None, # Format to return tensors (e.g., "pt" for PyTorch).
        **kwargs                              # Additional keyword arguments passed to the underlying tokenizer.
    ) -> dict[str, Any]:
        """
        Tokenizes text (or a batch of texts) and returns a dictionary
        containing `input_ids` and `attention_mask`. This method mimics the
        behavior of `transformers` library tokenizers.
        """
        return self._tokenizer( # Calls the underlying `PreTrainedTokenizerFast`'s __call__ method.
            text,                          # Text to tokenize.
            max_length=max_length,         # Maximum length.
            padding=padding,               # Padding strategy.
            truncation=truncation,         # Truncation flag.
            return_tensors=return_tensors, # Tensor return format.
            **kwargs                       # Any additional arguments.
        )
    
    def encode(
        self,
        text: Union[str, list[str]],     # The text or list of texts to encode.
        add_special_tokens: bool = True, # Whether to add special tokens (BOS, EOS) during encoding.
        **kwargs                         # Additional keyword arguments.
    ) -> Union[list[int], list[list[int]]]:
        """
        Encodes text into a list of token IDs. This is a convenience method
        that directly uses the underlying tokenizer's `encode` method.
        """
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs) # Calls underlying encode.
    
    def decode(
        self,
        token_ids: Union[list[int], torch.Tensor], # The token IDs (list or tensor) to decode.
        skip_special_tokens: bool = True,          # Whether to skip special tokens in the decoded text.
        **kwargs                                   # Additional keyword arguments.
    ) -> str:
        """
        Decodes a list of token IDs (or a 1D tensor) back into a single text string.
        """
        if isinstance(token_ids, torch.Tensor): # Checks if input is a PyTorch tensor.
            token_ids = token_ids.tolist()      # Converts tensor to a list of Python integers.
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs) # Calls underlying decode.
    
    def batch_decode(
        self,
        token_ids: Union[list[list[int]], torch.Tensor], # A list of lists of token IDs (or a 2D tensor) to decode.
        skip_special_tokens: bool = True,                # Whether to skip special tokens in the decoded texts.
        **kwargs                                         # Additional keyword arguments.
    ) -> list[str]:
        """
        Decodes a batch of token IDs (list of lists or a 2D tensor) back into a list of text strings.
        """
        if isinstance(token_ids, torch.Tensor): # Checks if input is a PyTorch tensor.
            token_ids = token_ids.tolist()      # Converts tensor to a list of lists of integers.
        return self._tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs) # Calls underlying batch_decode.
    
    @property # Decorator to make this method accessible as an attribute.
    def vocab_size(self) -> int:
        """
        Property to get the vocabulary size of the tokenizer.
        """
        return len(self._tokenizer) # Returns the length of the tokenizer's vocabulary.
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the tokenizer and its configuration to a specified directory on disk.
        This allows for later reloading without retraining.

        Inputs:
        - path (Union[str, Path]): The directory path where the tokenizer files will be saved.
        Outputs:
        - None.
        """
        path = Path(path)                       # Ensures the path is a Path object.
        path.mkdir(parents=True, exist_ok=True) # Creates the directory and any necessary parent directories.
        
        # This will save tokenizer.json, special_tokens_map.json, tokenizer_config.json
        # among others, to the specified directory using the Hugging Face Transformers' method.
        self._tokenizer.save_pretrained(str(path)) # Saves the tokenizer using the Transformers library's method.
        
        # For robustness, explicitly save our custom TokenizerConfig dataclass
        # as it might contain pipeline-specific fields not covered by standard HF saving.
        config_path = path / "tokenizer_config.json"                  # Defines the path for the custom config file.
        with open(config_path, "w") as f:                             # Opens the config file in write mode.
            json.dump(self.config.__dict__, f, indent=2, default=str) # Serializes the config dataclass to JSON.

    @classmethod # Decorator indicating this is a class method.
    def load(cls, path: Union[str, Path], config: Optional[TokenizerConfig] = None) -> "TokenizerWrapper":
        """
        Loads a tokenizer and its configuration from a specified directory on disk.

        Inputs:
        - path (Union[str, Path]): The directory containing the tokenizer files (e.g., tokenizer.json, tokenizer_config.json).
        - config (Optional[TokenizerConfig]): An optional TokenizerConfig object. If None, it will attempt to load it from `path`.

        Outputs:
        - TokenizerWrapper: An instance of the loaded TokenizerWrapper.
        """
        path = Path(path) # Ensures the path is a Path object.
        
        # Load the custom TokenizerConfig first.
        if config is None: # Checks if a config object was not provided.
            config_path = path / "tokenizer_config.json"    # Defines the path to the custom config file.
            # Checks if the custom config file exists.
            if config_path.exists(): 
                with open(config_path, "r") as f:           # Opens the config file in read mode.
                    config_dict = json.load(f)              # Loads the JSON content.
                    config = TokenizerConfig(**config_dict) # Reconstructs the TokenizerConfig object.
            # If no custom config file is found.
            else: 
                raise ValueError( # Raises an error.
                    f"No custom TokenizerConfig found at {config_path}. "
                    "Ensure the tokenizer directory contains tokenizer.json and tokenizer_config.json "
                    "to TokenizerWrapper.load()."
                )

        # Load the PreTrainedTokenizerFast from the directory.
        # Explicitly pass the tokenizer_file argument to tell transformers
        # where the main 'tokenizers' library serialization is, for robustness.
        # Attempts to load the tokenizer.
        try: 
            tokenizer = PreTrainedTokenizerFast.from_pretrained( # Loads the tokenizer using Transformers' method.
                str(path),                                       # The directory path.
                tokenizer_file=str(path / "tokenizer.json")      # Explicitly points to the main tokenizers serialization file.
            )
        # Catches any exceptions during loading.
        except Exception as e: 
            logger.error(f"Failed to load PreTrainedTokenizerFast from '{path}' with tokenizer_file. Error: {e}") # Logs the error.
            raise # Re-raises the exception.

        return cls(tokenizer, config) # Returns a new TokenizerWrapper instance with the loaded tokenizer and config.


def build_tokenizer(
    config: TokenizerConfig,                    # Configuration object for building the tokenizer.
    training_texts: Optional[list[str]] = None, # Optional list of texts to train the tokenizer on.
) -> TokenizerWrapper:
    """
    Constructs a tokenizer based on the provided configuration and optionally trains it.
    This function orchestrates the creation of different tokenizer models (BPE, WordPiece, Unigram)
    and sets up their pre-tokenizers, post-processors, and decoders.

    This function is the entry point for creating a tokenizer
    for the pipeline. It must correctly configure and, if necessary, train the tokenizer
    according to the specified parameters. Its correct functioning ensures that the model
    receives inputs tokenized consistently with the model's expected vocabulary.

    Inputs:
    - config (TokenizerConfig): Configuration for the tokenizer (type, vocab size, special tokens, etc.).
    - training_texts (Optional[list[str]]): A list of texts used for training the tokenizer vocabulary.

    Outputs:
    - TokenizerWrapper: An instance of the custom TokenizerWrapper containing the built (and possibly trained) tokenizer.
    """
    logger.info(f"Building {config.tokenizer_type} tokenizer...") # Logs the type of tokenizer being built.
    
    # Select tokenizer model based on configuration.
    # If BPE tokenizer type is specified.
    if config.tokenizer_type == "bpe": 
        model = BPE(unk_token=config.unk_token)       # Creates a BPE model with the unknown token.
    # If WordPiece tokenizer type is specified.
    elif config.tokenizer_type == "wordpiece": 
        model = WordPiece(unk_token=config.unk_token) # Creates a WordPiece model with the unknown token.
    # If Unigram tokenizer type is specified.
    elif config.tokenizer_type == "unigram": 
        model = Unigram()                             # Creates a Unigram model.
    # If an unsupported tokenizer type is specified.
    else: 
        raise ValueError(f"Unknown tokenizer type: {config.tokenizer_type}") # Raises an error.
    
    # Create the base tokenizer object.
    tokenizer = Tokenizer(model) # Instantiates the core `tokenizers` library Tokenizer.
    
    # Set up the pre-tokenizer (how raw text is initially split before tokenization).
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # Uses a simple whitespace pre-tokenizer.
    
    # Set up the post-processor (how special tokens are added around sequences).
    # This prepares the sequence in the format expected by the model (e.g., "[BOS] text [EOS]").
    special_tokens_for_post_processor = []                              # Initializes list for special tokens in post-processor.
    # If a beginning-of-sentence token is defined.
    if config.bos_token: 
        special_tokens_for_post_processor.append((config.bos_token, 1)) # Adds BOS token to list with its ID type.
    # If an end-of-sentence token is defined.
    if config.eos_token: 
        special_tokens_for_post_processor.append((config.eos_token, 2)) # Adds EOS token to list with its ID type.

    tokenizer.post_processor = processors.TemplateProcessing( # Sets up a template-based post-processor.
        single=f"{config.bos_token} $A {config.eos_token}",   # Defines the template for single sequences. `$A` is where the tokenized text goes.
        special_tokens=special_tokens_for_post_processor,     # Provides the special tokens and their types to the template.
    )
    
    # Set up the decoder (how token IDs are converted back to raw text).
    # If BPE tokenizer.
    if config.tokenizer_type == "bpe": 
        tokenizer.decoder = decoders.BPEDecoder() # Uses BPE-specific decoder.
    # If WordPiece tokenizer.
    elif config.tokenizer_type == "wordpiece": 
        tokenizer.decoder = decoders.WordPiece()  # Uses WordPiece-specific decoder.
    # Unigram doesn't always have a specific decoder needed beyond default.
    
    # Train tokenizer if requested and training texts are provided.
    if config.train_tokenizer and training_texts is not None: # Checks if training is enabled and texts are available.
        logger.info("Training tokenizer...") # Logs the start of tokenizer training.
        
        # Define special tokens from config for the trainer.
        # These tokens will be reserved in the vocabulary and not learned.
        defined_special_tokens_for_trainer = [] # Initializes list for special tokens in trainer.
        if config.unk_token: defined_special_tokens_for_trainer.append(config.unk_token)   # Adds unknown token.
        if config.pad_token: defined_special_tokens_for_trainer.append(config.pad_token)   # Adds padding token.
        if config.bos_token: defined_special_tokens_for_trainer.append(config.bos_token)   # Adds BOS token.
        if config.eos_token: defined_special_tokens_for_trainer.append(config.eos_token)   # Adds EOS token.
        if config.mask_token: defined_special_tokens_for_trainer.append(config.mask_token) # Adds mask token.
        
        # Select tokenizer trainer based on configuration.
        # If BPE tokenizer.
        if config.tokenizer_type == "bpe": 
            trainer = trainers.BpeTrainer(                         # Creates a BPE trainer.
                vocab_size=config.vocab_size,                      # Sets the target vocabulary size.
                min_frequency=config.min_frequency,                # Sets the minimum frequency for tokens to be included.
                special_tokens=defined_special_tokens_for_trainer, # Provides the list of special tokens.
            )
        # If WordPiece tokenizer.
        elif config.tokenizer_type == "wordpiece": 
            trainer = trainers.WordPieceTrainer(                   # Creates a WordPiece trainer.
                vocab_size=config.vocab_size,                      # Sets the target vocabulary size.
                min_frequency=config.min_frequency,                # Sets the minimum frequency.
                special_tokens=defined_special_tokens_for_trainer, # Provides the special tokens.
            )
        # If Unigram tokenizer.
        elif config.tokenizer_type == "unigram": 
            trainer = trainers.UnigramTrainer(                     # Creates a Unigram trainer.
                vocab_size=config.vocab_size,                      # Sets the target vocabulary size.
                special_tokens=defined_special_tokens_for_trainer, # Provides the special tokens.
            )
        else: # Should ideally not be reached due to initial validation.
            raise ValueError(f"No trainer implemented for tokenizer type: {config.tokenizer_type}") # Raises an error.
        
        # Train the tokenizer from the provided texts.
        tokenizer.train_from_iterator(training_texts, trainer) # Starts the training process.
        logger.info(f"Tokenizer trained with vocabulary size: {tokenizer.get_vocab_size()}") # Logs the resulting vocabulary size.
    
    # Create wrapper
    wrapper = TokenizerWrapper(tokenizer, config)
    
    # CRITICAL: Update the config with the actual vocabulary size
    actual_vocab_size = wrapper.vocab_size
    config.vocab_size = actual_vocab_size  # Update the config object
    
    logger.info(f"Final tokenizer vocab size: {actual_vocab_size}")
    logger.info(f"Updated config vocab_size to: {config.vocab_size}")
    
    # Verify special token IDs are valid
    special_token_info = {
        'pad': wrapper.pad_token_id,
        'unk': wrapper.unk_token_id,
        'bos': wrapper.bos_token_id,
        'eos': wrapper.eos_token_id,
        'mask': wrapper.mask_token_id
    }
    
    logger.info(f"Special token IDs: {special_token_info}")
    
    # Check for any special tokens that exceed vocab size
    for name, token_id in special_token_info.items():
        if token_id is not None and token_id >= actual_vocab_size:
            raise ValueError(
                f"Special token '{name}' has ID {token_id} >= vocab_size {actual_vocab_size}. "
                f"This will cause CUDA indexing errors during generation."
            )
    
    return wrapper