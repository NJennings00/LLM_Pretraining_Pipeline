# filename: src/llm_pipeline/data/collator.py
"""
This module provides data collation utilities specifically designed for language modeling tasks
within the LLM pretraining pipeline. Data collators are essential components that take a list
of individual data examples (typically produced by a `torch.utils.data.Dataset`) and combine
them into a single, batched tensor structure suitable for model input.

The primary class, `DataCollatorForLanguageModeling`, handles two main types of language modeling:
1.  **Causal Language Modeling (CLM):** Used for decoder-only models, where the goal is to predict
    the next token in a sequence. For CLM, the collator ensures sequences are padded and that
    labels are identical to `input_ids` (the loss function typically handles the shifting).
2.  **Masked Language Modeling (MLM):** Used for encoder-decoder or encoder-only models, where
    a certain percentage of tokens are randomly masked, and the model must predict them.
    The collator implements the masking strategy (80% mask, 10% random, 10% original)
    and prepares the corresponding labels.

Why data collation is essential for LLM pretraining:
-   **Batching for Efficiency:** Deep learning models, especially large LLMs, process data in batches
    to leverage parallel computation on GPUs, significantly speeding up training.
-   **Padding for Uniformity:** Sequences in a dataset often have varying lengths. Collators
    pad these sequences to a uniform maximum length within a batch, which is necessary for
    creating rectangular tensors (the input format for most neural networks).
-   **Label Generation:** For specific tasks like MLM, the collator dynamically creates
    the `labels` tensor based on the input `input_ids` and the masking strategy.

This module ensures that raw data examples from the datasets are correctly formatted and prepared
into batches, making them directly compatible with the model's forward pass and the training loop.
It confirms that data is efficiently organized for GPU processing and that the specific requirements
of different language modeling objectives are met.
"""

import torch                            # Imports PyTorch for tensor operations.
from typing import Optional, Union, Any # Imports typing hints for better code readability and validation.
from dataclasses import dataclass       # Imports dataclass for defining data classes with less boilerplate.
import numpy as np                      # Imports NumPy, although not directly used in the final version after `torch.full`.

from llm_pipeline.data.tokenizer import TokenizerWrapper # Imports the TokenizerWrapper to access special token IDs.


@dataclass # Decorator to automatically generate methods like __init__, __repr__, etc.
class DataCollator:
    """
    Base data collator class. Serves as an abstract interface for all data collators.
    Any concrete collator must inherit from this and implement the `__call__` method.

    Why it's needed: Provides a common interface for different collation strategies,
    making the data loading part of the pipeline modular and extensible.

    How it fits into the LLM pipeline: It's the blueprint for how individual
    processed examples from datasets are combined into mini-batches for the model.
    Inputs:
    - features (list[dict[str, Any]]): A list of individual data examples (dictionaries).
    Outputs:
    - dict[str, torch.Tensor]: A dictionary representing a batch of data.
    """
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]: # Defines the call method, making instances callable.
        """
        Collate a list of features (individual examples) into a single batch.
        This method must be implemented by subclasses.

        Inputs:
        - features (list[dict[str, Any]]): A list where each element is a dictionary
                                            representing one processed data example.
                                            Typically, each dictionary contains "input_ids"
                                            and "attention_mask".
        Outputs:
        - dict[str, torch.Tensor]: A dictionary containing batched PyTorch tensors,
                                   e.g., {"input_ids": ..., "attention_mask": ..., "labels": ...}.
        """
        raise NotImplementedError # Raises an error if the subclass does not implement this method.


@dataclass # Decorator to automatically generate methods.
class DataCollatorForLanguageModeling(DataCollator):
    """
    A specific data collator for language modeling tasks (Causal LM and Masked LM).
    It handles dynamic padding of sequences within a batch to the longest sequence
    in that batch (or to a multiple of a specified length). It also prepares
    labels based on the chosen language modeling objective (MLM or CLM).

    Why it's needed: This collator is crucial for preparing the exact input format
    required by Transformer-based LLMs. It standardizes sequence lengths for efficient
    GPU processing and dynamically applies masking strategies for MLM.

    How it fits into the LLM pipeline: This is the final step in the data preparation
    pipeline before data is fed into the model's training loop. It directly influences
    the computational efficiency and the specific learning objective of the model.
    Inputs:
    - features (list[dict[str, Any]]): A list of feature dictionaries, typically from `TextDataset` or `WikiTextDataset`.
                                      Each dict is expected to have "input_ids" (torch.Tensor)
                                      and optionally "attention_mask" (torch.Tensor).
    Outputs:
    - dict[str, torch.Tensor]: A batch dictionary containing:
        - "input_ids" (torch.Tensor): The padded and potentially masked input token IDs.
        - "attention_mask" (torch.Tensor): The padded attention mask.
        - "labels" (torch.Tensor): The labels for loss computation (either original input_ids for CLM, or masked labels for MLM).
    """
    
    tokenizer: TokenizerWrapper              # The tokenizer, used to get padding token ID and mask token ID.
    mlm: bool = False                        # If True, enables Masked Language Modeling; otherwise, it's Causal Language Modeling.
    mlm_probability: float = 0.15            # The probability of masking a token for MLM.
    pad_to_multiple_of: Optional[int] = None # If specified, pads sequences to a multiple of this value.
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]: # Defines the call method for batching.
        """
        Collate features for language modeling, applying padding and, if `mlm` is True, masking.
        """
        # Extract input_ids and attention_mask from the list of feature dictionaries.
        # It's assumed that `features` are individual examples from the dataset,
        # where "input_ids" and "attention_mask" are already 1D PyTorch tensors.
        input_ids = [f["input_ids"] for f in features] # Gathers all input_ids tensors from the features.
        attention_mask = [f.get("attention_mask", torch.ones_like(f["input_ids"], dtype=torch.long)) for f in features] # Gathers all attention_mask tensors, providing a default if missing.
        
        # Determine the maximum sequence length within the current batch.
        # All sequences in the batch will be padded to this length.
        max_length = max(len(ids) for ids in input_ids) # Finds the length of the longest sequence in the current batch.
        
        # If `pad_to_multiple_of` is specified, adjust `max_length` to be a multiple of that value.
        # This can improve GPU utilization by making batch dimensions more consistent.
        if self.pad_to_multiple_of is not None:            # Checks if padding to a multiple is required.
            max_length = (                                 # Calculates the new max_length.
                (max_length + self.pad_to_multiple_of - 1) # Adds `pad_to_multiple_of - 1` to round up.
                // self.pad_to_multiple_of                 # Integer division by the multiple.
                * self.pad_to_multiple_of                  # Multiplies by the multiple to get the next highest multiple.
            )
        
        # Pad each sequence in the batch to the determined `max_length`.
        padded_input_ids = []      # List to store padded input_ids.
        padded_attention_mask = [] # List to store padded attention_masks.
        
        for ids, mask in zip(input_ids, attention_mask): # Iterates through each input_ids and attention_mask pair.
            padding_length = max_length - len(ids)       # Calculates how many padding tokens are needed.
            
            if padding_length > 0: # If padding is necessary.
                # Create a tensor of padding IDs (`tokenizer.pad_token_id`).
                padding_ids_tensor = torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long, device=ids.device) # Creates a tensor filled with pad_token_id.
                # Concatenate the original IDs with padding IDs.
                ids = torch.cat((ids, padding_ids_tensor)) # Appends padding IDs to the input_ids.
                
                # Create a tensor of padding mask values (zeros).
                padding_mask_tensor = torch.full((padding_length,), 0, dtype=torch.long, device=mask.device) # Creates a tensor filled with zeros for the mask.
                # Concatenate the original mask with padding mask.
                mask = torch.cat((mask, padding_mask_tensor)) # Appends padding mask values to the attention_mask.
            
            padded_input_ids.append(ids)       # Adds the padded input_ids to the list.
            padded_attention_mask.append(mask) # Adds the padded attention_mask to the list.
        
        # Convert lists of 1D tensors into single 2D tensors, forming the batch.
        batch = {                                                        # Creates the batch dictionary.
            "input_ids": torch.stack(padded_input_ids, dim=0),           # Stacks all padded input_ids tensors along a new dimension (batch dimension).
            "attention_mask": torch.stack(padded_attention_mask, dim=0), # Stacks all padded attention_mask tensors.
        }
        
        # Prepare labels based on the language modeling objective.
        # If Masked Language Modeling is NOT enabled (i.e., Causal Language Modeling).
        if not self.mlm: 
            # For Causal LM, labels are typically the same as input_ids.
            # The model's loss computation will internally shift these labels by one position
            # to predict the next token.
            batch["labels"] = batch["input_ids"].clone() # Creates labels as a clone of input_ids.
        # If Masked Language Modeling IS enabled.
        else: 
            # For MLM, labels need to be created by masking tokens and setting unmasked positions to -100.
            # The `_mask_tokens` method handles this.
            batch["labels"] = batch["input_ids"].clone()             # Labels start as a clone of input_ids before masking.
            batch["input_ids"], batch["labels"] = self._mask_tokens( # Calls the private masking method.
                batch["input_ids"], batch["labels"]                  # Passes input_ids (will be modified) and labels (will be modified).
            )
        
        return batch # Returns the fully prepared batch.
    
    def _mask_tokens(
        self, 
        inputs: torch.Tensor, # The input IDs tensor (will be modified with mask tokens).
        labels: torch.Tensor  # The labels tensor (will be modified to indicate masked tokens).
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Private method to apply masking strategy for Masked Language Modeling (MLM).
        It randomly selects tokens to mask based on `mlm_probability` and replaces them
        with a mask token, a random token, or keeps them original, updating `inputs` and `labels`.
        Tokens not chosen for prediction have their label set to -100 so they are ignored by the loss function.

        Inputs:
        - inputs (torch.Tensor): The batched input_ids tensor.
        - labels (torch.Tensor): The batched labels tensor (a clone of input_ids initially).
        Outputs:
        - tuple[torch.Tensor, torch.Tensor]: A tuple containing the modified `inputs` tensor
                                            (with masked tokens) and the modified `labels` tensor.
        """
        # Create a probability matrix for masking, with `mlm_probability` for each token.
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=labels.device) # Fills a tensor with the masking probability.
        
        # Ensure special tokens (PAD, BOS, EOS) are never masked.
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device) # Initializes a boolean mask for special tokens.
        for special_id in [              # Iterates through the special token IDs.
            self.tokenizer.pad_token_id, # Padding token ID.
            self.tokenizer.bos_token_id, # Beginning of sentence token ID.
            self.tokenizer.eos_token_id, # End of sentence token ID.
        ]:
            if special_id is not None:                        # Checks if the special token ID is valid (not None).
                special_tokens_mask |= (labels == special_id) # Sets True for positions where labels match a special token ID.
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0) # Sets the masking probability to 0.0 for special tokens, ensuring they are not masked.
        
        # Randomly select indices to be masked based on the probability matrix.
        masked_indices = torch.bernoulli(probability_matrix).bool() # Generates a boolean mask; True means the token should be masked.
        
        # Set labels to -100 for tokens that are NOT masked.
        # This tells the loss function to ignore these positions during loss calculation.
        labels[~masked_indices] = -100 # Inverts the masked_indices and sets corresponding labels to -100.
        
        # Apply the masking strategy:
        # 1. 80% of the time: Replace the token with the MASK token.
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices # Selects 80% of the `masked_indices` for replacement.
        inputs[indices_replaced] = self.tokenizer.mask_token_id # Replaces the selected input tokens with the MASK token ID.
        
        # 2. 10% of the time: Replace the token with a random token from the vocabulary.
        # This must be applied to the remaining masked tokens that were not replaced.
        indices_random = ( # Selects remaining 10% of `masked_indices` for random replacement.
            torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() # 50% chance applied to the remaining masked tokens.
            & masked_indices    # Must be one of the originally masked tokens.
            & ~indices_replaced # Must NOT have been replaced in the previous step.
        )
        # Generate random token IDs from the tokenizer's vocabulary.
        # Use tokenizer.vocab_size to ensure random tokens are within model's vocabulary
        vocab_size = self.tokenizer.vocab_size
        random_words = torch.randint(  # Generates random integers.
            0,                         # Start from token ID 0
            vocab_size,                # End at vocab_size (exclusive), ensures max token ID = vocab_size-1
            labels.shape,              # Shape of the tensor to fill.
            dtype=torch.long,          # Data type of the random integers.
            device=labels.device       # Ensures the random tensor is on the same device.
        )
        inputs[indices_random] = random_words[indices_random] # Replaces the selected input tokens with random token IDs.
        
        # 3. 10% of the time: Keep the original token (implicitly done).
        # The tokens that are `masked_indices` but neither `indices_replaced` nor `indices_random`
        # are left as their original value in `inputs`. This behavior is useful for the model
        # to learn that it sometimes predicts the original token even if it was marked for masking.
        
        return inputs, labels # Returns the modified inputs and labels.