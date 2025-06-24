# filename: src/llm_pipeline/models/transformer.py
"""
Transformer model architecture.

This module defines the complete Transformer-based Language Model (LLM) architecture.
It encompasses:
1.  **TransformerConfig**: A dataclass for defining the hyperparameters and
    configuration settings of the Transformer model, including architectural
    parameters, dropout rates, and special token IDs.
2.  **TransformerLM (Transformer Language Model)**: The main model class,
    implementing a decoder-only transformer architecture suitable for
    autoregressive language modeling. It integrates:
    -   Token and positional embeddings.
    -   A stack of `TransformerDecoderLayer` instances.
    -   A final layer normalization and a language modeling head (linear layer)
        to predict the next token.
    -   Weight tying between input embeddings and the output projection layer.
    -   An `_init_weights` method for proper initialization of model parameters.
    -   A `forward` pass that handles causal masking, padding, and
        efficient key-value caching for inference.
    -   A `generate` method for performing text generation (greedy, top-k, top-p sampling).

In an LLM pretraining pipeline, this `TransformerLM` class represents the
entire model that is trained to predict the next token in a sequence. Its
design allows for:
-   **Scalability**: Parameters within `TransformerConfig` allow for easy
    scaling of model size (e.g., `hidden_size`, `num_hidden_layers`).
-   **Flexibility**: Support for different attention types (though only
    'standard' is implemented via `MultiHeadAttention` here, the config
    allows for expansion), and Rotary Embeddings.
-   **Efficient Inference**: The `use_cache` mechanism in the `forward`
    and `generate` methods significantly speeds up autoregressive decoding
    by reusing previously computed key and value states.
-   **Robust Initialization**: The `_init_weights` method helps in stable
    training of deep transformer networks.
"""

import logging                                  # Imports the logging module for emitting log messages.
from typing import Optional, Tuple, Union, Any  # Imports type hints for better code readability and type checking.
from dataclasses import dataclass, field        # Imports dataclass for defining configuration objects.
import torch                                    # Imports the PyTorch library, essential for defining and operating on neural networks.
import torch.nn as nn                           # Imports the neural network module from PyTorch.
import torch.nn.functional as F                 # Imports functional interface for common neural network operations.

from llm_pipeline.models.layers import (        # Imports custom transformer layer implementations.
    TransformerDecoderLayer,                    # Specifically imports the decoder layer.
)

from llm_pipeline.models.embeddings import TokenAndPositionalEmbedding # Imports the embedding layer.
from llm_pipeline.models.utils import generate_square_subsequent_mask  # Imports a utility function to generate causal masks.


logger = logging.getLogger(__name__) # Initializes a logger for this module.


@dataclass
class TransformerConfig:
    """
    Configuration for transformer model.

    Purpose:
        This dataclass serves as a centralized container for all hyperparameters
        and configuration settings required to build and operate the `TransformerLM`.
        It defines the model's architecture (e.g., size of layers, number of heads),
        dropout rates, initialization ranges, and special token IDs.

        A dedicated configuration class ensures consistency and ease of management
        of model parameters. It simplifies model instantiation and allows for
        quick experimentation with different architectural variations. The
        `from_model_config` class method provides flexibility to integrate with
        a more generic `ModelConfig` if such a base configuration exists.

    LLM Pipeline Fit:
        This configuration is the blueprint for creating any `TransformerLM`
        instance in the `llm_pipeline.models` package. During training, a
        `TransformerConfig` object is passed to the `TransformerLM` constructor,
        and its parameters dictate the size, complexity, and behavior of the
        LLM being trained.

    Attributes:
        - `vocab_size` (int): The size of the vocabulary.
        - `hidden_size` (int): Dimensionality of the embeddings and hidden layers.
        - `num_hidden_layers` (int): Number of transformer decoder layers.
        - `num_attention_heads` (int): Number of attention heads in each layer.
        - `intermediate_size` (int): Dimensionality of the "intermediate" (feed-forward) layer.
        - `max_position_embeddings` (int): Maximum sequence length the model can handle.
        - `hidden_dropout_prob` (float): Dropout probability for hidden states.
        - `attention_probs_dropout_prob` (float): Dropout probability for attention weights.
        - `layer_norm_eps` (float): Epsilon for layer normalization to prevent division by zero.
        - `initializer_range` (float): Standard deviation for weight initialization.
        - `use_cache` (bool): Whether to use key/value caching during inference.
        - `hidden_act` (str): Activation function name for the feed-forward network (e.g., "gelu").
        - `model_type` (str): Type of transformer model (e.g., "transformer_lm" for decoder-only).
        - `attention_type` (str): Type of attention mechanism (e.g., "standard", "flash", "sparse").
        - `use_rotary_embeddings` (bool): Whether to use Rotary Positional Embeddings.
        - `rotary_dim` (Optional[int]): Dimension for Rotary Embeddings. Derived from `head_dim` if None.
        - `tie_word_embeddings` (bool): Whether to tie the input token embeddings with the output language modeling head weights.
        - `pad_token_id` (int): ID for the padding token.
        - `bos_token_id` (int): ID for the beginning-of-sentence token.
        - `eos_token_id` (int): ID for the end-of-sentence token.

    Confirms/Verifies:
        - Ensures `hidden_size` is divisible by `num_attention_heads`.
        - Sets `rotary_dim` to `head_dim` if `use_rotary_embeddings` is true and `rotary_dim` is not explicitly set.
        - Provides a mechanism to initialize from a generic `ModelConfig` object, ensuring compatibility.
    """

    vocab_size: int = 32000                   # Default vocabulary size.
    hidden_size: int = 768                    # Default dimensionality of model states.
    num_hidden_layers: int = 12               # Default number of transformer layers.
    num_attention_heads: int = 12             # Default number of attention heads.
    intermediate_size: int = 3072             # Typically 4 * hidden_size # Default intermediate size for feed-forward network.
    max_position_embeddings: int = 512        # Default maximum sequence length.
    hidden_dropout_prob: float = 0.1          # Default dropout rate for hidden states.
    attention_probs_dropout_prob: float = 0.1 # Default dropout rate for attention probabilities.
    layer_norm_eps: float = 1e-12             # Default epsilon for Layer Normalization.
    initializer_range: float = 0.02           # Default range for weight initialization.
    use_cache: bool = True                    # Default to use key-value caching during inference.
    hidden_act: str = "gelu"                  # Default activation function.

    # Model type specific
    model_type: str = "transformer_lm" # "transformer_lm" (decoder-only) # Specifies the model type.

    # Attention specific
    attention_type: str = "standard"    # "standard", "flash", "sparse" # Specifies attention mechanism type.
    use_rotary_embeddings: bool = False # Flag for Rotary Positional Embeddings.
    rotary_dim: Optional[int] = None    # Dimension for RoPE.

    # Tie weights
    tie_word_embeddings: bool = True # Flag to tie input and output embeddings.

    # Added for generation/decoding
    pad_token_id: int = 0 # Assuming 0 is a reasonable default for padding # Default ID for padding token.
    bos_token_id: int = 1 # Assuming 1 is a reasonable default for beginning of sentence # Default ID for beginning of sequence token.
    eos_token_id: int = 2 # Assuming 2 is a reasonable default for end of sentence # Default ID for end of sequence token.

    def __post_init__(self):                                               # Post-initialization method for dataclass.
        if self.rotary_dim is None and self.use_rotary_embeddings:         # If RoPE is used but dim is not set.
            self.rotary_dim = self.hidden_size // self.num_attention_heads # Calculates rotary_dim as head_dim.

        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads" # Asserts that hidden_size is divisible by num_attention_heads.

    @classmethod
    def from_model_config(cls, config: Any) -> "TransformerConfig":
        """
        Create TransformerConfig from a generic ModelConfig.
        This assumes the ModelConfig object has the necessary attributes.

        Purpose:
            Provides a flexible way to instantiate `TransformerConfig` from
            a potentially more generic `ModelConfig` object, allowing for
            centralized configuration management across different model types.

        Inputs:
            - `config` (Any): A generic configuration object.

        Outputs:
            - `TransformerConfig`: An instance of `TransformerConfig` populated
              from the provided generic config.
        """
        return cls( # Creates and returns a new TransformerConfig instance.
            vocab_size=config.vocab_size,                                           # Populates vocab_size.
            hidden_size=config.hidden_size,                                         # Populates hidden_size.
            num_hidden_layers=config.num_hidden_layers,                             # Populates num_hidden_layers.
            num_attention_heads=config.num_attention_heads,                         # Populates num_attention_heads.
            intermediate_size=config.intermediate_size,                             # Populates intermediate_size.
            max_position_embeddings=config.max_position_embeddings,                 # Populates max_position_embeddings.
            hidden_dropout_prob=config.hidden_dropout_prob,                         # Populates hidden_dropout_prob.
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,       # Populates attention_probs_dropout_prob.
            initializer_range=config.initializer_range,                             # Populates initializer_range.
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-12),                # Safely gets layer_norm_eps with a default.
            use_cache=getattr(config, "use_cache", True),                           # Safely gets use_cache with a default.
            hidden_act=getattr(config, "hidden_act", "gelu"),                       # Safely gets hidden_act with a default.
            attention_type=getattr(config, "attention_type", "standard"),           # Safely gets attention_type with a default.
            use_rotary_embeddings=getattr(config, "use_rotary_embeddings", False),  # Safely gets use_rotary_embeddings with a default.
            rotary_dim=getattr(config, "rotary_dim", None),                         # Safely gets rotary_dim with a default.
            model_type=getattr(config, "model_type", "transformer_lm"),             # Safely gets model_type with a default.
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),       # Safely gets tie_word_embeddings with a default.
            pad_token_id=getattr(config, "pad_token_id", 0),                        # Safely gets pad_token_id with a default.
            bos_token_id=getattr(config, "bos_token_id", 1),                        # Safely gets bos_token_id with a default.
            eos_token_id=getattr(config, "eos_token_id", 2),                        # Safely gets eos_token_id with a default.
        )


class TransformerLM(nn.Module):
    """
    Transformer-based Language Model (decoder-only).

    Purpose:
        This is the main class that defines the full transformer language model
        architecture. It orchestrates the various components (embeddings,
        transformer decoder layers, and the language modeling head) to process
        input sequences and predict the next token. It is specifically designed
        as a decoder-only model, making it suitable for autoregressive tasks
        like text generation.

        This class is the core trainable and inference-capable model in the LLM
        pipeline. Its correct assembly and forward pass logic are paramount for
        the model to learn language patterns effectively and generate coherent text.
        The `generate` method is a key utility for showcasing the model's
        generative capabilities.

    LLM Pipeline Fit:
        This `TransformerLM` is the central "model" component in the `llm_pipeline`.
        It takes tokenized input, processes it through multiple layers of
        self-attention and feed-forward networks, and outputs logits over the
        vocabulary. It is the object that will be initialized, potentially loaded
        with pre-trained weights, and then trained or used for inference (text generation).

    Inputs to `__init__`:
        - `config` (TransformerConfig): An instance of `TransformerConfig`
          defining the model's architecture and hyperparameters.

    Outputs of `forward`:
        - `dict[str, torch.Tensor]`: A dictionary containing:
            - `logits` (torch.Tensor): Raw scores for each token in the vocabulary
              for each position in the sequence, shape `[batch_size, seq_length, vocab_size]`.
            - `loss` (Optional[torch.Tensor]): The computed language modeling loss
              if `labels` are provided during the forward pass.

    Outputs of `generate`:
        - `torch.Tensor`: A tensor of generated token IDs, shape
          `[batch_size, generated_sequence_length]`.

    Confirms/Verifies:
        - Correct sequential flow of data through embeddings, decoder layers,
          final LayerNorm, and output head.
        - Proper application of causal attention mask to ensure autoregressive property.
        - Effective handling of padding masks by combining them with the causal mask.
        - Efficient key-value caching during inference for faster generation.
        - Correct weight initialization strategy.
        - Optional tying of input and output embeddings.
        - Implementation of common decoding strategies: greedy, top-k, and top-p sampling.
        - Accurate loss computation for language modeling.
    """

    def __init__(self, config: TransformerConfig): # Constructor for TransformerLM.
        """
        Initialize the Transformer Language Model.

        Args:
            config: Transformer configuration
        """
        super().__init__()   # Calls the parent class (nn.Module) constructor.
        self.config = config # Stores the configuration object.

        # Embeddings
        self.embeddings = TokenAndPositionalEmbedding(config) # Initializes token and positional embeddings.
        self.drop = nn.Dropout(config.hidden_dropout_prob)    # Initializes a dropout layer.

        # Decoder layers (using TransformerDecoderLayer)
        self.h = nn.ModuleList([ # Creates a ModuleList to hold multiple TransformerDecoderLayers.
            TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers) # Instantiates N decoder layers.
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)     # Final Layer Normalization before the output head.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # Language modeling head: projects hidden states to vocabulary size.

        # Initialize weights
        self.apply(self._init_weights) # Applies the custom weight initialization method to all modules.

        # Tie weights if configured
        if config.tie_word_embeddings: # Checks if weight tying is enabled.
            if hasattr(self.embeddings, 'token_embedding') and hasattr(self.embeddings.token_embedding, 'embedding'): # Checks if the necessary embedding layers exist.
                self.lm_head.weight = self.embeddings.token_embedding.embedding.weight # Ties the weights of the language modeling head to the token embedding weights.
            else: # If embedding layers are not found.
                logger.warning( # Logs a warning message.
                    "Cannot tie word embeddings: "
                    "embeddings.token_embedding.embedding not found."
                )

        # Cache for generation (initialized to None)
        self.past_key_values = None # Initializes attribute to store cached key-value states for inference.

    def _init_weights(self, module):
        """
        Initialize the weights.

        Purpose:
            Applies a specific initialization strategy to different types of layers
            within the model. This helps in training stability and performance.
            Linear layers and embeddings are initialized with a normal distribution,
            and LayerNorm biases are zeroed while weights are set to one.

        Inputs:
            - `module` (nn.Module): The module to be initialized.
        """
        if isinstance(module, nn.Linear):                                           # If the module is a linear layer.
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) # Initializes weights with normal distribution.
            if module.bias is not None:                                             # If the linear layer has a bias.
                module.bias.data.zero_()                                            # Initializes bias to zeros.
        elif isinstance(module, nn.Embedding):                                      # If the module is an embedding layer.
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) # Initializes weights with normal distribution.
            if module.padding_idx is not None:                                      # If a padding index is specified.
                module.weight.data[module.padding_idx].zero_()                      # Sets padding token embedding to zeros.
        elif isinstance(module, nn.LayerNorm):                                      # If the module is a LayerNorm layer.
            module.bias.data.zero_()                                                # Initializes bias to zeros.
            module.weight.data.fill_(1.0)                                           # Initializes weight to ones.

    def forward(
        self,
        input_ids: torch.Tensor,                                                         # Input token IDs [batch_size, sequence_length].
        attention_mask: Optional[torch.Tensor] = None,                                   # This is the padding mask for input_ids [batch_size, sequence_length].
        labels: Optional[torch.Tensor] = None,                                           # Optional labels for loss computation [batch_size, sequence_length].
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None, # Optional past key/value states for efficient decoding.
        use_cache: Optional[bool] = None,                                                # Whether to return past key/value states.
    ) -> dict[str, torch.Tensor]:                                                        # Returns a dictionary with logits and optional loss.
        """
        Forward pass for the Transformer Language Model.

        Args:
            input_ids: Input token IDs (batch_size, sequence_length)
            attention_mask: Attention mask (batch_size, sequence_length)
            labels: Optional labels for loss computation (batch_size, sequence_length)
            past_key_values: Optional past key/value states for efficient decoding
            use_cache: Whether to return past key/value states

        Returns:
            Dictionary with logits and optional loss
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache # Determines whether to use caching based on input or config.

        # Calculate lengths
        current_batch_size, current_seq_length = input_ids.shape                           # Gets current batch size and sequence length.
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0 # Determines length of past sequence from cache.
        total_seq_length = past_length + current_seq_length                                # Calculates total sequence length.

        # Embeddings
        hidden_states = self.embeddings(input_ids) # Computes token and positional embeddings.
        hidden_states = self.drop(hidden_states)   # Applies dropout to embeddings.

        # 1. Create Causal Mask
        # This mask ensures that each token can only attend to previous tokens and itself.
        # Shape: (total_seq_length, total_seq_length)
        causal_mask = generate_square_subsequent_mask(total_seq_length, device=input_ids.device) # Generates a square subsequent (causal) mask.
        causal_mask_additive = (1 - causal_mask.float()) * torch.finfo(hidden_states.dtype).min  # Converts mask to additive form (-inf for masked positions).

        # 2. Slice the causal mask for the current query (new tokens only)
        # The attention_scores in MultiHeadAttention will be of shape
        # (batch_size, num_heads, current_seq_length, total_seq_length).
        # So the mask needs to match these last two dimensions.
        # Shape: (current_seq_length, total_seq_length)
        sliced_causal_mask_additive = causal_mask_additive[-current_seq_length:, :] # Slices the causal mask to apply to current queries.

        # Expand sliced_causal_mask_additive for broadcasting with batch and heads
        # Shape: (1, 1, current_seq_length, total_seq_length)
        final_mask_for_mha = sliced_causal_mask_additive[None, None, :, :] # Expands mask dimensions for broadcasting.

        # 3. Incorporate Padding Mask (if input attention_mask is provided)
        if attention_mask is not None: # Checks if a padding mask for input_ids is provided.
            # The input `attention_mask` is for `input_ids` only (current_seq_length).
            # We need to construct a full padding mask for `total_seq_length`.

            # Create a padding mask for past tokens (assuming they were all valid, represented by 1)
            if past_key_values is not None:     # If there are past key-value states.
                past_padding_mask = torch.ones( # Creates a mask for past tokens (all valid).
                    current_batch_size, past_length, dtype=attention_mask.dtype, device=input_ids.device
                )
                # Concatenate past padding mask with the current input_ids attention_mask
                full_padding_mask = torch.cat([past_padding_mask, attention_mask], dim=1) # Combines past and current padding masks.
            else: # If no past key-value states (first forward pass).
                full_padding_mask = attention_mask # Initial call, input_ids is full sequence.

            # Convert boolean/float padding mask to additive attention bias (-inf for padded)
            # Shape: (current_batch_size, total_seq_length) -> (current_batch_size, 1, 1, total_seq_length)
            padding_mask_additive = (1.0 - full_padding_mask.float()) * torch.finfo(hidden_states.dtype).min # Converts padding mask to additive form.
            padding_mask_additive = padding_mask_additive[:, None, None, :] # Expands padding mask dimensions for broadcasting.

            # Combine the padding mask with the (already expanded) sliced causal mask
            final_mask_for_mha = final_mask_for_mha + padding_mask_additive # Combines causal and padding masks.

        new_past_key_values = () if use_cache else None # Initializes tuple for new cached key-value states.

        for i, layer_module in enumerate(self.h): # Iterates through each TransformerDecoderLayer.
            past_key_value = past_key_values[i] if past_key_values is not None else None # Gets past key-value for the current layer if available.

            layer_outputs = layer_module( # Calls the forward pass of the current decoder layer.
                hidden_states,
                attention_mask=final_mask_for_mha, # Pass the carefully constructed mask.
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs["hidden_states"] # Updates hidden states with the output of the current layer.
            if use_cache: # If caching is enabled.
                new_past_key_values += (layer_outputs["past_key_value"],) # Appends the cached key-value of the current layer.

        # Store past_key_values at the model level if caching is enabled
        if use_cache: # If caching is enabled for the entire model.
            self.past_key_values = new_past_key_values # Stores the new cached key-value states.

        hidden_states = self.ln_f(hidden_states) # Applies final Layer Normalization.

        # Output projection
        logits = self.lm_head(hidden_states) # Projects hidden states to vocabulary logits.

        output = {"logits": logits} # Initializes output dictionary with logits.

        # Compute loss if labels are provided
        if labels is not None: # If labels are provided (during training).
            # Shift so that tokens predict next token
            # PyTorch CrossEntropyLoss expects (N, C, ...) where C is num_classes
            # and labels (N, ...)
            shift_logits = logits[..., :-1, :].contiguous() # Shifts logits to predict the next token.
            shift_labels = labels[..., 1:].contiguous()     # Shifts labels to align with shifted logits.

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Assuming -100 is ignore index for padding # Initializes CrossEntropyLoss, ignoring specific index for padding.
            loss = loss_fct(                                  # Computes the loss.
                shift_logits.view(-1, shift_logits.size(-1)), # Reshapes logits for CrossEntropyLoss.
                shift_labels.view(-1),                        # Reshapes labels for CrossEntropyLoss.
            )
            output["loss"] = loss # Adds computed loss to the output dictionary.

        return output # Returns the output dictionary.

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Returns the number of parameters in the model.

        Args:
            only_trainable: If True, only counts trainable parameters.
        """
        if only_trainable:                                                      # If only trainable parameters are requested.
            return sum(p.numel() for p in self.parameters() if p.requires_grad) # Sums elements of trainable parameters.
        else:                                                                   # If all parameters are requested.
            return sum(p.numel() for p in self.parameters())                    # Sums elements of all parameters.

    @torch.no_grad() # Decorator to disable gradient computation.
    def generate(
        self,
        input_ids: torch.Tensor,            # The sequence used as a prompt for the generation [batch_size, initial_sequence_length].
        max_length: int,                    # The maximum length of the sequence to be generated.
        temperature: float = 1.0,           # The value used to modulate the next token probabilities.
        do_sample: bool = False,            # Whether to use sampling; otherwise, use greedy decoding.
        top_k: Optional[int] = None,        # The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p: Optional[float] = None,      # If set to float < 1.0, only the most probable tokens with probabilities that add up to 'top_p' or higher are kept for generation.
        pad_token_id: Optional[int] = None, # The ID of the padding token. Defaults to config.pad_token_id.
        eos_token_id: Optional[int] = None, # The ID of the end-of-sequence token. Defaults to config.eos_token_id.
    ) -> torch.Tensor:                      # Returns a tensor containing the generated sequences.
        """
        Generates sequences of token IDs for models with a language modeling head.

        Args:
            input_ids: The sequence used as a prompt for the generation.
                       (batch_size, initial_sequence_length)
            max_length: The maximum length of the sequence to be generated.
            temperature: The value used to modulate the next token probabilities.
            do_sample: Whether to use sampling; otherwise, use greedy decoding.
            top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p: If set to float < 1.0, only the most probable tokens with probabilities
                   that add up to 'top_p' or higher are kept for generation.
            pad_token_id: The ID of the padding token. Defaults to config.pad_token_id.
            eos_token_id: The ID of the end-of-sequence token. Defaults to config.eos_token_id.

        Returns:
            A tensor containing the generated sequences.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id # Gets pad_token_id from input or config.
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id # Gets eos_token_id from input or config.

        # Initialize past_key_values for caching
        past_key_values = None # Initializes past_key_values to None.

        # Keep track of generated sequences
        generated_ids = input_ids # Starts generated_ids with the input prompt.

        # Set model to evaluation mode for generation
        self.eval() # Sets the model to evaluation mode.

        for _ in range(max_length - input_ids.shape[1]): # Loop until max_length is reached.
            # Get logits from the model's forward pass
            # For the first step, past_key_values is None, input_ids is full prompt
            # For subsequent steps, past_key_values holds prev states, input_ids is just the last token
            outputs = self.forward( # Performs a forward pass.
                input_ids=generated_ids[:, -1:] if past_key_values is not None else generated_ids, # Feeds only the last generated token if caching, else full sequence.
                past_key_values=past_key_values, # Passes cached key-value states.
                use_cache=True,                  # Always use cache during generation.
                attention_mask=None              # No padding mask needed for single token during generation, handled internally.
            )

            logits = outputs["logits"][:, -1, :]   # Logits for the last token.
            past_key_values = self.past_key_values # Retrieve updated past_key_values from model.

            # Apply temperature
            logits = logits / temperature # Divides logits by temperature for sampling control.

            # Apply top-k and top-p filtering
            if top_k is not None: # If top-k sampling is enabled.
                # Remove all tokens with a probability less than the last token in the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] # Identifies tokens to remove based on top-k threshold.
                logits[indices_to_remove] = -float("Inf") # Sets logits of removed tokens to negative infinity.

            if top_p is not None and top_p < 1.0: # If top-p sampling is enabled.
                # Sort by probability
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)       # Sorts logits and gets original indices.
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # Computes cumulative probabilities.

                # Remove tokens with cumulative probability above the threshold (token after the last threshold crossing token)
                sorted_indices_to_remove = cumulative_probs > top_p # Identifies sorted indices to remove.
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # Shifts indices to keep the first token above threshold.
                sorted_indices_to_remove[..., 0] = False # Ensures the highest probability token is not removed.

                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove) # Scatters removed indices back to original positions.
                logits[indices_to_remove] = -float("Inf") # Sets logits of removed tokens to negative infinity.


            # Sample or greedy decode
            if do_sample: # If sampling is enabled.
                probabilities = F.softmax(logits, dim=-1) # Computes probabilities from logits.
                next_token = torch.multinomial(probabilities, num_samples=1) # Samples the next token from the distribution.
            else: # If greedy decoding.
                next_token = torch.argmax(logits, dim=-1, keepdim=True) # Selects the token with the highest probability.

            # Append the next token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1) # Appends the chosen next token to the generated sequence.

            # Stop condition: if EOS token is generated
            if (next_token == eos_token_id).all(): # Checks if all generated tokens are EOS.
                break # Breaks the loop if EOS is generated.

        return generated_ids # Returns the complete generated sequence.