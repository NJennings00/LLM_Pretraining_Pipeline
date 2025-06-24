# filename: src/llm_pipeline/models/embeddings.py
"""
Embedding layers for transformer models.

This module defines various embedding layers crucial for the input processing
of transformer-based Large Language Models (LLMs). Embeddings convert discrete
input tokens (like words or subwords) into continuous vector representations
that the neural network can process. Positional embeddings are also included
to inject information about the relative or absolute position of tokens in a sequence,
which transformers inherently lack due to their self-attention mechanism.

In the context of an LLM pretraining pipeline, these embedding layers are fundamental:
1.  **Token Embedding**: Converts the raw numerical token IDs from the tokenizer
    into dense, low-dimensional vectors. This is the first step in turning
    text data into a format suitable for neural networks.
2.  **Positional Embedding**: Since transformers process sequences in parallel
    without inherent knowledge of token order, positional embeddings provide
    the necessary sequential information. Different types (learnable, sinusoidal,
    Rotary) offer varying trade-offs in terms of expressiveness, generalization
    to longer sequences, and computational cost.
3.  **Combination (TokenAndPositionalEmbedding)**: This layer typically combines
    token and positional embeddings (often by summing them) to create the final
    input representation for the subsequent transformer encoder/decoder layers.
    It often includes normalization and dropout for stable training.

The choice and implementation of these embedding layers directly impact
the model's ability to understand sequence order, handle varying sequence lengths,
and capture semantic meaning from the input text.
"""

import math                             # Imports the math module for mathematical functions, specifically used for `math.log` in sinusoidal embeddings.
from typing import Optional, Tuple, Any # Imports type hints for better code clarity and type checking.
import torch                            # Imports the PyTorch library, essential for building neural networks.
import torch.nn as nn                   # Imports the neural network module from PyTorch.


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Purpose:
        Converts input token IDs into dense, continuous vector representations.
        Each unique token in the vocabulary is mapped to a fixed-size vector.
        Includes a dropout layer to prevent overfitting on the embeddings.

        This is the initial layer for any transformer model that processes text.
        It transforms sparse categorical token IDs into a rich numerical format
        that the subsequent layers of the neural network can learn from.
        Proper initialization and dropout are crucial for stable training.

    LLM Pipeline Fit:
        This module is a core component of the `llm_pipeline.models` package.
        It's instantiated as part of the overall LLM architecture (e.g., within
        a `TransformerModel` or `TokenAndPositionalEmbedding` class) to process
        the raw tokenized input data before it enters the transformer blocks.

    Inputs:
        - `vocab_size` (int): The total number of unique tokens in the vocabulary.
        - `embedding_dim` (int): The dimensionality of the token embeddings.
        - `dropout_prob` (float): The dropout probability applied to the embeddings.
        - `padding_idx` (Optional[int]): If provided, the embeddings corresponding
          to this index will be zeroed out. Useful for ignoring padding tokens.

    Outputs:
        - `embeddings` (torch.Tensor): A tensor of shape
          `[batch_size, seq_length, embedding_dim]` representing the embedded tokens.

    Confirms/Verifies:
        - Correct mapping of token IDs to embeddings.
        - Application of dropout.
        - Proper weight initialization (normal distribution with small std,
          and zeroing out padding index if specified).
    """

    def __init__(
        self,
        vocab_size: int,                   # The size of the vocabulary.
        embedding_dim: int,                # The dimension of the embedding vectors.
        dropout_prob: float = 0.1,         # The probability for dropout.
        padding_idx: Optional[int] = None, # Optional index for padding tokens.
    ):
        super().__init__() # Calls the constructor of the parent class (nn.Module).

        self.embedding = nn.Embedding( # Initializes the PyTorch Embedding layer.
            vocab_size,                # Number of distinct embeddings (vocabulary size).
            embedding_dim,             # Size of each embedding vector.
            padding_idx=padding_idx    # Index to ignore (its embedding will be zero).
        )
        self.dropout = nn.Dropout(dropout_prob) # Initializes the Dropout layer.

        # Initialize weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)   # Initializes embedding weights from a normal distribution.
        if padding_idx is not None:                                  # Checks if a padding index is provided.
            nn.init.constant_(self.embedding.weight[padding_idx], 0) # Sets the embedding for the padding index to all zeros.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: # Defines the forward pass of the module.
        embeddings = self.embedding(input_ids)                  # Looks up embeddings for the given input IDs.
        embeddings = self.dropout(embeddings)                   # Applies dropout to the embeddings.
        return embeddings                                       # Returns the processed embeddings.


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    Purpose:
        Provides learnable vector representations for each position in a sequence.
        This allows the model to understand the order of tokens, as transformers
        are inherently permutation-invariant without such positional information.

        Learnable positional embeddings are a common way to inject sequence order
        into transformer models. They can adapt to specific datasets and tasks,
        potentially capturing more nuanced positional relationships than fixed embeddings.

    LLM Pipeline Fit:
        This module is part of the `llm_pipeline.models` and is used in conjunction
        with `TokenEmbedding` (e.g., within `TokenAndPositionalEmbedding`). It's
        responsible for adding spatial awareness to the token representations before
        they are fed into the self-attention layers.

    Inputs:
        - `max_position_embeddings` (int): The maximum sequence length the model
          is designed to handle, which determines the size of the positional
          embedding lookup table.
        - `embedding_dim` (int): The dimensionality of the positional embeddings.
          This should typically match the token embedding dimension.
        - `dropout_prob` (float): The dropout probability applied to the positional embeddings.

    Outputs:
        - `embeddings` (torch.Tensor): A tensor of shape
          `[batch_size, seq_length, embedding_dim]` representing the positional embeddings.

    Confirms/Verifies:
        - Correct lookup of positional embeddings based on `position_ids`.
        - Application of dropout.
        - Proper weight initialization.
    """

    def __init__(
        self,
        max_position_embeddings: int, # The maximum number of positions (sequence length) to embed.
        embedding_dim: int,           # The dimension of the positional embedding vectors.
        dropout_prob: float = 0.1,    # The probability for dropout.
    ):
        super().__init__() # Calls the constructor of the parent class (nn.Module).

        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_dim) # Initializes a standard embedding layer for positions.
        self.dropout = nn.Dropout(dropout_prob)                                         # Initializes the Dropout layer.

        # Initialize weights
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02) # Initializes positional embedding weights from a normal distribution.

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor: # Defines the forward pass.
        embeddings = self.position_embeddings(position_ids)        # Looks up positional embeddings for the given position IDs.
        embeddings = self.dropout(embeddings)                      # Applies dropout to the positional embeddings.
        return embeddings                                          # Returns the processed positional embeddings.


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings (non-learnable).

    Purpose:
        Implements fixed, non-learnable positional embeddings using sine and cosine
        functions of different frequencies. This approach, introduced in the
        original Transformer paper, allows for arbitrary sequence lengths
        during inference without extrapolation issues, as the patterns are
        mathematically defined.

        Sinusoidal embeddings offer an alternative to learnable ones. Their
        advantage is the ability to generalize to sequence lengths longer than
        those seen during training, as the embedding for any position is deterministically
        calculated. They don't add to the model's parameters, making them
        computationally lighter.

    LLM Pipeline Fit:
        This module provides another option for positional encoding within the
        `llm_pipeline.models` package. It can be selected based on model
        architecture choices or experimental requirements (e.g., when trying to
        minimize model size or enhance generalization to very long sequences).

    Inputs:
        - `max_position_embeddings` (int): The maximum expected sequence length.
          While sinusoidal embeddings can generalize, this parameter defines
          the pre-computed table size.
        - `embedding_dim` (int): The dimensionality of the positional embeddings.
        - `dropout_prob` (float): The dropout probability applied to the embeddings.

    Outputs:
        - `embeddings` (torch.Tensor): A tensor of shape
          `[batch_size, seq_length, embedding_dim]` representing the sinusoidal
          positional embeddings.

    Confirms/Verifies:
        - Correct generation of sinusoidal patterns.
        - Proper application of dropout.
        - Correct handling of sequence length by slicing the pre-computed table.
        - Registration of the embedding table as a buffer (not a trainable parameter).
    """

    def __init__(
        self,
        max_position_embeddings: int, # The maximum number of positions for which to pre-compute embeddings.
        embedding_dim: int,           # The dimension of the positional embedding vectors.
        dropout_prob: float = 0.1,    # The probability for dropout.
    ):
        super().__init__() # Calls the constructor of the parent class.

        self.dropout = nn.Dropout(dropout_prob) # Initializes the Dropout layer.

        # Create sinusoidal embeddings
        position = torch.arange(max_position_embeddings).unsqueeze(1)                # Creates a tensor of positions (0 to max_position_embeddings-1).
        div_term = torch.exp(                                                        # Calculates the division term for frequencies.
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim) # Exponential decay term.
        )

        pe = torch.zeros(max_position_embeddings, embedding_dim) # Initializes a tensor of zeros for positional embeddings.
        pe[:, 0::2] = torch.sin(position * div_term)             # Applies sine function to even indices.
        pe[:, 1::2] = torch.cos(position * div_term)             # Applies cosine function to odd indices.

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0)) # Registers `pe` as a buffer, meaning it's part of the model's state but not updated by optimizers. The unsqueeze(0) adds a batch dimension.

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor: # Defines the forward pass.
        batch_size, seq_length = position_ids.shape # Gets batch size and sequence length from input `position_ids`.
        # Ensure pe is on the same device as position_ids
        # Slices `self.pe` to match the current `seq_length` and expands it to match the batch size.
        # It's crucial to move `self.pe` to the same device as `position_ids` if they differ.
        embeddings = self.pe[:, :seq_length, :].expand(batch_size, -1, -1).to(position_ids.device)
        embeddings = self.dropout(embeddings) # Applies dropout.
        return embeddings # Returns the sinusoidal positional embeddings.


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings (RoPE).

    Purpose:
        Implements Rotary Positional Embeddings (RoPE), a method that
        encodes relative positional information directly into the self-attention
        mechanism by rotating the query and key vectors. This allows for
        modeling long dependencies and improved generalization to longer sequences
        compared to absolute positional embeddings.

        RoPE has become a popular choice in modern LLMs (e.g., LLaMA, GPT-NeoX)
        due to its theoretical advantages in handling long contexts and its empirical
        performance. It encodes relative position information, which is often
        more relevant for attention mechanisms.

    LLM Pipeline Fit:
        This is an advanced positional encoding strategy for the `llm_pipeline.models`
        package. It would be integrated directly into the attention mechanism
        (e.g., within `MultiHeadAttention` or `SelfAttention` modules) rather
        than simply added to token embeddings. It provides a state-of-the-art
        approach to positional information.

    Inputs:
        - `dim` (int): The dimension of the head (or embedding dimension if applied
          before splitting into heads) to which RoPE will be applied. This must be
          an even number.
        - `max_position_embeddings` (int): The maximum length for which to pre-compute
          the rotation angles. This can be extended dynamically.
        - `base` (int): The base value for the inverse frequencies, influencing
          how the rotation angles change with position. Commonly 10000.

    Outputs:
        - `query_rotated` (torch.Tensor): The query tensor with rotary embeddings applied.
        - `key_rotated` (torch.Tensor): The key tensor with rotary embeddings applied.

    Confirms/Verifies:
        - Correct calculation of inverse frequencies and rotation angles.
        - Proper application of rotation to query and key tensors.
        - Efficient caching and updating of `cos` and `sin` values based on sequence length.
        - Handling of device placement for cached tensors.
    """

    def __init__(
        self,
        dim: int,                            # The dimension of the vectors to which rotary embeddings will be applied (e.g., head dimension).
        max_position_embeddings: int = 2048, # The maximum sequence length for which to pre-compute rotation angles.
        base: int = 10000,                   # The base for the inverse frequency calculation.
    ):
        super().__init__() # Calls the constructor of the parent class.

        self.dim = dim # Stores the dimension.
        self.max_position_embeddings = max_position_embeddings # Stores max position.
        self.base = base # Stores the base.

        # Compute inverse frequencies
        # Calculates `1 / (base^(2i/dim))` for i from 0 to dim/2 - 1.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq) # Registers `inv_freq` as a buffer.

        # Initialize cache cos and sin values for first use
        # Initialize to 0 so comparison 'seq_len > self._seq_len_cached' works on first call.
        self._seq_len_cached = 0
        # Initialize to empty tensors on CPU. They will be moved to correct device on first use.
        self._cos_cached = torch.empty(0, device='cpu')
        self._sin_cached = torch.empty(0, device='cpu')

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device):
        """Update cached cos and sin values."""
        # Recompute only if cache is too small or device changed
        # self._seq_len_cached will be 0 on first call, so seq_len > 0 will be true.
        # For subsequent calls, self._cos_cached will be a tensor, so .device can be accessed.
        if seq_len > self._seq_len_cached or self._cos_cached.device != device: # Checks if the current sequence length exceeds the cached length or if the device has changed.
            self._seq_len_cached = seq_len # Updates the cached sequence length.

            # Compute positions
            t = torch.arange(seq_len, device=device).float() # Generates a tensor of positions from 0 to `seq_len-1` on the specified device.

            # Ensure inv_freq is on the correct device before multiplication
            inv_freq_device = self.inv_freq.to(device) # Moves `inv_freq` to the specified device.

            # Compute frequencies
            freqs = torch.einsum("i,j->ij", t, inv_freq_device) # Computes the outer product of positions `t` and `inv_freq` to get frequencies.
            emb = torch.cat((freqs, freqs), dim=-1)             # Concatenates `freqs` with itself to create `[..., dim]` for sine and cosine.

            # Cache cos and sin
            self._cos_cached = emb.cos()[None, None, :, :] # Computes cosine of frequencies and adds two unsqueezed dimensions for batch and heads.
            self._sin_cached = emb.sin()[None, None, :, :] # Computes sine of frequencies and adds two unsqueezed dimensions for batch and heads.

    def forward(
        self,
        query: torch.Tensor, # Query tensor [batch_size, num_heads, seq_len, head_dim].
        key: torch.Tensor # Key tensor [batch_size, num_heads, seq_len, head_dim].
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]

        Returns:
            Tuple of rotated query and key tensors
        """
        batch_size, num_heads, seq_len, head_dim = query.shape # Unpacks the dimensions of the query tensor.

        assert head_dim == self.dim, f"Head dim {head_dim} != rope dim {self.dim}" # Asserts that the head dimension matches the RoPE dimension.

        # Update cache if needed.
        # The condition here is `seq_len > self._seq_len_cached`.
        # The `_update_cos_sin_cache` method itself now handles the device consistency check.
        self._update_cos_sin_cache(seq_len, query.device) # Calls the helper to update the cached cos/sin values.

        # Apply rotary embeddings
        # Ensure cached tensors are sliced to the current seq_len before applying.
        # This is critical if the cached tensors are longer than the current seq_len.
        cos_sliced = self._cos_cached[:, :, :seq_len, :] # Slices the cached cosine values to the current sequence length.
        sin_sliced = self._sin_cached[:, :, :seq_len, :] # Slices the cached sine values to the current sequence length.

        query = self._apply_rotary_emb(query, cos_sliced, sin_sliced) # Applies rotary embedding to the query tensor.
        key = self._apply_rotary_emb(key, cos_sliced, sin_sliced)     # Applies rotary embedding to the key tensor.

        return query, key # Returns the rotated query and key tensors.

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,   # The input tensor (query or key).
        cos: torch.Tensor, # The cosine rotation tensor.
        sin: torch.Tensor  # The sine rotation tensor.
    ) -> torch.Tensor:
        """Apply rotary embedding to tensor."""
        # Split into two halves
        x1, x2 = x.chunk(2, dim=-1) # Splits the input tensor `x` into two halves along the last dimension.

        # Apply rotation
        # The cos and sin tensors are already sliced in the forward method
        # so we don't need to slice them again here.

        rotated = torch.cat((-x2, x1), dim=-1)  # Creates the rotated version of `x` ([-x2, x1]).
        x_rotated = (x * cos) + (rotated * sin) # Applies the rotary embedding formula: `x * cos + rotate(x) * sin`.

        return x_rotated # Returns the tensor with rotary embeddings applied.


class TokenAndPositionalEmbedding(nn.Module):
    """
    Combines token embeddings with positional embeddings.

    Purpose:
        This module integrates token embeddings with positional embeddings to
        produce the final input representation for a transformer model. It
        also applies layer normalization and dropout to this combined embedding.
        It handles different types of positional embeddings and sequence length
        management to prevent indexing errors.

        This is a standard input layer for most transformer architectures. It
        ensures that both the semantic meaning of tokens and their sequential
        order are captured in the input representation. The normalization and
        dropout steps are critical for stabilizing training and improving generalization.

    LLM Pipeline Fit:
        This module is the entry point for raw tokenized data into the main
        transformer layers of an LLM. It's typically the first layer of a
        `TransformerModel` and processes the `input_ids` from the data loader
        before they are fed into attention and feed-forward networks.

    Inputs:
        - `config` (Any): A configuration object (e.g., `ModelConfig`) that
          contains parameters like `vocab_size`, `hidden_size`,
          `max_position_embeddings`, `hidden_dropout_prob`, and `layer_norm_eps`.
          Using `Any` avoids circular imports if the config is defined in another module.

    Outputs:
        - `embeddings` (torch.Tensor): The combined and processed embeddings,
          of shape `[batch_size, sequence_length, hidden_size]`.

    Confirms/Verifies:
        - Correct combination of token and positional embeddings (summation).
        - Application of Layer Normalization.
        - Application of Dropout.
        - Robustness to `input_ids` with values outside the vocabulary size
          (by mapping them to an UNK token, ID 0, and issuing a warning).
        - Correct generation and handling of `position_ids`, including clamping
          for sequences longer than `max_position_embeddings`.
    """

    def __init__(self, config: Any): # Uses Any for generic config to avoid circular import if config classes are defined elsewhere
        super().__init__()           # Calls the constructor of the parent class.
        self.config = config         # Stores the configuration object.

        self.token_embedding = TokenEmbedding(       # Initializes the TokenEmbedding layer.
            vocab_size=config.vocab_size,            # Uses vocabulary size from config.
            embedding_dim=config.hidden_size,        # Uses hidden size as embedding dimension.
            dropout_prob=config.hidden_dropout_prob, # Uses dropout probability from config.
            padding_idx=None                         # Assuming no specific padding_idx for embedding layer here, or it would come from config.
        )

        # Choose positional embedding type
        # You can add logic here to choose between PositionalEmbedding (learnable)
        # and SinusoidalPositionalEmbedding (fixed) based on a config flag if desired.
        # For simplicity, let's default to PositionalEmbedding.
        self.position_embedding = PositionalEmbedding(              # Initializes the PositionalEmbedding layer (learnable).
            max_position_embeddings=config.max_position_embeddings, # Uses max position embeddings from config.
            embedding_dim=config.hidden_size,                       # Uses hidden size as embedding dimension.
            dropout_prob=config.hidden_dropout_prob,                # Uses dropout probability from config.
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # Initializes Layer Normalization.
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # Initializes Dropout layer for the combined embeddings.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token and positional embeddings.

        Args:
            input_ids: Input token IDs (batch_size, sequence_length)

        Returns:
            Combined embeddings (batch_size, sequence_length, hidden_size)
        """
        seq_length = input_ids.shape[1] # Gets the sequence length from the input.
        # Clamp sequence length to max_position_embeddings to avoid index errors
        max_pos = self.config.max_position_embeddings # Retrieves maximum position from config.
        clamped_seq_length = min(seq_length, max_pos) # Clamps the sequence length to prevent out-of-bounds indexing for positional embeddings.
        position_ids = torch.arange(clamped_seq_length, dtype=torch.long, device=input_ids.device) # Creates position IDs from 0 to clamped_seq_length-1.

        # For sequences longer than max_position_embeddings, repeat the last position
        if seq_length > max_pos: # If the sequence is longer than the defined max positions.
            # Extend position_ids by repeating the last valid position
            extra_positions = torch.full((seq_length - max_pos,), max_pos - 1,
                                         dtype=torch.long, device=input_ids.device) # Creates a tensor of the last valid position ID for the excess length.
            position_ids = torch.cat([position_ids, extra_positions]) # Concatenates the original position IDs with the extra ones.

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) # Expands `position_ids` to match the batch and sequence dimensions of `input_ids`.

        # Check for invalid token IDs and fix them
        invalid_tokens = input_ids >= self.config.vocab_size # Identifies token IDs that are out of bounds for the vocabulary.
        if invalid_tokens.any():                                                 # If any invalid tokens are found.
            print(f"WARNING: Found {invalid_tokens.sum()} invalid token IDs!")   # Prints a warning.
            print(f"Invalid token IDs: {input_tokens[invalid_tokens].unique()}") # Prints the unique invalid token IDs.
            print("Mapping invalid token IDs to UNK token (ID=0)")               # Informs about mapping to UNK.
            # Map invalid token IDs to UNK token (assuming ID 0 is UNK)
            input_ids = torch.where(input_ids >= self.config.vocab_size,
                                    torch.tensor(0, device=input_ids.device), # Maps to 0 (UNK) if invalid.
                                    input_ids)                                # Otherwise keeps the original ID.

        token_embeddings = self.token_embedding(input_ids) # Computes token embeddings.

        # DEBUG: Print values before position embedding
        # print(f"DEBUG: seq_length={seq_length}, max_pos={self.config.max_position_embeddings}") # Debug print for sequence length and max positions.
        # print(f"DEBUG: position_ids shape={position_ids.shape}, min={position_ids.min()}, max={position_ids.max()}") # Debug print for position IDs.
        # print(f"DEBUG: input_ids shape={input_ids.shape}, min={input_ids.min()}, max={input_ids.max()}") # Debug print for input IDs.
        # print(f"DEBUG: Expected vocab_size={self.config.vocab_size}") # Debug print for expected vocabulary size.

        position_embeddings = self.position_embedding(position_ids) # Computes positional embeddings.

        embeddings = token_embeddings + position_embeddings # Combines token and positional embeddings by summation.
        embeddings = self.layer_norm(embeddings) # Applies layer normalization.
        embeddings = self.dropout(embeddings) # Applies dropout.

        return embeddings # Returns the combined and processed embeddings.