# filename: src/llm_pipeline/models/layers.py
"""
Transformer layer implementations.

This module provides the building blocks for constructing transformer models,
specifically focusing on the individual layers that compose an LLM. It includes
implementations for:
1.  **Layer Normalization**: A normalization technique applied across the features
    of each sample independently, improving training stability.
2.  **Multi-Head Attention**: The core mechanism of transformers, allowing the
    model to weigh the importance of different parts of the input sequence when
    processing each token. It includes support for Rotary Positional Embeddings (RoPE).
3.  **Feed-Forward Network (FFN)**: A position-wise fully connected network
    applied to each token's representation independently, adding non-linearity
    to the model.
4.  **TransformerDecoderLayer (formerly TransformerBlock)**: Combines the
    multi-head attention and feed-forward network with residual connections
    and layer normalization to form a complete decoder layer, as typically
    found in autoregressive LLMs.

In an LLM pretraining pipeline, these layers are stacked together to form the
deep neural network that learns language representations. Their efficient and
correct implementation is crucial for:
-   **Learning Long-Range Dependencies**: Multi-Head Attention enables the model
    to capture relationships between distant tokens.
-   **Training Stability**: Layer Normalization helps prevent vanishing/exploding
    gradients during the training of deep networks.
-   **Model Capacity**: The Feed-Forward Networks provide the non-linearity
    and capacity for the model to learn complex functions.
-   **Autoregressive Generation**: The `TransformerDecoderLayer` is specifically
    designed for sequential generation, making it suitable for language modeling.
"""

import math                             # Imports the math module, although not directly used in this version, often useful for numerical operations.
from typing import Optional, Tuple, Any # Added Any for type hint in TransformerDecoderLayer init # Imports type hints for better code readability and type checking.
import torch                            # Imports the PyTorch library, fundamental for defining and operating on neural networks.
import torch.nn as nn                   # Imports the neural network module from PyTorch.
import torch.nn.functional as F         # Imports functional interface for common neural network operations, like softmax.

from llm_pipeline.models.embeddings import RotaryEmbedding # Imports RotaryEmbedding for advanced positional encoding.
from llm_pipeline.models.utils import get_activation_fn    # Imports a utility function to retrieve activation functions.


class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias.

    Purpose:
        Applies layer normalization to the input tensor. This technique normalizes
        the features across the hidden dimension for each individual sample in the batch,
        improving training stability and convergence speed in deep neural networks.
        It also includes learnable scale (`weight`) and optional shift (`bias`) parameters.

        Layer normalization is a standard component in modern transformer architectures.
        It helps to mitigate the vanishing/exploding gradient problem by maintaining
        mean activation close to zero and standard deviation close to one across layers.

    LLM Pipeline Fit:
        This module is used multiple times within each `TransformerDecoderLayer`
        (e.g., before self-attention and before the feed-forward network). It's a
        critical part of the foundational structure that enables the deep stacking
        of transformer blocks in an LLM pretraining pipeline.

    Inputs:
        - `hidden_size` (int): The size of the last dimension (feature dimension)
          to normalize.
        - `eps` (float): A small value added to the variance for numerical stability
          to prevent division by zero. Defaults to 1e-12.
        - `bias` (bool): Whether to include a learnable bias parameter. Defaults to True.

    Outputs:
        - `x` (torch.Tensor): The normalized tensor, with the same shape as the input.

    Confirms/Verifies:
        - Correct calculation of mean and variance across the last dimension.
        - Proper application of normalization formula.
        - Learnable scale and optional bias parameters are correctly applied.
        - Numerical stability via `eps`.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-12, bias: bool = True): # Defines the constructor for LayerNorm.
        super().__init__()                                                       # Calls the parent class (nn.Module) constructor.
        self.weight = nn.Parameter(torch.ones(hidden_size))                      # Initializes a learnable weight parameter (scale) as a tensor of ones.
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None     # Initializes a learnable bias parameter (shift) as a tensor of zeros if `bias` is True, otherwise None.
        self.eps = eps                                                           # Stores the epsilon value for numerical stability.

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Defines the forward pass of the layer.
        mean = x.mean(-1, keepdim=True)                 # Computes the mean of `x` along the last dimension, keeping the dimension for broadcasting.
        var = x.var(-1, keepdim=True, unbiased=False)   # Computes the variance of `x` along the last dimension, keeping the dimension. `unbiased=False` for population variance.
        x = (x - mean) / torch.sqrt(var + self.eps)     # Applies the normalization formula: (x - mean) / sqrt(variance + epsilon).
        x = self.weight * x                             # Multiplies the normalized `x` by the learnable `weight` (scaling).
        if self.bias is not None:                       # Checks if a bias parameter exists.
            x = x + self.bias                           # Adds the learnable `bias` if it exists.
        return x                                        # Returns the normalized tensor.


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Purpose:
        Implements the core self-attention mechanism, which allows the model
        to weigh the importance of different tokens in the input sequence when
        computing the representation for each token. It performs attention
        multiple times in parallel (multiple heads) and concatenates their
        outputs. It optionally supports Rotary Positional Embeddings (RoPE)
        for improved handling of positional information and caching of
        past key/value states for efficient autoregressive decoding.

        Multi-head attention is the defining component of transformer models.
        Its ability to capture long-range dependencies and complex relationships
        between tokens is central to the success of LLMs. Its correct implementation
        is paramount for model performance.

    LLM Pipeline Fit:
        This module is a fundamental building block within each `TransformerDecoderLayer`
        in the `llm_pipeline.models` package. It's where the model performs
        the crucial context-aware processing of input sequences, making it a
        central component in the deep learning architecture of an LLM.

    Inputs:
        - `hidden_size` (int): The dimensionality of the input and output features.
        - `num_attention_heads` (int): The number of attention heads. `hidden_size`
          must be divisible by `num_attention_heads`.
        - `attention_probs_dropout_prob` (float): Dropout probability for attention weights.
        - `use_rotary_embeddings` (bool): Whether to use RoPE.
        - `rotary_dim` (Optional[int]): The dimension for RoPE if used. Defaults to `head_dim`.
        - `max_position_embeddings` (int): Maximum sequence length for RoPE.

    Outputs:
        - `attention_output` (torch.Tensor): The output tensor after attention and
          projection, of shape `[batch_size, seq_length, hidden_size]`.
        - `present` (Optional[Tuple[torch.Tensor, torch.Tensor]]): A tuple
          containing the concatenated `(key, value)` tensors if `use_cache` is True,
          for efficient subsequent decoding steps.

    Confirms/Verifies:
        - Correct linear projections for Query, Key, and Value.
        - Proper reshaping for multi-head attention.
        - Accurate computation of scaled dot-product attention scores.
        - Correct application of attention mask for causality or padding.
        - Application of softmax and dropout to attention probabilities.
        - Proper concatenation and projection of attention heads.
        - Optional integration of Rotary Positional Embeddings.
        - Correct caching and retrieval of past key/value states for autoregression.
    """

    def __init__(
        self,
        hidden_size: int,                          # The dimension of the input and output features.
        num_attention_heads: int,                  # The number of parallel attention heads.
        attention_probs_dropout_prob: float = 0.1, # Dropout rate for attention probabilities.
        use_rotary_embeddings: bool = False,       # Flag to enable/disable Rotary Positional Embeddings.
        rotary_dim: Optional[int] = None,          # Dimension for RoPE; defaults to head_dim.
        max_position_embeddings: int = 512,        # Max sequence length for RoPE.
    ):
        super().__init__() # Calls the parent class constructor.

        assert hidden_size % num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads" # Ensures hidden_size can be evenly divided by num_attention_heads.

        self.hidden_size = hidden_size                     # Stores hidden size.
        self.num_attention_heads = num_attention_heads     # Stores number of heads.
        self.head_dim = hidden_size // num_attention_heads # Calculates the dimension of each attention head.
        self.scale = self.head_dim ** -0.5                 # Calculates the scaling factor for dot-product attention.

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size) # Linear layer for query projection.
        self.k_proj = nn.Linear(hidden_size, hidden_size) # Linear layer for key projection.
        self.v_proj = nn.Linear(hidden_size, hidden_size) # Linear layer for value projection.

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size) # Linear layer for projecting concatenated head outputs back to hidden_size.

        # Dropout
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob) # Dropout for attention probabilities.

        # Rotary embeddings
        if use_rotary_embeddings:                                # Checks if Rotary Embeddings are enabled.
            self.rotary_emb = RotaryEmbedding(                   # Initializes RotaryEmbedding.
                dim=rotary_dim or self.head_dim,                 # Uses `rotary_dim` if provided, else `head_dim`.
                max_position_embeddings=max_position_embeddings, # Passes max position embeddings.
            )
        else: # If not using rotary embeddings.
            self.rotary_emb = None # Sets `rotary_emb` to None.

    def forward(
        self,
        hidden_states: torch.Tensor,                                        # Input tensor representing the hidden states [batch_size, seq_length, hidden_size].
        attention_mask: Optional[torch.Tensor] = None,                      # Optional attention mask [batch_size, 1, seq_length, seq_length].
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Optional tuple of cached key and value tensors from previous steps.
        use_cache: bool = False,                                            # Flag indicating whether to return the current key and value for caching.
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:  # Returns a tuple: output tensor and optional cached key-value.
        """
        Forward pass of multi-head attention.

        Args:
            hidden_states: Input hidden states [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_length, seq_length]
            past_key_value: Cached key and value tensors
            use_cache: Whether to return key and value for caching

        Returns:
            Tuple of output tensor and optional cached key-value pair
        """
        batch_size, seq_length, _ = hidden_states.shape # Extracts batch size and sequence length from hidden_states.

        # Project to Q, K, V
        query = self.q_proj(hidden_states) # Projects hidden_states to query.
        key = self.k_proj(hidden_states)   # Projects hidden_states to key.
        value = self.v_proj(hidden_states) # Projects hidden_states to value.

        # Reshape for multi-head attention
        # [batch_size, seq_length, hidden_size] -> [batch_size, num_heads, seq_length, head_dim]
        query = query.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2) # Reshapes query for multi-head attention and transposes dimensions.
        key = key.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)     # Reshapes key.
        value = value.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2) # Reshapes value.

        # Apply rotary embeddings if enabled
        if self.rotary_emb is not None: # Checks if Rotary Embeddings are used.
            query, key = self.rotary_emb(query, key) # Applies RoPE to query and key.

        # Handle cached keys and values
        if past_key_value is not None:                    # Checks if past key-value states are provided (for inference).
            past_key, past_value = past_key_value         # Unpacks cached key and value.
            key = torch.cat([past_key, key], dim=2)       # Concatenates current key with past key along the sequence length dimension.
            value = torch.cat([past_value, value], dim=2) # Concatenates current value with past value.

        # Cache key and value if requested
        present = (key, value) if use_cache else None # Stores current key and value in `present` if caching is enabled.

        # Compute attention scores
        # [batch_size, num_heads, seq_length, head_dim] x [batch_size, num_heads, head_dim, key_length]
        # -> [batch_size, num_heads, seq_length, key_length]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale # Computes dot product of query and transposed key, then scales it.

        # Apply attention mask
        if attention_mask is not None: # Checks if an attention mask is provided.
            attention_scores = attention_scores + attention_mask # Adds the mask to attention scores (mask values are typically very large negative numbers for masked positions).

        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1) # Applies softmax to get attention probabilities.
        attention_probs = self.attn_dropout(attention_probs)  # Applies dropout to attention probabilities.

        # Apply attention to values
        # [batch_size, num_heads, seq_length, key_length] x [batch_size, num_heads, key_length, head_dim]
        # -> [batch_size, num_heads, seq_length, head_dim]
        attention_output = torch.matmul(attention_probs, value) # Multiplies attention probabilities by value.

        # Reshape back
        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, hidden_size]
        attention_output = attention_output.transpose(1, 2).contiguous().view( # Transposes dimensions back, makes contiguous, and reshapes to original hidden_size.
            batch_size, -1, self.hidden_size                                   # Use -1 instead of seq_length here, as seq_length can change with cache.
        )

        # Final projection
        attention_output = self.out_proj(attention_output) # Projects the concatenated attention output.

        return attention_output, present # Returns the final attention output and the cached key-value pair (if `use_cache` is True).


class FeedForward(nn.Module):
    """
    Feed-forward network.

    Purpose:
        Implements a position-wise feed-forward network (FFN), consisting of two
        linear transformations with an activation function in between. This layer
        is applied independently to each position in the sequence and adds
        non-linearity to the model, increasing its capacity to learn complex patterns.

        The FFN is a crucial component of each transformer layer, providing the
        computational capacity for the model to process the information aggregated
        by the attention mechanism. It's standard practice in transformer models.

    LLM Pipeline Fit:
        This module is used within each `TransformerDecoderLayer` in the
        `llm_pipeline.models` package. After the attention mechanism processes
        the input and generates a context-aware representation, the FFN
        further transforms this representation before it is passed to the next layer
        or the final output head.

    Inputs:
        - `hidden_size` (int): The input and output dimensionality of the FFN.
        - `intermediate_size` (int): The dimensionality of the hidden layer within the FFN.
          Typically, `intermediate_size` is 4 times `hidden_size`.
        - `hidden_act` (str): The name of the activation function to use (e.g., "gelu", "relu").
        - `hidden_dropout_prob` (float): The dropout probability applied within the FFN.

    Outputs:
        - `hidden_states` (torch.Tensor): The output tensor from the FFN, with
          the same shape as the input `hidden_size`.

    Confirms/Verifies:
        - Correct two-layer linear transformation.
        - Proper application of the specified activation function.
        - Application of dropout layers.
    """

    def __init__(
        self,
        hidden_size: int,                 # Input and output dimension of the FFN.
        intermediate_size: int,           # Dimension of the intermediate layer.
        hidden_act: str = "gelu",         # Name of the activation function to use.
        hidden_dropout_prob: float = 0.1, # Dropout probability.
    ):
        super().__init__() # Calls the parent class constructor.

        self.fc1 = nn.Linear(hidden_size, intermediate_size) # First linear layer: hidden_size -> intermediate_size.
        self.fc2 = nn.Linear(intermediate_size, hidden_size) # Second linear layer: intermediate_size -> hidden_size.
        self.act_fn = get_activation_fn(hidden_act)          # Retrieves the activation function based on its string name.
        self.dropout = nn.Dropout(hidden_dropout_prob)       # Initializes the Dropout layer.

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # Defines the forward pass.
        hidden_states = self.fc1(hidden_states)                     # Applies the first linear transformation.
        hidden_states = self.act_fn(hidden_states)                  # Applies the activation function.
        hidden_states = self.dropout(hidden_states)                 # Applies dropout after the activation.
        hidden_states = self.fc2(hidden_states)                     # Applies the second linear transformation.
        hidden_states = self.dropout(hidden_states)                 # Applies dropout again.
        return hidden_states                                        # Returns the output of the feed-forward network.


# Renamed TransformerBlock to TransformerDecoderLayer
class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with attention and feed-forward.

    Purpose:
        This class represents a single complete decoder layer of a transformer
        model. It encapsulates a self-attention sub-layer followed by a
        position-wise feed-forward network. Both sub-layers include
        residual connections and layer normalization, which are critical
        for training deep networks. This architecture is characteristic of
        decoder-only LLMs used for autoregressive language modeling.

        This layer is the fundamental repeatable unit that forms the deep
        architecture of transformer models. Its correct implementation and
        ability to pass information effectively across its sub-layers are
        crucial for the model's overall learning capacity and performance.

    LLM Pipeline Fit:
        Multiple instances of this `TransformerDecoderLayer` are stacked
        sequentially within the main `TransformerLM` model in the
        `llm_pipeline.models` package. Each layer refines the token representations
        by incorporating more context, ultimately leading to highly informed
        predictions for the next token. This modular design facilitates
        the creation of deep LLMs.

    Inputs:
        - `config` (Any): A configuration object (e.g., `TransformerConfig`)
          that contains all necessary hyperparameters for initializing the
          attention, feed-forward, and normalization layers (e.g., `hidden_size`,
          `num_attention_heads`, `hidden_act`, `attention_probs_dropout_prob`, etc.).
          Using `Any` avoids circular imports if the config classes are defined elsewhere.

    Outputs:
        - `dict[str, Any]`: A dictionary containing:
            - `hidden_states` (torch.Tensor): The output hidden states from
              this layer.
            - `past_key_value` (Optional[Tuple[torch.Tensor, torch.Tensor]]):
              The cached key and value states from the attention mechanism for
              efficient autoregressive decoding in subsequent calls.

    Confirms/Verifies:
        - Correct sequential application of LayerNorm, MultiHeadAttention, Dropout,
          LayerNorm, and FeedForward.
        - Proper implementation of residual connections.
        - Correct passing of `attention_mask` and `past_key_value` to the attention sub-layer.
        - Accurate caching mechanism for efficient inference.
    """

    def __init__(
        self,
        config: Any, # Use Any for generic config input, as it comes from TransformerConfig.
    ):
        super().__init__() # Calls the parent class constructor.

        self.config = config # Store config for easy access.

        # Attention
        self.attention = MultiHeadAttention(                                  # Initializes the MultiHeadAttention sub-layer.
            hidden_size=config.hidden_size,                                   # Passes hidden size from config.
            num_attention_heads=config.num_attention_heads,                   # Passes number of attention heads from config.
            attention_probs_dropout_prob=config.attention_probs_dropout_prob, # Passes attention dropout.
            use_rotary_embeddings=config.use_rotary_embeddings,               # Passes flag for RoPE.
            rotary_dim=config.rotary_dim,                                     # Passes RoPE dimension.
            max_position_embeddings=config.max_position_embeddings,           # Passes max positions for RoPE.
        )

        # Feed-forward
        self.feed_forward = FeedForward(                    # Initializes the FeedForward sub-layer.
            hidden_size=config.hidden_size,                 # Passes hidden size.
            intermediate_size=config.intermediate_size,     # Passes intermediate size.
            hidden_act=config.hidden_act,                   # Passes activation function name.
            hidden_dropout_prob=config.hidden_dropout_prob, # Passes dropout for FFN.
        )

        # Layer norms
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # First LayerNorm before attention.
        self.ln_2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # Second LayerNorm before feed-forward.

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # Dropout for residual connections.

    def forward(
        self,
        hidden_states: torch.Tensor,                                        # Input hidden states for the layer.
        attention_mask: Optional[torch.Tensor] = None,                      # Optional attention mask.
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Optional cached key-value pair.
        use_cache: bool = False,                                            # Flag to indicate if key-value pairs should be cached.
    ) -> dict[str, Any]:                                                    # Returns a dictionary containing the output hidden states and optional cached key-value pair.
        """
        Forward pass of transformer decoder layer.

        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            past_key_value: Cached key-value pair
            use_cache: Whether to cache key-value pairs

        Returns:
            Dictionary with output tensor and optional cached key-value pair
        """
        # Self-attention with residual connection
        residual = hidden_states                    # Stores the input hidden_states for the residual connection.
        hidden_states = self.ln_1(hidden_states)    # Applies Layer Normalization.
        attention_output, present = self.attention( # Passes through the MultiHeadAttention layer.
            hidden_states,                          # Input to attention.
            attention_mask=attention_mask,          # Passes attention mask.
            past_key_value=past_key_value,          # Passes cached key-value.
            use_cache=use_cache,                    # Passes use_cache flag.
        )
        hidden_states = residual + self.dropout(attention_output) # Adds residual connection and applies dropout to attention output.

        # Feed-forward with residual connection
        residual = hidden_states                               # Stores the current hidden_states for the second residual connection.
        hidden_states = self.ln_2(hidden_states)               # Applies Layer Normalization before the feed-forward network.
        feed_forward_output = self.feed_forward(hidden_states) # Passes through the FeedForward network.
        hidden_states = residual + feed_forward_output         # Adds the residual connection.

        return {"hidden_states": hidden_states, "past_key_value": present} # Returns a dictionary containing the final hidden states and the cached key-value pair.
