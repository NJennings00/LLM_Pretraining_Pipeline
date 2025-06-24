# filename: src/llm_pipeline/models/__init__.py
"""
Model modules for the LLM pipeline.

This `__init__.py` file serves as the public interface for the `llm_pipeline.models`
package. It aggregates and re-exports key components from its sub-modules, making
them easily accessible when `llm_pipeline.models` is imported.

Purpose:
    To provide a clean and organized way to import essential model-related classes,
    configurations, and utility functions without requiring users to navigate
    the specific sub-modules where they are defined. This simplifies the import
    statements for users of the library and clearly defines the public API of the
    `models` package.

    This file is crucial for package organization and usability. It dictates
    what symbols are exposed when `from llm_pipeline import models` or
    `from llm_pipeline.models import ...` is used. Its correctness ensures
    that all intended components are available for model construction, training,
    and inference. Changes here directly impact how other parts of the LLM
    pipeline (e.g., training scripts, inference scripts) interact with the model
    definitions.

LLM Pipeline Fit:
    This file acts as a central hub for all model-related components. When
    building an LLM, you'll typically import `TransformerLM` and `TransformerConfig`
    directly from `llm_pipeline.models`. Other components like `LayerNorm` or
    `MultiHeadAttention` might be imported for custom modifications or detailed
    inspections. The utility functions also become readily available for tasks
    like parameter counting or weight initialization.

Contents:
    - Re-exports `TransformerLM` and `TransformerConfig` from `transformer.py`.
    - Re-exports various embedding classes from `embeddings.py`.
    - Re-exports core layer components like `LayerNorm`, `MultiHeadAttention`,
      `FeedForward`, and `TransformerDecoderLayer` from `layers.py`.
    - Re-exports utility functions from `utils.py`.
    - Defines `__all__` to explicitly list the names that will be imported
      when `from llm_pipeline.models import *` is used. This is good practice
      for controlling the public API.

Changes from previous version:
    - `TransformerBlock` renamed to `TransformerDecoderLayer` for consistency
      with modern transformer terminology and its role in a decoder-only LM.
    - Added `generate_square_subsequent_mask` to the re-exports and `__all__`
      list, as it is a crucial utility for causal masking in the transformer decoder.
"""

from llm_pipeline.models.transformer import TransformerLM, TransformerConfig # Imports the main Transformer Language Model and its configuration.
from llm_pipeline.models.embeddings import (                                 # Imports various embedding related classes.
    TokenEmbedding,                                                          # Basic token embedding.
    PositionalEmbedding,                                                     # Base class for positional embeddings.
    SinusoidalPositionalEmbedding,                                           # Sinusoidal positional embedding.
    RotaryEmbedding,                                                         # Rotary positional embedding.
    TokenAndPositionalEmbedding,                                             # Combined token and positional embedding.
)
from llm_pipeline.models.layers import (  # Imports core transformer layer components.
    LayerNorm,                            # Layer normalization module.
    MultiHeadAttention,                   # Multi-head attention mechanism.
    FeedForward,                          # Feed-forward network.
    TransformerDecoderLayer,              # Changed from TransformerBlock to TransformerDecoderLayer # Single transformer decoder layer.
)
from llm_pipeline.models.utils import (  # Imports general utility functions for models.
    count_parameters,                    # Utility to count model parameters.
    freeze_parameters,                   # Utility to freeze model parameters.
    init_weights,                        # Utility for weight initialization (might be overridden by model-specific init).
    get_activation_fn,                   # Utility to retrieve activation functions.
    compute_model_stats,                 # Utility to compute various model statistics.
    generate_square_subsequent_mask,     # Added from utils.py for completeness if it's part of public API # Utility to generate causal masks.
)


__all__ = [ # Defines the public API for the 'models' package.
    "TransformerLM",                    # Main model class.
    "TransformerConfig",                # Model configuration class.
    "TokenEmbedding",                   # Embedding component.
    "PositionalEmbedding",              # Embedding component.
    "SinusoidalPositionalEmbedding",    # Embedding component.
    "RotaryEmbedding",                  # Embedding component.
    "TokenAndPositionalEmbedding",      # Embedding component.
    "LayerNorm",                        # Layer component.
    "MultiHeadAttention",               # Layer component.
    "FeedForward",                      # Layer component.
    "TransformerDecoderLayer",          # Layer component.
    "count_parameters",                 # Utility function.
    "freeze_parameters",                # Utility function.
    "init_weights",                     # Utility function.
    "get_activation_fn",                # Utility function.
    "compute_model_stats",              # Utility function.
    "generate_square_subsequent_mask",  # Added to __all__ # Utility function.
]