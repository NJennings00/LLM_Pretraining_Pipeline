# filename: src/llm_pipeline/models/utils.py
"""
This module provides general utility and helper functions specifically for
managing PyTorch neural network models within the LLM pretraining pipeline.
It includes functionalities for counting model parameters, freezing/unfreezing
parameters for fine-tuning, applying common weight initialization schemes,
retrieving activation functions by name, and computing comprehensive model statistics.
These utilities simplify common model-related operations and provide valuable insights.
It also includes a function to generate a causal attention mask for transformer decoders.
"""

## Imports
# Imports necessary modules and types for model utilities.
import math                                      # math: Provides mathematical functions.
from typing import Optional, Callable, Dict, Any # Optional: Type hint for parameters that can be None.
                                                 # Callable: Type hint for functions.
                                                 # Dict: Type hint for dictionaries.
                                                 # Any: Type hint for any type.
import torch                                     # torch: PyTorch deep learning framework.
import torch.nn as nn                            # torch.nn: PyTorch's neural network module.
import torch.nn.functional as F                  # torch.nn.functional: Functional interface for common neural network operations.


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """
    Count the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch neural network model.
        only_trainable (bool): If True, only counts parameters that have `requires_grad=True`.

    Returns:
        int: The total number of parameters (or trainable parameters) in the model.
    """
    if only_trainable:                                                       # If only trainable parameters are to be counted.
        return sum(p.numel() for p in model.parameters() if p.requires_grad) # Sums the number of elements (`numel()`) for parameters that require gradients.
    return sum(p.numel() for p in model.parameters())                        # Sums the number of elements for all parameters in the model.


def freeze_parameters(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze all parameters of a PyTorch model.

    Freezing parameters means setting their `requires_grad` attribute to `False`,
    preventing them from being updated during backpropagation. This is useful
    for fine-tuning, where you might want to keep the weights of a pre-trained
    backbone fixed.

    Args:
        model (nn.Module): The PyTorch neural network model.
        freeze (bool): If True, freezes the parameters (sets `requires_grad` to False).
                       If False, unfreezes them (sets `requires_grad` to True).

    Returns:
        None
    """
    for param in model.parameters():     # Iterates through all parameters in the model.
        param.requires_grad = not freeze # Sets `requires_grad` to the opposite of `freeze`.
                                         # If `freeze` is True, `requires_grad` becomes False (frozen).
                                         # If `freeze` is False, `requires_grad` becomes True (unfrozen).


def init_weights(module: nn.Module, initializer_range: float = 0.02) -> None:
    """
    Initialize the weights of different types of PyTorch modules.

    This function applies common initialization schemes (e.g., normal distribution
    for linear layers and embeddings, and constant for LayerNorm weights) to
    stabilize training and improve convergence. It's designed to be used with `model.apply()`.

    Args:
        module (nn.Module): The PyTorch module whose weights are to be initialized.
        initializer_range (float): The standard deviation for normal distribution initialization.
                                   Typically a small value (e.g., 0.02).

    Returns:
        None
    """
    if isinstance(module, nn.Linear):                                # Checks if the module is a linear layer (`nn.Linear`).
        module.weight.data.normal_(mean=0.0, std=initializer_range)  # Initialize weights of linear layers with a normal distribution.
        if module.bias is not None:                                  # If the linear layer has a bias.
            module.bias.data.zero_()                                 # Initialize bias to zeros.
    elif isinstance(module, nn.Embedding):                           # Checks if the module is an embedding layer (`nn.Embedding`).
        module.weight.data.normal_(mean=0.0, std=initializer_range)  # Initialize weights of embedding layers with a normal distribution.
        if module.padding_idx is not None:                           # If the embedding layer has a padding index.
            module.weight.data[module.padding_idx].zero_()           # Set the embedding vector for the padding index to zeros.
    elif isinstance(module, nn.LayerNorm):                           # Checks if the module is a layer normalization layer (`nn.LayerNorm`).
                                                                     # Initialize weights and bias of layer normalization.
        if hasattr(module, "bias") and module.bias is not None:      # Check for bias parameter.
            module.bias.data.zero_()                                 # Initialize bias to zeros.
        if hasattr(module, "weight"):                                # Check for weight parameter.
            module.weight.data.fill_(1.0)                            # Initialize weight to ones.


def get_activation_fn(activation: str) -> Callable:
    """
    Retrieve a PyTorch activation function based on its string name.

    This utility provides a mapping from common string names (e.g., "relu", "gelu")
    to their corresponding PyTorch functional API implementations.

    Args:
        activation (str): The string name of the desired activation function.

    Returns:
        Callable: The PyTorch functional activation function.

    Raises:
        ValueError: If an unknown activation function name is provided.
    """
    activation_functions = {                                 # Dictionary mapping activation names to their PyTorch functional implementations.
        "relu": F.relu,
        "gelu": F.gelu,
        "gelu_new": lambda x: F.gelu(x, approximate="tanh"), # Variant of GELU used in some models (e.g., original GPT-2).
        "swish": F.silu,                                     # Swish and SiLU are the same.
        "silu": F.silu,
        "mish": F.mish,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }

    if activation not in activation_functions:                         # Checks if the requested activation name is in the dictionary.
        raise ValueError(f"Unknown activation function: {activation}") # Raises an error if the name is not recognized.

    return activation_functions[activation] # Returns the corresponding activation function callable.


def compute_model_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about a PyTorch model, including total and trainable
    parameters, and detailed breakdown of parameters by module type.

    Args:
        model (nn.Module): The PyTorch neural network model.

    Returns:
        Dict[str, Any]: A dictionary containing model statistics:
                        - "total_parameters": Total number of parameters.
                        - "trainable_parameters": Total number of trainable parameters.
                        - "model_size_mb": Estimated model size in Megabytes (assuming float32).
                        - "layer_stats": Detailed statistics per module type (count, params, trainable_params).
    """
    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)

    # Estimate model size assuming float32 (4 bytes per parameter)
    model_size_mb = (total_params * 4) / (1024 * 1024)

    # Compute parameter statistics by layer type
    layer_stats = {}
    for name, module in model.named_modules():
        # Only consider leaf modules (modules that do not contain other nn.Module instances)
        # This prevents double-counting parameters of sub-modules.
        if len(list(module.children())) == 0:
            module_type = type(module).__name__
            if module_type not in layer_stats:
                layer_stats[module_type] = {
                    "count": 0,
                    "params": 0,
                    "trainable_params": 0,
                }

            layer_stats[module_type]["count"] += 1
            # Sum parameters directly owned by this module (recurse=False)
            module_params = sum(p.numel() for p in module.parameters(recurse=False))
            module_trainable_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

            layer_stats[module_type]["params"] += module_params
            layer_stats[module_type]["trainable_params"] += module_trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "layer_stats": layer_stats,
    }


def generate_square_subsequent_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate a square subsequent mask for causal attention.

    This mask ensures that each position in a sequence can only attend to
    previous positions and itself, which is crucial for decoder-only
    transformer models (causal language models).

    Args:
        size (int): The size of the square mask (sequence length).
        device (torch.device, optional): The device on which to create the mask.
                                        Defaults to None (will use default tensor device).

    Returns:
        torch.Tensor: A square upper-triangular mask of shape (size, size).
                      Elements with value True allow attention, False (or -inf) block it.
                      Here, it returns a boolean mask: True indicates allowed attention.
    """
    # Create an upper-triangular matrix of ones, then invert it
    # triu(1) creates a mask where the diagonal and elements above it are 1, rest are 0.
    # We want the opposite for causal mask (lower triangular should be True/0, upper should be False/-inf)
    # The mask should be `(seq_len, seq_len)` where `mask[i,j]` is True if token `i` can attend to token `j`.
    # For causal LM, `i` can only attend to `j` if `j <= i`.
    mask = torch.full((size, size), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1) # Set upper triangle (including diagonal+1) to -inf
    return mask # This mask will be added to attention scores
