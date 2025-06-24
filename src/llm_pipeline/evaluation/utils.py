# filename: src/llm_pipeline/evaluation/utils.py
"""Utilities for evaluation, including text generation and metric computation."""

import logging
from typing import Optional, Any, Union, List
import torch
import torch.nn.functional as F # Import F for torch.nn.functional
import numpy as np

from llm_pipeline.models import TransformerLM
from llm_pipeline.data.tokenizer import TokenizerWrapper


logger = logging.getLogger(__name__)


def generate_text(
    model: TransformerLM, # Explicitly use TransformerLM
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[torch.device] = None, # Device where the model is expected to be.
) -> str:
    """
    Generates text using the provided model and tokenizer via an iterative sampling loop.

    Args:
        model: The language model for generation.
        tokenizer: The tokenizer for encoding/decoding.
        prompt: The initial text prompt.
        max_length: Maximum length of the generated sequence (including prompt).
        temperature: Controls randomness in sampling. Lower values make output more deterministic.
        top_k: If set, only the top k most likely tokens are considered.
        top_p: If set, tokens with cumulative probability less than top_p are considered.
        device: Device the model is currently on (e.g., 'cuda', 'cpu'). Used to place new tensors.

    Returns:
        The generated text.
    """
    # Determine the device. Prioritize the device passed in, then model's device, then CUDA/CPU default.
    if device is None:
        device = model.device if hasattr(model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval() # Ensure model is in evaluation mode.

    # Encode the prompt and ensure it's on the correct device.
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Check for token ID overflow
    max_id = max(input_ids)
    vocab_size = model.config.vocab_size
    assert max_id < vocab_size, (
        f"Input prompt produced token ID {max_id} >= model vocab size {vocab_size}. "
        "Tokenizer and model vocab sizes are misaligned."
    )

    # Explicitly move input_ids to the model's device
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension and move to device.

    generated_ids = input_ids.tolist()[0] # Start with prompt tokens as a list.

    # Autoregressive generation loop
    # Generate up to max_length tokens, or until EOS token is generated.
    for _ in range(max_length - input_ids.shape[1]): 
        with torch.no_grad(): # Disable gradient calculations for inference.
            # Get logits for the next token.
            # Model outputs (batch_size, sequence_length, vocab_size).
            # We only need the logits for the last token to predict the next one.
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs["logits"][:, -1, :] # Shape: (batch_size, vocab_size).

            # Apply temperature to smooth or sharpen probabilities.
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k sampling: filter to the top_k most probable tokens.
            if top_k is not None and top_k > 0:
                # Set all values below the top_kth value to -inf.
                # This ensures only the top_k tokens have non-negative infinity logits after filtering.
                # `topk(dim=-1)` returns values and indices along the last dimension.
                # `[0][..., -1, None]` gets the k-th smallest value (threshold) for filtering.
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k, dim=-1)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf') # Set filtered logits to negative infinity.

            # Apply top-p (nucleus) sampling: keep the smallest set of tokens whose cumulative probability exceeds top_p.
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1) # Sort logits in descending order.
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)              # Calculate cumulative probabilities.

                # Remove tokens with cumulative probability above the threshold
                # and ensure at least one token is kept
                indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                # (i.e., keep the smallest set of tokens that sum up to top_p)
                if indices_to_remove.shape[-1] > 1: # Ensure there's a dimension to shift
                    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
                indices_to_remove[..., 0] = False # Ensure the highest probability token (index 0 in sorted list) is never removed.

                # Set logits of tokens to be removed to -infinity directly on the sorted_logits
                sorted_logits[indices_to_remove] = -float('Inf')

                # Re-order the logits back to their original positions using gather.
                # We need the inverse permutation of sorted_indices to unsort the logits.
                # `torch.argsort(sorted_indices, dim=-1)` gives this inverse permutation.
                next_token_logits.copy_( # Copy the re-ordered logits back into next_token_logits
                    torch.gather(
                        sorted_logits,                              # The tensor to gather from (filtered sorted logits)
                        dim=-1,                                     # Dimension along which to gather
                        index=torch.argsort(sorted_indices, dim=-1) # Indices to gather (inverse permutation)
                    )
                )
            
            # --- START OF NEW/MODIFIED LOGIC ---
            # Check if all logits are -Inf after filtering.
            # If so, force EOS token to prevent F.softmax from producing NaN or multinomial from failing.
            if torch.all(next_token_logits == -float('Inf')):
                logger.warning("All next_token_logits are -Inf after filtering. Forcing EOS token.")
                # Create a tensor with the EOS token ID on the correct device.
                next_token = torch.tensor([tokenizer.eos_token_id], device=device).squeeze(0)
            else:
                # Apply softmax to get probabilities.
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Check for NaNs or all zeros in probabilities.
                # This can happen if previous filters result in problematic logits (e.g., all -Inf, which might lead to NaNs in softmax).
                if torch.any(torch.isnan(probs)) or torch.all(probs == 0.0):
                    logger.warning("Probabilities are NaN or all zero after softmax. Forcing EOS token.")
                    next_token = torch.tensor([tokenizer.eos_token_id], device=device).squeeze(0)
                else:
                    # Sample one token from the distribution.
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    
                    # === SAFETY CHECK ===
                    # Ensure generated token ID is within model vocabulary
                    if next_token.item() >= model.config.vocab_size:
                        logger.warning(
                            f"Generated token ID {next_token.item()} >= model vocab size {model.config.vocab_size}. "
                            f"Using EOS token (ID: {tokenizer.eos_token_id}) instead."
                        )
                        # Use EOS token if it's valid, otherwise fallback to token 0
                        if tokenizer.eos_token_id < model.config.vocab_size:
                            next_token = torch.tensor([tokenizer.eos_token_id], device=device).squeeze(0)
                        else:
                            logger.error(f"EOS token ID {tokenizer.eos_token_id} also >= vocab size. Using token 0.")
                            next_token = torch.tensor([0], device=device).squeeze(0)
                    # === END SAFETY CHECK ===
            # --- END OF NEW/MODIFIED LOGIC ---

            # If an EOS token is generated (either by sampling or forced), stop generation.
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Append the new token to the input_ids for the next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1) # Use unsqueeze(1) for consistent shape
            generated_ids.append(next_token.item())

    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def sample_generations(
    model: TransformerLM,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    num_samples: int,
    max_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> list[str]:
    """
    Generates multiple text samples.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompts: List of prompts to generate from.
        num_samples: Number of samples to generate.
        max_length: Maximum length of each generated sequence.
        temperature: Controls randomness in sampling.
        top_k: Top K sampling parameter.
        top_p: Top P (nucleus) sampling parameter.
        device: Device to use.
        
    Returns:
        List of generated texts.
    """
    generated_texts = []
    # Ensure num_samples does not exceed the number of available prompts
    num_prompts_to_use = min(num_samples, len(prompts)) 

    for i in range(num_prompts_to_use):
        prompt = prompts[i]
        logger.info(f"Generating sample {i+1}/{num_samples} with prompt: '{prompt}'")
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        generated_texts.append(generated_text)
    return generated_texts


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return float(np.exp(loss)) # Ensure float return


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy.
    
    Args:
        predictions: Predicted token IDs
        labels: True token IDs
        
    Returns:
        Accuracy
    """
    # Filter out -100 (ignored tokens)
    mask = labels != -100
    correct_predictions = (predictions == labels)[mask].sum().item()
    total_predictions = mask.sum().item()
    
    if total_predictions == 0:
        return 0.0
    
    return float(correct_predictions / total_predictions) # Ensure float return
