# filename: src/llm_pipeline/cli/evaluate.py
"""
Evaluation command-line interface for the LLM pipeline.

This module provides a command-line interface (CLI) for evaluating a trained
language model using `click`. It allows users to specify the model path,
configuration, tokenizer, dataset split, batch size, and various evaluation
options, including text generation.

Purpose:
    To provide a convenient and standardized way for users to assess the
    performance of a trained LLM. It abstracts the underlying evaluation logic
    and makes it accessible via a simple command.

    Evaluation is a critical step in the machine learning workflow. This CLI:
    1. **Facilitates Model Assessment:** Allows quick and easy performance checks.
    2. **Supports Reproducibility:** By taking explicit arguments for model,
       config, and data, it helps ensure evaluations are consistent.
    3. **Enables Automation:** Can be easily integrated into larger scripts
       or continuous integration pipelines.
    It integrates components from `config`, `data`, `models`, `evaluation`,
    and `utils` sub-packages to perform a complete evaluation run.

LLM Pipeline Fit:
    This module is part of the `llm_pipeline.cli` package. It acts as an
    executable script that users can run from their terminal to evaluate a model.
    It loads necessary components, sets up the environment, initializes the
    `Evaluator`, and reports the results.
"""

import logging                    # For logging information and status messages.
import json                       # For loading and saving JSON configuration and results.
from pathlib import Path          # For handling file paths in an object-oriented way.
from typing import Optional, List # Type hinting.
import click                      # Command-line interface creation library.
import torch                      # PyTorch library, used for model loading and device management.

# Import necessary components from other sub-packages of llm_pipeline.
from llm_pipeline.config import Config, load_config # For loading general and model configurations.
from llm_pipeline.data import (                     # For dataset handling and tokenization.
    WikiTextDataset,
    TokenizerWrapper,
    DataCollatorForLanguageModeling,
)
from llm_pipeline.models import TransformerLM, TransformerConfig         # For defining and loading the model.
from llm_pipeline.evaluation import Evaluator, EvaluationConfig          # For performing the evaluation and defining its config.
from llm_pipeline.utils import setup_logger, set_seed, set_deterministic # General utilities for logging and reproducibility.


logger = logging.getLogger(__name__) # Initialize a logger for this module.


@click.command() # Decorator from click to define a command-line command.
@click.argument("model_path", type=click.Path(exists=True, path_type=Path)) # Required argument: path to the model.
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a configuration file (e.g., config.yaml or config.json). If not provided, attempts to load from model_path's directory.",
)
@click.option(
    "--tokenizer",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the tokenizer directory (e.g., a SentencePiece model). If not provided, attempts to load from model_path's directory or default data path.",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["validation", "test"]), # Restricts choice to "validation" or "test" splits.
    default="test",
    help="The dataset split to evaluate the model on (e.g., 'validation' or 'test').",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=16,
    help="The batch size to use during evaluation. Overrides any batch size specified in config.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Optional path to a directory or file where evaluation results (metrics, generated samples) will be saved.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "auto"]), # Choice of device for computation.
    default="auto",
    help="The device to use for evaluation (e.g., 'cuda' for GPU, 'cpu' for CPU, 'auto' to automatically detect).",
)
@click.option(
    "--generate-samples",
    is_flag=True, # A boolean flag. If present, it's True.
    help="If set, the model will generate text samples during evaluation.",
)
@click.option(
    "--num-samples",
    type=int,
    default=10,
    help="The number of text samples to generate if '--generate-samples' is enabled.",
)
@click.option(
    "--max-generate-length",
    type=int,
    default=100, # Default max generation length
    help="The maximum length (number of tokens) for generated text samples. Overrides config's setting and is capped by model's max position embeddings.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="The random seed for reproducibility of evaluation results (e.g., sampling, generation).",
)
def evaluate_command(
    model_path: Path,
    config: Optional[Path],
    tokenizer: Optional[Path],
    dataset: str,
    batch_size: int,
    output: Optional[Path],
    device: str,
    generate_samples: bool,
    num_samples: int,
    max_generate_length: int,
    seed: int,
):
    """
    Evaluate a trained language model.

    This command loads a pre-trained language model, its configuration, and tokenizer,
    then performs evaluation on a specified dataset split. It can compute
    perplexity, accuracy, and optionally generate text samples. Results are
    logged and can be saved to an output file.
    """
    # Setup logging for the CLI script.
    setup_logger(log_level=logging.INFO)
    
    # Set random seed and configure deterministic behavior for reproducibility.
    # Evaluation can often tolerate non-deterministic behavior for speed.
    set_seed(seed)
    set_deterministic(False) # Evaluation can be non-deterministic (e.g., for speed on GPU).

    # Determine the computing device (CUDA, CPU, or automatic detection).
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) # Directly create device from string.
    
    logger.info(f"Using device: {device}")
    
    # --- Load Configuration ---
    # Attempt to load the configuration from the provided path or from the model's directory.
    loaded_cfg_from_file = None 
    if config: # If a config file path is explicitly provided.
        logger.info(f"Loading configuration from {config}")
        loaded_cfg_from_file = load_config(config) # Load the full Config object.
        cfg = loaded_cfg_from_file 
    else: # If no config file path is provided, try to find it near the model.
        model_dir = model_path if model_path.is_dir() else model_path.parent # Get model's directory.
        config_path = model_dir / "config.json" # Expected config file name.
        
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, "r") as f:
                model_config_dict = json.load(f) # Load raw model config dictionary.
            
            # Create a minimal Config object and populate `model_config` from the loaded dict.
            cfg = Config() # Initialize a default Config.
            cfg.model = TransformerConfig(**model_config_dict) # Populate model config from JSON.
            cfg.data = Config().data # Use default DataConfig for dataset configuration, it will be updated later.
        else:
            raise ValueError("No configuration found. Please provide --config argument or ensure 'config.json' is alongside your model.")
    
    # --- Load Tokenizer ---
    # Determine tokenizer path: explicitly provided, or inferred from model directory, or default.
    if tokenizer:
        tokenizer_path = tokenizer
    else:
        model_dir = model_path if model_path.is_dir() else model_path.parent
        tokenizer_path = model_dir / "tokenizer" # Common location for tokenizer saved with model.
        
        if not tokenizer_path.exists():
            # Fallback to a default expected tokenizer path if not found with model.
            tokenizer_path = Path("data/preprocessed/tokenizer/") 
            if not tokenizer_path.exists():
                raise ValueError(
                    f"No tokenizer found. Please provide --tokenizer argument "
                    f"or ensure tokenizer is at {tokenizer_path} or within the model directory."
                )
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer_wrapper = TokenizerWrapper.load(tokenizer_path) # Load the tokenizer.
    
    # --- Load Model ---
    logger.info(f"Loading model from {model_path}")
    
    # Initialize the model architecture using the loaded configuration.
    # Ensure `TransformerConfig` is instantiated correctly from `cfg.model`.
    model_config = TransformerConfig.from_model_config(cfg.model)
    model = TransformerLM(model_config)
    
    # Load model weights (state dictionary).
    if model_path.is_file(): # If model_path points directly to a .bin file.
        state_dict = torch.load(model_path, map_location="cpu") # Load to CPU first.
    else: # If model_path points to a directory (e.g., a checkpoint folder).
        model_file = model_path / "pytorch_model.bin"
        if not model_file.exists():
            raise ValueError(f"Model file not found: {model_file} within {model_path}.")
        state_dict = torch.load(model_file, map_location="cpu")
    
    model.load_state_dict(state_dict) # Load the weights into the model.
    model.to(device)                  # Move the model to the specified evaluation device.
    model.eval()                      # Set the model to evaluation mode (e.g., disable dropout).
    
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    
    # --- Load Dataset ---
    logger.info(f"Loading {dataset} dataset...")
    data_config_for_dataset = cfg.data # Get data configuration from the loaded config.
    
    # Crucial step: Ensure the dataset's `max_seq_length` aligns with the model's
    # `max_position_embeddings` to prevent truncation issues or wasted computation.
    effective_max_seq_length = model_config.max_position_embeddings
    if data_config_for_dataset.max_seq_length != effective_max_seq_length:
        logger.warning(
            f"Dataset's configured max_seq_length ({data_config_for_dataset.max_seq_length}) "
            f"does not match model's max_position_embeddings ({effective_max_seq_length}). "
            f"Overriding dataset's max_seq_length to {effective_max_seq_length} for evaluation."
        )
        data_config_for_dataset.max_seq_length = effective_max_seq_length # Explicitly update the config.

    # Override evaluation batch size with the one provided via CLI.
    data_config_for_dataset.eval_batch_size = batch_size 

    eval_dataset = WikiTextDataset(
        config=data_config_for_dataset, # Pass the (potentially modified) data config.
        tokenizer=tokenizer_wrapper,
        split=dataset,                  # Use the specified dataset split (validation/test).
    )
    
    # Create data collator for language modeling (no MLM for standard LM evaluation).
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_wrapper,
        mlm=False,            # Standard (causal) language modeling, not masked.
        pad_to_multiple_of=8, # Pad to a multiple of 8 for potential GPU efficiency.
    )
    
    # Create DataLoader for efficient batching during evaluation.
    from torch.utils.data import DataLoader # Re-import DataLoader to ensure it's available.
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,            # No need to shuffle for evaluation.
        collate_fn=data_collator, # Use the custom data collator.
        num_workers=4,            # Use multiple workers for data loading to speed up I/O.
        pin_memory=True,          # Pin memory for faster transfer to GPU.
    )
    
    # --- Create Evaluation Configuration ---
    eval_config = EvaluationConfig(
        compute_perplexity=True,           # Always compute perplexity for LM evaluation.
        compute_accuracy=True,             # Always compute accuracy for LM evaluation.
        generate_samples=generate_samples, # Based on CLI flag.
        num_generate_samples=num_samples,  # Number of samples to generate.
        # Cap max generation length by model's max position embeddings.
        max_generate_length=min(max_generate_length, model_config.max_position_embeddings), 
        batch_size=batch_size,             # Use the CLI batch size for evaluation config as well.
    )
    
    # --- Create and Run Evaluator ---
    evaluator = Evaluator(
        model=model,
        dataloader=eval_dataloader,
        tokenizer=tokenizer_wrapper, # Pass tokenizer to Evaluator for text generation and token-based metrics.
        config=eval_config,
        device=device,
    )
    
    logger.info("Running evaluation...")
    metrics = evaluator.evaluate() # Execute the evaluation.
    
    # Add model-specific metrics (e.g., parameter count).
    model_metrics = evaluator.compute_model_metrics()
    metrics.update(model_metrics)
    
    # Benchmark inference speed.
    logger.info("Benchmarking inference speed...")
    speed_metrics = evaluator.benchmark_inference_speed(num_samples=100)
    metrics.update(speed_metrics)
    
    # --- Log and Save Results ---
    logger.info("\nEvaluation Results:")
    logger.info("-" * 50)
    # Log all metrics in a formatted way.
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}") # Format floats to 4 decimal places.
        elif isinstance(value, int):
            logger.info(f"{key}: {value:,}")   # Format integers with comma separators.
        elif key != "generations":             # Avoid logging raw generations directly to console if many.
            logger.info(f"{key}: {value}")
    
    if output: # If an output path is provided.
        output.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists.
        
        # Determine the metrics file path.
        metrics_file = output if output.suffix == ".json" else output / "evaluation_metrics.json"
        
        # Prepare metrics for saving: handle 'generations' separately to avoid large inline JSON.
        save_metrics = {}
        for k, v in metrics.items():
            if k == "generations" and isinstance(v, list):
                gen_file = metrics_file.parent / "generations.json"
                with open(gen_file, "w") as gf:
                    json.dump(v, gf, indent=2)           # Save generations to a separate file.
                save_metrics[k] = f"Saved to {gen_file}" # Reference the file in main metrics.
            else:
                save_metrics[k] = v # Include other metrics directly.
        
        # Save the main metrics dictionary.
        with open(metrics_file, "w") as f:
            json.dump(save_metrics, f, indent=2)
        
        logger.info(f"Results saved to {metrics_file}")
    
    return metrics # Return the evaluation metrics dictionary.


def main():
    """Main entry point for the evaluation CLI."""
    evaluate_command()


if __name__ == "__main__":
    main() # Call the main function when the script is executed.