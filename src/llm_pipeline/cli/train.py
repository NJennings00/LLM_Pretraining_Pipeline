# filename: src/llm_pipeline/cli/train.py
"""
Training command-line interface.

This module provides the main entry point for training transformer-based
language models using a comprehensive configuration system powered by Hydra.
It orchestrates the entire training process, including setting up logging,
initializing data loaders, building/loading the model, and running the training loop.

Purpose:
    To enable users to train an LLM with flexible configurations via the
    command line. It acts as the central control script that ties together
    all other components of the `llm_pipeline` for model training.

    This CLI is the core executable for the LLM pretraining pipeline. It is
    critical for:
    1. **Orchestration:** Manages the flow from configuration to actual training.
    2. **Reproducibility:** Leverages Hydra for consistent experiment setup and
       `set_seed`, `save_reproducibility_info` for ensuring repeatable runs.
    3. **Flexibility:** Allows extensive customization of training parameters
       (model, data, optimizer, logging) through a structured configuration.
    4. **Integration:** Connects various modules: `config` for settings, `data`
       for preprocessing and loading, `models` for architecture, `training`
       for the training loop, and `utils` for common helpers.

LLM Pipeline Fit:
    As part of `llm_pipeline.cli`, this script is the primary way users will
    initiate and manage training runs. It is designed to be run from the terminal
    with a Hydra configuration file (or overrides), making it highly adaptable
    for different experimental setups and distributed training environments.
"""

import logging                              # For standard logging.
import sys                                  # For system-specific parameters and functions.
from pathlib import Path                    # For object-oriented filesystem paths.
from typing import Optional, List           # Type hinting.
import hydra                                # Framework for managing configuration and composing applications.
from omegaconf import DictConfig, OmegaConf # Hydra's configuration object and utilities.
import torch                                # PyTorch, the deep learning framework.
import json

# Import core components from the llm_pipeline package.
from llm_pipeline.config import Config # Custom configuration class mapping from DictConfig.
from llm_pipeline.data import (        # Data-related modules.
    WikiTextDataset,
    build_tokenizer,
    DataCollatorForLanguageModeling,
    create_dataloaders,
    TokenizerWrapper,
)
from llm_pipeline.models import TransformerLM, TransformerConfig # Model-related modules.
from llm_pipeline.training import Trainer, TrainingArguments     # Training loop and arguments.
from llm_pipeline.utils import (                                 # General utilities.
    setup_logger,
    set_seed,
    set_deterministic,
    save_reproducibility_info,
)


logger = logging.getLogger(__name__) # Initialize a logger for this module.


# `@hydra.main` decorator initializes Hydra and loads the configuration.
# `version_base=None` means Hydra's automatic versioning is disabled for this entry point.
# `config_path` specifies the directory where configuration files are located relative to this script.
# `config_name` specifies the base configuration file to load.
@hydra.main(version_base=None, config_path="../config/hydra", config_name="config")
def train_command(cfg: DictConfig) -> None:
    """
    Train a language model.

    This function serves as the main entry point for the training process.
    It receives the Hydra-managed configuration, sets up the environment,
    prepares data and model, and then initiates the training loop.

    Args:
        cfg: A `DictConfig` object containing the merged configuration
             from Hydra (command-line overrides, config files, defaults).
    """

    # DEBUG: Print the configuration for inspection. This is useful during development
    # to verify that arguments and config files are correctly interpreted by Hydra.
    """     
    config_dict = OmegaConf.to_container(cfg, resolve=True) # Convert DictConfig to a Python dict.
    print("="*50)
    print("DEBUG: Full config dict:")
    print(json.dumps(config_dict, indent=2))
    print("="*50)
    print("DEBUG: Data section:")
    print(json.dumps(config_dict.get("data", {}), indent=2))
    print("="*50) """

    # Convert the Hydra DictConfig into the custom `Config` object.
    # `resolve=True` resolves interpolations (e.g., `${oc.env:VAR}`).
    config = Config.from_dict(OmegaConf.to_container(cfg, resolve=True))
    
    # Setup logging based on configuration (e.g., log file path, level).
    setup_logger(
        log_file=config.logging.log_file,
        log_level=config.logging.log_level,
    )
    
    # Log the complete configuration for the current run.
    logger.info("Starting training with configuration:")
    logger.info(OmegaConf.to_yaml(cfg)) # Print the config in YAML format for readability.
    
    # Set random seed for reproducibility across different runs.
    set_seed(config.training.seed)
    # Configure deterministic operations (e.g., CUDA algorithms) for stricter reproducibility.
    set_deterministic(config.training.deterministic)
    
    # Save information about the environment and dependencies for reproducibility.
    save_reproducibility_info(
        config.training.output_dir / "reproducibility_info.json"
    )
    
    # --- Initialize Tokenizer ---
    logger.info("Building tokenizer...")
    
    tokenizer_path = config.tokenizer.tokenizer_file
    if tokenizer_path and tokenizer_path.exists():
        # If a tokenizer file path is provided in config and it exists, load it.
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = TokenizerWrapper.load(tokenizer_path)
    else:
        # If no tokenizer file is found/provided, train a new one.
        logger.info("Tokenizer file not found or not provided. Training a new tokenizer.")
        # Load a sample of the training data in raw text format for tokenizer training.
        # `load_raw_only=True` ensures that the dataset is loaded without tokenization,
        # which is necessary when the tokenizer itself is being built.
        temp_dataset = WikiTextDataset(
            config=config.data,
            tokenizer=None,     # Pass None as tokenizer during initial raw data load.
            split="train",
            load_raw_only=True, # Crucial flag to prevent premature tokenization.
        )
        
        # Get sample texts from the raw dataset for tokenizer training.
        # Limiting sample size for efficiency.
        sample_size = min(config.tokenizer.sample_size, len(temp_dataset.dataset))
        sample_texts = [
            temp_dataset.dataset[i]["text"] 
            for i in range(sample_size)
        ]
        
        # Build the tokenizer using the sampled raw texts and tokenizer configuration.
        tokenizer = build_tokenizer(config.tokenizer, sample_texts)
        
        # Save the newly built tokenizer if a path is specified in the config.
        if config.tokenizer.tokenizer_file:
            tokenizer.save(config.tokenizer.tokenizer_file)
            logger.info(f"New tokenizer saved to {config.tokenizer.tokenizer_file}")
        else:
            logger.warning("No tokenizer_file specified in config. Trained tokenizer will not be saved.")
    
    # Ensure that the model's `vocab_size` configuration matches the actual tokenizer's vocab size.
    config.model.vocab_size = tokenizer.vocab_size
    config.tokenizer.vocab_size = tokenizer.vocab_size # Also update tokenizer config for consistency.
    
    # --- Load Datasets for Training/Evaluation ---
    # Now load the datasets for the actual training and evaluation,
    # passing the fully built and configured tokenizer.
    logger.info("Loading datasets for training and validation...")
    train_dataset = WikiTextDataset(
        config=config.data,
        tokenizer=tokenizer, # Pass the built tokenizer here.
        split="train",
    )
    
    eval_dataset = WikiTextDataset(
        config=config.data,
        tokenizer=tokenizer, # Pass the built tokenizer here.
        split="validation",
    )
    
    # Log statistics about the loaded datasets.
    train_stats = train_dataset.get_statistics()
    logger.info(f"Training dataset statistics: {train_stats}")
    
    eval_stats = eval_dataset.get_statistics()
    logger.info(f"Validation dataset statistics: {eval_stats}")
    
    # --- Create Data Collator and Dataloaders ---
    # The DataCollator is responsible for batching, padding, and potentially masking tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False for causal language modeling (next token prediction).
        # Pad to a multiple of 8 for potential efficiency on GPUs, especially with FP16.
        pad_to_multiple_of=8 if config.training.fp16 else None,
    )
    
    # Create PyTorch DataLoaders for training and evaluation.
    # Explicitly using parameter names `data_config` and `training_config` as defined in `create_dataloaders`.
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=data_collator,
        data_config=config.data,         # Pass the data-specific part of the config.
        training_config=config.training, # Pass the training-specific part for batch sizes, etc.
    )

    # --- Initialize Model ---
    logger.info("Initializing model...")
    # Create the model configuration from the loaded main configuration.
    model_config = TransformerConfig.from_model_config(config.model)
    model = TransformerLM(model_config)
    
    logger.info(f"Model initialized with {model.num_parameters():,} parameters")
    
    # --- Create Training Arguments ---
    # Map the training-related configurations to a `TrainingArguments` object.
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        warmup_ratio=config.training.warmup_ratio,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        logging_steps=config.training.logging_steps,
        logging_first_step=config.training.logging_first_step,
        seed=config.training.seed,
        dataloader_num_workers=config.data.num_workers, # Dataloader workers from data config.
        resume_from_checkpoint=config.training.resume_from_checkpoint,
    )
    
    # --- Setup Callbacks ---
    # Initialize a list for training callbacks (e.g., for logging to TensorBoard, Weights & Biases).
    callbacks = []
    
    if config.logging.use_tensorboard:
        # Import TensorBoardCallback only if enabled to avoid unnecessary dependencies.
        from llm_pipeline.training.callbacks import TensorBoardCallback
        callbacks.append(TensorBoardCallback(config.logging.tensorboard_dir))
    
    if config.logging.use_wandb:
        # Import WandbCallback only if enabled.
        from llm_pipeline.training.callbacks import WandbCallback
        callbacks.append(
            WandbCallback(
                project=config.logging.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True), # Pass the full resolved config to WandB.
                name=config.logging.wandb_run_name,
                tags=config.logging.wandb_tags,
            )
        )
    
    # --- Initialize Trainer ---
    # Instantiate the main `Trainer` class with all prepared components.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        tokenizer=tokenizer, # Pass the tokenizer to the Trainer, useful for tokenizing generated samples if needed.
    )
    
    # --- Start Training ---
    logger.info("Starting training...")
    metrics = trainer.train() # Begin the training loop.
    
    # --- Save Final Model ---
    # After training, save the final state of the model.
    trainer.save_model()
    
    # --- Log and Save Final Metrics ---
    logger.info("Training completed!")
    logger.info(f"Final metrics: {metrics}")
    
    # Save the final training metrics to a JSON file.
    metrics_path = config.training.output_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")


def main():
    """Main entry point for the training CLI. This function is typically called by Hydra."""
    train_command()


if __name__ == "__main__":
    # If the script is run directly (not via Hydra's `hydra.main()`),
    # this block will execute, but `hydra.main()` itself handles the execution.
    # This `main()` function effectively wraps the `hydra.main()` call.
    main()