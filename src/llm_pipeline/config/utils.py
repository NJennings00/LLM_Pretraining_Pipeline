# filename: src/llm_pipeline/config/utils.py
"""
Utilities for configuration management.

This module provides helper functions for loading, saving, merging, and validating
configuration files within the LLM pretraining pipeline. It supports both
standard YAML/JSON files and integrates with Hydra for advanced configuration
management.

Purpose:
    To streamline the process of handling configuration objects throughout the
    pipeline. This includes reading configuration from various file formats,
    applying overrides, and ensuring the configuration is valid before use.

    Robust configuration utilities are essential for reproducible and flexible
    machine learning experiments. These functions are crucial because they:
    1. **Standardize I/O:** Provide consistent methods to load and save configurations.
    2. **Enable Overrides:** Facilitate merging base configurations with experiment-specific
       or command-line overrides.
    3. **Ensure Correctness:** Implement validation checks to catch incompatible settings early.
    4. **Integrate with Hydra:** Offer a bridge between Hydra's powerful configuration
       system (`DictConfig`) and the custom `Config` dataclass, allowing seamless use
       of both.

LLM Pipeline Fit:
    These utilities are used extensively by the `llm_pipeline.cli` commands
    (e.g., `train.py`, `evaluate.py`, `preprocess.py`) to manage their runtime
    settings. They ensure that the correct parameters are passed to the data,
    model, and training components, making the entire pipeline configurable and debuggable.
"""

import json                                         # For reading and writing JSON configuration files.
import yaml                                         # For reading and writing YAML configuration files.
from pathlib import Path                            # For handling filesystem paths.
from typing import Dict, Any, Optional, Union       # Type hints.
import logging                                      # For logging messages.
from omegaconf import OmegaConf, DictConfig         # Hydra's configuration objects.
import hydra                                        # Hydra framework 
from hydra import compose, initialize_config_dir    # Specific Hydra functions for composing configurations.

from llm_pipeline.config.config import Config       # Import the main configuration dataclass.


logger = logging.getLogger(__name__) # Initialize a logger for this module.


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from a YAML or JSON file and convert it into a `Config` object.
    
    Args:
        config_path: Path to the configuration file (e.g., 'path/to/my_config.yaml').
                     Can be a string or a `Path` object.
        
    Returns:
        A `Config` object populated with the settings from the file.
        
    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the file extension is not supported (neither .yaml/.yml nor .json).
    """
    config_path = Path(config_path) # Ensure path is a Path object for consistent handling.
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load based on file extension.
    if config_path.suffix in (".yaml", ".yml"):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) # Safely load YAML.
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config_dict = json.load(f) # Load JSON.
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}. "
                         "Only .yaml, .yml, and .json are supported.")
    
    return Config.from_dict(config_dict) # Convert the loaded dictionary to a Config dataclass instance.


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save a `Config` object to a YAML or JSON file.
    
    Args:
        config: The `Config` object to save.
        config_path: The desired path to save the configuration file (e.g., 'output/run_config.yaml').
                     The parent directories will be created if they don't exist.
        
    Raises:
        ValueError: If the file extension is not supported.
    """
    config_path = Path(config_path) # Ensure path is a Path object.
    config_path.parent.mkdir(parents=True, exist_ok=True) # Create parent directories if needed.
    
    config_dict = config.to_dict() # Convert the Config object to a dictionary for serialization.
    
    # Save based on file extension.
    if config_path.suffix in (".yaml", ".yml"):
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False) # Dump to YAML, ensuring block style.
    elif config_path.suffix == ".json":
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2) # Dump to JSON with 2-space indentation.
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}. "
                         "Only .yaml, .yml, and .json are supported for saving.")
    
    logger.info(f"Configuration saved to {config_path}")


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """
    Merge an override dictionary into a base `Config` object.
    
    This function performs a deep merge, meaning it recursively merges nested dictionaries.
    Values in `override_config` will take precedence over `base_config` values.
    
    Args:
        base_config: The base `Config` object to merge into.
        override_config: A dictionary containing the configuration overrides.
        
    Returns:
        A new `Config` object with the merged settings.
    """
    base_dict = base_config.to_dict()                     # Convert base Config to dict.
    merged_dict = _deep_merge(base_dict, override_config) # Perform recursive merge.
    return Config.from_dict(merged_dict)                  # Convert merged dict back to Config object.


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    This helper function is used by `merge_configs` to combine nested dictionary structures.
    If a key exists in both `base` and `override`, and both values are dictionaries,
    they are recursively merged. Otherwise, the value from `override` takes precedence.
    """
    merged = base.copy() # Start with a copy of the base dictionary.
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # If both values for a key are dictionaries, merge them recursively.
            merged[key] = _deep_merge(merged[key], value)
        else:
            # Otherwise, the override value replaces the base value.
            merged[key] = value
    
    return merged


def validate_config(config: Config) -> None:
    """
    Validate the entire `Config` object for consistency and correctness.
    
    This performs both internal validations defined within the `Config` dataclass
    and additional cross-parameter checks.
    
    Args:
        config: The `Config` object to validate.
        
    Raises:
        ValueError: If any validation rule is violated.
    """
    config.validate() # Call the internal validation method of the Config object.
    
    # Additional specific validation checks.
    if config.training.warmup_steps > 0 and config.training.warmup_ratio > 0:
        logger.warning(
            "Both warmup_steps and warmup_ratio are set. "
            "warmup_steps will take precedence and be used for the learning rate scheduler."
        )
    
    if config.training.fp16 and config.training.bf16:
        raise ValueError("Cannot enable both fp16 and bf16 mixed precision training simultaneously. Choose one or none.")
    
    if config.training.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer (>= 1).")
    
    logger.info("Configuration validated successfully.")


def load_hydra_config(
    config_path: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[list] = None,
) -> DictConfig:
    """
    Load configuration using Hydra's composition capabilities.
    
    This function initializes Hydra's configuration system from a specified
    directory and composes a configuration, optionally applying command-line-style
    overrides.
    
    Args:
        config_path: Path to the Hydra configuration directory. If None, it defaults
                     to the 'hydra' directory within the current module's parent directory.
        config_name: The name of the primary configuration file (without extension)
                     within `config_path` to compose.
        overrides: A list of strings, each representing a Hydra override (e.g., ["training.learning_rate=1e-3"]).
        
    Returns:
        A `DictConfig` object, which is Hydra's structured dictionary-like configuration.
    """
    if config_path is None:
        # Default config path is 'src/llm_pipeline/config/hydra' relative to this file.
        config_path = str(Path(__file__).parent / "hydra")
    
    # `initialize_config_dir` sets up the config search path.
    # `version_base=None` is used to allow older Hydra features if needed; for new projects, it's often "1.3".
    with initialize_config_dir(config_dir=config_path, version_base=None):
        # `compose` loads the specified config file and applies any overrides.
        cfg = compose(config_name=config_name, overrides=overrides or [])
    
    return cfg


def hydra_to_config(hydra_config: DictConfig) -> Config:
    """
    Convert a Hydra `DictConfig` object to the custom `Config` dataclass.
    
    This is a bridging function, allowing code that uses the custom `Config`
    dataclass to receive configurations managed by Hydra.
    
    Args:
        hydra_config: The `DictConfig` object generated by Hydra.
        
    Returns:
        A `Config` object, with all values resolved (e.g., interpolations like `${oc.env:VAR}` are replaced).
    """
    # Convert DictConfig to a standard Python dictionary, resolving all interpolations.
    config_dict = OmegaConf.to_container(hydra_config, resolve=True)
    # Instantiate the custom Config dataclass from the resolved dictionary.
    return Config.from_dict(config_dict)