# filename: src/llm_pipeline/cli/__init__.py
"""
Command-line interface for the LLM pipeline.

This `__init__.py` file serves as the public interface for the `llm_pipeline.cli`
package. It aggregates and re-exports the main `click` commands defined in its
sub-modules (`train.py`, `evaluate.py`, `preprocess.py`).

Purpose:
    To provide a centralized and easily discoverable set of command-line entry
    points for users. By importing these commands here, they can be exposed
    through a single top-level CLI tool (e.g., via a `pyproject.toml` or `setup.py`
    entry point that registers a main script to this package). This allows users
    to run commands like `llm_pipeline train`, `llm_pipeline evaluate`, etc.,
    if properly configured.

    This file itself primarily deals with imports and `__all__` definition.
    It's crucial for:
    1. **CLI Structure:** Organizes the various CLI functionalities under a
       single package namespace.
    2. **User Experience:** Simplifies how users interact with the pipeline
       by exposing key operations directly.
    3. **Package API:** Clearly defines which CLI commands are part of the
       public interface of the `cli` package.

LLM Pipeline Fit:
    This module is foundational for the user-facing interaction with the LLM
    pipeline. It bundles together the discrete steps of the pipeline (data
    preprocessing, model training, and model evaluation) into a coherent and
    accessible command-line interface, making the entire workflow manageable
    from the terminal.
"""

# Re-export the `train_command` function from the `train` module.
# This command is responsible for initiating the language model training process.
from llm_pipeline.cli.train import train_command

# Re-export the `evaluate_command` function from the `evaluate` module.
# This command is used to assess the performance of a trained language model.
from llm_pipeline.cli.evaluate import evaluate_command

# Re-export the `preprocess_command` function from the `preprocess` module.
# This command handles the preparation of raw text data, including tokenization.
from llm_pipeline.cli.preprocess import preprocess_command

# Define `__all__` to explicitly list the public API of the `llm_pipeline.cli` package.
# This tells Python which names should be imported when `from llm_pipeline.cli import *` is used.
# More importantly, it clarifies which functions are intended to be callable as CLI commands.
__all__ = [
    "train_command",      # The command for training the LLM.
    "evaluate_command",   # The command for evaluating the LLM.
    "preprocess_command", # The command for preprocessing data and building the tokenizer.
]