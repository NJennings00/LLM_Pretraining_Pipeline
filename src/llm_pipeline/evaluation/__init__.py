# filename: src/llm_pipeline/evaluation/__init__.py
"""
Evaluation modules.

This `__init__.py` file serves as the package initializer for the `evaluation`
sub-package within the `llm_pipeline`. Its primary purpose is to control
what is exposed when `from llm_pipeline.evaluation import *` is used, and to
provide a cleaner import path for key components of the evaluation system.

In an LLM pretraining pipeline, a well-structured `evaluation` package is critical
for standardizing how model performance is assessed. This `__init__.py` facilitates:
1. **Centralized Access**: Allows other parts of the pipeline (e.g., training scripts,
   main execution scripts) to import necessary evaluation tools directly from
   `llm_pipeline.evaluation` without needing to know the exact sub-module paths
   (e.g., `llm_pipeline.evaluation.evaluator` or `llm_pipeline.evaluation.metrics`).
2. **API Definition**: The `__all__` list explicitly defines the public interface
   of the `evaluation` package, making it clear which functions and classes are
   intended for external use.
3. **Modularity**: While exposing key components, it maintains the internal
   modularity of the `evaluator`, `metrics`, and `utils` sub-modules.
"""

from llm_pipeline.evaluation.evaluator import Evaluator, EvaluationConfig # Imports the `Evaluator` class and `EvaluationConfig` dataclass from the `evaluator` module. These are core components for setting up and running evaluations.
from llm_pipeline.evaluation.metrics import (                             # Imports specific functions and a class from the `metrics` module.
    compute_perplexity,                                                   # Function to calculate perplexity from loss.
    compute_accuracy,                                                     # Function to calculate token-level accuracy.
    EvaluationMetrics,                                                    # Class for collecting and managing evaluation metrics.
)
from llm_pipeline.evaluation.utils import (  # Imports utility functions from the `utils` module.
    generate_text,                           # Function for generating text samples from the model.
    sample_generations,                      # Utility for sampling generations (e.g., from a dataset).
)

__all__ = [               # Defines the public API of the `llm_pipeline.evaluation` package. 
                          # When `from llm_pipeline.evaluation import *` is used, only the names in this list will be imported.
    "Evaluator",          # Exposes the main Evaluator class.
    "EvaluationConfig",   # Exposes the configuration class for evaluation.
    "compute_perplexity", # Exposes the perplexity computation function.
    "compute_accuracy",   # Exposes the accuracy computation function.
    "EvaluationMetrics",  # Exposes the metrics container class.
    "generate_text",      # Exposes the text generation utility.
    "sample_generations", # Exposes the sample generations utility.
]