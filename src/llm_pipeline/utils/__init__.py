# filename: src/llm_pipeline/utils/__init__.py
"""
Utility modules for the LLM pipeline.

This `__init__.py` file serves as the public interface for the `llm_pipeline.utils`
package. It aggregates and re-exports key functions and classes from various
sub-modules (e.g., `logging`, `reproducibility`, `profiling`, `visualization`,
`distributed`).

Purpose:
    To provide a convenient, centralized access point for common utility functions
    used throughout the LLM pipeline. Instead of importing from specific sub-modules,
    users can import directly from `llm_pipeline.utils`, simplifying code and
    improving readability. This also defines what is considered part of the
    public API of the `utils` package.

    This file itself doesn't contain executable logic that needs testing beyond
    ensuring that the imports are correct and that the `__all__` list accurately
    reflects the re-exported symbols. It's crucial for:
    1. **Ease of Use:** Simplifies imports for downstream modules.
    2. **API Definition:** Clearly indicates which functions/classes are intended
       for external use.
    3. **Modularity:** While providing a unified interface, it still allows
       for internal organization of utilities into logical sub-modules.

LLM Pipeline Fit:
    The `llm_pipeline.utils` package is a core component providing foundational
    helper functions for various stages of the LLM lifecycle, including:
    - **Configuration and Logging:** Setting up logging, recording metrics and configurations.
    - **Reproducibility:** Ensuring experimental results can be replicated.
    - **Performance Monitoring:** Profiling model performance and resource usage.
    - **Data Visualization:** Creating plots to understand training dynamics and model behavior.
    - **Distributed Training:** Handling multi-GPU and multi-node setups.
    This `__init__.py` makes all these capabilities easily accessible.
"""

# Re-export functions from the `logging` utility module.
# These functions handle setting up loggers, retrieving them, and logging
# structured metrics and configuration details.
from llm_pipeline.utils.logging import (
    setup_logger,
    get_logger,
    log_metrics,
    log_config,
)

# Re-export functions from the `reproducibility` utility module.
# These are crucial for ensuring that experiments can be reliably repeated,
# covering random seeds and system/environment information.
from llm_pipeline.utils.reproducibility import (
    set_seed,
    set_deterministic,
    get_reproducibility_info,
    save_reproducibility_info,
)

# Re-export functions from the `profiling` utility module.
# These provide tools for analyzing the performance, speed, and memory consumption
# of models and operations.
from llm_pipeline.utils.profiling import (
    profile_model,
    profile_forward_pass,
    profile_memory_usage,
    get_gpu_memory_info,
)

# Re-export functions from the `visualization` utility module.
# These are used for generating various plots and figures to help understand
# model training, attention patterns, and embeddings.
from llm_pipeline.utils.visualization import (
    plot_training_curves,
    plot_attention_heatmap,
    plot_token_embeddings,
    save_figure,
)

# Re-export functions from the `distributed` utility module.
# These are essential for managing and coordinating processes in
# distributed training environments (e.g., multi-GPU, multi-node setups).
from llm_pipeline.utils.distributed import (
    get_rank,
    get_world_size,
    is_main_process,
    wait_for_everyone,
)

# Define `__all__` to explicitly list the public API of the `llm_pipeline.utils` package.
# When a user does `from llm_pipeline.utils import *`, only the names in `__all__`
# will be imported. This makes the public interface clear and prevents accidental imports
# of internal helper functions or variables.
__all__ = [
    # Logging Utilities
    "setup_logger",
    "get_logger",
    "log_metrics",
    "log_config",
    # Reproducibility Utilities
    "set_seed",
    "set_deterministic",
    "get_reproducibility_info",
    "save_reproducibility_info",
    # Profiling Utilities
    "profile_model",
    "profile_forward_pass",
    "profile_memory_usage",
    "get_gpu_memory_info",
    # Visualization Utilities
    "plot_training_curves",
    "plot_attention_heatmap",
    "plot_token_embeddings",
    "save_figure",
    # Distributed Training Utilities
    "get_rank",
    "get_world_size",
    "is_main_process",
    "wait_for_everyone",
]