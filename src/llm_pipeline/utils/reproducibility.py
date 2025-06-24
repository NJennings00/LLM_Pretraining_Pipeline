# filename: src/llm_pipeline/utils/reproducibility.py
"""
Reproducibility utilities.

This module provides a comprehensive set of tools to ensure and document
reproducibility for experiments, especially crucial in machine learning.
It covers setting random seeds, configuring deterministic operations in PyTorch,
and collecting detailed information about the execution environment (system, GPU,
Git, installed packages).

Purpose:
    To enable researchers and developers to reliably reproduce experimental results.
    Reproducibility is paramount in ML for validating findings, debugging,
    and comparing different models or hyperparameter configurations.

    Without proper reproducibility measures, ML experiments can be difficult to
    debug, verify, or build upon. This module's functions are used to:
    1. **Eliminate Randomness:** By fixing random seeds across different libraries.
    2. **Control Determinism:** For operations that might otherwise yield
       slightly different results on different runs (e.g., CUDA convolutions).
    3. **Document Environment:** Capture details of the hardware, software,
       and code version used, which are common causes of non-reproducibility.

LLM Pipeline Fit:
    This module is part of the `llm_pipeline.utils` package. Its `set_seed` and
    `set_deterministic` functions are typically called at the very beginning of
    any training or evaluation script. `save_reproducibility_info` can be called
    at the start or end of a run to log the environment.
"""

import random                                        # For Python's built-in pseudo-random number generator.
import os                                            # For interacting with the operating system (e.g., environment variables).
import json                                          # For serializing information into JSON format.
from pathlib import Path                             # For object-oriented filesystem paths.
from typing import Dict, Any, Optional, Union, List  # Type hinting.
import numpy as np                                   # For numerical operations, specifically for NumPy's random seed.
import torch                                         # For PyTorch-specific random seed and deterministic settings.
import logging                                       # For logging information, warnings.
import platform                                      # For getting system platform information.
import subprocess                                    # For running external commands (e.g., Git commands).
import sys                                           # For getting Python version.
from datetime import datetime                        # For timestamping reproducibility info.


logger = logging.getLogger(__name__) # Initializes a logger specifically for this module.


# Helper function to make NumPy types JSON serializable
def _json_serializable_default(obj):
    """
    Default JSON serializer for objects not serializable by default `json` module.

    Purpose:
        The standard `json` module cannot serialize NumPy integer or float types,
        nor `pathlib.Path` objects directly. This helper function provides a custom
        serializer to handle these common data types encountered in ML contexts.

    Args:
        obj: The object to be serialized.

    Returns:
        A Python int, float, or string representation of the object.

    Raises:
        TypeError: If the object's type is still not JSON serializable after
                   attempting conversions for known types.
    """
    if isinstance(obj, (np.integer, np.int_)):      # Handles NumPy integer types (e.g., np.int64, np.uint32).
        return int(obj)                             # Convert to standard Python int.
    elif isinstance(obj, (np.floating, np.float_)): # Handles NumPy floating point types (e.g., np.float32, np.float64).
        return float(obj)                           # Convert to standard Python float.
    elif isinstance(obj, Path):                     # Handles `pathlib.Path` objects.
        return str(obj)                             # Convert to string representation.
    # If the object is none of the above, raise a TypeError.
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python's `random`, NumPy, and PyTorch (CPU and CUDA).

    Purpose:
        To ensure that any operations relying on pseudo-random number generation
        (e.g., data shuffling, weight initialization, dropout) produce the same
        sequence of "random" numbers across different runs, contributing to reproducibility.

    Args:
        seed: The integer seed value to use.
    """
    random.seed(seed)       # Set seed for Python's built-in `random` module.
    np.random.seed(seed)    # Set seed for NumPy's random number generator.
    torch.manual_seed(seed) # Set seed for PyTorch on CPU.
    
    # Set seed for PyTorch on all CUDA GPUs if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set PYTHONHASHSEED environment variable to ensure consistent hash values
    # across runs, which affects dictionary iteration order and other hashed operations.
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set to {seed}")


def set_deterministic(deterministic: bool = True) -> None:
    """
    Configure PyTorch to use deterministic algorithms where possible.

    Purpose:
        Even with fixed random seeds, some operations (especially on GPUs)
        might not be perfectly deterministic due to floating-point precision
        or the nature of parallel computations. This function attempts to
        force deterministic behavior in PyTorch.

    Args:
        deterministic: If `True`, enable deterministic operations. If `False`,
                       disable them (which might improve performance but reduce reproducibility).
    """
    if deterministic:
        # `torch.backends.cudnn.deterministic = True` makes cuDNN operations deterministic.
        # `torch.backends.cudnn.benchmark = False` disables cuDNN autotuner, which can
        # choose non-deterministic algorithms for speed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # `torch.use_deterministic_algorithms(True)` (available from PyTorch 1.8)
        # enables deterministic behavior for more PyTorch operations beyond cuDNN.
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        # Note: Depending on the PyTorch version and CUDA version, some operations
        # might still not have deterministic implementations, and this might raise a warning/error.

        logger.info("Deterministic mode enabled")
    else:
        # Revert to default (potentially non-deterministic but faster) settings.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)

        logger.info("Deterministic mode disabled")


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Retrieve information about available GPU devices.

    Purpose:
        To document the specific GPUs used for an experiment, including their
        names and memory characteristics. This is crucial for hardware-related
        reproducibility.

    Returns:
        A list of dictionaries, where each dictionary contains details for
        one detected CUDA-enabled GPU. Returns an empty list if no GPUs are available.
    """
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "name": props.name,
                "memory_total_bytes": props.total_memory, # Total memory on the GPU in bytes.
                # These metrics reflect current usage; they are not fixed properties.
                # Including them gives a snapshot, but might not be perfectly reproducible
                # as they depend on what's running on the GPU at call time.
                "memory_allocated_bytes": torch.cuda.memory_allocated(i),
                "memory_cached_bytes": torch.cuda.memory_reserved(i),
            })
    return gpu_info


def get_git_info() -> Dict[str, str]:
    """
    Retrieve information about the current Git repository state.

    Purpose:
        To record the exact version of the code that was executed, which is
        fundamental for perfect code reproducibility. This includes branch,
        commit hash, commit message, and any uncommitted changes.

    Returns:
        A dictionary containing Git details: "branch", "commit_hash",
        "commit_message", and "diff" (for uncommitted changes).
        If not in a Git repository or Git is not found, returns "N/A" values.
    """
    git_info = {}
    try:
        # Get current branch name.
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        # Get full commit hash.
        git_info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        # Get the last commit message.
        git_info["commit_message"] = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        
        # Get the diff of uncommitted changes.
        diff_output = subprocess.check_output(
            ["git", "diff", "--submodule=diff"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if diff_output:
            git_info["diff"] = diff_output # Store the actual diff if changes exist.
        else:
            git_info["diff"] = "No uncommitted changes." # Indicate no changes.

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Handle cases where `git` command fails or is not found.
        logger.warning("Not a git repository or git not found. Git information will be 'N/A'.")
        git_info["branch"] = "N/A"
        git_info["commit_hash"] = "N/A"
        git_info["commit_message"] = "N/A"
        git_info["diff"] = "Git information not available."
    return git_info


def get_environment_info() -> Dict[str, Any]:
    """
    Retrieve general system and Python environment information.

    Purpose:
        To capture broader environmental factors that can influence reproducibility,
        such as operating system details, Python version, and current working directory.

    Returns:
        A dictionary containing various environment details.
    """
    env_info = {
        "python_version": sys.version,      # Full Python version string.
        "platform": platform.platform(),    # Generic system name (e.g., 'Linux-5.15.0-78-generic-x86_64-with-glibc2.35').
        "processor": platform.processor(),  # Processor name (e.g., 'x86_64').
        "hostname": platform.node(),        # Network name of the computer.
        "cwd": os.getcwd(),                 # Current working directory.
        "os_release": platform.release(),   # OS release (e.g., '5.15.0-78-generic').
        "os_version": platform.version(),   # OS version information.
        "architecture": platform.machine(), # Machine architecture (e.g., 'x86_64').
    }
    return env_info


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of commonly used Python packages in an LLM pipeline.

    Purpose:
        To document the exact versions of critical dependencies. Different package
        versions can introduce subtle behavior changes or bugs, making this
        information vital for debugging reproducibility issues.

    Returns:
        A dictionary where keys are package names and values are their version strings,
        or "N/A" if the package is not found or its version cannot be determined.
    """
    package_versions = {}
    # Define a comprehensive list of key packages relevant to LLM pipelines.
    key_packages = [
        "torch", "transformers", "datasets", "tokenizers", "accelerate",
        "numpy", "scipy", "scikit-learn", "pandas", "hydra-core",
        "omegaconf", "python-dotenv", "colorlog", "tqdm", "psutil",
        "GPUtil", "py-cpuinfo", "wandb", "tensorboard", "nltk", "rouge_score" 
    ]
    for pkg_name in key_packages:
        try:
            # Dynamically import the package and try to get its __version__ attribute.
            package_versions[pkg_name] = __import__(pkg_name).__version__
        except (ImportError, AttributeError):
            package_versions[pkg_name] = "N/A" # Package not found or no `__version__` attribute.
    return package_versions


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Collect comprehensive information for ensuring reproducibility of an experiment.

    Purpose:
        To aggregate all relevant system, environment, code, and random state
        information into a single, structured dictionary that can be saved.

    Returns:
        A dictionary containing:
        - `timestamp`: When the info was collected.
        - `environment`: Output of `get_environment_info()`.
        - `gpu_info`: Output of `get_gpu_info()`.
        - `git_info`: Output of `get_git_info()`.
        - `package_versions`: Output of `get_package_versions()`.
        - `random_seed_set`: Current state of Python's `random` module seed.
        - `numpy_seed_set`: Current state of NumPy's random module seed.
        - `torch_seed_set`: Current state of PyTorch's initial random seed.
    """
    info = {
        "timestamp": datetime.now().isoformat(), # Current time in ISO format.
        "environment": get_environment_info(),
        "gpu_info": get_gpu_info(),
        "git_info": get_git_info(),
        "package_versions": get_package_versions(),
        # Get the internal state of random number generators for full reproducibility.
        # Note: These might not be the *exact* original seeds passed to `set_seed`
        # if the generators have been advanced, but they capture the current state.
        # Convert to Python int for JSON serialization.
        "random_seed_set": int(random.getstate()[0]),       # Python `random` state.
        "numpy_seed_set": int(np.random.get_state()[1][0]), # NumPy `random` state.
        "torch_seed_set": int(torch.initial_seed()),        # PyTorch CPU random seed.
    }
    return info


def save_reproducibility_info(save_path: Union[str, Path]) -> None:
    """
    Save the collected reproducibility information to a JSON file.

    Purpose:
        To persistently store the reproducibility details alongside experiment results,
        allowing for later review and recreation of the exact experimental conditions.

    Args:
        save_path: The full path to the JSON file where the information will be saved.
                   Parent directories will be created if they do not exist.
    """
    save_path = Path(save_path) # Ensure `save_path` is a `Path` object.
    
    # Create parent directories for the save_path if they don't exist.
    # This prevents `FileNotFoundError` if the directory structure isn't pre-made.
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    info = get_reproducibility_info() # Collect all information.
    
    with open(save_path, "w") as f:
        # Use `json.dump` with the custom `_json_serializable_default` function
        # to handle non-standard (e.g., NumPy) types, and indent for readability.
        json.dump(info, f, indent=2, default=_json_serializable_default)
    
    logger.info(f"Reproducibility info saved to {save_path}")