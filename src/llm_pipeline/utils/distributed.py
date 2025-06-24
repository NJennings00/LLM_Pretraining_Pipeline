# filename: src/llm_pipeline/utils/distributed.py
"""
Distributed training utilities.

This module provides a set of helper functions designed to facilitate
distributed training with PyTorch's `torch.distributed` package.
It abstracts away common distributed operations like getting rank/world size,
initializing/cleaning up the process group, and performing collective
communications (reduce, gather, broadcast).

Purpose:
    To simplify the implementation of distributed training within the LLM pipeline
    by providing a centralized and easy-to-use set of abstractions over `torch.distributed`.
    This helps in writing device-agnostic and scalable training code.

    Distributed training is essential for training large language models due to
    their immense computational and memory requirements. These utilities ensure
    that models can be effectively trained across multiple GPUs or machines,
    managing data and gradient synchronization correctly. They are critical
    for performance and scalability.

LLM Pipeline Fit:
    This module is part of the `llm_pipeline.utils` package. It is typically
    imported and used by the `Trainer` class or other components of the
    `llm_pipeline.training` module when distributed training is enabled.
    It provides the foundational distributed communication primitives required
    for data parallelism or other parallelization strategies.
"""

import os                        # Used to access environment variables like RANK, WORLD_SIZE, LOCAL_RANK.
import torch                     # Core PyTorch library.
import torch.distributed as dist # PyTorch's distributed communication package.
from typing import Optional, Any # Type hinting for optional values and any type.


def get_rank() -> int:
    """
    Get the rank of the current process in a distributed training setup.

    Purpose:
        To identify the unique ID of the current process within the distributed group.
        Rank 0 is typically designated as the "main" or "master" process.

    Returns:
        The integer rank of the current process. Returns 0 if distributed
        training is not available or not initialized (e.g., single-process training).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0 # Default to rank 0 for non-distributed or uninitialized scenarios.


def get_world_size() -> int:
    """
    Get the total number of processes participating in the distributed training.

    Purpose:
        To determine the total number of workers/GPUs involved in the distributed setup.
        This is crucial for operations like averaging gradients or distributing data.

    Returns:
        The integer world size. Returns 1 if distributed training is not available
        or not initialized (e.g., single-process training).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1 # Default to world size 1 for non-distributed or uninitialized scenarios.


def is_main_process() -> bool:
    """
    Check if the current process is the main (rank 0) process.

    Purpose:
        To control operations that should only be performed once (e.g., logging,
        saving checkpoints, downloading data) to avoid redundancy or conflicts
        across multiple processes.

    Returns:
        `True` if the current process's rank is 0, `False` otherwise.
    """
    return get_rank() == 0


def wait_for_everyone():
    """
    Synchronize all processes in the distributed group.

    Purpose:
        To ensure that all participating processes have reached a certain point
        in the execution before proceeding. This is critical for maintaining
        consistency in distributed operations (e.g., before/after a collective
        communication, before loading data).
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier() # PyTorch's barrier primitive.


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> None:
    """
    Initialize the PyTorch distributed training environment.

    Purpose:
        To set up the communication group required for multi-process training.
        This function typically reads environment variables set by launch scripts
        (e.g., `torch.distributed.launch` or `accelerate`).

    Args:
        backend: The communication backend to use (e.g., "nccl" for GPU, "gloo" for CPU).
                 "nccl" is generally preferred for GPU training due to its high performance.
        init_method: The method used to initialize the process group.
                     If `None`, it defaults to "env://" which uses environment variables
                     like `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE`.
    """
    # Check if necessary environment variables for distributed training are set.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if init_method is None:
            init_method = "env://" # Default to environment variable initialization.
        
        # Initialize the distributed process group.
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        
        # If CUDA is available, set the device for the current process
        # to its corresponding local GPU based on LOCAL_RANK environment variable.
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)


def cleanup_distributed():
    """
    Destroy the distributed process group.

    Purpose:
        To properly clean up resources allocated for distributed training
        at the end of the script execution.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all processes (e.g., sum or average).

    Purpose:
        To aggregate values from the same tensor across all participating processes.
        Commonly used for summing gradients across GPUs or aggregating loss values.

    Args:
        tensor: The `torch.Tensor` to be reduced.
        average: If `True`, the reduced values will be averaged across the world size.
                 If `False`, they will be summed.

    Returns:
        The reduced `torch.Tensor`. If distributed training is not active or
        world size is 1, the original tensor is returned.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor # Return original tensor if not in a distributed environment.
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor # No reduction needed for a single process.
    
    with torch.no_grad():       # Perform operation without tracking gradients.
        dist.all_reduce(tensor) # Sums the tensor across all processes by default.
        if average:
            tensor /= world_size # Divide by world size to get the average.
    
    return tensor


def gather_tensor(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Gather tensors from all processes to the main process.

    Purpose:
        To collect all parts of a distributed tensor (e.g., predictions, labels)
        onto a single process (rank 0) for consolidated evaluation or logging.
        This operation can be memory-intensive on the main process if tensors are large.

    Args:
        tensor: The `torch.Tensor` to be gathered from the current process.
                It is assumed to have a batch dimension as its first dimension (size 0).

    Returns:
        The concatenated `torch.Tensor` on the main process (`is_main_process()` returns True).
        Returns `None` on non-main processes.
        If distributed training is not active or world size is 1, the original tensor is returned.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    # 1. Gather sizes: Each process shares the size of its local tensor's first dimension.
    local_size = torch.tensor([tensor.size(0)], device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size) # All processes get all sizes.
    
    # 2. Pad tensors to the maximum size: This is necessary for `all_gather`
    #    as it requires tensors to be of equal size.
    max_size = max(size.item() for size in sizes)
    # Create a padded tensor filled with zeros, matching the largest local tensor's size.
    padded_tensor = torch.zeros(max_size, *tensor.shape[1:], device=tensor.device)
    padded_tensor[:tensor.size(0)] = tensor # Copy original tensor into the padded one.
    
    # 3. Gather padded tensors: Each process sends its padded tensor to all others.
    gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded_tensor)
    
    # 4. Trim padding: Remove the padding from each gathered tensor using the original sizes.
    gathered_tensors = [
        tensor[:size.item()] 
        for tensor, size in zip(gathered_tensors, sizes)
    ]
    
    # 5. Concatenate only on the main process.
    if is_main_process():
        return torch.cat(gathered_tensors, dim=0) # Concatenate along the batch dimension.
    return None # Non-main processes don't need the concatenated result.


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcasts an arbitrary Python object from a source process to all other processes.

    Purpose:
        To synchronize configuration, data structures, or other non-tensor Python objects
        across all distributed processes. Useful for ensuring all processes have the same
        hyperparameters or initial state.

    Args:
        obj: The Python object to broadcast. This object must be picklable.
        src: The rank of the source process that will broadcast the object.

    Returns:
        The broadcasted object on all processes. If distributed training is not active
        or world size is 1, the original object is returned.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return obj # Return original object if not in a distributed environment.
    
    if get_world_size() == 1:
        return obj # No broadcast needed for a single process.
    
    # Use pickle for serialization and deserialization of the object.
    import pickle
    import io
    
    buffer = io.BytesIO()
    if get_rank() == src:
        pickle.dump(obj, buffer) # Serialize the object on the source process.
    
    # Broadcast the size of the serialized object.
    if get_rank() == src:
        size = torch.tensor([buffer.tell()], dtype=torch.long) # Get size of serialized data.
    else:
        size = torch.tensor([0], dtype=torch.long) # Placeholder size for receivers.
    
    dist.broadcast(size, src=src) # All processes get the size from the source.
    
    # Broadcast the serialized object data.
    if get_rank() == src:
        data = buffer.getvalue() # Get bytes from buffer on source.
    else:
        data = bytearray(size.item()) # Create a bytearray of the correct size on receivers.
    
    # Wrap bytearray in a torch.ByteTensor for `dist.broadcast`.
    data_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(data))
    dist.broadcast(data_tensor, src=src) # All processes receive the data.
    
    # Deserialize the object.
    buffer = io.BytesIO(data_tensor.numpy().tobytes()) # Convert ByteTensor back to bytes.
    obj = pickle.load(buffer) # Deserialize the object.
    
    return obj