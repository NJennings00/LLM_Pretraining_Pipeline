# filename: src/llm_pipeline/config/config.py
"""
Configuration dataclasses for the LLM pretraining pipeline.

This module defines a structured way to manage all configuration parameters
required for the end-to-end LLM pretraining pipeline. It uses Python's
`dataclasses` for type-hinted, immutable, and easily readable configuration
objects.

Purpose:
    To centralize and organize all tunable parameters of the LLM pipeline,
    from data loading and preprocessing to model architecture, training
    hyperparameters, evaluation metrics, and logging settings. This makes
    configurations explicit, validates them where necessary, and simplifies
    passing them around the codebase.

    Configuration management is vital for reproducible and manageable ML
    experiments. These dataclasses are used throughout the `llm_pipeline` to:
    1. **Define Parameters:** Clearly state all configurable options.
    2. **Validate Inputs:** Enforce constraints on parameter values (e.g.,
       `hidden_size` divisibility, path types).
    3. **Ensure Type Safety:** Provide clear type hints for all parameters.
    4. **Facilitate Serialization:** Easily convert configurations to/from
       dictionaries (and thus JSON/YAML).
    5. **Integrate with Hydra:** Designed to work seamlessly with Hydra for
       flexible command-line overrides and composition.

LLM Pipeline Fit:
    These configuration objects are the backbone of the entire pipeline.
    Instances of `Config` (and its nested dataclasses) are passed to almost
    every major component, including `WikiTextDataset`, `TransformerLM`,
    `Trainer`, `Evaluator`, and utility functions, ensuring that all parts
    of the system operate under a consistent set of parameters.
"""

from dataclasses import dataclass, field            # Used for defining data classes. `field` for default factories.
from typing import Optional, List, Dict, Any, Union # Type hints for various data structures.
from pathlib import Path                            # For handling filesystem paths in an OS-agnostic way.
import torch                                        # Imported for `torch.cuda.is_available()` in TrainingConfig's __post_init__.


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.

    This dataclass holds parameters related to where the raw dataset is loaded from,
    how it's preprocessed, and how data loaders are configured.

    Why it's needed: To define the source and properties of the data.
    How it fits into the LLM pipeline: This configuration is used by `WikiTextDataset`
    and other data-related modules to fetch, preprocess, and prepare the dataset.
    Inputs/Outputs: Defines dataset name, splits, sequence length, preprocessing
    batching, and DataLoader worker settings.
    Ensures consistency in data handling across training and evaluation.
    """
    
    dataset_name: str = "wikitext"            # Name of the dataset (e.g., 'wikitext').
    dataset_config: str = "wikitext-2-raw-v1" # Specific configuration of the dataset.
    train_split: str = "train"                # Name of the training split.
    validation_split: str = "validation"      # Name of the validation split.
    test_split: str = "test"                  # Name of the test split.
    
    # Preprocessing parameters
    max_seq_length: int = 512            # Maximum sequence length for tokenized inputs.
    min_seq_length: int = 10             # Minimum sequence length to filter out very short texts.
    preprocessing_num_workers: int = 4   # Number of processes for parallel preprocessing.
    preprocessing_batch_size: int = 1000 # Batch size for dataset mapping during preprocessing.
    
    # Attributes for dataset chunking (how long texts are broken down)
    keep_last_incomplete_batch: bool = True # Whether to keep the last batch if it's smaller than full batch size.
    stride: Optional[int] = None            # Overlap when chunking long documents. If None, no overlap.
    
    # Data loading (dataloader settings) for PyTorch DataLoader
    num_workers: int = 4     # Number of subprocesses to use for data loading.
    pin_memory: bool = True  # If True, the data loader will copy Tensors to CUDA pinned memory.
    drop_last: bool = True   # If True, drop the last incomplete batch.
    
    # Caching settings for Hugging Face Datasets
    cache_dir: Optional[Path] = field(default_factory=lambda: Path("./cache")) # Directory for caching datasets.
    use_cache: bool = True                                                     # Whether to use caching for processed datasets.
    
    # Data augmentation settings 
    use_augmentation: bool = False  # Flag to enable data augmentation.
    augmentation_prob: float = 0.1  # Probability of applying an augmentation.
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Convert cache_dir to a Path object if it's not already, and ensure it's absolute.
        if self.cache_dir is not None and not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)
            

@dataclass
class TokenizerConfig:
    """
    Configuration for tokenizer creation and behavior.

    This dataclass defines parameters for building a new tokenizer or loading an
    existing one, as well as its special tokens and processing rules.

    Why it's needed: Tokenization is fundamental for transforming text into numerical IDs.
    How it fits into the LLM pipeline: This config is used by `build_tokenizer` and
    `TokenizerWrapper` to create or load the correct tokenizer for both preprocessing
    and model input preparation.
    Inputs/Outputs: Specifies tokenizer type, vocabulary size, special tokens,
    and paths.
    Ensures consistent tokenization across all stages.
    """
    
    tokenizer_type: str = "bpe"  # Type of subword tokenizer: 'bpe', 'wordpiece', or 'unigram'.
    vocab_size: int = 32000      # Desired size of the tokenizer vocabulary.
    min_frequency: int = 2       # Minimum frequency for a token to be included in the vocabulary during training.
    
    # Special tokens used by the tokenizer and model.
    pad_token: str = "<pad>"   # Token for padding sequences to uniform length.
    unk_token: str = "<unk>"   # Token for unknown words.
    bos_token: str = "<bos>"   # Beginning-of-sentence token.
    eos_token: str = "<eos>"   # End-of-sentence token.
    mask_token: str = "<mask>" # Mask token (primarily for masked language modeling, if used).
    
    # Tokenizer behavior flags.
    add_prefix_space: bool = True  # Whether to add a leading space to the first token (common for BPE).
    lowercase: bool = False        # Whether to lowercase text before tokenization.
    strip_accents: bool = False    # Whether to strip accents from characters.
    
    # Tokenizer training/loading
    train_tokenizer: bool = True          # Whether to train a new tokenizer if one isn't found.
    tokenizer_file: Optional[Path] = None # Path to save/load the tokenizer model.
    
    def __post_init__(self):
        """Validate configuration."""
        # Convert tokenizer_file to a Path object if it's not already.
        if self.tokenizer_file is not None and not isinstance(self.tokenizer_file, Path):
            self.tokenizer_file = Path(self.tokenizer_file)
            

@dataclass
class ModelConfig:
    """
    Configuration for transformer model architecture.

    This dataclass defines the core architectural parameters of the transformer
    language model, such as hidden sizes, number of layers, attention heads, etc.

    Why it's needed: To define the structure and capacity of the neural network.
    How it fits into the LLM pipeline: This configuration is passed to the
    `TransformerLM` class to instantiate the model with the specified architecture.
    Inputs/Outputs: Specifies model dimensions, dropout rates, activation functions.
    Ensures the model architecture is consistently defined.
    """
    
    # Core Architecture parameters
    model_type: str = "gpt2"            # Type of transformer architecture: 'gpt2', 'gpt', 'transformer'.
    vocab_size: int = 32000             # Size of the vocabulary. IMPORTANT: Must match tokenizer's vocab_size.
    hidden_size: int = 768              # Dimensionality of the embeddings and transformer layers.
    num_hidden_layers: int = 12         # Number of hidden layers (Transformer blocks).
    num_attention_heads: int = 12       # Number of attention heads in each multi-head attention block.
    intermediate_size: int = 3072       # Dimensionality of the "intermediate" (feed-forward) layer in transformer blocks.
    max_position_embeddings: int = 512  # The maximum sequence length that the model can handle.
    
    # Dropout probabilities
    hidden_dropout_prob: float = 0.1          # Dropout probability for hidden states.
    attention_probs_dropout_prob: float = 0.1 # Dropout probability for attention weights.
    embedding_dropout_prob: float = 0.1       # Dropout probability for embedding layer.
    
    # Layer normalization and initialization
    layer_norm_eps: float = 1e-12    # Epsilon for layer normalization.
    initializer_range: float = 0.02  # Standard deviation for weight initialization.
    use_cache: bool = True           # Whether the model should use its K/V cache during inference/generation.
    
    # Attention specific configurations
    attention_type: str = "standard"     # Type of attention mechanism: 'standard', 'flash', 'sparse'.
    use_rotary_embeddings: bool = False  # Whether to use Rotary Positional Embeddings (RoPE).
    rotary_dim: Optional[int] = None     # Dimension for RoPE. If None and use_rotary_embeddings, it's derived.
    
    # Activation function
    hidden_act: str = "gelu"  # Activation function to use in the feed-forward layers: 'gelu', 'relu', 'swish', 'gelu_new'.
    
    # Output layer
    tie_word_embeddings: bool = True # Whether to tie input and output word embeddings.
    
    def __post_init__(self):
        """Validate model configuration after initialization."""
        # Ensure hidden_size is divisible by num_attention_heads for proper multi-head attention.
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        # If using rotary embeddings and dim is not specified, derive it.
        if self.rotary_dim is None and self.use_rotary_embeddings:
            self.rotary_dim = self.hidden_size // self.num_attention_heads
            

@dataclass
class TrainingConfig:
    """
    Configuration for the training process.

    This dataclass defines all hyperparameters and settings related to the
    optimization, scheduling, evaluation, logging, and hardware usage during training.

    Why it's needed: To control the entire training loop behavior.
    How it fits into the LLM pipeline: This configuration is passed to the
    `Trainer` class, which uses these parameters to run the optimization process.
    Inputs/Outputs: Specifies epochs, batch sizes, learning rates, logging frequency,
    output directories, and mixed precision settings.
    
    Ensures consistent and reproducible training runs.
    """
    
    # Basic training parameters
    num_train_epochs: int = 3             # Total number of training epochs to perform.
    max_steps: int = -1                   # If set to a positive number, the total number of training steps to perform. -1 means use `num_train_epochs`.
    gradient_accumulation_steps: int = 1  # Number of updates steps to accumulate gradients before performing a backward/update pass.
    
    # Batch sizes specific to training and evaluation during the training loop.
    train_batch_size: int = 8  # Batch size per device during training.
    eval_batch_size: int = 16  # Batch size per device during evaluation.

    # Optimization parameters (AdamW is common)
    learning_rate: float = 5e-4  # The initial learning rate for the AdamW optimizer.
    weight_decay: float = 0.01   # The weight decay to apply.
    adam_beta1: float = 0.9      # The beta1 parameter for the AdamW optimizer.
    adam_beta2: float = 0.999    # The beta2 parameter for the AdamW optimizer.
    adam_epsilon: float = 1e-8   # The epsilon parameter for the AdamW optimizer.
    max_grad_norm: float = 1.0   # Maximum gradient norm (for gradient clipping).
    
    dataloader_num_workers: int = 0 # Number of subprocesses to use for data loading in the Trainer. Default to 0 for simpler debugging, can be increased for production.

    # Learning rate schedule parameters
    lr_scheduler_type: str = "cosine"  # Type of learning rate scheduler: 'linear', 'cosine', 'cosine_with_restarts'.
    warmup_steps: int = 500            # Number of steps for the warmup phase of the learning rate scheduler.
    warmup_ratio: float = 0.0          # Ratio of total training steps for the warmup phase (alternative to `warmup_steps`).
    
    # Mixed precision training
    fp16: bool = False          # Whether to use float16 (mixed precision) training.
    bf16: bool = False          # Whether to use bfloat16 (mixed precision) training.
    fp16_opt_level: str = "O1"  # Optimization level for NVIDIA Apex `fp16_opt_level` (O0, O1, O2, O3).
    
    # Gradient checkpointing to save memory
    gradient_checkpointing: bool = False      # Whether to use gradient checkpointing to save memory at the cost of speed.
    gradient_checkpointing_ratio: float = 0.5 # Percentage of layers to checkpoint.
    
    # Evaluation and checkpoint saving strategies
    eval_steps: int = 500         # Number of steps between evaluations.
    eval_strategy: str = "steps"  # Evaluation strategy: 'steps', 'epoch', 'no'.
    save_steps: int = 1000        # Number of steps between saving checkpoints.
    save_strategy: str = "steps"  # Checkpoint saving strategy: 'steps', 'epoch', 'no'.
    save_total_limit: int = 3     # Maximum number of checkpoints to keep. Oldest are deleted.
    
    # Logging parameters
    logging_steps: int = 50              # Number of steps between logging metrics.
    logging_first_step: bool = True      # Whether to log metrics at the very first step.
    logging_nan_inf_filter: bool = True  # Whether to filter NaN/Inf values from logged metrics.
    
    # Early stopping parameters
    early_stopping: bool = False           # Whether to enable early stopping.
    early_stopping_patience: int = 3       # Number of evaluation steps/epochs to wait for improvement before stopping.
    early_stopping_threshold: float = 0.0  # Minimum change in the monitored metric to qualify as an improvement.
    
    # Distributed training settings
    local_rank: int = -1                       # For distributed training: rank of the current process. -1 indicates not in DDP.
    ddp_backend: Optional[str] = None          # Distributed Data Parallel backend (e.g., 'nccl', 'gloo').
    deepspeed: Optional[Dict[str, Any]] = None # DeepSpeed configuration (if using DeepSpeed).
    
    # Hardware settings
    no_cuda: bool = False        # Whether to disable CUDA and force CPU usage.
    device: Optional[str] = None # Explicit device string (e.g., 'cuda:0', 'cpu'). Auto-detected if None.
    n_gpu: int = 1               # Number of GPUs to use (automatically detected if not specified in distributed setup).
    
    # Reproducibility
    seed: int = 42             # Random seed for all random operations.
    deterministic: bool = True # Whether to enforce deterministic algorithms (can impact performance).
    
    # Experiment tracking and output management
    run_name: Optional[str] = None                                      # Name of the current training run.
    output_dir: Path = field(default_factory=lambda: Path("./outputs")) # Directory to save models, logs, and checkpoints.
    overwrite_output_dir: bool = False                                  # Whether to overwrite the output directory if it already exists.
    
    # Resume training
    resume_from_checkpoint: Optional[Path] = None # Path to a checkpoint to resume training from.
    
    def __post_init__(self):
        """Process and validate training configuration after initialization."""
        # Convert output_dir and resume_from_checkpoint to Path objects.
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if self.resume_from_checkpoint is not None and not isinstance(self.resume_from_checkpoint, Path):
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)
            
        # Set default device if not explicitly provided.
        if self.device is None:
            if not self.no_cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
        # Validate mixed precision settings.
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16 mixed precision at the same time.")
            
@dataclass
class EvaluationConfig:
    """
    Configuration for the evaluation process of the LLM.
    Defines which metrics to compute, whether to generate samples,
    and parameters for text generation.

    Why it's needed: Centralizes all parameters related to how the model's
    performance is measured and how text generation is controlled during evaluation.
    How it fits into the LLM pipeline: This config is passed to the `Evaluator` class,
    which runs inference on evaluation data and computes metrics. It dictates the
    scope and nature of the evaluation.
    Inputs/Outputs: Stores evaluation process parameters like `compute_perplexity`,
    `generate_samples`, `max_generate_length`, `temperature`, `top_k`, `top_p`.
    
    Defines a structured way to configure the LLM evaluation process.
    """
    
    compute_perplexity: bool = True   # Whether to compute perplexity during evaluation.
    compute_accuracy: bool = True     # Whether to compute token accuracy during evaluation.
    generate_samples: bool = True     # Whether to generate text samples during evaluation.
    num_generate_samples: int = 10    # Number of text samples to generate.
    max_generate_length: int = 100    # Maximum length of generated text sequences.
    temperature: float = 1.0          # Sampling temperature for text generation. A higher value makes output more random.
    top_k: Optional[int] = 50         # Top-K sampling parameter for text generation. Considers only top_k most probable tokens.
    top_p: Optional[float] = 0.9      # Top-P (nucleus) sampling parameter for text generation. Considers smallest set of tokens whose cumulative probability exceeds top_p.
    use_cache: bool = True            # Whether the model should use its K/V cache during generation (speeds up sequential decoding).
    batch_size: Optional[int] = None  # Batch size for evaluation DataLoader. If None, it often defaults to `TrainingConfig.eval_batch_size`.


@dataclass
class LoggingConfig:
    """
    Configuration for logging and monitoring.

    This dataclass defines settings for various logging backends (TensorBoard,
    Weights & Biases, MLflow) and console/file logging.

    Why it's needed: To track experiment progress, visualize metrics, and debug issues.
    How it fits into the LLM pipeline: This configuration is used by the `setup_logger`
    utility and by various callbacks (e.g., `TensorBoardCallback`, `WandbCallback`)
    to initialize and send data to monitoring tools.
    Inputs/Outputs: Specifies logging destinations, run names, log levels, and
    what metrics to log.
    
    Provides a consistent logging setup across different runs.
    """
    
    # Logging backends to enable/disable
    use_tensorboard: bool = True  # Whether to use TensorBoard for logging.
    use_wandb: bool = False       # Whether to use Weights & Biases for logging.
    use_mlflow: bool = False      # Whether to use MLflow for logging.
    
    # Tensorboard specific settings
    tensorboard_dir: Path = field(default_factory=lambda: Path("./runs")) # Directory for TensorBoard logs.
    
    # Weights & Biases specific settings
    wandb_project: str = "llm-pretraining"  # Weights & Biases project name.
    wandb_entity: Optional[str] = None      # Weights & Biases entity (username or team).
    wandb_run_name: Optional[str] = None    # Name for the Weights & Biases run.
    wandb_tags: Optional[List[str]] = None  # List of tags for the Weights & Biases run.
    wandb_notes: Optional[str] = None       # Notes for the Weights & Biases run.
    
    # MLflow specific settings
    mlflow_tracking_uri: Optional[str] = None       # MLflow tracking server URI.
    mlflow_experiment_name: str = "llm-pretraining" # MLflow experiment name.
    
    # Console and file logging settings
    log_level: str = "INFO"  # Minimum logging level for console and file: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    log_to_file: bool = True # Whether to save logs to a file.
    log_file: Path = field(default_factory=lambda: Path("./logs/training.log")) # Path to the log file.
    
    # Specific metrics to log
    log_model_stats: bool = True     # Whether to log model architecture and parameter counts.
    log_gradient_stats: bool = True  # Whether to log gradient norms/statistics.
    log_parameter_stats: bool = True # Whether to log parameter values/statistics.
    log_learning_rate: bool = True   # Whether to log the current learning rate.
    
    def __post_init__(self):
        """Process paths after initialization."""
        # Convert path strings to Path objects.
        if not isinstance(self.tensorboard_dir, Path):
            self.tensorboard_dir = Path(self.tensorboard_dir)
        if not isinstance(self.log_file, Path):
            self.log_file = Path(self.log_file)
            

@dataclass
class Config:
    """
    Main configuration containing all sub-configurations for the LLM pretraining pipeline.

    This top-level dataclass serves as a single source of truth for all
    experiment parameters by composing the individual configuration dataclasses.

    Why it's needed: Provides a hierarchical and comprehensive configuration object
    that can be easily serialized/deserialized and managed (e.g., by Hydra).
    How it fits into the LLM pipeline: An instance of `Config` is the primary
    input to the `train_command` and other CLI entry points, encapsulating all
    settings for a given run.
    Inputs/Outputs: Composes `DataConfig`, `TokenizerConfig`, `ModelConfig`,
    `TrainingConfig`, `EvaluationConfig`, and `LoggingConfig`.
    
    The central configuration object for the entire LLM pretraining system.
    """
    
    # Nested sub-configurations, initialized with their default values.
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Placeholder for Hydra-specific configuration, if needed.
    hydra: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create a `Config` instance from a dictionary.

        This class method is useful for converting dictionary-based configurations
        (e.g., loaded from JSON/YAML or provided by Hydra) into the structured
        dataclass format.
        """
        # Instantiate each sub-configuration dataclass from its corresponding dictionary section.
        data_config = DataConfig(**config_dict.get("data", {}))
        tokenizer_config = TokenizerConfig(**config_dict.get("tokenizer", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        
        # Return a new `Config` instance with the populated sub-configurations.
        return cls(
            data=data_config,
            tokenizer=tokenizer_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config,
            logging=logging_config,
            hydra=config_dict.get("hydra", None),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the `Config` instance to a dictionary.

        This method allows for easy serialization of the entire configuration,
        for instance, to save it as a JSON file or pass it to logging frameworks.
        It handles `Path` objects by converting them to strings for serialization.
        """
        # Recursively convert sub-dataclasses to dictionaries.
        # Handle Path objects specifically by converting them to strings.
        return {
            "data": self.data.__dict__,
            "tokenizer": self.tokenizer.__dict__,
            "model": self.model.__dict__,
            "training": {k: str(v) if isinstance(v, Path) else v 
                         for k, v in self.training.__dict__.items()},
            "evaluation": {k: str(v) if isinstance(v, Path) else v 
                           for k, v in self.evaluation.__dict__.items()},
            "logging": {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.logging.__dict__.items()},
            "hydra": self.hydra, # Hydra config might contain complex objects, but usually simple dict.
        }
    
    def validate(self) -> None:
        """
        Perform cross-component validation of the entire configuration.

        This method enforces logical consistency between different parts of the
        configuration, ensuring that incompatible settings are caught early.
        """
        # Ensure that the model's vocabulary size matches the tokenizer's vocabulary size.
        assert self.model.vocab_size == self.tokenizer.vocab_size, \
            "Model and tokenizer vocab sizes must match for proper embedding and output layers."
        
        # Ensure that the maximum sequence length for data preprocessing does not exceed
        # the maximum position embeddings supported by the model (i.e., the model's context window).
        assert self.data.max_seq_length <= self.model.max_position_embeddings, \
            "Data max_seq_length cannot exceed model max_position_embeddings; " \
            "model cannot process sequences longer than its maximum positional encoding."