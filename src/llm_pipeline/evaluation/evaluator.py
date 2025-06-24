# filename: src/llm_pipeline/evaluation/evaluator.py
"""
Main evaluator class for language models.

This file defines the `Evaluator` class, which is a crucial component in an LLM
pretraining pipeline. Its general role is to assess the performance and characteristics
of a trained or partially trained language model. This includes computing standard
language model metrics like loss and perplexity, evaluating prediction accuracy,
generating text samples to qualitatively assess generation capabilities, and
benchmarking model-specific attributes like parameter count, FLOPs, and inference speed.

Within an LLM pretraining pipeline, evaluation is a continuous process that
provides feedback on the model's learning progress. This `Evaluator` class allows
for systematic and comprehensive evaluation at various stages (e.g., end of epoch,
after a certain number of training steps, or for final model assessment).
It helps monitor for overfitting, track improvements, and understand the model's
strengths and weaknesses.
"""

import logging                           # Imports the logging module for logging messages.
from typing import Optional, Any, Union  # Imports specific types for type hinting, improving code readability and maintainability.
from dataclasses import dataclass        # Imports `dataclass` for easily creating data classes, though not directly used in this file, it's often used for configurations.
import torch                             # Imports the PyTorch library, fundamental for deep learning operations.
import torch.nn as nn                    # Imports the neural network module from PyTorch.
from torch.utils.data import DataLoader  # Imports DataLoader for efficient batching and loading of datasets.
from tqdm import tqdm                    # Imports tqdm for displaying progress bars during iterations.
import numpy as np                       # Imports the NumPy library for numerical operations, though not heavily used here directly.

from llm_pipeline.models import TransformerLM  # Imports the TransformerLM model definition from the local models module.
from llm_pipeline.evaluation.metrics import (  # Imports various metrics functions and classes from the local evaluation.metrics module.
    compute_perplexity,                        # Function to compute perplexity from loss.
    compute_accuracy,                          # Function to compute accuracy (though the logic is now inline).
    EvaluationMetrics,                         # Class to track and manage evaluation metrics.
)
from llm_pipeline.evaluation.utils import generate_text   # Only generate_text is needed # Imports the generate_text utility function from the local evaluation.utils module.
from llm_pipeline.data.tokenizer import TokenizerWrapper  # Imports the TokenizerWrapper for handling tokenization.
from llm_pipeline.config import EvaluationConfig          # Imports the EvaluationConfig dataclass for evaluation-specific settings.

logger = logging.getLogger(__name__) # Initializes a logger for this module, enabling structured logging of events and messages.


class Evaluator:
    """
    Evaluator for language models.

    Purpose:
        This class provides a comprehensive framework for evaluating the performance
        of a Transformer-based Language Model (TransformerLM). It encapsulates logic
        for computing quantitative metrics like loss, perplexity, and accuracy, as well
        as qualitative aspects like text generation, and system-level metrics such as
        parameter counts and inference speed.

        This class is used to systematically measure and report on the model's capabilities
        during and after the training process. It ensures that the model's performance
        is tracked against established benchmarks and that its text generation
        abilities are qualitatively assessed.

    LLM Pipeline Fit:
        In an LLM pretraining pipeline, the `Evaluator` serves as the post-training
        or in-training evaluation component. After a certain number of training steps
        or epochs, the trainer calls this evaluator to get an objective measure of
        how well the model is learning the language. The metrics provided by this
        class inform decisions about hyperparameter tuning, model architecture changes,
        and early stopping. It's essential for validating the effectiveness of the
        training process and the quality of the resulting model.

    Inputs:
        - `model` (TransformerLM): The language model instance to be evaluated.
        - `dataloader` (DataLoader): A PyTorch DataLoader providing the evaluation dataset.
        - `tokenizer` (TokenizerWrapper): An instance of the tokenizer used by the model
          for encoding and decoding text.
        - `config` (Optional[EvaluationConfig]): An optional configuration object
          specifying evaluation parameters (e.g., whether to compute perplexity,
          number of samples to generate). If None, a default `EvaluationConfig` is used.
        - `device` (Optional[torch.device]): The computing device (e.g., 'cuda', 'cpu')
          to which the model and data should be moved for evaluation. If None, it
          defaults to CUDA if available, otherwise CPU.

    Outputs:
        The `Evaluator` class itself does not return a value upon initialization.
        Its methods, such as `evaluate()`, return dictionaries of computed metrics.

    Confirms/Verifies:
        - The model's ability to minimize loss on unseen data.
        - The model's ability to predict the next token accurately (via perplexity and accuracy).
        - The model's capacity for coherent and relevant text generation.
        - Basic architectural properties like parameter count and estimated computational cost.
        - Inference efficiency on the specified hardware.
    """

    def __init__(                                  # Defines the constructor method for the Evaluator class.
        self,                                      # The instance of the class.
        model: TransformerLM,                      # Type hint: the model to be evaluated, expected to be a TransformerLM.
        dataloader: DataLoader,                    # Type hint: the DataLoader providing evaluation data.
        tokenizer: TokenizerWrapper,               # Type hint: the tokenizer wrapper.
        config: Optional[EvaluationConfig] = None, # Type hint: optional evaluation configuration. Defaults to None.
        device: Optional[torch.device] = None,     # Type hint: optional PyTorch device. Defaults to None.
    ): 
        """
        Initialize evaluator.

        Purpose:
            Initializes the Evaluator object by setting up the model, data loader,
            tokenizer, configuration, and device. It also moves the model to the
            specified device once and initializes the metrics tracker.

            This method is crucial for preparing the `Evaluator` for its tasks.
            It ensures that all necessary components (model, data, tokenizer, config)
            are correctly linked and that the model is placed on the appropriate
            compute device, optimizing for performance.

        LLM Pipeline Fit:
            This initialization happens at the beginning of an evaluation run within
            the LLM pipeline. It's a setup phase that prepares all the components
            needed for subsequent metric computation and text generation.

        Inputs:
            - `model` (TransformerLM): The language model instance to be evaluated.
            - `dataloader` (DataLoader): A PyTorch DataLoader providing the evaluation dataset.
            - `tokenizer` (TokenizerWrapper): An instance of the tokenizer used by the model
              for encoding and decoding text.
            - `config` (Optional[EvaluationConfig]): An optional configuration object
              specifying evaluation parameters.
            - `device` (Optional[torch.device]): The computing device (e.g., 'cuda', 'cpu')
              to which the model and data should be moved for evaluation.

        Outputs:
            None. The primary effect is the initialization of the `Evaluator` instance's
            internal state.

        Confirms/Verifies:
            - Correct assignment of the model, dataloader, tokenizer, and configuration.
            - Proper device selection and model transfer to that device.
            - Initialization of the internal metrics tracker.
        """
        self.model = model                                                                   # Assigns the input model to an instance variable.
        self.dataloader = dataloader                                                         # Assigns the input dataloader to an instance variable.
        self.tokenizer = tokenizer                                                           # Assigns the input tokenizer to an instance variable.
        self.config = config or EvaluationConfig()                                           # Assigns the provided config, or a default EvaluationConfig if none is given.
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determines the device: 'cuda' if a GPU is available, otherwise 'cpu'.

        # Move model to device during initialization (only once)
        self.model.to(self.device) # Moves the entire model to the selected device (GPU or CPU).

        # Metrics tracker
        self.metrics = EvaluationMetrics() # Initializes an instance of EvaluationMetrics to store and manage computed metrics.

    @torch.no_grad() # Decorator that disables gradient calculation. This is crucial for evaluation to save memory and computations.
    def evaluate(self) -> dict[str, float]: # Defines the public `evaluate` method. It takes `self` and returns a dictionary with string keys and float values.
        """
        Run evaluation.

        Purpose:
            Orchestrates the entire evaluation process. It sets the model to evaluation
            mode, computes loss-based metrics (perplexity, accuracy) if configured,
            and generates text samples if enabled in the configuration. It then
            aggregates and returns all computed metrics.

            This is the main entry point for running an evaluation. It ensures that
            the evaluation steps are performed in the correct order and that all
            relevant metrics are collected and presented.

        LLM Pipeline Fit:
            This method is called periodically during or at the end of the training
            loop in an LLM pretraining pipeline. It provides a snapshot of the
            model's performance, which can be used for logging, visualization,
            and decision-making regarding the training process.

        Inputs:
            - `self`: The instance of the `Evaluator` class.

        Outputs:
            - `dict[str, float]`: A dictionary containing various evaluation metrics,
              such as 'loss', 'perplexity', 'accuracy', and 'num_generated_samples'.

        Confirms/Verifies:
            - The correct execution flow of the evaluation process.
            - The aggregation of all configured evaluation metrics.
            - The model's performance on the evaluation dataset.
        """
        logger.info("Running evaluation...") # Logs an informational message indicating the start of evaluation.
        self.model.eval()                    # Sets the model to evaluation mode. This disables dropout and batch normalization updates.

        # Compute loss and perplexity
        if self.config.compute_perplexity or self.config.compute_accuracy: # Checks if either perplexity or accuracy computation is enabled in the configuration.
            loss_metrics = self._compute_loss_metrics()                    # Calls a private method to compute loss-based metrics.
            self.metrics.update(loss_metrics)                              # Updates the internal metrics tracker with the computed loss-based metrics.

        # Generate samples
        if self.config.generate_samples: # Checks if text sample generation is enabled in the configuration.
            # _generate_samples will now return ONLY numerical metrics for tracking
            # The actual generated text will be logged internally by _generate_samples
            generation_metrics = self._generate_samples() # Calls a private method to generate text samples and returns numerical metrics related to generation.
            self.metrics.update(generation_metrics)       # Updates the internal metrics tracker with numerical metrics from sample generation.

        return self.metrics.get_metrics() # Returns the accumulated metrics as a dictionary.

    def _compute_loss_metrics(self) -> dict[str, float]: # Defines a private method to compute loss-based metrics. It returns a dictionary of string keys and float values.
        """
        Compute loss-based metrics.

        Purpose:
            Iterates through the evaluation dataloader, performs a forward pass
            to calculate the loss and accuracy for each batch, and accumulates
            these values. It then computes the average loss, perplexity, and
            overall accuracy.

            This method is fundamental for quantitative evaluation of the language
            model's core task: predicting the next token. Loss and perplexity
            directly indicate how well the model predicts the training data
            distribution, while accuracy measures correct token predictions.

        LLM Pipeline Fit:
            This is a core part of the quantitative evaluation within the LLM pipeline.
            It provides numerical feedback on the model's ability to learn the
            underlying language patterns. The results are critical for monitoring
            convergence and identifying issues like underfitting or overfitting.

        Inputs:
            - `self`: The instance of the `Evaluator` class.

        Outputs:
            - `dict[str, float]`: A dictionary containing 'loss', 'perplexity'
              (if configured), and 'accuracy' (if configured).

        Confirms/Verifies:
            - The model's ability to compute a valid loss on unseen data.
            - The model's perplexity, indicating its uncertainty in predicting sequences.
            - The model's next-token prediction accuracy on the evaluation set.
            - Robustness to potential invalid token IDs by mapping them to UNK.
        """
        total_loss = 0.0       # Initializes a variable to accumulate the total loss across all batches.
        total_tokens = 0       # Initializes a variable to count the total number of tokens processed for loss calculation.
        total_correct = 0      # Initializes a variable to count the total number of correctly predicted tokens for accuracy calculation.
        total_valid_tokens = 0 # Initializes a variable to count the total number of valid (non-masked) tokens for accuracy calculation.

        progress_bar = tqdm(                                  # Initializes a tqdm progress bar for visualizing the evaluation loop.
            self.dataloader,                                  # The iterable (dataloader) over which to show progress.
            desc="Evaluating",                                # Description displayed next to the progress bar.
            disable=logging.getLogger().level > logging.INFO, # Disables the progress bar if the logging level is higher than INFO (e.g., WARNING, ERROR).
        )

        for batch in progress_bar: # Iterates through each batch provided by the dataloader.
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()} # Moves all tensors in the current batch to the specified device.

            # Fix invalid token IDs in batch (both input_ids and labels)
            vocab_size = self.model.config.vocab_size        # Retrieves the vocabulary size from the model's configuration.
            for key in ['input_ids', 'labels']:              # Iterates through 'input_ids' and 'labels' keys in the batch.
                if key in batch:                             # Checks if the current key exists in the batch.
                    invalid_mask = batch[key] >= vocab_size  # Creates a boolean mask where token IDs are greater than or equal to the vocabulary size (i.e., invalid).
                    if invalid_mask.any():                   # Checks if there are any invalid token IDs in the current tensor.
                        logger.warning(f"Found {invalid_mask.sum()} invalid token IDs in {key}, mapping to UNK (0)") # Logs a warning if invalid token IDs are found.
                        # Map invalid token IDs to UNK token (ID=0)
                        # For labels, keep -100 (ignore index) unchanged
                        if key == 'labels':                                        # Special handling for 'labels'.
                            batch[key] = torch.where(                              # Conditionally replaces elements in the labels tensor.
                                (batch[key] >= vocab_size) & (batch[key] != -100), # Condition: if ID is invalid AND not the ignore index (-100).
                                torch.tensor(0, device=batch[key].device),         # Value to replace with: UNK token ID (0) on the same device.
                                batch[key]                                         # Original value if condition is false.
                            )
                        else:                                              # For 'input_ids' or other non-label tensors.
                            batch[key] = torch.where(                      # Conditionally replaces elements in the input_ids tensor.
                                batch[key] >= vocab_size,                  # Condition: if ID is invalid.
                                torch.tensor(0, device=batch[key].device), # Value to replace with: UNK token ID (0) on the same device.
                                batch[key]                                 # Original value if condition is false.
                            )

            # Forward pass
            outputs = self.model(**batch) # Performs a forward pass through the model with the prepared batch, unpacking the dictionary.

            # Get loss
            if "loss" in outputs:                        # Checks if the model output contains a 'loss' key.
                loss = outputs["loss"]                   # Retrieves the loss value from the model's output.
                batch_size = batch["input_ids"].shape[0] # Gets the batch size from the input_ids tensor.
                seq_length = batch["input_ids"].shape[1] # Gets the sequence length from the input_ids tensor.

                # Account for sequence length in loss
                # Note: Labels with -100 are ignored by CrossEntropyLoss, so scaling by actual non-masked tokens is preferred
                # For consistency with the existing logic, we keep current scaling if loss is per batch mean:
                total_loss += loss.item() * batch_size * seq_length # Accumulates the total loss, scaling by batch size and sequence length (assuming loss is per-token or per-sequence averaged).
                total_tokens += batch_size * seq_length             # Accumulates the total number of tokens processed.

            # Compute accuracy if requested
            if self.config.compute_accuracy and "labels" in batch: # Checks if accuracy computation is enabled and if 'labels' are present in the batch.
                logits = outputs["logits"]                         # Retrieves the logits (raw unnormalized scores) from the model's output.
                labels = batch["labels"]                           # Retrieves the true labels from the batch.

                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous() # Shifts logits: removes the last token's logits as they correspond to predicting a non-existent next token. `.contiguous()` ensures memory contiguity for correct reshaping.
                shift_labels = labels[..., 1:].contiguous()     # Shifts labels: removes the first label as it's the input for the first token's prediction, and the labels are for the *next* token.

                # Get predictions
                predictions = shift_logits.argmax(dim=-1) # Gets the predicted token IDs by taking the argmax along the last dimension (vocabulary dimension) of the shifted logits.

                # Compute accuracy (ignoring padding and -100 labels)
                mask = shift_labels != -100                                       # Creates a mask to identify valid labels (i.e., not the ignore index -100).
                total_correct += (predictions == shift_labels)[mask].sum().item() # Counts correctly predicted tokens by applying the mask to a comparison of predictions and shifted labels, then sums them up.
                total_valid_tokens += mask.sum().item()                           # Counts the total number of valid (non-masked) tokens for accuracy calculation.

        # Compute final metrics
        metrics = {} # Initializes an empty dictionary to store the final computed metrics.

        if total_tokens > 0:                     # Checks if any tokens were processed to avoid division by zero.
            avg_loss = total_loss / total_tokens # Calculates the average loss per token.
            metrics["loss"] = avg_loss           # Stores the average loss in the metrics dictionary.

            if self.config.compute_perplexity:                       # Checks if perplexity computation is enabled.
                metrics["perplexity"] = compute_perplexity(avg_loss) # Computes and stores perplexity using the average loss.

            if self.config.compute_accuracy: # Checks if accuracy computation is enabled.
                # To avoid division by zero if all labels are -100
                if total_valid_tokens > 0:                                   # Checks if there were any valid tokens for accuracy calculation.
                    metrics["accuracy"] = total_correct / total_valid_tokens # Calculates and stores the accuracy.
                else:                                                        # If no valid tokens were found.
                    metrics["accuracy"] = 0.0                                # Sets accuracy to 0.0 to avoid division by zero.

        return metrics # Returns the dictionary of computed loss-based metrics.

    def _generate_samples(self) -> dict[str, Any]: # Defines a private method to generate text samples. It returns a dictionary.
        """
        Generate text samples.

        Purpose:
            Generates a specified number of text samples from the language model
            using either default prompts or prompts extracted from the dataset.
            The generated texts are logged directly to the console or log file.
            It returns a numerical metric indicating the number of samples generated.

            Text generation provides a qualitative assessment of the model's linguistic
            capabilities beyond numerical metrics. It helps to understand the coherence,
            fluency, and relevance of the text produced by the model, which is crucial
            for evaluating its real-world utility.

        LLM Pipeline Fit:
            This method is part of the qualitative evaluation in the LLM pipeline.
            It helps developers visually inspect the model's output, giving insights
            into potential issues like repetition, nonsensical output, or lack of
            coherence, complementing the quantitative metrics.

        Inputs:
            - `self`: The instance of the `Evaluator` class.

        Outputs:
            - `dict[str, Any]`: A dictionary containing a single numerical metric:
              'num_generated_samples'. The actual generated text is logged internally.

        Confirms/Verifies:
            - The model's ability to generate coherent and plausible text given a prompt.
            - The integration of the `generate_text` utility with the evaluator.
            - The logging mechanism for generated samples.
        """
        logger.info(f"Generating {self.config.num_generate_samples} samples...") # Logs an informational message about the number of samples to be generated.

        generations = [] # Initializes an empty list to store details of generated texts (though not returned directly).
        # Get sample prompts from dataset or use defaults
        prompts = self._get_sample_prompts() # Calls a private method to obtain a list of prompts for text generation.

        # Generate and log samples directly here
        for i, prompt in enumerate(prompts[:self.config.num_generate_samples]): # Iterates through a subset of prompts, up to the configured number of samples.
            generated_text = generate_text(                                     # Calls the external `generate_text` utility function.
                model=self.model,                                               # Passes the language model.
                tokenizer=self.tokenizer,                                       # Passes the tokenizer.
                prompt=prompt,                                                  # Passes the current prompt.
                max_length=self.config.max_generate_length,                     # Passes the maximum length for generated text.
                temperature=self.config.temperature,                            # Passes the sampling temperature.
                top_k=self.config.top_k,                                        # Passes the top-k sampling parameter.
                top_p=self.config.top_p,                                        # Passes the top-p (nucleus) sampling parameter.
                device=self.device,                                             # Pass the evaluator's device # Passes the device the model is on.
            )
            generations.append({             # Appends a dictionary containing the prompt and generated text to the `generations` list.
                "prompt": prompt,            # The prompt used for generation.
                "generated": generated_text, # The text generated by the model.
            })

            # Log samples as they are generated
            logger.info(f"Sample {i+1}:")                  # Logs the sample number.
            logger.info(f"   Prompt: {prompt}")            # Logs the prompt.
            logger.info(f"   Generated: {generated_text}") # Logs the generated text.

        return {                                       # Returns a dictionary containing only numerical metrics.
            "num_generated_samples": len(generations), # The number of samples successfully generated.
        }

    def _get_sample_prompts(self) -> list[str]: # Defines a private method to obtain sample prompts for text generation. It returns a list of strings.
        """
        Get sample prompts for generation.

        Purpose:
            Retrieves a list of prompts to be used for text generation. It first
            attempts to get prompts directly from the dataloader's dataset
            (if the dataset has a `get_sample_texts` method). If that fails
            or is not available, it falls back to a predefined list of default prompts.

            Having relevant prompts is essential for meaningful text generation.
            This method ensures that the text generation uses either domain-specific
            prompts from the dataset or sensible general-purpose prompts, improving
            the quality of the qualitative evaluation.

        LLM Pipeline Fit:
            This supports the `_generate_samples` functionality by providing the
            starting points for text generation. It ensures flexibility in prompt
            selection, allowing the evaluation to be tailored to the specific
            dataset being used in the LLM pipeline.

        Inputs:
            - `self`: The instance of the `Evaluator` class.

        Outputs:
            - `list[str]`: A list of strings, each representing a prompt for text generation.

        Confirms/Verifies:
            - The ability to dynamically retrieve prompts from a compatible dataset.
            - The fallback mechanism to default prompts if dataset-specific prompts are unavailable.
            - The availability of prompts for text generation.
        """
        # Default prompts
        default_prompts = [                           # Defines a list of default prompts.
            "The future of artificial intelligence",  # A common topic.
            "Once upon a time",                       # A classic story starter.
            "In a world where",                       # Another story starter.
            "The most important thing about",         # A philosophical prompt.
            "Scientists have discovered",             # A news-like prompt.
            "Breaking news:",                         # Another news-like prompt.
            "The secret to happiness is",             # A reflective prompt.
            "Technology has changed",                 # A societal impact prompt.
            "The best way to learn",                  # An educational prompt.
            "In conclusion,",                         # A concluding remark prompt.
        ]

        # Try to get prompts from dataset
        if hasattr(self.dataloader.dataset, "get_sample_texts") and \
            callable(getattr(self.dataloader.dataset, "get_sample_texts")): # Checks if the dataloader's dataset object has an attribute named "get_sample_texts" and if it is callable.
            try: # Attempts to execute the code within the try block.
                 # Assuming get_sample_texts returns raw strings
                return self.dataloader.dataset.get_sample_texts( # Calls the `get_sample_texts` method on the dataset.
                    self.config.num_generate_samples             # Passes the number of samples to generate as an argument.
                )
            except Exception as e: # Catches any exception that occurs during the try block.
                logger.warning(f"Could not get sample texts from dataset: {e}. Using default prompts.") # Logs a warning message if an exception occurs.
                return default_prompts # Returns the default prompts as a fallback.

        return default_prompts # If the dataset does not have the `get_sample_texts` method or it's not callable, returns the default prompts.

    @torch.no_grad() # Decorator to disable gradient calculations, optimizing memory and speed for this method.
    def compute_model_metrics(self) -> dict[str, Any]: # Defines a public method to compute model-specific metrics. It returns a dictionary.
        """
        Compute model-specific metrics.

        Purpose:
            Calculates various static metrics related to the model's architecture,
            such as the total number of parameters, trainable parameters,
            estimated model size in MB, and estimated FLOPs (if the model
            provides an `estimate_flops` method).

            These metrics provide insights into the model's complexity and resource
            requirements. Parameter count and model size are important for deploying
            models, while FLOPs give an estimate of computational cost during inference.

        LLM Pipeline Fit:
            These metrics are valuable for high-level analysis and comparison of
            different model architectures within the LLM pipeline. They help
            determine the computational footprint and scalability of the models,
            informing decisions about model selection and optimization.

        Inputs:
            - `self`: The instance of the `Evaluator` class.

        Outputs:
            - `dict[str, Any]`: A dictionary containing:
                - 'total_parameters' (int): Total number of parameters in the model.
                - 'trainable_parameters' (int): Number of parameters that are trainable.
                - 'model_size_mb' (float): Estimated model size in megabytes (assuming float32).
                - 'estimated_flops' (float, optional): Estimated floating-point operations,
                  if the model supports it.

        Confirms/Verifies:
            - Correct calculation of model parameter counts.
            - Reasonable estimation of model memory footprint.
            - Integration with model-specific FLOPs estimation if available.
        """
        metrics = {} # Initializes an empty dictionary to store model-specific metrics.

        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())                        # Calculates the total number of parameters in the model by summing the number of elements in each parameter tensor.
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) # Calculates the number of trainable parameters by summing elements only for parameters that require gradients.

        metrics["total_parameters"] = total_params                  # Stores the total parameter count.
        metrics["trainable_parameters"] = trainable_params          # Stores the trainable parameter count.
        metrics["model_size_mb"] = total_params * 4 / (1024 * 1024) # Calculates and stores the estimated model size in megabytes, assuming each parameter is a 4-byte float (float32).

        # Compute FLOPs estimate
        if hasattr(self.model, "estimate_flops"):                    # Checks if the model object has an `estimate_flops` method.
            metrics["estimated_flops"] = self.model.estimate_flops() # Calls the `estimate_flops` method on the model and stores its result.

        return metrics # Returns the dictionary of model-specific metrics.

    def benchmark_inference_speed(self, num_samples: int = 100) -> dict[str, float]: # Defines a method to benchmark inference speed. It takes the number of samples and returns a dictionary.
        """
        Benchmark model inference speed.

        Purpose:
            Measures the forward pass inference speed of the model by running
            a fixed number of dummy inputs through it. It reports the average
            time per batch and the throughput in tokens per second.

            Inference speed is a critical performance metric, especially for
            deploying LLMs in real-time applications. Benchmarking helps
            identify bottlenecks and assess the efficiency of the model
            on a given hardware device.

        LLM Pipeline Fit:
            This method is used at the later stages of the LLM pipeline,
            particularly when evaluating models for deployment. It provides
            concrete numbers on how quickly the model can process input,
            which is vital for resource planning and user experience.

        Inputs:
            - `self`: The instance of the `Evaluator` class.
            - `num_samples` (int): The number of dummy inference runs to perform
              for benchmarking. Defaults to 100.

        Outputs:
            - `dict[str, float]`: A dictionary containing:
                - 'inference_time_per_batch_ms' (float): Average inference time per batch in milliseconds.
                - 'inference_throughput_tokens_per_second' (float): Inferred tokens per second.
                - 'total_benchmark_time_seconds' (float): Total time taken for the benchmark.

        Confirms/Verifies:
            - The model's ability to perform forward passes efficiently.
            - The consistency of inference timing on the chosen device.
            - The calculation of meaningful throughput metrics.
        """
        import time # Imports the `time` module for measuring execution time.

        self.model.eval() # Sets the model to evaluation mode to ensure consistent timing (disables dropout, etc.).

        # Prepare dummy input
        batch_size = self.config.batch_size or 1 # Gets the batch size from the config, defaulting to 1 if not set.
        seq_length = 128                         # Defines a fixed sequence length for the dummy input.
        dummy_input = torch.randint(             # Generates a tensor of random integers to serve as dummy input IDs.
            0, self.model.config.vocab_size,     # Random integers between 0 and the vocabulary size (exclusive).
            (batch_size, seq_length),            # Shape of the tensor (batch_size, sequence_length).
            device=self.device,                  # Places the dummy input on the selected device.
        )

        # Warmup
        for _ in range(10):                       # Runs 10 warm-up iterations to ensure that GPU operations are fully initialized and cached.
            _ = self.model(input_ids=dummy_input) # Performs a forward pass with the dummy input; the output is discarded.

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None # Ensures all pending CUDA operations complete before starting the timer, if CUDA is available.
        start_time = time.time()                                        # Records the starting time of the benchmark.

        for _ in range(num_samples):              # Runs the specified number of inference samples.
            _ = self.model(input_ids=dummy_input) # Performs a forward pass; output is discarded.

        torch.cuda.synchronize() if torch.cuda.is_available() else None # Ensures all pending CUDA operations complete before ending the timer.
        end_time = time.time()                                          # Records the ending time of the benchmark.

        # Compute metrics
        total_time = end_time - start_time                # Calculates the total time taken for all benchmark samples.
        avg_time = total_time / num_samples               # Calculates the average time per sample (batch).
        throughput = (batch_size * seq_length) / avg_time # Calculates the throughput in tokens per second: (tokens per batch) / (time per batch).

        return {                                                  # Returns a dictionary of benchmarking metrics.
            "inference_time_per_batch_ms": avg_time * 1000,       # Average inference time per batch in milliseconds.
            "inference_throughput_tokens_per_second": throughput, # Throughput in tokens per second.
            "total_benchmark_time_seconds": total_time,           # Total time spent on the benchmark.
        }