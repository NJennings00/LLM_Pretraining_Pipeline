# filename: src/llm_pipeline/evaluation/metrics.py
"""
Evaluation metrics for language models.

This file provides a collection of functions and a class designed for quantitative
and qualitative evaluation of Large Language Models (LLMs). It includes implementations
for standard natural language processing (NLP) metrics such as perplexity and accuracy,
and more advanced generation metrics like BLEU and ROUGE, as well as model calibration
metrics. The `EvaluationMetrics` class acts as a central container to aggregate,
store, and summarize these metrics over time.

In an LLM pretraining pipeline, these metrics are essential for:
1. **Monitoring Training Progress**: Perplexity and accuracy directly reflect
   how well the model is learning to predict the next token, indicating
   convergence or issues like overfitting/underfitting.
2. **Assessing Generation Quality**: BLEU and ROUGE scores, alongside diversity
   metrics, provide quantitative ways to evaluate the coherence, relevance,
   and originality of generated text.
3. **Understanding Model Reliability**: Calibration metrics help in assessing
   whether the model's predicted probabilities truly reflect the likelihood of
   correct predictions, which is crucial for safety and trustworthiness.
4. **Overall Model Comparison**: By providing a standardized set of metrics,
   this module enables fair comparison between different model architectures,
   training strategies, or hyperparameter settings.
"""

import logging                                              # Imports the logging module for emitting log messages.
from typing import Dict, List, Optional, Any, Union, Tuple  # Imports specific type hints for better code readability and maintainability.
import numpy as np                                          # Imports the NumPy library for numerical operations, especially for mathematical functions and array manipulation.
import torch                                                # Imports the PyTorch library, essential for tensor operations and deep learning models.
from collections import defaultdict                         # Imports defaultdict from the collections module, useful for creating dictionaries with default values for missing keys.


logger = logging.getLogger(__name__) # Initializes a logger for this module, enabling logging of events and warnings.


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.

    Purpose:
        Calculates the perplexity score from a given cross-entropy loss value.
        Perplexity is a common metric in language modeling that measures how
        well a probability distribution predicts a sample. Lower perplexity
        indicates a better model.

        Perplexity is a fundamental metric for evaluating language models. It
        provides a direct measure of the model's uncertainty in predicting the
        next word/token, making it invaluable for tracking training progress
        and comparing different models.

    LLM Pipeline Fit:
        This function is used within the evaluation phase of an LLM pretraining
        pipeline (e.g., by the `Evaluator` class). After computing the cross-entropy
        loss on the validation set, perplexity is derived to give an intuitive
        understanding of the model's performance.

    Inputs:
        - `loss` (float): The cross-entropy loss value, typically averaged over
          tokens or batches.

    Outputs:
        - `float`: The calculated perplexity value. Returns `float("inf")` if
          the exponentiation results in an OverflowError (e.g., for very high loss).

    Confirms/Verifies:
        - The correct transformation of negative log-likelihood (loss) into perplexity.
        - Robustness to very high loss values that might cause an overflow.
    """
    try:                            # Starts a try block to handle potential errors.
        return float(np.exp(loss))  # Computes perplexity as e to the power of the loss. Converts to float explicitly.
    except OverflowError:           # Catches an OverflowError, which occurs if exp(loss) is too large to represent.
        return float("inf")         # Returns positive infinity to indicate an extremely high (bad) perplexity.


def compute_accuracy(
    predictions: torch.Tensor,   # Type hint: a PyTorch tensor containing predicted token IDs.
    labels: torch.Tensor,        # Type hint: a PyTorch tensor containing true token IDs (labels).
    ignore_index: int = -100,    # Type hint: an integer representing the index to ignore in the labels (e.g., padding tokens). Defaults to -100.
) -> float:                      # Type hint: the function returns a float.
    """
    Compute token-level accuracy.

    Purpose:
        Calculates the token-level prediction accuracy by comparing predicted
        token IDs with true labels, while ignoring specified indices (e.g., padding tokens).

        Accuracy is a straightforward and intuitive metric that complements
        perplexity. It directly measures the proportion of correctly predicted
        individual tokens, offering a different perspective on model performance.

    LLM Pipeline Fit:
        This function is used during the quantitative evaluation stage of the
        LLM pipeline. It's applied to batches of model outputs and corresponding
        labels to get an aggregate measure of how precisely the model predicts
        the subsequent tokens in a sequence.

    Inputs:
        - `predictions` (torch.Tensor): A tensor of predicted token IDs,
          typically of shape `[batch_size, seq_length]`.
        - `labels` (torch.Tensor): A tensor of true token IDs, also typically
          of shape `[batch_size, seq_length]`.
        - `ignore_index` (int): The integer value in `labels` that should be
          ignored during accuracy calculation (e.g., padding tokens). Defaults to -100.

    Outputs:
        - `float`: The calculated token-level accuracy. Returns 0.0 if there
          are no valid tokens to consider (to prevent division by zero).

    Confirms/Verifies:
        - The correct calculation of token-level accuracy, accounting for ignored indices.
        - Robustness to empty sets of valid tokens.
    """
    mask = labels != ignore_index                        # Creates a boolean mask to identify elements in `labels` that are not equal to the `ignore_index`.
    correct = (predictions == labels)[mask].sum().item() # Compares predictions with labels, applies the mask, sums the True values (correct predictions), and converts to a Python int.
    total = mask.sum().item()                            # Sums the True values in the mask to get the total number of valid (non-ignored) tokens, and converts to a Python int.

    return correct / total if total > 0 else 0.0 # Returns the accuracy (correct / total) if total is greater than 0, otherwise returns 0.0.


def compute_bleu_score(
    predictions: List[str], # Type hint: a list of predicted text strings.
    references: List[str],  # Type hint: a list of reference (ground truth) text strings.
    max_n: int = 4,         # Type hint: the maximum n-gram order to consider for BLEU calculation. Defaults to 4.
    smooth: bool = True,    # Type hint: a boolean indicating whether to apply smoothing to BLEU scores. Defaults to True.
) -> Dict[str, float]:      # Type hint: the function returns a dictionary with string keys and float values.
    """
    Compute BLEU score.

    Purpose:
        Calculates the Bilingual Evaluation Understudy (BLEU) score, a metric
        for evaluating the quality of text generated by a machine translation
        system, or, in this context, any text generation system, against
        human-produced reference translations. It considers the precision of
        n-grams between the generated text and the reference.

        BLEU is a widely recognized metric for evaluating the quality of generated
        text, especially for tasks like summarization and machine translation,
        where a "reference" output exists. It provides an objective, quantitative
        measure of similarity to reference texts.

    LLM Pipeline Fit:
        This function can be incorporated into the LLM pipeline for evaluating
        the quality of the model's text generation capabilities when there are
        known reference texts. This is common in fine-tuning scenarios or specific
        downstream tasks where a gold-standard output is available.

    Inputs:
        - `predictions` (List[str]): A list of predicted text strings.
        - `references` (List[str]): A list of corresponding reference text strings.
        - `max_n` (int): The maximum n-gram order to compute BLEU for (e.g., 4 for BLEU-4).
        - `smooth` (bool): Whether to apply smoothing to avoid zero precision for
          missing n-grams, which is common for short sentences.

    Outputs:
        - `Dict[str, float]`: A dictionary containing BLEU scores for different
          n-gram orders (e.g., 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4'). Returns
          an empty dictionary if NLTK is not installed.

    Confirms/Verifies:
        - The correct computation of BLEU scores for text generation.
        - Handling of missing NLTK library (graceful fallback).
        - Ability to compute BLEU for various n-gram orders and apply smoothing.
    """
    try:                                                                       # Starts a try block to import NLTK modules.
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # Imports `sentence_bleu` for calculating BLEU and `SmoothingFunction` for smoothing.
        import nltk                                                            # Imports the main NLTK library.
        nltk.download("punkt", quiet=True)                                     # Downloads the 'punkt' tokenizer models from NLTK, usually needed for text processing, quietly.
    except ImportError:                                                        # Catches an ImportError if NLTK is not installed.
        logger.warning("NLTK not installed, BLEU computation skipped")         # Logs a warning indicating NLTK is missing.
        return {}                                                              # Returns an empty dictionary if NLTK is not available.

    smoothing = SmoothingFunction().method1 if smooth else None # Initializes a smoothing function (method1) if `smooth` is True, otherwise sets it to None.

    bleu_scores = defaultdict(list) # Initializes a defaultdict to store BLEU scores for different n-gram orders, where each value is a list.

    for pred, ref in zip(predictions, references): # Iterates through corresponding predicted and reference texts.
        # Tokenize
        pred_tokens = pred.split() # Splits the predicted text into a list of words (simple whitespace tokenization).
        ref_tokens = ref.split()   # Splits the reference text into a list of words.

        # Compute BLEU for different n-gram orders
        for n in range(1, min(max_n + 1, len(pred_tokens) + 1)): # Iterates from 1 up to `max_n`, but not exceeding the length of the predicted tokens.
            weights = [1/n] * n + [0] * (4 - n)                  # Creates a list of weights for the n-grams. For BLEU-n, the first 'n' weights are 1/n, and the rest are 0.
            score = sentence_bleu(                               # Calculates the BLEU score for the current sentence pair.
                [ref_tokens],                                    # Reference should be a list of lists of tokens (even if only one reference).
                pred_tokens,                                     # Predicted tokens.
                weights=weights,                                 # N-gram weights.
                smoothing_function=smoothing,                    # The smoothing function to apply.
            )
            bleu_scores[f"bleu_{n}"].append(score)                # Appends the calculated score to the list for the corresponding BLEU-n metric.

    # Average scores
    return {                                   # Returns a dictionary of averaged BLEU scores.
        key: float(np.mean(scores))            # Computes the mean of the scores for each n-gram order and converts it to float.
        for key, scores in bleu_scores.items() # Iterates through the stored BLEU scores.
    }


def compute_rouge_scores(
    predictions: List[str], # Type hint: a list of predicted text strings.
    references: List[str],  # Type hint: a list of reference text strings.
) -> Dict[str, float]:      # Type hint: the function returns a dictionary with string keys and float values.
    """
    Compute ROUGE scores.

    Purpose:
        Calculates ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
        scores, a set of metrics used for evaluating summarization and machine
        translation software. ROUGE measures the overlap of n-grams, word
        sequences, and word pairs between the generated text and the reference.

        ROUGE is particularly important for summarization tasks, where recall
        (how much of the reference is covered by the prediction) is crucial.
        It complements BLEU by focusing on recall, providing a more balanced
        evaluation for tasks where a subset of information is expected.

    LLM Pipeline Fit:
        Similar to BLEU, ROUGE scores are valuable in the LLM pipeline when
        evaluating models fine-tuned for tasks like text summarization,
        where the generated output is expected to capture key information
        from a source text, which is reflected in a reference summary.

    Inputs:
        - `predictions` (List[str]): A list of predicted text strings.
        - `references` (List[str]): A list of corresponding reference text strings.

    Outputs:
        - `Dict[str, float]`: A dictionary containing various ROUGE scores
          (e.g., 'rouge-1_f', 'rouge-2_p', 'rouge-l_r'). Returns an empty
          dictionary if the `rouge-score` library is not installed.

    Confirms/Verifies:
        - The correct computation of ROUGE scores.
        - Handling of missing `rouge-score` library.
        - The structure of the returned scores (flattened for easy access).
    """
    try:                                                                       # Starts a try block to import the `Rouge` class.
        from rouge import Rouge                                                # Imports the Rouge class from the `rouge` library (often `rouge-score`).
    except ImportError:                                                        # Catches an ImportError if the `rouge-score` library is not installed.
        logger.warning("rouge-score not installed, ROUGE computation skipped") # Logs a warning indicating the library is missing.
        return {}                                                              # Returns an empty dictionary if the library is not available.

    rouge = Rouge() # Creates an instance of the Rouge evaluator.

    # Compute scores
    scores = rouge.get_scores(predictions, references, avg=True) # Computes ROUGE scores for all prediction-reference pairs and averages them.

    # Flatten scores
    result = {}                                # Initializes an empty dictionary to store the flattened ROUGE scores.
    for metric, values in scores.items():      # Iterates through the top-level ROUGE metrics (e.g., 'rouge-1', 'rouge-2', 'rouge-l').
        for key, value in values.items():      # Iterates through the sub-metrics for each ROUGE metric (e.g., 'f', 'p', 'r' for f-score, precision, recall).
            result[f"{metric}_{key}"] = value  # Stores the score in the result dictionary with a flattened key (e.g., 'rouge-1_f').

    return result # Returns the dictionary of flattened ROUGE scores.


class EvaluationMetrics:
    """
    Container for evaluation metrics.

    Purpose:
        This class acts as a central repository for storing and managing
        various evaluation metrics during a model's lifecycle (e.g., during
        training epochs or evaluation runs). It allows for updating current
        metric values, keeping a history of numerical metrics, and generating
        summaries (mean, std, min, max) of the historical data.

        This class is essential for tracking model performance over time.
        It simplifies the process of collecting, viewing, and analyzing metrics,
        which is critical for understanding training dynamics, debugging,
        and making informed decisions about model development.

    LLM Pipeline Fit:
        In an LLM pretraining pipeline, an instance of `EvaluationMetrics`
        is typically used by the `Evaluator` or `Trainer` classes. It
        accumulates metrics reported after each evaluation step (e.g., per epoch)
        and provides methods to query the current state, historical trends,
        and summary statistics, which are then often logged or visualized.

    Inputs:
        - `self`: The instance of the `EvaluationMetrics` class. (No explicit
          arguments in `__init__`).

    Outputs:
        The class itself does not return a value upon initialization. Its
        methods return dictionaries of metrics or historical data.

    Confirms/Verifies:
        - Correct storage and retrieval of current metric values.
        - Accurate maintenance of historical data for numerical metrics.
        - Proper calculation of summary statistics (mean, std, min, max).
        - Functionality for logging and serializing/deserializing metrics.
    """

    def __init__(self):                  # Defines the constructor for the EvaluationMetrics class.
        self.metrics = {}                # Initializes an empty dictionary to store the most recent (current) values of various metrics.
        self.history = defaultdict(list) # Initializes a defaultdict where keys are metric names and values are lists, used to store historical values of numerical metrics.

    def update(self, metrics: Dict[str, Any]): # Defines the `update` method, which takes a dictionary of metrics.
        """
        Update metrics.

        Purpose:
            Updates the current `metrics` dictionary with new values and
            appends numerical metric values to their respective lists in the `history`.

            This method is the primary way to feed new metric results into the
            `EvaluationMetrics` container. It ensures that the current state
            is up-to-date and that a running history of numerical performance
            is maintained.

        LLM Pipeline Fit:
            Called by the `Evaluator` (or `Trainer`) after each evaluation run
            or training step to log the latest performance figures. It's the
            mechanism by which the system's performance over time is recorded.

        Inputs:
            - `metrics` (Dict[str, Any]): A dictionary where keys are metric names
              (e.g., "loss", "perplexity") and values are their corresponding
              (current) measurements.

        Outputs:
            - None. The method modifies the internal state of the `EvaluationMetrics`
              instance.

        Confirms/Verifies:
            - Correct assignment of current metric values.
            - Proper appending of numerical values to their history lists.
            - Discrimination between numerical and non-numerical metrics for history tracking.
        """
        for key, value in metrics.items():      # Iterates through each key-value pair in the input `metrics` dictionary.
            self.metrics[key] = value           # Updates the current metric value in `self.metrics`.
            if isinstance(value, (int, float)): # Checks if the value is an integer or a float (i.e., a numerical metric).
                self.history[key].append(value) # If it's numerical, appends the value to its corresponding list in `self.history`.

    def get_metrics(self) -> Dict[str, Any]: # Defines the `get_metrics` method, which returns a dictionary.
        """
        Get current metrics.

        Purpose:
            Returns a copy of the most recently updated metrics.

            This method allows other parts of the system (e.g., logging modules,
            reporting functions) to access the current performance snapshot.

        LLM Pipeline Fit:
            Used when only the immediate performance is needed for logging or
            display after a single evaluation step in the LLM pipeline.

        Inputs:
            - `self`: The instance of the `EvaluationMetrics` class.

        Outputs:
            - `Dict[str, Any]`: A dictionary containing the most current values
              of all tracked metrics. A copy is returned to prevent external modification.

        Confirms/Verifies:
            - Accurate retrieval of the current metric state.
            - That a copy of the dictionary is returned, ensuring data encapsulation.
        """
        return self.metrics.copy() # Returns a shallow copy of the `self.metrics` dictionary.

    def get_history(self) -> Dict[str, List[float]]: # Defines the `get_history` method, which returns a dictionary of lists of floats.
        """
        Get metrics history.

        Purpose:
            Returns the complete historical list of numerical metric values.

            This is used when detailed historical trends are required, for example,
            to plot performance curves over training epochs.

        LLM Pipeline Fit:
            Provides the raw data for visualizing training/evaluation curves
            over time in the LLM pipeline, which is crucial for deep analysis
            of model behavior.

        Inputs:
            - `self`: The instance of the `EvaluationMetrics` class.

        Outputs:
            - `Dict[str, List[float]]`: A dictionary where keys are metric names
              and values are lists of their historical numerical values.

        Confirms/Verifies:
            - Accurate retrieval of all historical numerical metric data.
        """
        return dict(self.history) # Returns a regular dictionary conversion of the defaultdict `self.history`.

    def get_summary(self) -> Dict[str, Any]: # Defines the `get_summary` method, which returns a dictionary.
        """
        Get metrics summary.

        Purpose:
            Provides a summary of all metrics, including the current values
            and statistical aggregates (mean, standard deviation, min, max)
            for numerical metrics that have more than one historical entry.

            This method offers a condensed view of performance, useful for
            high-level reporting and comparing overall stability or range
            of a metric over multiple evaluations.

        LLM Pipeline Fit:
            Useful for generating final reports or summary logs at the end of
            an entire training run or major evaluation phase in the LLM pipeline,
            providing a concise overview of the model's performance.

        Inputs:
            - `self`: The instance of the `EvaluationMetrics` class.

        Outputs:
            - `Dict[str, Any]`: A dictionary combining current metric values with
              statistical summaries (mean, std, min, max) for historical numerical metrics.

        Confirms/Verifies:
            - Correct aggregation of current metrics and statistical summaries.
            - Accurate calculation of mean, standard deviation, minimum, and maximum for historical data.
            - Handling of metrics with insufficient history for statistics.
        """
        summary = self.metrics.copy() # Starts with a copy of the current metrics.

        # Add statistics for numeric metrics
        for key, values in self.history.items():                # Iterates through each numerical metric and its historical values.
            if len(values) > 1:                                 # Checks if there is more than one historical value to compute statistics.
                summary[f"{key}_mean"] = float(np.mean(values)) # Calculates and stores the mean of the historical values.
                summary[f"{key}_std"] = float(np.std(values))   # Calculates and stores the standard deviation of the historical values.
                summary[f"{key}_min"] = float(np.min(values))   # Calculates and stores the minimum of the historical values.
                summary[f"{key}_max"] = float(np.max(values))   # Calculates and stores the maximum of the historical values.

        return summary # Returns the dictionary containing current metrics and historical summaries.

    def log_metrics(self, prefix: str = ""): # Defines the `log_metrics` method, which takes an optional string prefix.
        """
        Log metrics.

        Purpose:
            Logs the current numerical metric values using the configured logger.

            This method provides a convenient way to output performance metrics
            to the console or a log file, making it easy to monitor progress
            during long-running training or evaluation processes.

        LLM Pipeline Fit:
            Integral to the logging and monitoring aspects of the LLM pipeline,
            providing real-time or periodic updates on the model's performance.

        Inputs:
            - `prefix` (str): A string prefix to prepend to each logged metric
              name (e.g., "eval_"). Defaults to an empty string.

        Outputs:
            - None. The primary effect is printing messages to the logger.

        Confirms/Verifies:
            - The correct formatting and output of numerical metrics to the logger.
            - The application of the specified prefix.
        """
        for key, value in self.metrics.items():            # Iterates through the current metrics.
            if isinstance(value, (int, float)):            # Checks if the metric value is numerical.
                logger.info(f"{prefix}{key}: {value:.4f}") # Logs the metric name (with prefix) and its value, formatted to 4 decimal places.

    def to_dict(self) -> Dict[str, Any]: # Defines the `to_dict` method, which returns a dictionary.
        """
        Convert to dictionary.

        Purpose:
            Serializes the current state of the `EvaluationMetrics` instance
            (current metrics and history) into a dictionary format.

            This is useful for saving the evaluation state (e.g., to a file)
            or passing it between different components of a system, enabling
            resumption or later analysis.

        LLM Pipeline Fit:
            Supports persistence in the LLM pipeline, allowing metric tracking
            to be saved and loaded, which is important for long-running experiments
            or when distributing evaluation results.

        Inputs:
            - `self`: The instance of the `EvaluationMetrics` class.

        Outputs:
            - `Dict[str, Any]`: A dictionary containing two keys:
                - 'metrics': The current metrics dictionary.
                - 'history': The historical metrics dictionary (converted from defaultdict to dict).

        Confirms/Verifies:
            - Accurate serialization of current and historical metrics.
            - Correct conversion of `defaultdict` to `dict`.
        """
        return {                           # Returns a dictionary.
            "metrics": self.metrics,       # Includes the current metrics dictionary.
            "history": dict(self.history), # Includes the historical metrics dictionary, converted to a regular dictionary.
        }

    def from_dict(self, data: Dict[str, Any]): # Defines the `from_dict` method, which takes a dictionary as input.
        """
        Load from dictionary.

        Purpose:
            Deserializes the state of an `EvaluationMetrics` instance from a
            dictionary, typically one created by `to_dict()`.

            This method allows loading previously saved evaluation states,
            which is crucial for resuming analysis, loading checkpoints,
            or re-evaluating results without recomputing everything.

        LLM Pipeline Fit:
            Enables checkpointing and loading of evaluation results in the
            LLM pipeline, facilitating reproducible research and efficient
            workflow management.

        Inputs:
            - `data` (Dict[str, Any]): A dictionary containing 'metrics' and
              'history' keys, representing the serialized state.

        Outputs:
            - None. The method modifies the internal state of the `EvaluationMetrics`
              instance.

        Confirms/Verifies:
            - Accurate deserialization and restoration of the metrics state.
            - Proper conversion back to `defaultdict` for the history.
            - Robustness to missing 'metrics' or 'history' keys in the input data.
        """
        self.metrics = data.get("metrics", {})                    # Sets `self.metrics` from the 'metrics' key in `data`, defaulting to an empty dict if not found.
        self.history = defaultdict(list, data.get("history", {})) # Sets `self.history` from the 'history' key, re-initializing it as a defaultdict with the loaded data.


def compute_generation_metrics(
    generations: List[Dict[str, str]], # Type hint: a list of dictionaries, where each dict has "prompt" and "generated" keys.
    tokenizer: Any,                    # Type hint: an object that has an `encode` method (e.g., TokenizerWrapper).
) -> Dict[str, float]:                 # Type hint: the function returns a dictionary with string keys and float values.
    """
    Compute metrics for generated texts.

    Purpose:
        Calculates various metrics specifically related to the characteristics
        of generated text, such as average prompt and generated text lengths
        (in tokens) and n-gram diversity ratios. This provides insights into
        the structural and linguistic properties of the model's output.

        These metrics are important for qualitative assessment of text generation.
        Length statistics help understand if the model generates outputs of
        expected lengths, and diversity metrics (unique n-gram ratio) help
        detect issues like repetition or lack of creativity.

    LLM Pipeline Fit:
        This function is part of the comprehensive evaluation suite in the LLM
        pipeline. It's called after text samples have been generated to
        quantify aspects of their quality that go beyond simple fluency,
        helping to refine generation strategies and model training.

    Inputs:
        - `generations` (List[Dict[str, str]]): A list of dictionaries, where
          each dictionary contains at least "prompt" and "generated" text strings.
        - `tokenizer` (Any): An object (e.g., `TokenizerWrapper`) that has an
          `encode(text: str)` method which returns a list of token IDs.

    Outputs:
        - `Dict[str, float]`: A dictionary containing:
            - 'avg_prompt_length': Average length of prompts in tokens.
            - 'avg_generated_length': Average length of generated texts in tokens.
            - 'avg_total_length': Average total length (prompt + generated) in tokens.
            - 'unique_1gram_ratio', 'unique_2gram_ratio', 'unique_3gram_ratio':
              Ratios of unique n-grams in generated texts (if enough generations exist).

    Confirms/Verifies:
        - Accurate calculation of average lengths based on tokenization.
        - Correct computation of n-gram diversity metrics for generated text.
        - Robustness to empty generation lists or insufficient data for n-gram calculation.
    """
    metrics = {} # Initializes an empty dictionary to store generation-specific metrics.

    # Length statistics
    prompt_lengths = []    # Initializes a list to store prompt lengths.
    generated_lengths = [] # Initializes a list to store generated text lengths.
    total_lengths = []     # Initializes a list to store total (prompt + generated) lengths.

    for gen in generations:                                   # Iterates through each generation dictionary.
        prompt_tokens = tokenizer.encode(gen["prompt"])       # Encodes the prompt text into token IDs using the provided tokenizer.
        generated_tokens = tokenizer.encode(gen["generated"]) # Encodes the generated text into token IDs.

        prompt_lengths.append(len(prompt_tokens))                        # Appends the length of prompt tokens to the list.
        generated_lengths.append(len(generated_tokens))                  # Appends the length of generated tokens to the list.
        total_lengths.append(len(prompt_tokens) + len(generated_tokens)) # Appends the combined length to the list.

    metrics["avg_prompt_length"] = float(np.mean(prompt_lengths))       # Calculates and stores the average prompt length.
    metrics["avg_generated_length"] = float(np.mean(generated_lengths)) # Calculates and stores the average generated text length.
    metrics["avg_total_length"] = float(np.mean(total_lengths))         # Calculates and stores the average total length.

    # Diversity metrics
    if len(generations) > 1: # Checks if there is more than one generation to compute diversity.
        # Unique n-grams
        for n in [1, 2, 3]:                             # Iterates for 1-gram, 2-gram, and 3-gram diversity.
            all_ngrams = []                             # Initializes a list to collect all n-grams across all generated texts.
            for gen in generations:                     # Iterates through each generated text.
                tokens = gen["generated"].split()       # Splits the generated text into words (simple tokenization for n-grams).
                ngrams = [                              # Generates n-grams for the current text.
                    tuple(tokens[i:i+n])                # Creates a tuple of n tokens.
                    for i in range(len(tokens) - n + 1) # Iterates to create n-grams of length `n`.
                ]
                all_ngrams.extend(ngrams) # Adds the generated n-grams to the overall list.

            if all_ngrams:                                            # Checks if any n-grams were found.
                unique_ratio = len(set(all_ngrams)) / len(all_ngrams) # Calculates the ratio of unique n-grams to total n-grams.
                metrics[f"unique_{n}gram_ratio"] = unique_ratio       # Stores the unique n-gram ratio in the metrics dictionary.

    return metrics # Returns the dictionary of generation-specific metrics.


def compute_calibration_metrics(
    logits: torch.Tensor, # Type hint: a PyTorch tensor of model logits.
    labels: torch.Tensor, # Type hint: a PyTorch tensor of true labels.
    num_bins: int = 10,   # Type hint: an integer specifying the number of bins for calibration calculation. Defaults to 10.
) -> Dict[str, float]:    # Type hint: the function returns a dictionary with string keys and float values.
    """
    Compute calibration metrics.

    Purpose:
        Evaluates the calibration of a language model's predictions. Calibration
        refers to how well the predicted probabilities align with the true
        correctness likelihood. For example, if a model predicts with 70%
        confidence, it should be correct 70% of the time. This function
        computes the Expected Calibration Error (ECE) and mean confidence.

        Calibration is vital for trustworthy AI. A well-calibrated model's
        confidence can be trusted, which is crucial for applications requiring
        reliability, safety, or decision-making based on model outputs (e.g., in medical or financial domains).

    LLM Pipeline Fit:
        This function can be integrated into the evaluation phase of the LLM
        pipeline, particularly for models that will be used in high-stakes
        applications. It provides a measure beyond simple accuracy to
        understand the model's reliability and its ability to quantify
        its own uncertainty.

    Inputs:
        - `logits` (torch.Tensor): Model's raw output logits, typically of
          shape `[batch_size, seq_length, vocab_size]`.
        - `labels` (torch.Tensor): True labels (token IDs), typically of
          shape `[batch_size, seq_length]`.
        - `num_bins` (int): The number of confidence bins to use for ECE calculation.

    Outputs:
        - `Dict[str, float]`: A dictionary containing:
            - 'expected_calibration_error': The ECE value.
            - 'mean_confidence': The average confidence the model assigned to
              the true labels.

    Confirms/Verifies:
        - The correct calculation of Expected Calibration Error (ECE).
        - The computation of mean confidence for true labels.
        - The model's ability to provide well-calibrated probability estimates.
    """
    # Get probabilities
    probs = torch.softmax(logits, dim=-1) # Applies softmax to the logits along the last dimension (vocabulary) to convert them into probabilities.

    # Get predicted probabilities for true labels
    batch_size, seq_length, _ = logits.shape   # Unpacks the dimensions of the logits tensor.
    true_probs = probs[                        # Selects the probabilities corresponding to the true labels.
        torch.arange(batch_size).unsqueeze(1), # Uses advanced indexing: selects all batch indices.
        torch.arange(seq_length).unsqueeze(0), # Selects all sequence indices.
        labels,                                # Selects the probability for the actual true label at each position.
    ]

    # Flatten
    true_probs = true_probs.view(-1)            # Flattens the `true_probs` tensor into a 1D array.
    labels_binary = torch.ones_like(true_probs) # Creates a tensor of ones with the same shape as `true_probs`. In the context of ECE, we assume `true_probs` corresponds to the correct class, so the accuracy for these selected probabilities is always 1.

    # Compute ECE (Expected Calibration Error)
    ece = expected_calibration_error( # Calls the helper function `expected_calibration_error`.
        true_probs.cpu().numpy(),     # Converts `true_probs` to a NumPy array on the CPU.
        labels_binary.cpu().numpy(),  # Converts `labels_binary` to a NumPy array on the CPU.
        num_bins=num_bins,            # Passes the number of bins.
    )

    return {                                         # Returns a dictionary of calibration metrics.
        "expected_calibration_error": ece,           # The calculated ECE.
        "mean_confidence": float(true_probs.mean()), # The mean of the probabilities assigned to the true labels.
    }


def expected_calibration_error(
    confidences: np.ndarray, # Type hint: a NumPy array of predicted confidences.
    accuracies: np.ndarray,  # Type hint: a NumPy array of binary accuracy values (0 or 1).
    num_bins: int = 10,      # Type hint: an integer specifying the number of bins for the calibration plot. Defaults to 10.
) -> float:                  # Type hint: the function returns a float.
    """
    Compute Expected Calibration Error (ECE).

    Purpose:
        Calculates the Expected Calibration Error (ECE) given arrays of
        predicted confidences and corresponding binary accuracy values (0 or 1).
        ECE quantifies the difference between the average confidence and the
        actual accuracy within different confidence bins. A lower ECE indicates
        better calibration.

        ECE is the standard metric for assessing model calibration. It is crucial
        for ensuring that a model's self-reported confidence is reliable, which
        is particularly important in real-world applications where decisions are
        made based on model predictions and their associated uncertainty.

    LLM Pipeline Fit:
        This is a lower-level helper function utilized by `compute_calibration_metrics`
        within the LLM pipeline's evaluation stage. It provides the core logic
        for quantifying how "honest" the model is about its predictions.

    Inputs:
        - `confidences` (np.ndarray): A NumPy array of confidence scores
          (probabilities) for each prediction.
        - `accuracies` (np.ndarray): A NumPy array of binary values (0 or 1),
          where 1 indicates a correct prediction and 0 indicates an incorrect one,
          corresponding to each confidence score.
        - `num_bins` (int): The number of uniform bins to divide the confidence
          interval [0, 1] into.

    Outputs:
        - `float`: The calculated ECE value.

    Confirms/Verifies:
        - The correct partitioning of confidences into bins.
        - Accurate calculation of average confidence and accuracy within each bin.
        - Proper summation of the weighted absolute differences for ECE.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1) # Creates evenly spaced bin boundaries from 0 to 1.
    bin_lowers = bin_boundaries[:-1]                 # Gets the lower bounds of the bins.
    bin_uppers = bin_boundaries[1:]                  # Gets the upper bounds of the bins.

    ece = 0.0 # Initializes ECE to 0.0.
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):            # Iterates through each bin defined by its lower and upper bounds.
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) # Creates a boolean mask for confidences that fall within the current bin.
        prop_in_bin = in_bin.mean()                                     # Calculates the proportion of predictions that fall into this bin.

        if prop_in_bin > 0:                                                   # Proceeds only if there are predictions in the current bin to avoid division by zero.
            accuracy_in_bin = accuracies[in_bin].mean()                       # Calculates the average accuracy of predictions within this bin.
            avg_confidence_in_bin = confidences[in_bin].mean()                # Calculates the average confidence of predictions within this bin.
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin) # Adds the weighted absolute difference between average confidence and accuracy in the bin to ECE.

    return float(ece) # Returns the final ECE value, cast to a float.
