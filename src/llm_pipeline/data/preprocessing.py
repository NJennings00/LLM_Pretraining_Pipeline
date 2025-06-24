# filename: src/llm_pipeline/data/preprocessing.py
"""
This module provides a collection of utility functions and a `PreprocessingPipeline` class
designed for cleaning and normalizing raw text data. These preprocessing steps are crucial
for preparing high-quality input for Language Model (LLM) pretraining.

The `TextPreprocessor` class contains static methods for common text cleaning tasks
like normalizing whitespace, removing control characters, handling Unicode,
converting to lowercase, and removing common unwanted patterns like URLs and emails.
The `PreprocessingPipeline` class allows for composing these individual steps
into a configurable and reusable sequence of operations.

In the LLM pretraining pipeline, **text preprocessing** is a fundamental step that occurs
before tokenization. Its importance cannot be overstated, as raw text often contains noise,
inconsistencies, or irrelevant information (e.g., HTML tags, special characters, redundant spaces).
Clean and consistent input data is vital for:
1.  **Effective Tokenization:** Ensures that the tokenizer creates meaningful tokens without being
    confused by artifacts.
2.  **Improved Model Learning:** Reduces noise in the input, allowing the model to focus on
    learning actual linguistic patterns and semantic relationships.
3.  **Reduced Vocabulary Size:** Normalizing text can lead to a more compact and efficient vocabulary.
4.  **Enhanced Model Robustness:** Makes the model less sensitive to variations in input data
    that are not semantically meaningful.

By providing a robust preprocessing layer, this module ensures that the LLM is trained on
the cleanest and most relevant data possible, which directly contributes to its overall
performance and generalization capabilities.
"""

import re                                           # Imports the regular expression module for pattern-based text manipulation.
import unicodedata                                  # Imports unicodedata for handling and normalizing Unicode strings.
from typing import List, Optional, Callable, Union  # Imports typing hints for better code readability and validation.
import logging                                      # Imports the logging library for structured logging.


logger = logging.getLogger(__name__) # Initializes a logger for this module to log information, warnings, or errors.


class TextPreprocessor:
    """
    A collection of static methods for common text preprocessing operations.
    Each method takes a string and returns a cleaned/modified string.

    Why it's needed: This class centralizes individual text cleaning functions,
    making them easily accessible and reusable. 
    Its role is to provide atomic, well-defined cleaning steps for the pipeline.

    How it fits into the LLM pipeline: These static methods represent the fundamental
    building blocks of text cleanliness. Before any text is tokenized and fed into
    the model, these functions are applied to ensure uniformity, remove noise, and
    prepare the text for optimal tokenization and subsequent model understanding.
    """
    
    @staticmethod # Decorator indicating that this is a static method, not requiring an instance of the class.
    def normalize_whitespace(text: str) -> str:
        """
        Normalizes all types of whitespace characters in a given text string.
        It replaces sequences of one or more whitespace characters (spaces, tabs, newlines, etc.)
        with a single space and removes leading/trailing whitespace.

        Inputs:
        - text (str): The input text string.
        Outputs:
        - str: The text string with normalized whitespace.
        """
        text = re.sub(r'\s+', ' ', text)  # Replaces any sequence of one or more whitespace characters with a single space.
        text = text.strip()               # Removes any whitespace from the beginning and end of the string.
        return text                       # Returns the text with normalized whitespace.
    
    @staticmethod # Decorator indicating a static method.
    def remove_control_characters(text: str) -> str:
        """
        Removes ASCII control characters (non-printable characters) from the text,
        which can often cause display issues or interfere with tokenization.

        Inputs:
        - text (str): The input text string.
        Outputs:
        - str: The text string with control characters removed.
        """
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text) # Uses regex to remove characters in ASCII ranges for control characters (0-31 and 127-159).
        return text # Returns the cleaned text.
    
    @staticmethod # Decorator indicating a static method.
    def normalize_unicode(text: str) -> str:
        """
        Normalizes Unicode characters to their canonical composed form (NFC).
        This helps ensure consistency for characters that can be represented
        in multiple ways (e.g., 'é' vs. 'e' + combining acute accent).

        Inputs:
        - text (str): The input text string.
        Outputs:
        - str: The text string with Unicode characters normalized to NFC form.
        """
        text = unicodedata.normalize('NFC', text) # Applies NFC (Normalization Form C) normalization to the text.
        return text # Returns the normalized text.
    
    @staticmethod # Decorator indicating a static method.
    def lowercase(text: str) -> str:
        """
        Converts all characters in the text to their lowercase equivalent.
        This is a common step to reduce vocabulary size and treat "The" and "the" as the same token.

        Inputs:
        - text (str): The input text string.
        Outputs:
        - str: The text string converted to lowercase.
        """
        return text.lower() # Returns the lowercase version of the text.
    
    @staticmethod # Decorator indicating a static method.
    def remove_urls(text: str) -> str:
        """
        Removes URLs (web addresses) from the text, replacing them with a single space.
        URLs often contain special characters that can complicate tokenization or are irrelevant
        for the core language modeling task.

        Inputs:
        - text (str): The input text string.
        Outputs:
        - str: The text string with URLs replaced by spaces.
        """
        url_pattern = re.compile( # Compiles a regular expression pattern for matching URLs.
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # The regex pattern for URLs.
        )
        text = url_pattern.sub(' ', text) # Substitutes all found URLs with a single space.
        return text # Returns the modified text.
    
    @staticmethod # Decorator indicating a static method.
    def remove_emails(text: str) -> str:
        """
        Removes email addresses from the text, replacing them with a single space.
        Similar to URLs, email addresses are structured patterns that might not contribute
        to general language understanding and can introduce noise.

        Inputs:
        - text (str): The input text string.
        Outputs:
        - str: The text string with email addresses replaced by spaces.
        """
        email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+') # Compiles a regular expression pattern for matching email addresses.
        text = email_pattern.sub(' ', text)                   # Substitutes all found email addresses with a single space.
        return text # Returns the modified text.
    
    @staticmethod # Decorator indicating a static method.
    def remove_extra_punctuation(text: str) -> str:
        """
        Reduces sequences of identical punctuation marks (e.g., "!!!" to "!", "..." to ".")
        to a single instance of that punctuation. This cleans up informal writing or repeated
        punctuation.

        Inputs:
        - text (str): The input text string.
        Outputs:
        - str: The text string with excessive punctuation normalized.
        """
        text = re.sub(r'([.!?]){2,}', r'\1', text) # Uses regex to find 2 or more occurrences of '.', '!', or '?' and replaces them with a single instance.
        return text # Returns the modified text.
    
    @staticmethod # Decorator indicating a static method.
    def fix_encoding_errors(text: str) -> str:
        """
        Corrects common encoding errors (mojibake) that might appear from mismatched
        character encodings during data ingestion. These are often seen when text
        intended to be UTF-8 is read with a different encoding.

        Inputs:
        - text (str): The input text string that might contain encoding errors.
        Outputs:
        - str: The text string with common encoding errors replaced.
        """
        replacements = { # Defines a dictionary of common mojibake patterns and their correct replacements.
            'â€™': "'",  # Replacement for common apostrophe error.
            'â€œ': '"',  # Replacement for common left double quote error.
            'â€': '"',   # Replacement for common right double quote error (often appears with â€œ).
            'â€"': '-',  # Replacement for common dash error.
            'â€"': '--', # Another common dash error replacement (consider order if one subsumes another).
            'Ã©': 'é',   # Replacement for common 'e' with acute accent error.
            'Ã¨': 'è',   # Replacement for common 'e' with grave accent error.
            'Ã ': 'à',   # Replacement for common 'a' with grave accent error.
        }
        for old, new in replacements.items(): # Iterates through each old/new replacement pair.
            text = text.replace(old, new)     # Replaces all occurrences of the old pattern with the new one.
        return text                           # Returns the text with fixed encoding errors.


class PreprocessingPipeline:
    """
    A class that composes multiple text preprocessing functions into a sequential pipeline.
    It allows for flexible customization of preprocessing steps and applies them
    in a defined order to an input text.

    Why it's needed: This class provides a structured and extensible way to apply
    a series of cleaning operations to raw text. Instead of manually calling each
    static method from `TextPreprocessor`, a pipeline can be configured once and
    then reused throughout the data loading process. This ensures consistency
    and simplifies managing preprocessing logic.

    How it fits into the LLM pipeline: This pipeline is typically integrated
    within the dataset loading process (e.g., `WikiTextDataset`), applied to
    raw text documents after they are loaded but before they are tokenized.
    It ensures that all incoming text data undergoes the necessary cleaning
    and normalization, which is a prerequisite for effective tokenization
    and subsequent model training.

    Inputs:
    - steps (Optional[List[Callable[[str], str]]]): An optional list of callable
      functions (like those in `TextPreprocessor`) that each take a string and return a string.
      If None, a default set of essential steps is used.

    Outputs:
    - __call__ (str): Applies all configured preprocessing steps to an input text string.
    - add_step (None): Adds a new preprocessing function to the end of the pipeline.
    - remove_step (None): Removes an existing preprocessing function from the pipeline.
    - create_default (PreprocessingPipeline): A class method to create a pipeline with a standard set of steps.
    """
    
    def __init__(self, steps: Optional[List[Callable[[str], str]]] = None):
        """
        Initializes the PreprocessingPipeline with a list of preprocessing functions.
        If no steps are provided, a default set of fundamental cleaning operations is used.
        """
        # Checks if no custom steps were provided.
        if steps is None: 
                                                            # Default preprocessing steps for basic text cleaning.
            steps = [                                       # Assigns a default list of preprocessing functions.
                TextPreprocessor.fix_encoding_errors,       # Fixes common character encoding issues.
                TextPreprocessor.normalize_unicode,         # Normalizes Unicode characters.
                TextPreprocessor.remove_control_characters, # Removes non-printable control characters.
                TextPreprocessor.normalize_whitespace,      # Consolidates and strips whitespace.
            ]
        self.steps = steps # Stores the list of preprocessing functions.
    
    def add_step(self, step: Callable[[str], str]) -> None:
        """
        Adds a new preprocessing function to the end of the pipeline's sequence of steps.

        Inputs:
        - step (Callable[[str], str]): The function to add, which should take a string and return a string.
        Outputs:
        - None.
        """
        self.steps.append(step) # Appends the new step to the list of steps.
    
    def remove_step(self, step: Callable[[str], str]) -> None:
        """
        Removes a specified preprocessing function from the pipeline.

        Inputs:
        - step (Callable[[str], str]): The function to remove.
        Outputs:
        - None.
        """
        if step in self.steps:      # Checks if the step exists in the pipeline.
            self.steps.remove(step) # Removes the first occurrence of the step.
    
    def __call__(self, text: str) -> str:
        """
        Applies all the configured preprocessing steps sequentially to the input text.
        Each step receives the output of the previous step as its input.
        Includes error handling for individual steps to prevent pipeline failure.

        Inputs:
        - text (str): The raw input text string.
        Outputs:
        - str: The processed text string after all steps have been applied.
        """
        # Iterates through each preprocessing function in the pipeline.
        for step in self.steps: 
            # Begins a try block to catch exceptions during step execution.
            try: 
                text = step(text)  # Applies the current preprocessing step to the text.
            # Catches any exception that occurs during the step.
            except Exception as e: 
                logger.warning(f"Error in preprocessing step {step.__name__}: {e}") # Logs a warning about the error.
                continue # Continues to the next step, skipping the problematic one.
        return text # Returns the final processed text.
    
    @classmethod # Decorator indicating this is a class method.
    def create_default(cls, lowercase: bool = False) -> "PreprocessingPipeline":
        """
        Creates a default preprocessing pipeline with a comprehensive set of common steps.
        Allows an option to include a lowercase conversion step.

        Inputs:
        - lowercase (bool): If True, adds the `TextPreprocessor.lowercase` step to the pipeline.
        Outputs:
        - PreprocessingPipeline: An instance of the `PreprocessingPipeline` configured with default steps.
        """
        steps = [                                       # Defines a standard set of preprocessing steps.
            TextPreprocessor.fix_encoding_errors,       # Fixes common character encoding issues.
            TextPreprocessor.normalize_unicode,         # Normalizes Unicode characters.
            TextPreprocessor.remove_control_characters, # Removes non-printable control characters.
            TextPreprocessor.remove_urls,               # Removes URLs.
            TextPreprocessor.remove_emails,             # Removes email addresses.
            TextPreprocessor.normalize_whitespace,      # Normalizes whitespace.
            TextPreprocessor.remove_extra_punctuation,  # Removes excessive punctuation.
        ]
        
        if lowercase: # Checks if the lowercase option is enabled.
            steps.append(TextPreprocessor.lowercase) # Adds the lowercase step if requested.
        
        return cls(steps) # Returns a new `PreprocessingPipeline` instance with the constructed list of steps.
