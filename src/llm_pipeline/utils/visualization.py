# filename: src/llm_pipeline/utils/visualization.py
"""
Visualization utilities for plotting training metrics, model insights, and reports.

This module provides functions to create various types of plots relevant to
Large Language Model (LLM) training and analysis, including training curves,
attention heatmaps, token embedding visualizations, and a simplified loss landscape.
It aims to offer clear and informative visual representations of complex data.

Purpose:
    To aid in understanding, debugging, and communicating the progress and results
    of LLM experiments. Visualizations are critical for quickly identifying trends,
    anomalies, and patterns that might be difficult to discern from raw data alone.

    Visualizations are an indispensable part of the machine learning workflow.
    These utilities are used because they:
    1. **Accelerate Insight:** Provide immediate understanding of training dynamics.
    2. **Facilitate Debugging:** Help pinpoint issues like overfitting, underfitting,
       or unstable training.
    3. **Enhance Communication:** Make complex model behaviors (e.g., attention)
       interpretable to a broader audience.
    4. **Streamline Reporting:** Automate the creation of visual summaries.

LLM Pipeline Fit:
    This module is part of the `llm_pipeline.utils` package. Its functions are
    typically invoked at the end of training runs, during evaluation, or in
    dedicated analysis scripts to generate visual artifacts.
"""

import logging                                              # For logging warnings and errors related to plotting.
from typing import Dict, List, Optional, Union, Any, Tuple  # Type hinting for clarity.
from pathlib import Path                                    # For handling file paths in an object-oriented way.
import numpy as np                                          # For numerical operations, especially for data manipulation before plotting.
import matplotlib                                           # The core plotting library.
import matplotlib.pyplot as plt                             # The pyplot interface for creating plots.
import torch                                                # Used for handling PyTorch tensors, especially in loss landscape.

logger = logging.getLogger(__name__)

# --- Matplotlib and Seaborn Configuration ---
# Set matplotlib backend and style with fallbacks for better aesthetics.
try:
    import seaborn as sns # A statistical data visualization library based on matplotlib.
    # Try new seaborn style format first (available in recent seaborn versions),
    # then fallback to old format, and finally to default matplotlib style.
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            plt.style.use("default")
            logger.warning("Seaborn style not available, using default matplotlib style.")
    
    sns.set_palette("husl")  # Set a pleasant default color palette.
    SEABORN_AVAILABLE = True # Flag to indicate if seaborn is successfully imported.
except ImportError:
    # If seaborn is not installed, use default matplotlib style and log a warning.
    plt.style.use("default")
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available, using matplotlib only.")

# Set default figure parameters for consistent plot appearance.
plt.rcParams.update({
    'figure.figsize': (10, 6), # Default figure size (width, height) in inches.
    'font.size': 12,           # Default font size for text elements.
    'axes.linewidth': 1.2,     # Thickness of the axes lines.
    'grid.alpha': 0.3          # Transparency of grid lines.
})


def plot_training_curves(
    metrics: Dict[str, List[Tuple[int, float]]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 10),
    smoothing_window: int = 10,
) -> matplotlib.figure.Figure:
    """
    Plot training curves (e.g., loss, accuracy) over training steps with smoothing.

    Purpose:
        To visualize the progression of various metrics during training, helping
        to identify trends, convergence, and potential issues (e.g., oscillations).

    Args:
        metrics: A dictionary where keys are metric names (e.g., "loss", "eval_accuracy")
                 and values are lists of `(step, value)` tuples representing the
                 metric's value at a given training step.
        save_path: Optional. The file path to save the generated plot. If `None`,
                   the plot is not saved.
        title: The overall title of the plot.
        figsize: A tuple `(width, height)` specifying the figure size in inches.
        smoothing_window: The window size for applying a moving average to smooth
                          the curves, reducing noise and highlighting overall trends.
                          If `1` or less, no smoothing is applied.

    Returns:
        A `matplotlib.figure.Figure` object representing the generated plot.
    """
    # Filter out any metrics that have no data to avoid plotting empty subplots.
    available_metrics = {k: v for k, v in metrics.items() if v}
    
    if not available_metrics:
        logger.warning("No metrics available for plotting.")
        # Create an empty figure with a message if no data is provided.
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No metrics available", ha='center', va='center', 
                        transform=ax.transAxes, fontsize=16)
        ax.set_title(title)
        return fig
    
    # Determine an appropriate subplot grid layout based on the number of metrics.
    n_metrics = len(available_metrics)
    if n_metrics <= 2:
        rows, cols = 1, n_metrics
    elif n_metrics <= 4:
        rows, cols = 2, 2
    elif n_metrics <= 6:
        rows, cols = 2, 3
    else: # Max 9 plots to avoid clutter.
        rows, cols = 3, 3
    
    # Create the figure and subplots. `squeeze=False` ensures `axes` is always an array.
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes into a 1D array for easier iteration.
    
    # Define a color scheme for the plots.
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    # Iterate through each metric and plot it in its own subplot.
    for idx, (metric_key, data) in enumerate(available_metrics.items()):
        ax = axes[idx] # Get the current subplot axis.
        
        if not data: # Skip if somehow an empty list made it here (should be filtered above).
            continue
            
        steps, values = zip(*data) # Unpack (step, value) tuples into separate lists.
        steps = np.array(steps)
        values = np.array(values)
        
        # Apply moving average smoothing if `smoothing_window` is greater than 1
        # and there's enough data for smoothing.
        if smoothing_window > 1 and len(values) > smoothing_window:
            # `np.convolve` with `valid` mode returns only full convolutions.
            smoothed_values = np.convolve(values, np.ones(smoothing_window)/smoothing_window, mode='valid')
            # The steps for smoothed values are shifted by `smoothing_window - 1`.
            smoothed_steps = steps[smoothing_window-1:]
            
            # Plot both original (faint) and smoothed (bold) curves.
            ax.plot(steps, values, alpha=0.3, color=colors[idx % len(colors)], linewidth=0.8, label="Original")
            ax.plot(smoothed_steps, smoothed_values, color=colors[idx % len(colors)], linewidth=2, label="Smoothed")
            ax.legend(loc='best')
        else:
            # If no smoothing, just plot the raw values.
            ax.plot(steps, values, color=colors[idx % len(colors)], linewidth=2)
        
        # Set labels and title for the current subplot.
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(metric_key.replace('_', ' ').title()) # Format metric name nicely.
        ax.set_title(metric_key.replace('_', ' ').replace('/', ' ').title())
        ax.grid(True, alpha=0.3) # Add a grid for readability.
        
        # Use scientific notation on y-axis if values are very large.
        if values.max() > 10000:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Remove any unused subplots from the grid.
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(title, fontsize=16, y=0.98)  # Set the main title for the entire figure.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent titles/labels from overlapping.
    
    if save_path:
        save_figure(fig, save_path) # Save the figure if a path is provided.
    
    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (12, 10),
    head_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
) -> matplotlib.figure.Figure:
    """
    Plot attention weights as a heatmap.

    Purpose:
        To visualize how different tokens attend to each other within a sequence
        for a given attention layer or head, providing insights into model reasoning.

    Args:
        attention_weights: A NumPy array of attention weights.
                           Expected shapes: `[seq_length, seq_length]` for a single head/average,
                           or `[num_heads, seq_length, seq_length]` for multi-head attention.
        tokens: Optional. A list of strings representing the tokens corresponding to
                the sequence indices. Used for axis labels.
        save_path: Optional. The file path to save the generated plot.
        title: The overall title of the plot.
        figsize: A tuple `(width, height)` specifying the figure size in inches.
        head_idx: If `attention_weights` is 3D, specifies which attention head to plot.
                  If `None`, weights are averaged across heads.
        layer_idx: Optional. An integer representing the layer index, added to the title.

    Returns:
        A `matplotlib.figure.Figure` object.
    """
    # Handle multi-head attention weights (3D array).
    if attention_weights.ndim == 3:
        if head_idx is not None:
            attention_weights = attention_weights[head_idx] # Select a specific head.
            title += f" (Head {head_idx})"
        else:
            attention_weights = attention_weights.mean(axis=0) # Average across all heads.
            title += " (Average across heads)"
    
    # Add layer index to the title if provided.
    if layer_idx is not None:
        title += f" Layer {layer_idx}"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn for a more aesthetically pleasing heatmap if available.
    if SEABORN_AVAILABLE:
        sns.heatmap(
            attention_weights,
            cmap="Blues", # Color map (darker color means higher weight).
            cbar=True,    # Show color bar.
            square=True,  # Make cells square.
            # Limit labels to prevent overcrowding for long sequences.
            xticklabels=tokens[:50] if tokens else False,
            yticklabels=tokens[:50] if tokens else False,
            ax=ax,
            cbar_kws={"shrink": 0.8} # Shrink color bar size.
        )
    else:
        # Fallback to pure matplotlib `imshow`.
        im = ax.imshow(attention_weights, cmap="Blues", aspect="equal")
        plt.colorbar(im, ax=ax, shrink=0.8) # Manually add color bar.
        
        if tokens:
            # Manually set ticks and labels, limiting for readability.
            limited_tokens = tokens[:min(50, len(tokens))]
            ax.set_xticks(range(len(limited_tokens)))
            ax.set_yticks(range(len(limited_tokens)))
            ax.set_xticklabels(limited_tokens, rotation=45, ha='right') # Rotate x-axis labels.
            ax.set_yticklabels(limited_tokens)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    
    plt.tight_layout() # Adjust layout.
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_token_embeddings(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = "tsne",
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Token Embeddings",
    figsize: Tuple[int, int] = (12, 10),
    sample_size: int = 1000,
    **kwargs,
) -> matplotlib.figure.Figure:
    """
    Plot token embeddings in 2D using dimensionality reduction techniques.

    Purpose:
        To visualize the high-dimensional token embeddings in a 2D space,
        helping to understand semantic relationships between tokens (e.g.,
        clusters of related words).

    Args:
        embeddings: A NumPy array of token embeddings with shape `[num_tokens, embedding_dim]`.
        labels: Optional. A list of strings corresponding to the tokens, used for annotation.
        method: The dimensionality reduction method to use: "tsne", "pca", or "umap".
        save_path: Optional. The file path to save the generated plot.
        title: The overall title of the plot.
        figsize: A tuple `(width, height)` specifying the figure size.
        sample_size: The maximum number of tokens to sample and plot. Useful for very
                     large vocabularies to keep plots manageable.
        **kwargs: Additional keyword arguments passed to the dimensionality
                  reduction method (e.g., `perplexity` for t-SNE).

    Returns:
        A `matplotlib.figure.Figure` object.

    Raises:
        ValueError: If an unknown dimensionality reduction method is specified,
                    or if embeddings are 1D and cannot be plotted in 2D.
    """
    # Sample embeddings if the number of tokens exceeds `sample_size`.
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        if labels:
            labels = [labels[i] for i in indices]
    
    # Perform dimensionality reduction with error handling for missing libraries.
    embeddings_2d = None
    try:
        if method == "tsne":
            from sklearn.manifold import TSNE
            kwargs.setdefault('perplexity', min(30, len(embeddings) // 4)) # Sensible default for perplexity.
            kwargs.setdefault('random_state', 42) # For reproducibility.
            reducer = TSNE(n_components=2, **kwargs)
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, **kwargs)
        elif method == "umap":
            try:
                import umap # UMAP is often not installed by default.
                kwargs.setdefault('random_state', 42)
                reducer = umap.UMAP(n_components=2, **kwargs)
            except ImportError:
                logger.warning("UMAP not installed. Falling back to PCA for token embeddings plot.")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                method = "pca" # Update method name for title.
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}. Choose from 'tsne', 'pca', 'umap'.")
        
        # Apply the chosen dimensionality reduction.
        embeddings_2d = reducer.fit_transform(embeddings)
        
    except ImportError as e:
        logger.error(f"Required library not installed for {method} plotting: {e}")
        # Fallback to using the first two dimensions directly if reduction fails.
        if embeddings.shape[1] >= 2:
            embeddings_2d = embeddings[:, :2]
            method = "first_2_dims"
            logger.info("Using first two dimensions of embeddings as fallback for plotting.")
        else:
            raise ValueError("Embeddings must have at least 2 dimensions to be plotted directly.")
    
    # Create the scatter plot.
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color points based on their original index (or some other categorical variable if provided in labels).
    colors = np.arange(len(embeddings_2d)) # Using simple index as color for now.
    
    scatter = ax.scatter(
        embeddings_2d[:, 0], # X-coordinates from the first component.
        embeddings_2d[:, 1], # Y-coordinates from the second component.
        c=colors,            # Color mapping.
        cmap="viridis",      # Color map.
        alpha=0.7,           # Transparency.
        s=20                 # Marker size.
    )
    
    # Add text labels for a subset of tokens to avoid overcrowding the plot.
    if labels is not None and len(labels) > 0:
        n_labels = min(100, len(labels)) # Limit the number of labels.
        indices_to_label = np.random.choice(len(labels), n_labels, replace=False) # Randomly select labels.
        for i in indices_to_label:
            ax.annotate(
                labels[i][:20], # Truncate long labels for readability.
                (embeddings_2d[i, 0], embeddings_2d[i, 1]), # Position of the annotation.
                fontsize=8,
                alpha=0.8,
                xytext=(5, 5), # Offset text from the point.
                textcoords='offset points'
            )
    
    ax.set_title(f"{title} ({method.upper()})", fontsize=14)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    
    plt.colorbar(scatter, ax=ax, label="Token Index", shrink=0.8) # Add a color bar.
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_loss_landscape(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Loss Landscape",
    figsize: Tuple[int, int] = (10, 8),
    resolution: int = 20,
) -> matplotlib.figure.Figure:
    """
    Plot a 2D projection of the loss landscape around the current model parameters.

    Purpose:
        To visualize the shape of the loss function in a small region around the
        current model weights. This can provide insights into optimization
        dynamics (e.g., flatness, sharpness of minima).
        This method is simplified and uses random directions for projection.

    Args:
        model: The PyTorch model whose loss landscape is to be plotted.
        dataloader: A PyTorch `DataLoader` providing data for loss computation.
                    The batch must contain `input_ids` and potentially `labels`
                    or other inputs depending on the model's forward pass.
        device: The `torch.device` (e.g., "cuda" or "cpu") on which to perform computations.
        save_path: Optional. The file path to save the generated plot.
        title: The overall title of the plot.
        figsize: A tuple `(width, height)` specifying the figure size.
        resolution: The number of points along each dimension of the grid (e.g., 20x20 grid).

    Returns:
        A `matplotlib.figure.Figure` object.

    Raises:
        ValueError: If the model has no trainable parameters or if the dataloader
                    does not provide a 'loss' in its output.
    """
    model.eval()     # Set model to evaluation mode to disable dropout, batch norm updates, etc.
    model.to(device) # Move model to the specified device.
    
    # Collect all trainable parameters into a single flat vector.
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p.view(-1))
    
    if not params:
        raise ValueError("No trainable parameters found in the model to plot loss landscape.")
    
    param_vector = torch.cat(params).detach().cpu() # Detach and move to CPU for operations.
    
    # Generate two random, orthonormal directions in the parameter space.
    # These directions define the 2D plane through the original parameters.
    direction1 = torch.randn_like(param_vector)
    direction1 = direction1 / direction1.norm() # Normalize to unit vector.
    
    direction2 = torch.randn_like(param_vector)
    # Orthogonalize `direction2` with respect to `direction1` using Gram-Schmidt.
    direction2 = direction2 - (direction2 @ direction1) * direction1
    direction2 = direction2 / direction2.norm() # Normalize.
    
    # Define the grid for the landscape plot.
    alpha_range = np.linspace(-1, 1, resolution)   # Range for the first direction.
    beta_range = np.linspace(-1, 1, resolution)    # Range for the second direction.
    loss_grid = np.zeros((resolution, resolution)) # Initialize grid to store loss values.
    
    # Store original parameters to restore them after plotting.
    original_params = param_vector.clone().to(device) # Keep original params on device for restoration.
    
    # Compute loss at each point on the grid.
    # This loop can be computationally expensive for large models/datasets.
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Calculate new parameter vector by moving in the 2D plane.
            # Convert to device as model parameters are on device.
            new_params_cpu = param_vector + alpha * direction1 + beta * direction2
            new_params_gpu = new_params_cpu.to(device)
            
            # Temporarily set the model's parameters to `new_params_gpu`.
            idx = 0
            with torch.no_grad(): # Ensure no gradients are computed during parameter manipulation.
                for p in model.parameters():
                    if p.requires_grad:
                        size = p.numel()
                        p.data = new_params_gpu[idx:idx+size].view(p.shape)
                        idx += size
            
            # Compute average loss over a batch or a subset of the dataloader.
            total_loss = 0.0
            total_samples = 0
            
            with torch.no_grad(): # Ensure no gradients are computed for the forward pass.
                for batch in dataloader:
                    # Move batch tensors to the specified device.
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
                    
                    outputs = model(**batch) # Perform forward pass.
                    if "loss" in outputs:
                        # Accumulate loss, weighted by batch size if appropriate.
                        current_batch_size = batch["input_ids"].size(0) if "input_ids" in batch else 1
                        total_loss += outputs["loss"].item() * current_batch_size
                        total_samples += current_batch_size
                    else:
                        raise ValueError(f"Model outputs do not contain 'loss' key. "
                                         f"Ensure your model's forward pass returns a dictionary with 'loss'. "
                                         f"Output keys: {outputs.keys()}")
            
            # Store the average loss for this grid point.
            loss_grid[i, j] = total_loss / total_samples if total_samples > 0 else float('inf')
    
    # Restore original model parameters.
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                size = p.numel()
                # Use `original_params` from the device.
                p.data = original_params[idx:idx+size].view(p.shape)
                idx += size
    
    # Create the heatmap plot of the loss grid.
    fig, ax = plt.subplots(figsize=figsize)
    
    if SEABORN_AVAILABLE:
        sns.heatmap(loss_grid, xticklabels=False, yticklabels=False, 
                            cmap="viridis", ax=ax, cbar_kws={"label": "Loss"})
    else:
        im = ax.imshow(loss_grid, cmap="viridis", aspect="equal")
        plt.colorbar(im, ax=ax, label="Loss")
    
    # Mark the center point (where the original model parameters lie).
    center_idx = resolution // 2
    ax.plot(center_idx, center_idx, 'r*', markersize=15, label="Current Parameters", zorder=5) # zorder to ensure visibility.
    ax.legend(loc='best')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def save_figure(
    fig: matplotlib.figure.Figure,
    save_path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = "tight",
    formats: Optional[List[str]] = None,
) -> None:
    """
    Save a Matplotlib figure to one or more specified file formats.

    Purpose:
        A utility function to encapsulate the saving logic for all plots,
        ensuring consistent quality (DPI), layout, and error handling. It
        also automatically creates necessary directories.

    Args:
        fig: The `matplotlib.figure.Figure` object to save.
        save_path: The base path (without extension) to save the figure.
                   E.g., "results/my_plot" will result in "results/my_plot.png".
        dpi: The dots per inch (resolution) for raster formats (e.g., PNG).
        bbox_inches: How to handle the bounding box of the figure. "tight"
                     tries to make sure all elements fit without extra whitespace.
        formats: A list of file extensions (e.g., ['png', 'pdf', 'svg'])
                 to save the figure in. Defaults to `['png']`.
    """
    if formats is None:
        formats = ['png']
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True) # Create parent directories if they don't exist.
    
    for fmt in formats:
        # Construct the full file path with the appropriate suffix.
        fig_path = save_path.with_suffix(f'.{fmt}')
        
        try:
            fig.savefig(fig_path, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
            logger.info(f"Figure saved to {fig_path}")
        except Exception as e:
            logger.error(f"Failed to save figure as {fmt}: {e}")
    
    plt.close(fig) # Close the figure to free up memory, important when generating many plots.


def create_training_summary_report(
    metrics: Dict[str, Any],
    model_config: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Create a comprehensive summary report figure combining key training metrics and model configuration.

    Purpose:
        To provide a single, digestible visual summary of an LLM training run,
        useful for quick comparison across experiments. This report will combine
        various data points into a single multi-panel plot.

    Args:
        metrics: A dictionary containing key scalar training and evaluation metrics
                 (e.g., "loss", "final_eval/perplexity", "total_parameters",
                 "inference_throughput_tokens_per_second").
        model_config: A dictionary containing key model configuration parameters
                      (e.g., "hidden_size", "num_hidden_layers", "num_attention_heads",
                      "vocab_size", "max_position_embeddings").
        output_path: The file path (without extension) to save the generated report figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 15)) # Create a large figure for the report.
    
    # Subplot 1: Summary Statistics (Bar Chart)
    ax1 = plt.subplot(3, 3, 1) # Position in a 3x3 grid.
    summary_data = [
        metrics.get('loss', 0), # Get values with default 0 if not present.
        metrics.get('final_eval/perplexity', 0),
        metrics.get('total_parameters', 0) / 1e6, # Convert parameters to millions.
        metrics.get('inference_throughput_tokens_per_second', 0)
    ]
    labels = ['Final Loss', 'Perplexity', 'Params (M)', 'Throughput (tok/s)']
    
    bars = ax1.bar(labels, summary_data, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']) # Different colors for bars.
    ax1.set_title('Model Summary', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability.
    
    # Add value labels on top of the bars.
    for bar, value in zip(bars, summary_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05, # Position text slightly above bar.
                 f'{value:.2f}', ha='center', va='bottom')               # Format to 2 decimal places.
    ax1.grid(axis='y', alpha=0.3) # Add horizontal grid.
    
    # Subplot 2: Model Configuration (Text Box)
    ax2 = plt.subplot(3, 3, 2)
    # Format model configuration parameters into a multi-line string.
    config_text = '\n'.join([
        f"Hidden Size: {model_config.get('hidden_size', 'N/A')}",
        f"Layers: {model_config.get('num_hidden_layers', 'N/A')}",
        f"Attention Heads: {model_config.get('num_attention_heads', 'N/A')}",
        f"Vocab Size: {model_config.get('vocab_size', 'N/A')}",
        f"Max Position: {model_config.get('max_position_embeddings', 'N/A')}"
    ])
    ax2.text(0.1, 0.5, config_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray")) # Text box styling.
    ax2.set_title('Model Configuration', fontsize=14, fontweight='bold')
    ax2.axis('off') # Hide axes for text plot.

    plt.suptitle('Training Summary Report', fontsize=18, fontweight='bold', y=0.98) # Main title for the entire report.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for the main title.
    
    save_figure(fig, output_path) # Save the combined report figure.