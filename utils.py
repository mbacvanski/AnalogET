import json
from typing import Dict, List

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

ModelParams = Dict[str, jax.Array]


def save_params(params: ModelParams, fname: str):
    np.savez(fname, **{k: np.array(v) for k, v in params.items()})  # type: ignore


def load_params(fname: str) -> ModelParams:
    loaded = np.load(fname)
    return {k: jnp.array(loaded[k]) for k in loaded.files}


def save_metrics(obj: Dict[str, List[int] | List[float]], fname: str):
    with open(fname, "w") as f:
        json.dump(obj, f)


def load_metrics(fname: str) -> dict:
    with open(fname, "r") as f:
        data = json.load(f)
    return data


def generate_text(
    params,
    seed_context: jnp.ndarray,
    idx_to_char: Dict[int, str],
    cfg,
    num_chars: int = 32,
    temperature: float = 1.0,
    key: jax.Array = None,
) -> str:
    """
    Generate text autoregressively by feeding generated characters back into context.
    
    Args:
        params: Model parameters
        seed_context: Initial context of shape (L,) with token indices
        idx_to_char: Mapping from token indices to characters
        cfg: Configuration object
        num_chars: Number of characters to generate
        temperature: Sampling temperature (higher = more random)
        key: Random key for sampling
    
    Returns:
        Generated text string
    """
    # Import here to avoid circular dependency
    from model import infer_forward_euler_with_force, logits_from_v
    
    if key is None:
        key = jr.PRNGKey(0)
    
    # Start with seed context
    context = seed_context.copy()
    generated_chars = []
    
    for _ in range(num_chars):
        # Prepare batch of size 1
        ctx_batch = context.reshape(1, -1)  # (1, L)
        
        # Run inference
        V_T, _ = infer_forward_euler_with_force(
            params, jnp.zeros((1, cfg.D), jnp.float32), ctx_batch, cfg
        )
        logits = logits_from_v(params, V_T)[0]  # (vocab_size,)
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample next token
        key, subkey = jr.split(key)
        next_token = jr.categorical(subkey, logits)
        
        # Add to generated text
        generated_chars.append(idx_to_char[int(next_token)])
        
        # Update context: shift left and append new token
        context = jnp.concatenate([context[1:], jnp.array([next_token])])
    
    return "".join(generated_chars)


def calculate_perplexity(
    params, valid_X: jnp.ndarray, valid_y: jnp.ndarray, cfg, batch_size: int = 512
) -> float:
    """
    Calculate perplexity on validation set.
    
    Perplexity = exp(average cross-entropy loss)
    """
    # Import here to avoid circular dependency
    from model import infer_forward_euler_with_force, logits_from_v
    
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    n = int(valid_y.shape[0])
    if n == 0:
        return float("inf")

    ce_sum = 0.0
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x_batch = valid_X[start:stop]
        y_batch = valid_y[start:stop]
        V_T, _ = infer_forward_euler_with_force(
            params, jnp.zeros((y_batch.shape[0], cfg.D), jnp.float32), x_batch, cfg
        )
        logits = logits_from_v(params, V_T)
        ce_losses = optax.softmax_cross_entropy_with_integer_labels(logits, y_batch)
        ce_sum += float(jnp.sum(ce_losses))

    avg_ce = ce_sum / n
    
    # Perplexity is exp(cross-entropy)
    perplexity = jnp.exp(jnp.array(avg_ce, dtype=jnp.float32))

    return float(perplexity)


def plot_training_metrics(
    losses_steps: List[int],
    losses_all: List[float],
    accs_steps: List[int],
    accs_all: List[float],
    perplexities_steps: List[int],
    perplexities_all: List[float],
    output_path: str = "data/training_metrics_shakespeare.png",
) -> str:
    """
    Create and save comprehensive training metrics plots.
    
    Args:
        losses_steps: Training steps for loss values
        losses_all: Training loss values
        accs_steps: Training steps for accuracy values
        accs_all: Validation accuracy values
        perplexities_steps: Training steps for perplexity values
        perplexities_all: Validation perplexity values
        output_path: Path to save the plot
    
    Returns:
        Path to the saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(losses_steps, losses_all, alpha=0.6, linewidth=0.5)
    # Add smoothed line
    if len(losses_all) > 100:
        window = min(100, len(losses_all) // 10)
        smoothed = jnp.convolve(jnp.array(losses_all), jnp.ones(window) / window, mode='valid')
        ax1.plot(losses_steps[:len(smoothed)], smoothed, 'r-', linewidth=2, label='Smoothed')
        ax1.legend()
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax2 = axes[0, 1]
    ax2.plot(accs_steps, accs_all, 'o-', markersize=3)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Perplexity
    ax3 = axes[1, 0]
    ax3.plot(perplexities_steps, perplexities_all, 'o-', markersize=3, color='green')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Perplexity')
    ax3.set_title('Validation Perplexity')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Loss (log scale)
    ax4 = axes[1, 1]
    ax4.plot(losses_steps, losses_all, alpha=0.6, linewidth=0.5)
    if len(losses_all) > 100:
        window = min(100, len(losses_all) // 10)
        smoothed = jnp.convolve(jnp.array(losses_all), jnp.ones(window) / window, mode='valid')
        ax4.plot(losses_steps[:len(smoothed)], smoothed, 'r-', linewidth=2, label='Smoothed')
        ax4.legend()
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('Training Loss Over Time (Log Scale)')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path
