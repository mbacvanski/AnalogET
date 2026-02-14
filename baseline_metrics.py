"""
Calculate baseline metrics for Shakespeare character prediction.

This script computes:
1. Random guessing baseline (uniform distribution over vocabulary)
2. Most frequent character baseline (always predict mode)
3. Unigram distribution baseline (predict according to training frequencies)
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp
from collections import Counter

from data import load_shakespeare_dataset


def calculate_cross_entropy(predictions, targets):
    """
    Calculate cross-entropy loss.
    
    Args:
        predictions: (N, vocab_size) probability distributions
        targets: (N,) target indices
    
    Returns:
        Average cross-entropy loss
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    log_probs = jnp.log(predictions + eps)
    
    # Get log probability of correct class for each sample
    correct_log_probs = log_probs[jnp.arange(len(targets)), targets]
    
    # Cross-entropy is negative log likelihood
    ce = -jnp.mean(correct_log_probs)
    return float(ce)


def calculate_accuracy(predictions, targets):
    """
    Calculate accuracy.
    
    Args:
        predictions: (N, vocab_size) probability distributions
        targets: (N,) target indices
    
    Returns:
        Accuracy (fraction correct)
    """
    predicted_classes = jnp.argmax(predictions, axis=1)
    accuracy = jnp.mean(predicted_classes == targets)
    return float(accuracy)


def calculate_perplexity(cross_entropy):
    """Calculate perplexity from cross-entropy."""
    return float(np.exp(cross_entropy))


def random_baseline(train_y, valid_y, vocab_size):
    """
    Baseline: Random guessing (uniform distribution).
    
    Each character has equal probability 1/vocab_size.
    """
    print("\n" + "="*60)
    print("RANDOM GUESSING BASELINE")
    print("="*60)
    print(f"Strategy: Predict uniform distribution over {vocab_size} characters")
    
    # Create uniform distribution
    uniform_probs = jnp.ones(vocab_size) / vocab_size
    
    # Replicate for all samples
    train_predictions = jnp.tile(uniform_probs, (len(train_y), 1))
    valid_predictions = jnp.tile(uniform_probs, (len(valid_y), 1))
    
    # Calculate metrics
    train_ce = calculate_cross_entropy(train_predictions, train_y)
    train_acc = calculate_accuracy(train_predictions, train_y)
    train_ppl = calculate_perplexity(train_ce)
    
    valid_ce = calculate_cross_entropy(valid_predictions, valid_y)
    valid_acc = calculate_accuracy(valid_predictions, valid_y)
    valid_ppl = calculate_perplexity(valid_ce)
    
    print(f"\nTraining Set:")
    print(f"  Cross-Entropy: {train_ce:.4f}")
    print(f"  Perplexity:    {train_ppl:.4f}")
    print(f"  Accuracy:      {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    print(f"\nValidation Set:")
    print(f"  Cross-Entropy: {valid_ce:.4f}")
    print(f"  Perplexity:    {valid_ppl:.4f}")
    print(f"  Accuracy:      {valid_acc:.4f} ({valid_acc*100:.2f}%)")
    
    return {
        'train_ce': train_ce,
        'train_ppl': train_ppl,
        'train_acc': train_acc,
        'valid_ce': valid_ce,
        'valid_ppl': valid_ppl,
        'valid_acc': valid_acc,
    }


def most_frequent_baseline(train_y, valid_y, vocab_size):
    """
    Baseline: Always predict the most frequent character.
    
    Find the mode in training data and always predict it.
    """
    print("\n" + "="*60)
    print("MOST FREQUENT CHARACTER BASELINE")
    print("="*60)
    
    # Find most frequent character in training data
    char_counts = Counter(np.array(train_y))
    most_frequent_char = char_counts.most_common(1)[0][0]
    most_frequent_count = char_counts[most_frequent_char]
    most_frequent_freq = most_frequent_count / len(train_y)
    
    print(f"Most frequent character index: {most_frequent_char}")
    print(f"Frequency in training: {most_frequent_freq:.4f} ({most_frequent_freq*100:.2f}%)")
    print(f"Strategy: Always predict character {most_frequent_char}")
    
    # Create probability distribution (all mass on most frequent)
    mode_probs = jnp.zeros(vocab_size)
    mode_probs = mode_probs.at[most_frequent_char].set(1.0)
    
    # Replicate for all samples
    train_predictions = jnp.tile(mode_probs, (len(train_y), 1))
    valid_predictions = jnp.tile(mode_probs, (len(valid_y), 1))
    
    # Calculate metrics
    train_ce = calculate_cross_entropy(train_predictions, train_y)
    train_acc = calculate_accuracy(train_predictions, train_y)
    train_ppl = calculate_perplexity(train_ce)
    
    valid_ce = calculate_cross_entropy(valid_predictions, valid_y)
    valid_acc = calculate_accuracy(valid_predictions, valid_y)
    valid_ppl = calculate_perplexity(valid_ce)
    
    print(f"\nTraining Set:")
    print(f"  Cross-Entropy: {train_ce:.4f}")
    print(f"  Perplexity:    {train_ppl:.4f}")
    print(f"  Accuracy:      {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    print(f"\nValidation Set:")
    print(f"  Cross-Entropy: {valid_ce:.4f}")
    print(f"  Perplexity:    {valid_ppl:.4f}")
    print(f"  Accuracy:      {valid_acc:.4f} ({valid_acc*100:.2f}%)")
    
    return {
        'train_ce': train_ce,
        'train_ppl': train_ppl,
        'train_acc': train_acc,
        'valid_ce': valid_ce,
        'valid_ppl': valid_ppl,
        'valid_acc': valid_acc,
        'most_frequent_char': int(most_frequent_char),
        'most_frequent_freq': float(most_frequent_freq),
    }


def unigram_baseline(train_y, valid_y, vocab_size):
    """
    Baseline: Predict according to unigram (character frequency) distribution.
    
    Use the empirical character distribution from training data.
    """
    print("\n" + "="*60)
    print("UNIGRAM DISTRIBUTION BASELINE")
    print("="*60)
    print("Strategy: Predict according to character frequencies in training data")
    
    # Calculate character frequencies from training data
    char_counts = Counter(np.array(train_y))
    
    # Create probability distribution
    unigram_probs = jnp.zeros(vocab_size)
    for char_idx, count in char_counts.items():
        unigram_probs = unigram_probs.at[char_idx].set(count / len(train_y))
    
    # Show top 5 most frequent characters
    print("\nTop 5 most frequent characters:")
    for i, (char_idx, count) in enumerate(char_counts.most_common(5), 1):
        freq = count / len(train_y)
        print(f"  {i}. Character {char_idx}: {freq:.4f} ({freq*100:.2f}%)")
    
    # Replicate for all samples
    train_predictions = jnp.tile(unigram_probs, (len(train_y), 1))
    valid_predictions = jnp.tile(unigram_probs, (len(valid_y), 1))
    
    # Calculate metrics
    train_ce = calculate_cross_entropy(train_predictions, train_y)
    train_acc = calculate_accuracy(train_predictions, train_y)
    train_ppl = calculate_perplexity(train_ce)
    
    valid_ce = calculate_cross_entropy(valid_predictions, valid_y)
    valid_acc = calculate_accuracy(valid_predictions, valid_y)
    valid_ppl = calculate_perplexity(valid_ce)
    
    print(f"\nTraining Set:")
    print(f"  Cross-Entropy: {train_ce:.4f}")
    print(f"  Perplexity:    {train_ppl:.4f}")
    print(f"  Accuracy:      {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    print(f"\nValidation Set:")
    print(f"  Cross-Entropy: {valid_ce:.4f}")
    print(f"  Perplexity:    {valid_ppl:.4f}")
    print(f"  Accuracy:      {valid_acc:.4f} ({valid_acc*100:.2f}%)")
    
    return {
        'train_ce': train_ce,
        'train_ppl': train_ppl,
        'train_acc': train_acc,
        'valid_ce': valid_ce,
        'valid_ppl': valid_ppl,
        'valid_acc': valid_acc,
    }


def main():
    print("="*60)
    print("SHAKESPEARE CHARACTER PREDICTION BASELINE METRICS")
    print("="*60)
    
    # Load dataset
    print("\nLoading Shakespeare dataset...")
    filename_prefix = "shakespeare_data"
    train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = load_shakespeare_dataset(
        filename_prefix=filename_prefix
    )
    
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(train_y):,}")
    print(f"Validation samples: {len(valid_y):,}")
    
    # Calculate baselines
    random_results = random_baseline(train_y, valid_y, vocab_size)
    freq_results = most_frequent_baseline(train_y, valid_y, vocab_size)
    unigram_results = unigram_baseline(train_y, valid_y, vocab_size)
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON (Validation Set)")
    print("="*60)
    print(f"\n{'Baseline':<30} {'Loss':<12} {'Perplexity':<12} {'Accuracy':<12}")
    print("-" * 66)
    print(f"{'Random Guessing':<30} {random_results['valid_ce']:<12.4f} {random_results['valid_ppl']:<12.4f} {random_results['valid_acc']:<12.4f}")
    print(f"{'Most Frequent Character':<30} {freq_results['valid_ce']:<12.4f} {freq_results['valid_ppl']:<12.4f} {freq_results['valid_acc']:<12.4f}")
    print(f"{'Unigram Distribution':<30} {unigram_results['valid_ce']:<12.4f} {unigram_results['valid_ppl']:<12.4f} {unigram_results['valid_acc']:<12.4f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
Your model should achieve:
- Loss < {:.4f} (better than unigram baseline)
- Perplexity < {:.4f} (better than unigram baseline)
- Accuracy > {:.4f} (better than most frequent baseline)

If your model performs worse than these baselines, there may be an issue
with the model architecture, training, or hyperparameters.

The unigram baseline represents a simple frequency-based predictor.
A good language model should learn context and perform significantly
better than this baseline.
""".format(
        unigram_results['valid_ce'],
        unigram_results['valid_ppl'],
        freq_results['valid_acc']
    ))
    
    # Save results
    import json
    results = {
        'vocab_size': vocab_size,
        'train_samples': int(len(train_y)),
        'valid_samples': int(len(valid_y)),
        'random_guessing': random_results,
        'most_frequent_character': freq_results,
        'unigram_distribution': unigram_results,
    }
    
    output_file = 'data/baseline_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
