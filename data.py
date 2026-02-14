import json
import os
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import urllib.request


def generate_and_save_datasets(
    key: jax.Array,
    ctx_length: int = 8,
    train_ratio: float = 0.8,
    filename_prefix: str = "parity_data",
):
    n = 2**ctx_length
    # Create numbers 0 .. n - 1
    ints = jnp.arange(n, dtype=jnp.uint32)
    # Extract bits by shifting and masking
    bits = (ints[:, None] >> jnp.arange(ctx_length - 1, -1, -1)) & 1
    bits = bits.astype(jnp.int16)

    # Labels are parity (XOR of bits)
    labels = jnp.sum(bits, axis=1) % 2
    labels = labels.astype(jnp.int16)

    # shuffle bits and labels
    perm_key, _ = jr.split(key)
    perm = jr.permutation(perm_key, n)
    bits = bits[perm]
    labels = labels[perm]

    train_X = bits[: int(n * train_ratio)]
    train_y = labels[: int(n * train_ratio)]
    test_X = bits[int(n * train_ratio) :]
    test_y = labels[int(n * train_ratio) :]

    # Save to disk as txt
    os.makedirs("data", exist_ok=True)
    np.savetxt(f"data/{filename_prefix}_train_X.txt", train_X, fmt="%d")
    np.savetxt(f"data/{filename_prefix}_train_y.txt", train_y, fmt="%d")
    np.savetxt(f"data/{filename_prefix}_test_X.txt", test_X, fmt="%d")
    np.savetxt(f"data/{filename_prefix}_test_y.txt", test_y, fmt="%d")


def load_dataset(
    filename_prefix: str = "parity_data",
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    train_X = jnp.array(
        np.loadtxt(f"data/{filename_prefix}_train_X.txt", dtype=np.int32)
    )
    train_y = jnp.array(
        np.loadtxt(f"data/{filename_prefix}_train_y.txt", dtype=np.int32)
    )
    test_X = jnp.array(np.loadtxt(f"data/{filename_prefix}_test_X.txt", dtype=np.int32))
    test_y = jnp.array(np.loadtxt(f"data/{filename_prefix}_test_y.txt", dtype=np.int32))
    return train_X, train_y, test_X, test_y


def download_shakespeare_text(url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt") -> str:
    """Download Shakespeare text from URL and return as string."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "shakespeare_input.txt")
    
    if os.path.exists(cache_path):
        print(f"Loading cached Shakespeare text from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    
    print(f"Downloading Shakespeare text from {url}")
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    return text


def create_char_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create character-to-index and index-to-character mappings."""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def text_to_sequences(
    text: str,
    char_to_idx: Dict[str, int],
    ctx_length: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert text to sequences of context and target.
    
    Args:
        text: Input text string
        char_to_idx: Character to index mapping
        ctx_length: Length of context (default 16)
    
    Returns:
        X: (N, ctx_length) array of context sequences
        y: (N,) array of target character indices
    """
    # Convert text to integer indices
    data = np.array([char_to_idx[ch] for ch in text], dtype=np.int32)
    
    # Create sequences: for each position i, use chars[i:i+ctx_length] as context
    # and char[i+ctx_length] as target
    sequences = []
    targets = []
    
    for i in range(len(data) - ctx_length):
        sequences.append(data[i : i + ctx_length])
        targets.append(data[i + ctx_length])
    
    X = np.array(sequences, dtype=np.int32)  # (N, ctx_length)
    y = np.array(targets, dtype=np.int32)  # (N,)
    
    return X, y


def prepare_shakespeare_dataset(
    ctx_length: int = 16,
    train_ratio: float = 0.9,
    url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt",
    filename_prefix: str = "shakespeare_data",
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Dict[str, int], Dict[int, str]]:
    """
    Download and prepare Shakespeare dataset for character prediction.
    
    Args:
        ctx_length: Length of context sequence (default 16)
        train_ratio: Ratio of data to use for training (default 0.9)
        url: URL to download Shakespeare text from
        filename_prefix: Prefix for saved dataset files (default "shakespeare_data")
    
    Returns:
        train_X: (N_train, ctx_length) training contexts
        train_y: (N_train,) training targets
        valid_X: (N_valid, ctx_length) validation contexts
        valid_y: (N_valid,) validation targets
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
    """
    # Download text
    text = download_shakespeare_text(url)
    
    # Create vocabulary
    char_to_idx, idx_to_char = create_char_vocab(text)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size} unique characters")
    
    # Convert to sequences
    X, y = text_to_sequences(text, char_to_idx, ctx_length)
    print(f"Total sequences: {len(X)}")
    
    # Split into train/validation
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    
    train_X = jnp.array(X[:n_train])
    train_y = jnp.array(y[:n_train])
    valid_X = jnp.array(X[n_train:])
    valid_y = jnp.array(y[n_train:])
    
    print(f"Training sequences: {len(train_X)}")
    print(f"Validation sequences: {len(valid_X)}")
    
    # Save to disk
    os.makedirs("data", exist_ok=True)
    np.savetxt(f"data/{filename_prefix}_ctx{ctx_length}_train_X.txt", np.array(train_X), fmt="%d")
    np.savetxt(f"data/{filename_prefix}_ctx{ctx_length}_train_y.txt", np.array(train_y), fmt="%d")
    np.savetxt(f"data/{filename_prefix}_ctx{ctx_length}_valid_X.txt", np.array(valid_X), fmt="%d")
    np.savetxt(f"data/{filename_prefix}_ctx{ctx_length}_valid_y.txt", np.array(valid_y), fmt="%d")
    
    # Save vocabulary mappings as JSON
    # Convert keys to strings for JSON serialization
    char_to_idx_str = {ch: idx for ch, idx in char_to_idx.items()}
    idx_to_char_str = {str(idx): ch for idx, ch in idx_to_char.items()}
    
    with open(f"data/{filename_prefix}_char_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(char_to_idx_str, f, ensure_ascii=False)
    
    with open(f"data/{filename_prefix}_idx_to_char.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_char_str, f, ensure_ascii=False)
    
    print(f"Saved dataset to data/{filename_prefix}_ctx{ctx_length}_*.txt and vocabulary to data/{filename_prefix}_*.json")
    
    return train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char


def load_shakespeare_dataset(
    ctx_length: int = 16,
    filename_prefix: str = "shakespeare_data",
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Dict[str, int], Dict[int, str]]:
    """
    Load Shakespeare dataset from saved files.
    
    Args:
        ctx_length: Context length used when dataset was created (default 16)
        filename_prefix: Prefix for saved dataset files (default "shakespeare_data")
    
    Returns:
        train_X: (N_train, ctx_length) training contexts
        train_y: (N_train,) training targets
        valid_X: (N_valid, ctx_length) validation contexts
        valid_y: (N_valid,) validation targets
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
    """
    # Load datasets
    train_X = jnp.array(
        np.loadtxt(f"data/{filename_prefix}_ctx{ctx_length}_train_X.txt", dtype=np.int32)
    )
    train_y = jnp.array(
        np.loadtxt(f"data/{filename_prefix}_ctx{ctx_length}_train_y.txt", dtype=np.int32)
    )
    valid_X = jnp.array(
        np.loadtxt(f"data/{filename_prefix}_ctx{ctx_length}_valid_X.txt", dtype=np.int32)
    )
    valid_y = jnp.array(
        np.loadtxt(f"data/{filename_prefix}_ctx{ctx_length}_valid_y.txt", dtype=np.int32)
    )
    
    # Load vocabulary mappings
    with open(f"data/{filename_prefix}_char_to_idx.json", "r", encoding="utf-8") as f:
        char_to_idx_str = json.load(f)
    
    with open(f"data/{filename_prefix}_idx_to_char.json", "r", encoding="utf-8") as f:
        idx_to_char_str = json.load(f)
    
    # Convert back to proper types
    char_to_idx = {ch: int(idx) for ch, idx in char_to_idx_str.items()}
    idx_to_char = {int(idx): ch for idx, ch in idx_to_char_str.items()}
    
    print(f"Loaded dataset: {len(train_X)} training sequences, {len(valid_X)} validation sequences")
    print(f"Context length: {ctx_length}")
    print(f"Vocabulary size: {len(char_to_idx)} unique characters")
    
    return train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char
