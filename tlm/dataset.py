"""
Dataset generation for the Library of Babel.

Generates random text sequences representing a "mini Library of Babel"
where every possible sequence of characters exists in pure randomness.
"""

import numpy as np
import random
from typing import List, Tuple


# Full alphabet: lowercase letters + space + punctuation
ALPHABET = list("abcdefghijklmnopqrstuvwxyz ,.")
ALPHABET_SIZE = len(ALPHABET)
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALPHABET)}


def random_babel_sequence(length: int = 200, seed: int = None) -> str:
    """
    Generate a random sequence from the Library of Babel.
    
    Args:
        length: Length of the sequence
        seed: Random seed for reproducibility
        
    Returns:
        Random string from the alphabet
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    return "".join(np.random.choice(ALPHABET, size=length))


def text_to_indices(text: str) -> np.ndarray:
    """Convert text string to index array."""
    indices = []
    for char in text:
        if char in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[char])
        else:
            # Fallback: use space if character not in alphabet
            indices.append(CHAR_TO_IDX.get(' ', 26))
    return np.array(indices, dtype=np.int32)


def indices_to_text(indices: np.ndarray) -> str:
    """Convert index array to text string."""
    text = []
    for idx in indices:
        idx_int = int(idx)
        if 0 <= idx_int < len(IDX_TO_CHAR):
            text.append(IDX_TO_CHAR[idx_int])
        else:
            # Fallback: use space if index out of range
            text.append(' ')
    return "".join(text)


def generate_babel_dataset(
    n_sequences: int = 1000,
    length: int = 200,
    seed: int = None
) -> List[np.ndarray]:
    """
    Generate a dataset of random Babel sequences.
    
    Args:
        n_sequences: Number of sequences to generate
        length: Length of each sequence
        seed: Random seed
        
    Returns:
        List of sequences as index arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    sequences = []
    for _ in range(n_sequences):
        seq = random_babel_sequence(length, seed=None)
        sequences.append(text_to_indices(seq))
    
    return sequences


def generate_babel_strings(
    n_sequences: int = 1000,
    length: int = 200,
    seed: int = None
) -> List[str]:
    """
    Generate a dataset of random Babel sequences as strings.
    
    Args:
        n_sequences: Number of sequences to generate
        length: Length of each sequence
        seed: Random seed
        
    Returns:
        List of sequences as strings
    """
    if seed is not None:
        np.random.seed(seed)
    
    return [random_babel_sequence(length, seed=None) for _ in range(n_sequences)]


if __name__ == "__main__":
    # Example usage
    print("Generating sample Babel sequences...")
    sequences = generate_babel_strings(n_sequences=5, length=50)
    for i, seq in enumerate(sequences):
        print(f"Sequence {i+1}: {seq[:50]}...")

