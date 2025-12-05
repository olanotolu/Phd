"""
Training loop for the Babel EBM using KL-gradient estimation.

Implements moment matching and pseudo-likelihood training methods
compatible with THRML's gradient estimation approach.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Optional, Tuple
from tqdm import tqdm
from babel_library import BabelEBM
from dataset import generate_babel_dataset


def estimate_kl_gradient(
    model: BabelEBM,
    data: List[np.ndarray],
    n_samples: int = 100,
    temperature: float = 1.0,
    key: Optional[jax.random.PRNGKey] = None
) -> np.ndarray:
    """
    Estimate KL divergence gradient using moment matching.
    
    This is the THRML-style gradient estimator that matches:
    - Data moments (from training data)
    - Model moments (from samples)
    
    Args:
        model: BabelEBM model
        data: List of training sequences
        n_samples: Number of samples for model moments
        temperature: Sampling temperature
        key: JAX random key
        
    Returns:
        Gradient of weights
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Convert data to numpy array
    data_array = np.array(data)  # Shape: (n_sequences, sequence_length)
    n_data = len(data)
    
    # Compute data moments (bigram frequencies)
    data_moments = compute_bigram_moments(data_array, model.sequence_length, model.alphabet_size)
    
    # Sample from model
    key, subkey = jax.random.split(key)
    model_samples = model.sample(
        n_samples=n_samples,
        n_burnin=50,
        temperature=temperature,
        key=subkey
    )
    
    # Convert samples to indices
    from dataset import text_to_indices
    model_indices = np.array([text_to_indices(s) for s in model_samples])
    
    # Compute model moments
    model_moments = compute_bigram_moments(model_indices, model.sequence_length, model.alphabet_size)
    
    # Gradient is difference in moments (moment matching)
    gradient = data_moments - model_moments
    
    return gradient


def compute_bigram_moments(
    sequences: np.ndarray,
    sequence_length: int,
    alphabet_size: int
) -> np.ndarray:
    """
    Compute bigram moment matrix from sequences.
    
    Returns:
        Matrix of shape (sequence_length-1, alphabet_size, alphabet_size)
        representing bigram frequencies
    """
    n_sequences = sequences.shape[0]
    actual_seq_length = sequences.shape[1]
    max_positions = min(sequence_length - 1, actual_seq_length - 1)
    moments = np.zeros((sequence_length - 1, alphabet_size, alphabet_size))
    
    for seq in sequences:
        # Ensure valid indices
        seq = seq[:sequence_length].copy()
        seq = np.clip(seq, 0, alphabet_size - 1).astype(int)
        
        for i in range(max_positions):
            if i >= len(seq) - 1 or i >= moments.shape[0]:
                break
            prev_char = max(0, min(int(seq[i]), alphabet_size - 1))
            next_char = max(0, min(int(seq[i + 1]), alphabet_size - 1))
            moments[i, prev_char, next_char] += 1
    
    # Normalize
    if n_sequences > 0:
        moments = moments / n_sequences
    
    return moments


def pseudo_likelihood_gradient(
    model: BabelEBM,
    data: List[np.ndarray],
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute pseudo-likelihood gradient (simpler alternative).
    
    This is faster than full KL-gradient but less accurate.
    Good for initial training or when sampling is expensive.
    
    Args:
        model: BabelEBM model
        data: List of training sequences
        temperature: Temperature for logits
        
    Returns:
        Gradient of weights
    """
    data_array = np.array(data)
    n_data = len(data)
    gradient = np.zeros_like(model.weights)
    
    # Get actual sequence length from data (may differ from model)
    actual_seq_length = data_array.shape[1]
    max_positions = min(model.sequence_length - 1, actual_seq_length - 1)
    
    for seq in data_array:
        # Ensure sequence is the right length and indices are valid
        seq = seq[:model.sequence_length].copy()  # Truncate if too long
        seq = np.clip(seq, 0, model.alphabet_size - 1).astype(int)  # Clip indices to valid range
        
        for i in range(max_positions):
            if i >= len(seq) - 1 or i >= gradient.shape[0]:
                break
                
            prev_char = int(seq[i])
            next_char = int(seq[i + 1])
            
            # Validate indices (double-check after clipping)
            prev_char = max(0, min(prev_char, model.alphabet_size - 1))
            next_char = max(0, min(next_char, model.alphabet_size - 1))
            
            # Positive gradient: observed bigram
            gradient[i, prev_char, next_char] += 1.0
            
            # Negative gradient: expected bigram (softmax over next_char)
            logits = model.weights[i, prev_char, :] / temperature
            probs = jax.nn.softmax(logits)
            gradient[i, prev_char, :] -= probs
    
    return gradient / n_data


def train_ebm(
    model: BabelEBM,
    data: List[np.ndarray],
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    method: str = "kl_gradient",
    n_samples: int = 50,
    temperature: float = 1.0,
    verbose: bool = True,
    key: Optional[jax.random.PRNGKey] = None
) -> BabelEBM:
    """
    Train the Babel EBM model.
    
    Args:
        model: BabelEBM model to train
        data: Training sequences
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        method: "kl_gradient" or "pseudo_likelihood"
        n_samples: Number of samples for KL-gradient
        temperature: Sampling temperature
        verbose: Print progress
        key: JAX random key
        
    Returns:
        Trained model
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    if verbose:
        print(f"Training Babel EBM with {method} method...")
        print(f"Data: {len(data)} sequences, Length: {model.sequence_length}")
        print(f"Epochs: {n_epochs}, Learning rate: {learning_rate}")
    
    history = {
        'loss': [],
        'energy_data': [],
        'energy_model': []
    }
    
    for epoch in tqdm(range(n_epochs), disable=not verbose):
        # Compute gradient
        if method == "kl_gradient":
            key, subkey = jax.random.split(key)
            gradient = estimate_kl_gradient(
                model, data, n_samples=n_samples, temperature=temperature, key=subkey
            )
        elif method == "pseudo_likelihood":
            gradient = pseudo_likelihood_gradient(model, data, temperature=temperature)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Update weights
        model.weights = model.weights + learning_rate * gradient
        
        # Compute loss (negative log-likelihood on data)
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            data_energies = model.energy(np.array(data[:100]))  # Sample for speed
            avg_energy = np.mean(data_energies)
            history['loss'].append(-avg_energy)  # Negative energy = log prob
            history['energy_data'].append(avg_energy)
            
            # Sample from model to compare
            key, subkey = jax.random.split(key)
            model_samples = model.sample(n_samples=10, n_burnin=30, key=subkey)
            from dataset import text_to_indices
            model_indices = np.array([text_to_indices(s) for s in model_samples])
            model_energies = model.energy(model_indices)
            history['energy_model'].append(np.mean(model_energies))
    
    if verbose:
        print("\nTraining complete!")
        print(f"Final data energy: {history['energy_data'][-1]:.4f}")
        print(f"Final model energy: {history['energy_model'][-1]:.4f}")
    
    return model, history


def train_with_annealing(
    model: BabelEBM,
    data: List[np.ndarray],
    n_epochs: int = 100,
    initial_lr: float = 0.1,
    final_lr: float = 0.001,
    method: str = "kl_gradient",
    verbose: bool = True
) -> Tuple[BabelEBM, dict]:
    """
    Train with learning rate annealing.
    
    Args:
        model: BabelEBM model
        data: Training sequences
        n_epochs: Number of epochs
        initial_lr: Initial learning rate
        final_lr: Final learning rate
        method: Training method
        verbose: Print progress
        
    Returns:
        Trained model and history
    """
    if verbose:
        print("Training with learning rate annealing...")
    
    history = {'loss': [], 'learning_rate': []}
    
    for epoch in tqdm(range(n_epochs), disable=not verbose):
        # Anneal learning rate
        progress = epoch / n_epochs
        lr = initial_lr * (1 - progress) + final_lr * progress
        history['learning_rate'].append(lr)
        
        # Compute gradient
        if method == "kl_gradient":
            gradient = estimate_kl_gradient(model, data, n_samples=50)
        else:
            gradient = pseudo_likelihood_gradient(model, data)
        
        # Update weights
        model.weights = model.weights + lr * gradient
        
        # Track loss
        if epoch % 10 == 0:
            data_energies = model.energy(np.array(data[:100]))
            history['loss'].append(-np.mean(data_energies))
    
    return model, history


if __name__ == "__main__":
    # Example training
    print("Generating Babel dataset...")
    data = generate_babel_dataset(n_sequences=500, length=100, seed=42)
    
    print("Initializing model...")
    from dataset import ALPHABET_SIZE
    model = BabelEBM(sequence_length=100, alphabet_size=ALPHABET_SIZE)
    
    print("Training model...")
    trained_model, history = train_ebm(
        model,
        data,
        n_epochs=50,
        learning_rate=0.01,
        method="pseudo_likelihood",  # Faster for testing
        verbose=True
    )
    
    print("\nSampling from trained model...")
    samples = trained_model.sample(n_samples=5, n_burnin=50)
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample[:80]}...")

