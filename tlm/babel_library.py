"""
Core EBM implementation for the Library of Babel.

Implements THRML-compatible Energy-Based Model with:
- CategoricalNode representation
- CategoricalEBMFactor for bigram interactions
- BlockGibbsSpec for sampling
- Energy functions modeling text structure
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional
from dataclasses import dataclass
from dataset import ALPHABET, ALPHABET_SIZE, CHAR_TO_IDX, IDX_TO_CHAR, indices_to_text


@dataclass
class CategoricalNode:
    """THRML-style categorical node representing a single character."""
    n_categories: int
    value: Optional[int] = None


@dataclass
class Block:
    """Block of nodes for block Gibbs sampling."""
    nodes: List[CategoricalNode]
    indices: List[int]
    
    def __init__(self, indices: List[int], n_categories: int):
        self.indices = indices
        self.nodes = [CategoricalNode(n_categories) for _ in indices]
        self.n_categories = n_categories


class CategoricalEBMFactor:
    """
    Energy-based factor for categorical variables.
    Models bigram interactions between adjacent characters.
    """
    
    def __init__(self, blocks: List[Block], weights: np.ndarray):
        """
        Args:
            blocks: List of blocks (typically [prev_chars, next_chars])
            weights: Weight matrix of shape (n_positions, n_categories, n_categories)
        """
        self.blocks = blocks
        self.weights = jnp.array(weights)
        self.n_positions = weights.shape[0]
    
    def energy(self, states: jnp.ndarray) -> jnp.ndarray:
        """
        Compute energy for given states.
        
        Args:
            states: Array of shape (batch_size, sequence_length) with category indices
            
        Returns:
            Energy values of shape (batch_size,)
        """
        batch_size = states.shape[0]
        energies = jnp.zeros(batch_size)
        
        # Bigram energy: sum over adjacent pairs
        for i in range(self.n_positions):
            prev_chars = states[:, i]
            next_chars = states[:, i + 1]
            
            # Extract energy from weight matrix
            energy_contrib = self.weights[i, prev_chars, next_chars]
            energies = energies + energy_contrib
        
        return -energies  # Negative for log-probability interpretation
    
    def log_prob(self, states: jnp.ndarray) -> jnp.ndarray:
        """Compute log probability (negative energy)."""
        return -self.energy(states)


class CategoricalGibbsConditional:
    """
    Gibbs sampling conditional for categorical variables.
    Uses softmax to sample from conditional distribution.
    """
    
    def __init__(self, n_categories: int, temperature: float = 1.0):
        self.n_categories = n_categories
        self.temperature = temperature
    
    def sample(self, key: jax.random.PRNGKey, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Sample from categorical distribution.
        
        Args:
            key: JAX random key
            logits: Logits of shape (batch_size, n_categories)
            
        Returns:
            Sampled indices of shape (batch_size,)
        """
        logits = logits / self.temperature
        return jax.random.categorical(key, logits, axis=-1)


@dataclass
class BlockGibbsSpec:
    """
    Specification for block Gibbs sampling.
    Defines which blocks to update and in what order.
    """
    update_blocks: List[Block]
    clamped_blocks: List[Block]
    
    def __init__(self, update_blocks: List[Block], clamped_blocks: List[Block] = None):
        self.update_blocks = update_blocks
        self.clamped_blocks = clamped_blocks or []


@dataclass
class SamplingSchedule:
    """Schedule for Gibbs sampling with temperature annealing."""
    n_burnin: int
    n_samples: int
    n_steps_per_sample: int
    initial_temperature: float = 2.0
    final_temperature: float = 1.0
    
    def get_temperature(self, step: int) -> float:
        """Get temperature for current step (linear annealing)."""
        total_steps = self.n_burnin + self.n_samples * self.n_steps_per_sample
        if step >= total_steps:
            return self.final_temperature
        
        progress = step / total_steps
        return self.initial_temperature * (1 - progress) + self.final_temperature * progress


class BabelEBM:
    """
    Main EBM model for the Library of Babel.
    Learns structure in random text through energy-based modeling.
    """
    
    def __init__(
        self,
        sequence_length: int = 200,
        alphabet_size: int = ALPHABET_SIZE,
        n_bigram_factors: int = None,
        init_scale: float = 0.1
    ):
        """
        Initialize the Babel EBM.
        
        Args:
            sequence_length: Length of text sequences
            alphabet_size: Number of characters in alphabet
            n_bigram_factors: Number of bigram factors (default: sequence_length - 1)
            init_scale: Scale for weight initialization
        """
        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.alphabet = ALPHABET
        
        if n_bigram_factors is None:
            n_bigram_factors = sequence_length - 1
        
        # Initialize bigram weight matrix
        # Shape: (n_positions, alphabet_size, alphabet_size)
        self.weights = np.random.randn(
            n_bigram_factors, alphabet_size, alphabet_size
        ) * init_scale
        
        # Create blocks for block Gibbs sampling
        # Alternating pattern: even indices, odd indices
        even_indices = list(range(0, sequence_length, 2))
        odd_indices = list(range(1, sequence_length, 2))
        
        self.block_even = Block(even_indices, alphabet_size)
        self.block_odd = Block(odd_indices, alphabet_size)
        
        # Create factor (weights will be converted to JAX in factor)
        self.factor = CategoricalEBMFactor(
            [self.block_even, self.block_odd],
            self.weights
        )
        
        # Sampling spec
        self.sampling_spec = BlockGibbsSpec(
            update_blocks=[self.block_even, self.block_odd],
            clamped_blocks=[]
        )
    
    def energy(self, states: np.ndarray) -> np.ndarray:
        """Compute energy for given states."""
        states_jax = jnp.array(states)
        return np.array(self.factor.energy(states_jax))
    
    def log_prob(self, states: np.ndarray) -> np.ndarray:
        """Compute log probability for given states."""
        states_jax = jnp.array(states)
        return np.array(self.factor.log_prob(states_jax))
    
    def sample(
        self,
        n_samples: int = 10,
        n_burnin: int = 100,
        n_steps_per_sample: int = 10,
        temperature: float = 1.0,
        key: Optional[jax.random.PRNGKey] = None
    ) -> List[str]:
        """
        Sample sequences from the model using block Gibbs sampling.
        
        Args:
            n_samples: Number of samples to generate
            n_burnin: Number of burn-in steps
            n_steps_per_sample: Steps between samples
            temperature: Sampling temperature
            key: JAX random key
            
        Returns:
            List of sampled sequences as strings
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Initialize random state
        init_state = jax.random.randint(
            key, (1, self.sequence_length), 0, self.alphabet_size
        )
        
        schedule = SamplingSchedule(
            n_burnin=n_burnin,
            n_samples=n_samples,
            n_steps_per_sample=n_steps_per_sample,
            initial_temperature=temperature,
            final_temperature=temperature
        )
        
        sampler = CategoricalGibbsConditional(
            self.alphabet_size, temperature=temperature
        )
        
        # Run block Gibbs sampling
        samples = self._block_gibbs_sample(
            key, init_state, schedule, sampler
        )
        
        # Convert to strings
        return [indices_to_text(sample[0]) for sample in samples]
    
    def _block_gibbs_sample(
        self,
        key: jax.random.PRNGKey,
        init_state: jnp.ndarray,
        schedule: SamplingSchedule,
        sampler: CategoricalGibbsConditional
    ) -> List[np.ndarray]:
        """
        Internal block Gibbs sampling implementation.
        """
        current_state = init_state
        samples = []
        step = 0
        
        # Burn-in
        for _ in range(schedule.n_burnin):
            temp = schedule.get_temperature(step)
            sampler.temperature = temp
            current_state = self._gibbs_step(key, current_state, sampler)
            step += 1
        
        # Sampling
        for _ in range(schedule.n_samples):
            for _ in range(schedule.n_steps_per_sample):
                temp = schedule.get_temperature(step)
                sampler.temperature = temp
                current_state = self._gibbs_step(key, current_state, sampler)
                step += 1
            samples.append(np.array(current_state))
        
        return samples
    
    def _gibbs_step(
        self,
        key: jax.random.PRNGKey,
        state: jnp.ndarray,
        sampler: CategoricalGibbsConditional
    ) -> jnp.ndarray:
        """
        Single Gibbs step: update alternating blocks.
        """
        new_state = state.copy()
        
        # Update even indices
        key, subkey = jax.random.split(key)
        new_state = self._update_block(
            subkey, new_state, self.block_even, self.block_odd, sampler
        )
        
        # Update odd indices
        key, subkey = jax.random.split(key)
        new_state = self._update_block(
            subkey, new_state, self.block_odd, self.block_even, sampler
        )
        
        return new_state
    
    def _update_block(
        self,
        key: jax.random.PRNGKey,
        state: jnp.ndarray,
        update_block: Block,
        context_block: Block,
        sampler: CategoricalGibbsConditional
    ) -> jnp.ndarray:
        """
        Update a single block given context from other blocks.
        """
        new_state = state.copy()
        batch_size = state.shape[0]
        
        for idx in update_block.indices:
            # Compute logits for this position based on neighbors
            logits = self._compute_logits(state, idx)
            
            # Sample new value
            key, subkey = jax.random.split(key)
            new_value = sampler.sample(subkey, logits.reshape(1, -1))
            new_state = new_state.at[0, idx].set(new_value[0])
        
        return new_state
    
    def _compute_logits(self, state: jnp.ndarray, position: int) -> jnp.ndarray:
        """
        Compute logits for a position based on neighboring positions.
        Uses the bigram weights.
        """
        weights_jax = jnp.array(self.weights)
        logits = jnp.zeros(self.alphabet_size)
        
        # Left neighbor (if exists)
        if position > 0:
            left_char = int(state[0, position - 1])
            logits = logits + weights_jax[position - 1, left_char, :]
        
        # Right neighbor (if exists)
        if position < self.sequence_length - 1:
            right_char = int(state[0, position + 1])
            logits = logits + weights_jax[position, :, right_char]
        
        return logits


if __name__ == "__main__":
    # Test the model
    print("Initializing Babel EBM...")
    model = BabelEBM(sequence_length=50, alphabet_size=ALPHABET_SIZE)
    
    print("Sampling from untrained model...")
    samples = model.sample(n_samples=3, n_burnin=50, temperature=1.0)
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample}")

