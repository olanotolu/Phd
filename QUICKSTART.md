# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running the Example

### Option 1: Quick Example Script
```bash
python example.py
```

This runs a complete workflow:
- Generates random Babel data
- Trains an EBM model
- Samples from the trained model
- Analyzes patterns

### Option 2: Interactive Jupyter Notebook
```bash
jupyter notebook babel_exploration.ipynb
```

This provides an interactive exploration with:
- Step-by-step workflow
- Visualizations
- Pattern analysis
- Discussion prompts

### Option 3: Custom Script

```python
from dataset import generate_babel_dataset
from babel_library import BabelEBM
from training import train_ebm

# Generate data
data = generate_babel_dataset(n_sequences=1000, length=200)

# Create model
from dataset import ALPHABET_SIZE
model = BabelEBM(sequence_length=200, alphabet_size=ALPHABET_SIZE)

# Train
trained_model, history = train_ebm(
    model, data, n_epochs=100, method="pseudo_likelihood"
)

# Sample
samples = trained_model.sample(n_samples=10, n_burnin=100)
print(samples[0])  # First sample
```

## Project Structure

```
Phd/
├── dataset.py              # Babel string generation
├── babel_library.py        # Core EBM implementation (THRML-compatible)
├── training.py             # Training with KL-gradient estimation
├── visualization.py        # Pattern discovery visualizations
├── babel_exploration.ipynb # Interactive Jupyter notebook
├── example.py              # Quick example script
├── file.py                 # Original babel_string function
├── requirements.txt        # Dependencies
└── README.md              # Full documentation
```

## Key Components

### 1. Dataset Generation (`dataset.py`)
- `generate_babel_dataset()`: Generate index arrays
- `generate_babel_strings()`: Generate text strings
- Alphabet: `{a-z, space, period, comma}` (28 characters)

### 2. EBM Model (`babel_library.py`)
- `BabelEBM`: Main model class
- `CategoricalNode`: Character representation
- `CategoricalEBMFactor`: Bigram interactions
- `BlockGibbsSpec`: Block Gibbs sampling
- `SamplingSchedule`: Temperature annealing

### 3. Training (`training.py`)
- `train_ebm()`: Main training function
- `estimate_kl_gradient()`: THRML-style moment matching
- `pseudo_likelihood_gradient()`: Faster alternative

### 4. Visualization (`visualization.py`)
- `visualize_bigram_weights()`: Learned transitions
- `analyze_patterns()`: Pattern statistics
- `create_babel_map()`: Comprehensive visualization

## Training Methods

### Pseudo-Likelihood (Fast)
```python
trained_model, history = train_ebm(
    model, data, method="pseudo_likelihood"
)
```

### KL-Gradient (THRML-style, Slower)
```python
trained_model, history = train_ebm(
    model, data, method="kl_gradient", n_samples=100
)
```

## Common Issues

### JAX Installation
If JAX installation fails, try:
```bash
pip install --upgrade pip
pip install jax jaxlib
```

### Memory Issues
For longer sequences, reduce:
- `n_sequences` in dataset generation
- `n_samples` in training
- `sequence_length` in model

## Next Steps

1. **Explore the notebook**: Run `babel_exploration.ipynb` for full analysis
2. **Experiment with parameters**: Try different temperatures, learning rates
3. **Extend the model**: Add trigram interactions, longer sequences
4. **Analyze results**: What patterns emerge? Do they have meaning?

## Research Questions

- What patterns emerge from pure noise?
- Does the model "invent" meaningful clusters?
- How does this connect to Borges's Library of Babel?
- What does structure-from-noise tell us about meaning?

## Citation

If you use this project in your research, consider citing:

```
Sampling the Library of Babel: Can a probabilistic computer find meaning in chaos?
A computational exploration using Energy-Based Models and Extropic-style THRML framework.
```

