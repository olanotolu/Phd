# Getting Started with TLM / Babel Engine

## Installation

```bash
# Clone the repository
git clone https://github.com/yourname/tlm-babel.git
cd tlm-babel

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Example

```bash
python example.py
```

This will:
- Generate random Babel sequences
- Train a TLM model
- Sample from the trained model
- Analyze emergent patterns

### 2. Interactive Exploration

Open the Jupyter notebook:

```bash
jupyter notebook babel_exploration.ipynb
```

This provides:
- Step-by-step workflow
- Visualizations
- Pattern analysis
- Discussion prompts

### 3. Custom Experiment

```python
from dataset import generate_babel_dataset, ALPHABET_SIZE
from babel_library import BabelEBM
from training import train_ebm

# Generate data
data = generate_babel_dataset(n_sequences=1000, length=200)

# Create model
model = BabelEBM(sequence_length=200, alphabet_size=ALPHABET_SIZE)

# Train
trained_model, history = train_ebm(
    model, data, n_epochs=100, method="pseudo_likelihood"
)

# Sample
samples = trained_model.sample(n_samples=10, n_burnin=100)
print(samples[0])  # First sample
```

## Understanding the Output

### Training Output

- **Epoch progress**: Shows training iterations
- **Data energy**: Average energy of training sequences
- **Model energy**: Average energy of sampled sequences
- **Loss**: Negative log-likelihood (lower is better)

### Pattern Analysis

The analyzer shows:
- **Character frequencies**: Distribution of characters
- **Vowel/space/punctuation ratios**: Structural patterns
- **Top bigrams**: Most common character pairs
- **Repeating patterns**: Emergent motifs

### What to Look For

After training, you should see:
- Model samples have different patterns than random data
- Emergent repetition in model samples
- Stabilized bigram frequencies
- Lower energy in model samples vs random data

## Next Steps

1. **Experiment with parameters**:
   - Different sequence lengths
   - Different temperatures
   - Different learning rates

2. **Try different training methods**:
   - `pseudo_likelihood` (faster)
   - `kl_gradient` (more accurate, slower)

3. **Explore visualizations**:
   - Bigram heatmaps
   - Pattern emergence plots
   - Energy comparisons

4. **Read the architecture**:
   - See `ARCHITECTURE.md` for technical details
   - Understand the energy function
   - Learn about Gibbs sampling

## Common Issues

### JAX Installation

If JAX fails to install:

```bash
pip install --upgrade pip
pip install jax jaxlib
```

### Memory Issues

For longer sequences, reduce:
- `n_sequences` in dataset generation
- `n_samples` in training
- `sequence_length` in model

### Slow Training

- Use `method="pseudo_likelihood"` for faster training
- Reduce `n_epochs` for quick experiments
- Use smaller datasets for testing

## Resources

- **Architecture**: See `ARCHITECTURE.md`
- **Branding**: See `BRANDING.md`
- **Citation**: See `CITATION.md`
- **Contributing**: See `CONTRIBUTING.md`

## Questions?

Open an issue on GitHub or check the documentation.

