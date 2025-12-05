"""
Thermodynamic Language Model (TLM) - Babel Engine v0.1

Quick example script demonstrating the TLM/Babel Engine project.

This is a minimal example that shows the full workflow:
1. Generate random Babel data
2. Train a Thermodynamic Language Model (TLM)
3. Sample from the trained model using Gibbs sampling
4. Analyze emergent patterns

Tagline: "From chaos to structure."
"""

from dataset import generate_babel_dataset, generate_babel_strings, ALPHABET_SIZE
from babel_library import BabelEBM
from training import train_ebm
from visualization import analyze_patterns, visualize_character_frequencies

def main():
    print("=" * 60)
    print("Thermodynamic Language Model (TLM) - Babel Engine v0.1")
    print("From chaos to structure.")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating random Babel sequences...")
    data = generate_babel_dataset(n_sequences=500, length=100, seed=42)
    data_strings = generate_babel_strings(n_sequences=500, length=100, seed=42)
    print(f"   Generated {len(data)} sequences of length 100")
    
    # 2. Initialize model
    print("\n2. Initializing EBM model...")
    model = BabelEBM(sequence_length=100, alphabet_size=ALPHABET_SIZE, init_scale=0.1)
    print(f"   Model initialized with {model.sequence_length} positions, alphabet size: {model.alphabet_size}")
    
    # 3. Train model
    print("\n3. Training model (this may take a minute)...")
    trained_model, history = train_ebm(
        model,
        data,
        n_epochs=50,
        learning_rate=0.01,
        method="pseudo_likelihood",  # Faster than kl_gradient
        verbose=True
    )
    print("   Training complete!")
    
    # 4. Sample from trained model
    print("\n4. Sampling from trained model...")
    samples = trained_model.sample(n_samples=10, n_burnin=50, temperature=1.0)
    print(f"   Generated {len(samples)} samples")
    
    # 5. Analyze patterns
    print("\n5. Analyzing patterns...")
    print("\n   Random data patterns:")
    analyze_patterns(data_strings[:50], verbose=True)
    
    print("\n   Model sample patterns:")
    analyze_patterns(samples, verbose=True)
    
    # 6. Show samples
    print("\n6. Sample sequences from trained model:")
    print("   " + "-" * 56)
    for i, sample in enumerate(samples[:5], 1):
        print(f"   {i}. {sample[:70]}...")
    
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Open babel_exploration.ipynb for interactive exploration")
    print("  - Try different training methods (kl_gradient vs pseudo_likelihood)")
    print("  - Experiment with different temperatures and sequence lengths")
    print("  - Explore the visualization functions in visualization.py")

if __name__ == "__main__":
    main()

