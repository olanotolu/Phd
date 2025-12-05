"""
Visualization utilities for pattern discovery in the Library of Babel.

Helps visualize:
- Emergent patterns in sampled sequences
- Bigram structure learned by the model
- Character clustering
- Pattern emergence over training
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from babel_library import BabelEBM
from dataset import ALPHABET, ALPHABET_SIZE


def visualize_bigram_weights(
    model: BabelEBM,
    position: int = 0,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Visualize learned bigram weights as a heatmap.
    
    Args:
        model: Trained BabelEBM
        position: Which position in sequence to visualize
        figsize: Figure size
        save_path: Optional path to save figure
    """
    weights = model.weights[position]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        weights,
        xticklabels=ALPHABET,
        yticklabels=ALPHABET,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Weight'}
    )
    plt.title(f'Learned Bigram Weights at Position {position}')
    plt.xlabel('Next Character')
    plt.ylabel('Previous Character')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def visualize_character_frequencies(
    sequences: List[str],
    title: str = "Character Frequencies",
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Visualize character frequency distribution in sequences.
    
    Args:
        sequences: List of text sequences
        title: Plot title
        figsize: Figure size
    """
    all_chars = ''.join(sequences)
    char_counts = {char: all_chars.count(char) for char in ALPHABET}
    
    chars = list(char_counts.keys())
    counts = list(char_counts.values())
    
    plt.figure(figsize=figsize)
    plt.bar(chars, counts, color='steelblue', alpha=0.7)
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_pattern_emergence(
    sequences: List[str],
    window_size: int = 10,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Visualize repeating patterns in sequences.
    Shows n-gram frequencies to detect emergent structure.
    
    Args:
        sequences: List of text sequences
        window_size: Size of n-grams to analyze
        figsize: Figure size
    """
    # Extract n-grams
    ngrams = {}
    for seq in sequences:
        for i in range(len(seq) - window_size + 1):
            ngram = seq[i:i+window_size]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
    
    # Get top n-grams
    top_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)[:20]
    
    if not top_ngrams:
        print("No patterns found.")
        return
    
    ngram_texts = [ng[0] for ng in top_ngrams]
    counts = [ng[1] for ng in top_ngrams]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(ngram_texts)), counts, color='coral', alpha=0.7)
    plt.yticks(range(len(ngram_texts)), ngram_texts)
    plt.xlabel('Frequency')
    plt.title(f'Top {len(top_ngrams)} Repeating Patterns (n-gram size: {window_size})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def visualize_sequence_comparison(
    original: List[str],
    sampled: List[str],
    n_examples: int = 5,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Compare original random sequences with model samples.
    
    Args:
        original: Original random Babel sequences
        sampled: Sequences sampled from trained model
        n_examples: Number of examples to show
        figsize: Figure size
    """
    fig, axes = plt.subplots(n_examples, 2, figsize=figsize)
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(n_examples, len(original), len(sampled))):
        # Original
        axes[i, 0].text(0.1, 0.5, original[i][:100], fontsize=10, family='monospace')
        axes[i, 0].set_title(f'Original Random Sequence {i+1}')
        axes[i, 0].axis('off')
        
        # Sampled
        axes[i, 1].text(0.1, 0.5, sampled[i][:100], fontsize=10, family='monospace')
        axes[i, 1].set_title(f'Model Sample {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_training_history(
    history: Dict,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Visualize training history (loss, energies).
    
    Args:
        history: Training history dictionary
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    if 'loss' in history and history['loss']:
        axes[0].plot(history['loss'], 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch (x10)')
        axes[0].set_ylabel('Negative Log-Likelihood')
        axes[0].set_title('Training Loss')
        axes[0].grid(alpha=0.3)
    
    # Energies
    if 'energy_data' in history and 'energy_model' in history:
        epochs = range(len(history['energy_data']))
        axes[1].plot(epochs, history['energy_data'], 'g-', label='Data Energy', linewidth=2)
        axes[1].plot(epochs, history['energy_model'], 'r-', label='Model Energy', linewidth=2)
        axes[1].set_xlabel('Epoch (x10)')
        axes[1].set_ylabel('Energy')
        axes[1].set_title('Energy Comparison')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_patterns(
    sequences: List[str],
    verbose: bool = True
) -> Dict:
    """
    Analyze patterns in sequences and return statistics.
    
    Args:
        sequences: List of text sequences
        verbose: Print analysis
        
    Returns:
        Dictionary with pattern statistics
    """
    all_text = ''.join(sequences)
    
    stats = {
        'total_chars': len(all_text),
        'char_frequencies': {char: all_text.count(char) for char in ALPHABET},
        'vowel_ratio': sum(all_text.count(v) for v in 'aeiou') / len(all_text) if all_text else 0,
        'space_ratio': all_text.count(' ') / len(all_text) if all_text else 0,
        'punctuation_ratio': sum(all_text.count(p) for p in '.,') / len(all_text) if all_text else 0,
    }
    
    # Find most common bigrams
    bigrams = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigram = seq[i:i+2]
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
    
    stats['top_bigrams'] = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Find repeating substrings
    repeating = {}
    for seq in sequences:
        for length in [3, 4, 5]:
            for i in range(len(seq) - length + 1):
                substr = seq[i:i+length]
                if substr.count(substr[0]) >= length - 1:  # Mostly repeating
                    repeating[substr] = repeating.get(substr, 0) + 1
    
    stats['repeating_patterns'] = sorted(repeating.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if verbose:
        print("=== Pattern Analysis ===")
        print(f"Total characters: {stats['total_chars']}")
        print(f"Vowel ratio: {stats['vowel_ratio']:.3f}")
        print(f"Space ratio: {stats['space_ratio']:.3f}")
        print(f"Punctuation ratio: {stats['punctuation_ratio']:.3f}")
        print("\nTop 10 Bigrams:")
        for bigram, count in stats['top_bigrams']:
            print(f"  '{bigram}': {count}")
        print("\nRepeating Patterns:")
        for pattern, count in stats['repeating_patterns']:
            print(f"  '{pattern}': {count}")
    
    return stats


def create_babel_map(
    sequences: List[str],
    model: BabelEBM,
    save_path: Optional[str] = None
):
    """
    Create a "Library of Babel Map" showing emergent domains.
    Visualizes character clustering and pattern formation.
    
    Args:
        sequences: Sampled sequences
        model: Trained model
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Character frequency comparison
    from dataset import generate_babel_strings
    random_seqs = generate_babel_strings(n_sequences=len(sequences), length=len(sequences[0]))
    
    all_random = ''.join(random_seqs)
    all_sampled = ''.join(sequences)
    
    random_counts = [all_random.count(c) for c in ALPHABET]
    sampled_counts = [all_sampled.count(c) for c in ALPHABET]
    
    x = np.arange(len(ALPHABET))
    width = 0.35
    axes[0, 0].bar(x - width/2, random_counts, width, label='Random', alpha=0.7)
    axes[0, 0].bar(x + width/2, sampled_counts, width, label='Model Samples', alpha=0.7)
    axes[0, 0].set_xlabel('Character')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Character Frequency: Random vs Model')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(ALPHABET, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Bigram heatmap (average across positions)
    avg_weights = np.mean(model.weights, axis=0)
    im = axes[0, 1].imshow(avg_weights, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_xlabel('Next Character')
    axes[0, 1].set_ylabel('Previous Character')
    axes[0, 1].set_title('Average Bigram Weights')
    axes[0, 1].set_xticks(range(len(ALPHABET)))
    axes[0, 1].set_xticks(range(len(ALPHABET)))
    axes[0, 1].set_xticklabels(ALPHABET, rotation=45)
    axes[0, 1].set_yticklabels(ALPHABET)
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Pattern emergence (top n-grams)
    ngrams = {}
    for seq in sequences:
        for i in range(len(seq) - 5):
            ngram = seq[i:i+5]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
    
    top_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)[:15]
    if top_ngrams:
        ngram_texts = [ng[0] for ng in top_ngrams]
        counts = [ng[1] for ng in top_ngrams]
        axes[1, 0].barh(range(len(ngram_texts)), counts, color='coral', alpha=0.7)
        axes[1, 0].set_yticks(range(len(ngram_texts)))
        axes[1, 0].set_yticklabels(ngram_texts, fontsize=8)
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_title('Top Emergent Patterns (5-grams)')
        axes[1, 0].invert_yaxis()
    
    # 4. Sample sequences display
    axes[1, 1].axis('off')
    sample_text = "\n\n".join([f"Sample {i+1}: {seq[:80]}..." for i, seq in enumerate(sequences[:5])])
    axes[1, 1].text(0.05, 0.95, sample_text, transform=axes[1, 1].transAxes,
                    fontsize=9, family='monospace', verticalalignment='top')
    axes[1, 1].set_title('Sample Sequences from Model')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities ready!")
    print("Import this module and use functions like:")
    print("  - visualize_bigram_weights(model)")
    print("  - analyze_patterns(sequences)")
    print("  - create_babel_map(sequences, model)")

