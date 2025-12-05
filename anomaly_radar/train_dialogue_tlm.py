"""
Train TLM on Restaurant Booking Dialogues

This trains a TLM to learn "normal" restaurant booking patterns.
After training, the TLM can detect anomalies (weird, hallucinated, or unsafe outputs).
"""

import numpy as np
import sys
import os
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from babel_library import BabelEBM
from training import train_ebm
from dataset import text_to_indices, ALPHABET_SIZE
from anomaly_radar.dialogue_data import (
    collect_training_data,
    load_dialogues,
    load_real_dialogues,
    load_real_dialogues_simple,
    format_dialogue_for_tlm
)


def prepare_dialogue_data(dialogues: List[str], max_length: int = 200) -> List[np.ndarray]:
    """
    Prepare dialogue data for TLM training.
    
    Args:
        dialogues: List of dialogue strings
        max_length: Maximum sequence length (pad or truncate)
        
    Returns:
        List of index arrays
    """
    data = []
    
    for dialogue in dialogues:
        # Format dialogue
        formatted = format_dialogue_for_tlm(dialogue)
        
        # Convert to indices
        indices = text_to_indices(formatted)
        
        # Pad or truncate to max_length
        if len(indices) < max_length:
            # Pad with spaces (index 26)
            padding = np.full(max_length - len(indices), 26, dtype=np.int32)
            indices = np.concatenate([indices, padding])
        else:
            # Truncate
            indices = indices[:max_length]
        
        data.append(indices)
    
    return data


def train_dialogue_tlm(
    dialogues: List[str],
    sequence_length: int = 200,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    method: str = "pseudo_likelihood",
    save_path: str = "anomaly_radar/dialogue_tlm_weights.npy"
) -> BabelEBM:
    """
    Train a TLM on restaurant booking dialogues.
    
    Args:
        dialogues: List of dialogue strings
        sequence_length: Length of sequences for TLM
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        method: Training method ("pseudo_likelihood" or "kl_gradient")
        save_path: Path to save trained weights
        
    Returns:
        Trained TLM model
    """
    print("=" * 60)
    print("Training TLM on Restaurant Booking Dialogues")
    print("=" * 60)
    
    # Prepare data
    print("\n1. Preparing dialogue data...")
    data = prepare_dialogue_data(dialogues, max_length=sequence_length)
    print(f"   Prepared {len(data)} sequences of length {sequence_length}")
    
    # Initialize model
    print("\n2. Initializing TLM model...")
    model = BabelEBM(
        sequence_length=sequence_length,
        alphabet_size=ALPHABET_SIZE,
        init_scale=0.1
    )
    print(f"   Model initialized: {sequence_length} positions, {ALPHABET_SIZE} characters")
    
    # Train
    print("\n3. Training TLM...")
    trained_model, history = train_ebm(
        model,
        data,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        method=method,
        verbose=True
    )
    
    # Save weights
    print(f"\n4. Saving trained model...")
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    np.save(save_path, trained_model.weights)
    print(f"   Saved weights to {save_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return trained_model, history


if __name__ == "__main__":
    import sys
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TLM on restaurant booking dialogues")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to real dialogue data file (if not provided, uses synthetic data)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force use of synthetic data (even if real data file exists)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=200,
        help="Sequence length for TLM"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pseudo_likelihood",
        choices=["pseudo_likelihood", "kl_gradient"],
        help="Training method"
    )
    
    args = parser.parse_args()
    
    # Load data
    if args.data and not args.synthetic:
        # Use real data
        print(f"Using REAL dialogue data from {args.data}")
        if os.path.exists(args.data):
            # Try simple format first (one per line)
            try:
                dialogues = load_real_dialogues_simple(args.data)
            except:
                # Fall back to multi-line format
                dialogues = load_real_dialogues(args.data)
        else:
            print(f"Warning: File {args.data} not found. Using synthetic data.")
            dialogue_file = "anomaly_radar/dialogues.txt"
            if os.path.exists(dialogue_file):
                dialogues = load_dialogues(dialogue_file)
            else:
                dialogues = collect_training_data(n_dialogues=500)
                from anomaly_radar.dialogue_data import save_dialogues
                save_dialogues(dialogues, dialogue_file)
    else:
        # Use synthetic data
        print("Using SYNTHETIC dialogue data")
        dialogue_file = "anomaly_radar/dialogues.txt"
        if os.path.exists(dialogue_file):
            dialogues = load_dialogues(dialogue_file)
        else:
            dialogues = collect_training_data(n_dialogues=500)
            from anomaly_radar.dialogue_data import save_dialogues
            save_dialogues(dialogues, dialogue_file)
    
    # Train TLM
    trained_model, history = train_dialogue_tlm(
        dialogues,
        sequence_length=args.sequence_length,
        n_epochs=args.epochs,
        learning_rate=0.01,
        method=args.method
    )
    
    print("\n" + "=" * 60)
    print("TLM is now trained on restaurant booking dialogues!")
    print("Next step: Use this model to detect anomalies in LLM outputs!")
    print("=" * 60)

