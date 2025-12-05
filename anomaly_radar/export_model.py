"""
Export TLM Model to Pickle

Saves the trained model in a portable format.
"""

import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from babel_library import BabelEBM
from dataset import ALPHABET_SIZE


def export_model(weights_path="anomaly_radar/dialogue_tlm_weights.npy", 
                 output_path="anomaly_radar/tlm_model.pkl"):
    """Export trained model to pickle."""
    print("=" * 70)
    print("Exporting TLM Model to Pickle")
    print("=" * 70)
    
    # Load weights
    weights = np.load(weights_path)
    sequence_length = weights.shape[0] + 1
    
    # Create model
    model = BabelEBM(sequence_length=sequence_length, alphabet_size=ALPHABET_SIZE)
    model.weights = weights
    
    # Save model
    model_data = {
        "weights": weights,
        "sequence_length": sequence_length,
        "alphabet_size": ALPHABET_SIZE,
        "model_type": "BabelEBM"
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"\n✓ Model exported to {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Alphabet size: {ALPHABET_SIZE}")
    print(f"  Weight matrix shape: {weights.shape}")
    
    # Verify load
    with open(output_path, "rb") as f:
        loaded = pickle.load(f)
    print(f"\n✓ Verification: Model loads successfully")
    
    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)


if __name__ == "__main__":
    export_model()

