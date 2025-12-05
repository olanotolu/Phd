"""
Train TLM on Restaurant Booking Dialogues and Analyze Learned Patterns

This script:
1. Trains TLM on dialogue data
2. Analyzes what patterns it learned (bigrams, dialogue structure)
3. Shows the "normal conversation energy" baseline
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_radar.dialogue_data import load_dialogues, load_real_dialogues_simple
from anomaly_radar.train_dialogue_tlm import train_dialogue_tlm, prepare_dialogue_data
from babel_library import BabelEBM
from dataset import ALPHABET, ALPHABET_SIZE, indices_to_text


def analyze_learned_patterns(model: BabelEBM, dialogues: list):
    """
    Analyze what patterns the TLM learned from training.
    """
    print("\n" + "=" * 70)
    print("ANALYZING LEARNED PATTERNS")
    print("=" * 70)
    
    # 1. Analyze bigram weights
    print("\n1. Top Learned Bigrams (by average weight):")
    print("-" * 70)
    
    # Average weights across all positions
    avg_weights = np.mean(model.weights, axis=0)  # Shape: (K, K)
    
    # Find top bigrams
    top_bigrams = []
    for i in range(ALPHABET_SIZE):
        for j in range(ALPHABET_SIZE):
            weight = avg_weights[i, j]
            char1 = ALPHABET[i]
            char2 = ALPHABET[j]
            bigram = f"{char1}{char2}"
            top_bigrams.append((bigram, weight))
    
    # Sort by weight (highest = most learned/stable)
    top_bigrams.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Learned Bigrams (highest weights = most stable):")
    for i, (bigram, weight) in enumerate(top_bigrams[:20], 1):
        print(f"  {i:2d}. '{bigram}' : {weight:7.3f}")
    
    # 2. Analyze position-specific patterns
    print("\n2. Position-Specific Patterns:")
    print("-" * 70)
    
    # Look at first few positions (beginning of dialogues)
    print("\nTop bigrams at dialogue start (position 0):")
    pos0_weights = model.weights[0]  # First position
    top_start = []
    for i in range(ALPHABET_SIZE):
        for j in range(ALPHABET_SIZE):
            weight = pos0_weights[i, j]
            char1 = ALPHABET[i]
            char2 = ALPHABET[j]
            bigram = f"{char1}{char2}"
            top_start.append((bigram, weight))
    top_start.sort(key=lambda x: x[1], reverse=True)
    for i, (bigram, weight) in enumerate(top_start[:10], 1):
        print(f"  {i:2d}. '{bigram}' : {weight:7.3f}")
    
    # 3. Analyze common dialogue patterns
    print("\n3. Common Dialogue Patterns Detected:")
    print("-" * 70)
    
    # Look for common restaurant booking phrases
    common_phrases = [
        "customer",
        "agent",
        "reservation",
        "table",
        "time",
        "people",
        "confirmed",
        "pm",
    ]
    
    print("\nChecking if TLM learned common phrases:")
    for phrase in common_phrases:
        # Convert phrase to indices
        indices = [ALPHABET.index(c) if c in ALPHABET else -1 for c in phrase.lower()]
        if -1 in indices:
            continue
        
        # Compute average energy contribution for this phrase
        total_weight = 0.0
        count = 0
        for pos in range(min(len(phrase) - 1, model.weights.shape[0])):
            if pos < len(indices) - 1:
                i, j = indices[pos], indices[pos + 1]
                total_weight += model.weights[pos, i, j]
                count += 1
        
        if count > 0:
            avg_weight = total_weight / count
            print(f"  '{phrase}': avg weight = {avg_weight:7.3f} ({'learned' if avg_weight > 0 else 'penalized'})")
    
    # 4. Energy baseline
    print("\n4. Energy Baseline (Normal Conversation Energy):")
    print("-" * 70)
    
    # Sample some training dialogues and compute their energy
    sample_dialogues = dialogues[:20]
    from dataset import text_to_indices
    energies = []
    for dialogue in sample_dialogues:
        indices = text_to_indices(dialogue)
        if len(indices) < model.sequence_length:
            padding = np.full(model.sequence_length - len(indices), 26, dtype=np.int32)
            indices = np.concatenate([indices, padding])
        else:
            indices = indices[:model.sequence_length]
        
        energy = model.energy(indices.reshape(1, -1))[0]
        energies.append(energy)
    
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    
    print(f"\n  Mean energy (normal dialogues): {mean_energy:.2f}")
    print(f"  Std energy: {std_energy:.2f}")
    print(f"  Range: [{mean_energy - 2*std_energy:.2f}, {mean_energy + 2*std_energy:.2f}]")
    print(f"\n  → Dialogues with energy > {mean_energy + 2*std_energy:.2f} are likely anomalous")
    print(f"  → Dialogues with energy < {mean_energy - 2*std_energy:.2f} are very normal")
    
    return mean_energy, std_energy


def main():
    print("=" * 70)
    print("TLM Training & Pattern Analysis - Restaurant Booking Dialogues")
    print("=" * 70)
    
    # Load dialogues
    dialogue_file = "anomaly_radar/dialogues.txt"
    real_data_file = "anomaly_radar/real_dialogues.txt"
    
    if os.path.exists(real_data_file):
        print(f"\n✓ Using REAL dialogue data from {real_data_file}")
        dialogues = load_real_dialogues_simple(real_data_file)
    elif os.path.exists(dialogue_file):
        print(f"\n✓ Loading dialogues from {dialogue_file}")
        dialogues = load_dialogues(dialogue_file)
    else:
        print("\n❌ No dialogue data found!")
        print("   Please run: python -m anomaly_radar.dialogue_data")
        return
    
    print(f"✓ Loaded {len(dialogues)} dialogues")
    print(f"  Sample: {dialogues[0][:80]}...")
    
    # Train TLM
    print("\n" + "=" * 70)
    print("TRAINING TLM")
    print("=" * 70)
    
    weights_path = "anomaly_radar/dialogue_tlm_weights.npy"
    
    if os.path.exists(weights_path):
        print(f"\n✓ Found existing trained model")
        print("  Loading weights...")
        weights = np.load(weights_path)
        sequence_length = weights.shape[0] + 1
        from babel_library import BabelEBM
        model = BabelEBM(sequence_length=sequence_length, alphabet_size=ALPHABET_SIZE)
        model.weights = weights
        print("  ✓ Model loaded")
    else:
        print("\n✓ Training TLM on dialogues...")
        print("  (This may take a few minutes)")
        model, history = train_dialogue_tlm(
            dialogues,
            sequence_length=200,
            n_epochs=100,
            learning_rate=0.01,
            method="pseudo_likelihood",
            save_path=weights_path
        )
        print("\n✓ Training complete!")
    
    # Analyze learned patterns
    baseline_mean, baseline_std = analyze_learned_patterns(model, dialogues)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ TLM has learned:")
    print("  • Common bigrams in restaurant booking dialogues")
    print("  • Dialogue structure patterns")
    print("  • 'Reservation language shape'")
    print("  • Normal conversation energy baseline")
    print(f"\n✓ Baseline energy: {baseline_mean:.2f} ± {baseline_std:.2f}")
    print(f"  → Use this to detect anomalies (energy > {baseline_mean + 2*baseline_std:.2f})")
    print("\n✓ Ready for anomaly detection!")
    print("=" * 70)


if __name__ == "__main__":
    main()

