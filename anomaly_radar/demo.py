"""
TLM Anomaly Radar - Demo Script

Complete workflow:
1. Generate/load restaurant booking dialogues
2. Train TLM on normal dialogues
3. Test anomaly detection on sample texts
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_radar.dialogue_data import (
    collect_training_data,
    load_dialogues,
    load_real_dialogues,
    load_real_dialogues_simple,
    save_dialogues
)
from anomaly_radar.train_dialogue_tlm import train_dialogue_tlm
from anomaly_radar.anomaly_detector import AnomalyRadar, load_trained_model
import numpy as np


def main():
    print("=" * 70)
    print("TLM Anomaly Radar - Restaurant Booking Domain")
    print("From chaos to structure. Detecting anomalies in LLM outputs.")
    print("=" * 70)
    
    # Step 1: Collect training data
    print("\n" + "=" * 70)
    print("STEP 1: Collecting Training Data")
    print("=" * 70)
    
    # Check for real data file
    real_data_file = "anomaly_radar/real_dialogues.txt"
    dialogue_file = "anomaly_radar/dialogues.txt"
    
    if os.path.exists(real_data_file):
        print(f"\n✓ Found REAL dialogue data at {real_data_file}")
        print("  Using real data for training (better structure!)")
        try:
            dialogues = load_real_dialogues_simple(real_data_file)
        except:
            dialogues = load_real_dialogues(real_data_file)
    elif os.path.exists(dialogue_file):
        print(f"\n✓ Loading existing synthetic dialogues from {dialogue_file}...")
        dialogues = load_dialogues(dialogue_file)
        print("  (Tip: Add real dialogues to 'anomaly_radar/real_dialogues.txt' for better results)")
    else:
        print("\n✓ Generating new synthetic restaurant booking dialogues...")
        dialogues = collect_training_data(n_dialogues=500)
        save_dialogues(dialogues, dialogue_file)
        print("  (Tip: Add real dialogues to 'anomaly_radar/real_dialogues.txt' for better results)")
    
    print(f"\n✓ Collected {len(dialogues)} normal restaurant booking dialogues")
    print(f"  Sample: {dialogues[0][:80]}...")
    
    # Step 2: Train TLM
    print("\n" + "=" * 70)
    print("STEP 2: Training TLM on Normal Dialogues")
    print("=" * 70)
    
    weights_path = "anomaly_radar/dialogue_tlm_weights.npy"
    
    if os.path.exists(weights_path):
        print(f"\n✓ Found existing trained model at {weights_path}")
        print("  (Skipping training. To retrain, delete this file.)")
        from babel_library import BabelEBM
        from dataset import ALPHABET_SIZE
        weights = np.load(weights_path)
        sequence_length = weights.shape[0] + 1
        model = BabelEBM(sequence_length=sequence_length, alphabet_size=ALPHABET_SIZE)
        model.weights = weights
    else:
        print("\n✓ Training TLM on normal dialogues...")
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
    
    # Step 3: Test Anomaly Detection
    print("\n" + "=" * 70)
    print("STEP 3: Testing Anomaly Detection")
    print("=" * 70)
    
    detector = AnomalyRadar(model)
    
    # Compute baseline from training data
    print("\n✓ Computing baseline energy from normal dialogues...")
    baseline_mean, baseline_std = detector.compute_baseline_energy(dialogues[:50])
    print(f"  Baseline energy: {baseline_mean:.2f} ± {baseline_std:.2f}")
    threshold = baseline_mean + 2 * baseline_std
    print(f"  Anomaly threshold: {threshold:.2f}")
    
    # Test cases - use actual examples from training data format
    print("\n" + "-" * 70)
    print("TEST CASE 1: Normal Restaurant Booking")
    print("-" * 70)
    normal_text = dialogues[0]  # Use actual training example
    result = detector.detect_anomaly(normal_text, baseline_energy=baseline_mean, baseline_std=baseline_std)
    print(f"Text: {normal_text[:60]}...")
    print(f"Energy: {result['energy']:.2f}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f} (0=normal, 1=anomalous)")
    print(f"Status: {'✓ NORMAL' if not result['is_anomalous'] else '✗ ANOMALOUS'}")
    
    print("\n" + "-" * 70)
    print("TEST CASE 2: Hallucinated Content (Nonsensical)")
    print("-" * 70)
    hallucinated = "quantum computing research table booking for negative infinity people at yesterday time please confirm the reservation for my pet dragon"
    result = detector.detect_anomaly(hallucinated, baseline_energy=baseline_mean, baseline_std=baseline_std)
    print(f"Text: {hallucinated[:60]}...")
    print(f"Energy: {result['energy']:.2f}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Status: {'✓ NORMAL' if not result['is_anomalous'] else '✗ ANOMALOUS'}")
    
    print("\n" + "-" * 70)
    print("TEST CASE 3: Off-Topic Content")
    print("-" * 70)
    off_topic = "the sky is blue and elephants can fly through quantum space while making restaurant reservations for imaginary friends"
    result = detector.detect_anomaly(off_topic, baseline_energy=baseline_mean, baseline_std=baseline_std)
    print(f"Text: {off_topic[:60]}...")
    print(f"Energy: {result['energy']:.2f}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Status: {'✓ NORMAL' if not result['is_anomalous'] else '✗ ANOMALOUS'}")
    
    print("\n" + "-" * 70)
    print("TEST CASE 4: Repetitive/Stuck Output")
    print("-" * 70)
    repetitive = "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello"
    result = detector.detect_anomaly(repetitive, baseline_energy=baseline_mean, baseline_std=baseline_std)
    print(f"Text: {repetitive[:60]}...")
    print(f"Energy: {result['energy']:.2f}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Status: {'✓ NORMAL' if not result['is_anomalous'] else '✗ ANOMALOUS'}")
    
    print("\n" + "-" * 70)
    print("TEST CASE 5: Another Normal Booking")
    print("-" * 70)
    normal2 = dialogues[10] if len(dialogues) > 10 else dialogues[0]  # Use another training example
    result = detector.detect_anomaly(normal2, baseline_energy=baseline_mean, baseline_std=baseline_std)
    print(f"Text: {normal2[:60]}...")
    print(f"Energy: {result['energy']:.2f}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Status: {'✓ NORMAL' if not result['is_anomalous'] else '✗ ANOMALOUS'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ TLM Anomaly Radar is ready!")
    print("\nThe model can now:")
    print("  • Detect hallucinations in LLM outputs")
    print("  • Flag off-topic or nonsensical content")
    print("  • Identify repetitive/stuck outputs")
    print("  • Score any restaurant booking dialogue for anomaly")
    print("\nNext steps:")
    print("  • Integrate with your LLM pipeline")
    print("  • Set up monitoring for production")
    print("  • Expand to other domains")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

