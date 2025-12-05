"""
Example: Training TLM on Real Restaurant Booking Dialogues

This shows how to use your actual dialogue data instead of synthetic data.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_radar.dialogue_data import load_real_dialogues_simple
from anomaly_radar.train_dialogue_tlm import train_dialogue_tlm


def main():
    print("=" * 70)
    print("Training TLM on REAL Restaurant Booking Dialogues")
    print("=" * 70)
    
    # Path to your real dialogue data
    real_data_file = "anomaly_radar/real_dialogues.txt"
    
    # Check if file exists
    if not os.path.exists(real_data_file):
        print(f"\n❌ File not found: {real_data_file}")
        print("\nTo use real data:")
        print("1. Create a file: anomaly_radar/real_dialogues.txt")
        print("2. Add your dialogues (one per line):")
        print("   customer: hi i'd like to make a reservation agent: sure what time customer: 7pm agent: how many people customer: two agent: your reservation is confirmed")
        print("   customer: hello can i book a table agent: of course what time customer: 8pm agent: how many guests customer: four agent: your table is booked")
        print("\n3. Run this script again")
        return
    
    # Load real dialogues
    print(f"\n✓ Loading real dialogues from {real_data_file}...")
    dialogues = load_real_dialogues_simple(real_data_file)
    
    if len(dialogues) == 0:
        print("❌ No dialogues found in file. Please add some dialogues.")
        return
    
    print(f"✓ Loaded {len(dialogues)} real dialogues")
    print(f"  Sample: {dialogues[0][:80]}...")
    
    # Train TLM
    print("\n" + "=" * 70)
    print("Training TLM on Real Data")
    print("=" * 70)
    
    trained_model, history = train_dialogue_tlm(
        dialogues,
        sequence_length=200,
        n_epochs=100,
        learning_rate=0.01,
        method="pseudo_likelihood",
        save_path="anomaly_radar/dialogue_tlm_weights.npy"
    )
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    print("\nYour TLM is now trained on REAL restaurant booking dialogues.")
    print("This will give you better anomaly detection than synthetic data!")
    print("\nNext: Use the trained model for anomaly detection:")
    print("  python -m anomaly_radar.anomaly_detector")


if __name__ == "__main__":
    main()

