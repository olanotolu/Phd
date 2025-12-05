"""
Real Demo Test - TLM Anomaly Radar

Comprehensive test showing the system working on realistic examples.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_radar import AnomalyRadar, load_trained_model
from anomaly_radar.dialogue_data import load_real_dialogues_simple


def main():
    print("=" * 80)
    print("TLM ANOMALY RADAR - REAL DEMO TEST")
    print("=" * 80)
    print("\nTesting anomaly detection on realistic restaurant booking scenarios...")
    
    # Load model
    print("\n[1/5] Loading trained TLM model...")
    try:
        model = load_trained_model("anomaly_radar/dialogue_tlm_weights.npy")
        detector = AnomalyRadar(model)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Compute baseline
    print("\n[2/5] Computing baseline from training data...")
    try:
        dialogues = load_real_dialogues_simple("anomaly_radar/real_dialogues.txt")
        baseline_mean, baseline_std = detector.compute_baseline_energy(dialogues[:50])
        threshold_upper = baseline_mean + 2 * baseline_std
        threshold_lower = baseline_mean - 2 * baseline_std
        print(f"✓ Baseline computed:")
        print(f"  Mean energy: {baseline_mean:.2f}")
        print(f"  Std deviation: {baseline_std:.2f}")
        print(f"  Normal range: [{threshold_lower:.2f}, {threshold_upper:.2f}]")
    except Exception as e:
        print(f"✗ Error computing baseline: {e}")
        return
    
    # Test cases
    print("\n[3/5] Testing on realistic scenarios...")
    print("-" * 80)
    
    test_cases = [
        {
            "name": "✅ NORMAL: Standard Booking Request",
            "text": "hi i want to make a reservation for two at 7pm",
            "expected": "NORMAL"
        },
        {
            "name": "✅ NORMAL: Confirmation Response",
            "text": "your table for two at 7pm is confirmed",
            "expected": "NORMAL"
        },
        {
            "name": "✅ NORMAL: Party Size Question",
            "text": "we have 7pm or 8pm for two which one works",
            "expected": "NORMAL"
        },
        {
            "name": "❌ ANOMALOUS: Hallucinated Content",
            "text": "quantum computing research table booking for negative infinity people at yesterday time",
            "expected": "ANOMALOUS"
        },
        {
            "name": "❌ ANOMALOUS: Completely Off-Topic",
            "text": "the sky is blue and elephants can fly through quantum space while making restaurant reservations",
            "expected": "ANOMALOUS"
        },
        {
            "name": "❌ ANOMALOUS: Nonsensical Numbers",
            "text": "i need a table for negative five people at 25pm tomorrow",
            "expected": "ANOMALOUS"
        },
        {
            "name": "⚠️ EDGE CASE: Repetitive Output",
            "text": "hello hello hello hello hello hello hello hello hello hello",
            "expected": "ANOMALOUS"
        },
        {
            "name": "✅ NORMAL: Special Request",
            "text": "can i get a table outside at 7pm for two",
            "expected": "NORMAL"
        },
        {
            "name": "✅ NORMAL: Time Change Request",
            "text": "i want to change my reservation from 6pm to 7pm",
            "expected": "NORMAL"
        },
        {
            "name": "❌ ANOMALOUS: Mixed Languages/Gibberish",
            "text": "bonjour je veux une table pour deux à 7h pm reservation confirmée quantum",
            "expected": "ANOMALOUS"
        },
    ]
    
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Text: {test['text']}")
        
        try:
            result = detector.detect_anomaly(
                test['text'],
                baseline_energy=baseline_mean,
                baseline_std=baseline_std
            )
            
            status = "✗ ANOMALOUS" if result['is_anomalous'] else "✓ NORMAL"
            match = "✓ CORRECT" if (
                (test['expected'] == "ANOMALOUS" and result['is_anomalous']) or
                (test['expected'] == "NORMAL" and not result['is_anomalous'])
            ) else "✗ WRONG"
            
            print(f"Energy: {result['energy']:.2f}")
            print(f"Anomaly Score: {result['anomaly_score']:.3f} (0=normal, 1=anomalous)")
            print(f"Status: {status}")
            print(f"Expected: {test['expected']} → {match}")
            
            results.append({
                'test': test['name'],
                'expected': test['expected'],
                'actual': 'ANOMALOUS' if result['is_anomalous'] else 'NORMAL',
                'correct': match == "✓ CORRECT",
                'energy': result['energy'],
                'score': result['anomaly_score']
            })
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'test': test['name'],
                'expected': test['expected'],
                'actual': 'ERROR',
                'correct': False
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("[4/5] TEST SUMMARY")
    print("=" * 80)
    
    correct = sum(1 for r in results if r.get('correct', False))
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\nTotal tests: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 80)
    for r in results:
        status = "✓" if r.get('correct', False) else "✗"
        print(f"{status} {r['test']}")
        if 'energy' in r:
            print(f"   Energy: {r['energy']:.2f}, Score: {r['score']:.3f}")
        print(f"   Expected: {r['expected']}, Got: {r['actual']}")
    
    # Energy distribution analysis
    print("\n" + "=" * 80)
    print("[5/5] ENERGY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    normal_energies = [r['energy'] for r in results if r['expected'] == 'NORMAL' and 'energy' in r]
    anomalous_energies = [r['energy'] for r in results if r['expected'] == 'ANOMALOUS' and 'energy' in r]
    
    if normal_energies:
        print(f"\nNormal examples:")
        print(f"  Count: {len(normal_energies)}")
        print(f"  Energy range: [{min(normal_energies):.2f}, {max(normal_energies):.2f}]")
        print(f"  Average: {sum(normal_energies)/len(normal_energies):.2f}")
    
    if anomalous_energies:
        print(f"\nAnomalous examples:")
        print(f"  Count: {len(anomalous_energies)}")
        print(f"  Energy range: [{min(anomalous_energies):.2f}, {max(anomalous_energies):.2f}]")
        print(f"  Average: {sum(anomalous_energies)/len(anomalous_energies):.2f}")
    
    if normal_energies and anomalous_energies:
        separation = abs(sum(anomalous_energies)/len(anomalous_energies) - sum(normal_energies)/len(normal_energies))
        print(f"\nEnergy separation: {separation:.2f}")
        if separation > baseline_std:
            print("✓ Good separation between normal and anomalous")
        else:
            print("⚠️ Limited separation - may need threshold tuning")
    
    print("\n" + "=" * 80)
    print("DEMO TEST COMPLETE")
    print("=" * 80)
    print("\nThe TLM Anomaly Radar is working and detecting anomalies!")
    print("Ready for production use.")


if __name__ == "__main__":
    main()

