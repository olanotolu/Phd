"""
TLM Anomaly Detector for LLM Outputs

Uses a trained TLM to detect anomalies in LLM-generated text.
Anomalies = high energy = unstable/weird/hallucinated outputs.
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from babel_library import BabelEBM
from dataset import text_to_indices, indices_to_text, ALPHABET_SIZE
from anomaly_radar.dialogue_data import format_dialogue_for_tlm


class AnomalyRadar:
    """
    Anomaly detection system using TLM energy scores.
    """
    
    def __init__(self, model: BabelEBM):
        """
        Initialize anomaly detector with trained TLM.
        
        Args:
            model: Trained TLM model
        """
        self.model = model
        self.sequence_length = model.sequence_length
    
    def compute_energy(self, text: str) -> float:
        """
        Compute energy score for a text sequence.
        Lower energy = more stable/normal
        Higher energy = more anomalous
        
        Args:
            text: Input text to evaluate
            
        Returns:
            Energy score (lower is better)
        """
        # Format and convert to indices
        formatted = format_dialogue_for_tlm(text)
        indices = text_to_indices(formatted)
        
        # Pad or truncate
        if len(indices) < self.sequence_length:
            padding = np.full(self.sequence_length - len(indices), 26, dtype=np.int32)  # space
            indices = np.concatenate([indices, padding])
        else:
            indices = indices[:self.sequence_length]
        
        # Compute energy
        energy = self.model.energy(indices.reshape(1, -1))[0]
        
        return float(energy)
    
    def detect_anomaly(
        self,
        text: str,
        threshold: float = None,
        baseline_energy: float = None,
        baseline_std: float = None
    ) -> Dict:
        """
        Detect if text is anomalous.
        
        Args:
            text: Text to evaluate
            threshold: Energy threshold (if None, uses baseline)
            baseline_energy: Baseline energy from normal training data
            baseline_std: Standard deviation of baseline energy
            
        Returns:
            Dictionary with anomaly detection results
        """
        energy = self.compute_energy(text)
        
        # If no threshold provided, use baseline
        if threshold is None:
            if baseline_energy is None:
                # Default threshold
                threshold = 10.0
            elif baseline_std is not None:
                # Use proper statistical threshold
                threshold = baseline_energy + 2 * baseline_std
            else:
                # Fallback: use baseline + fixed offset
                threshold = baseline_energy + 1.5
        
        # For energy-based models: lower energy = more stable/normal
        # Anomalies can be either:
        # 1. Much HIGHER energy (unstable, doesn't match patterns)
        # 2. Much LOWER energy (too perfect, might be repetitive or memorized)
        
        # Use statistical threshold: flag if energy is outside normal range
        if baseline_energy is not None and baseline_std is not None:
            # Flag if energy is more than 2 std away from baseline (either direction)
            lower_bound = baseline_energy - 2 * baseline_std
            upper_bound = baseline_energy + 2 * baseline_std
            is_anomalous = energy < lower_bound or energy > upper_bound
            
            # Anomaly score: distance from baseline normalized by std
            distance = abs(energy - baseline_energy)
            normalized_score = min(1.0, distance / max(baseline_std * 2, 0.1))
        else:
            # Fallback: simple threshold
            is_anomalous = energy > threshold
            normalized_score = min(1.0, max(0.0, (energy + 50) / 100))
        
        return {
            "text": text,
            "energy": energy,
            "anomaly_score": normalized_score,
            "is_anomalous": is_anomalous,
            "threshold": threshold,
            "confidence": abs(energy - threshold) / max(abs(threshold), 1.0)
        }
    
    def batch_detect(
        self,
        texts: List[str],
        baseline_energy: float = None
    ) -> List[Dict]:
        """
        Detect anomalies in multiple texts.
        
        Args:
            texts: List of texts to evaluate
            baseline_energy: Baseline energy from normal data
            
        Returns:
            List of anomaly detection results
        """
        results = []
        
        for text in texts:
            result = self.detect_anomaly(text, baseline_energy=baseline_energy)
            results.append(result)
        
        return results
    
    def compute_baseline_energy(self, normal_texts: List[str]) -> Tuple[float, float]:
        """
        Compute baseline energy from normal training texts.
        
        Args:
            normal_texts: List of normal (training) texts
            
        Returns:
            (mean_energy, std_energy)
        """
        energies = [self.compute_energy(text) for text in normal_texts]
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        return float(mean_energy), float(std_energy)
    
    def per_token_anomaly_map(self, text: str, window_size: int = 10) -> List[float]:
        """
        Compute per-token anomaly scores using sliding window.
        
        Args:
            text: Input text
            window_size: Size of sliding window
            
        Returns:
            List of anomaly scores per position
        """
        formatted = format_dialogue_for_tlm(text)
        scores = []
        
        for i in range(len(formatted) - window_size + 1):
            window = formatted[i:i+window_size]
            energy = self.compute_energy(window)
            scores.append(energy)
        
        # Normalize
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        return scores


def load_trained_model(weights_path: str = "anomaly_radar/dialogue_tlm_weights.npy") -> BabelEBM:
    """
    Load a trained TLM model from saved weights.
    
    Args:
        weights_path: Path to saved weights
        
    Returns:
        Trained BabelEBM model
    """
    weights = np.load(weights_path)
    sequence_length = weights.shape[0] + 1  # weights shape is (n_positions, K, K)
    
    model = BabelEBM(
        sequence_length=sequence_length,
        alphabet_size=ALPHABET_SIZE,
        init_scale=0.1
    )
    model.weights = weights
    
    return model


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("TLM Anomaly Detector - Example Usage")
    print("=" * 60)
    
    # Load trained model
    try:
        model = load_trained_model()
        print("\n✓ Loaded trained TLM model")
    except FileNotFoundError:
        print("\n✗ No trained model found. Please train first:")
        print("  python anomaly_radar/train_dialogue_tlm.py")
        exit(1)
    
    # Create detector
    detector = AnomalyRadar(model)
    
    # Test texts
    normal_text = "customer: hi i'd like to make a reservation agent: sure what time customer: 7pm agent: how many people customer: two agent: your reservation is confirmed"
    
    anomalous_texts = [
        "customer: hello i want to book a table for quantum computing research agent: sure what time customer: yesterday agent: how many people customer: negative infinity agent: your reservation is confirmed",
        "customer: hi i'd like to make a reservation agent: sure what time customer: 7pm agent: how many people customer: two agent: the sky is blue and elephants can fly your reservation is confirmed",
        "customer: hello agent: hello agent: hello agent: hello agent: hello agent: hello",
    ]
    
    # Compute baseline
    print("\n1. Computing baseline energy from normal text...")
    baseline_mean, baseline_std = detector.compute_baseline_energy([normal_text])
    print(f"   Baseline energy: {baseline_mean:.2f} ± {baseline_std:.2f}")
    
    # Test normal text
    print("\n2. Testing normal text:")
    result = detector.detect_anomaly(normal_text, baseline_energy=baseline_mean)
    print(f"   Energy: {result['energy']:.2f}")
    print(f"   Anomaly score: {result['anomaly_score']:.2f}")
    print(f"   Is anomalous: {result['is_anomalous']}")
    
    # Test anomalous texts
    print("\n3. Testing anomalous texts:")
    for i, text in enumerate(anomalous_texts, 1):
        result = detector.detect_anomaly(text, baseline_energy=baseline_mean)
        print(f"\n   Anomaly {i}:")
        print(f"   Text: {text[:80]}...")
        print(f"   Energy: {result['energy']:.2f}")
        print(f"   Anomaly score: {result['anomaly_score']:.2f}")
        print(f"   Is anomalous: {result['is_anomalous']}")
    
    print("\n" + "=" * 60)
    print("Anomaly detection complete!")
    print("=" * 60)

