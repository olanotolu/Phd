"""
Create Benchmark CSV with Metrics

Generates a CSV with:
- text
- energy
- anomaly_score
- label (normal/anomalous)
- anomaly_type

Then computes metrics:
- AUROC
- Precision/Recall
- False Positive Rate
"""

import sys
import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_radar import AnomalyRadar, load_trained_model
from anomaly_radar.dialogue_data import load_real_dialogues_simple, format_dialogue_for_tlm


def load_anomalies(filepath: str = "anomaly_radar/anomalous_dataset.txt"):
    """Load anomalous examples."""
    anomalies = []
    with open(filepath, "r") as f:
        for line in f:
            if "\t" in line:
                text, anomaly_type = line.strip().split("\t", 1)
                anomalies.append((text, anomaly_type))
    return anomalies


def create_benchmark_csv(output_file: str = "anomaly_radar/benchmark_results.csv"):
    """Create benchmark CSV with all metrics."""
    print("=" * 70)
    print("Creating Benchmark CSV")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading model...")
    model = load_trained_model("anomaly_radar/dialogue_tlm_weights.npy")
    detector = AnomalyRadar(model)
    
    # Load normal data
    print("\n[2/4] Loading normal dialogues...")
    normal_dialogues = load_real_dialogues_simple("anomaly_radar/real_dialogues.txt")
    # Use subset for benchmark
    normal_dialogues = normal_dialogues[:200]  # Match anomaly count
    
    # Load anomalies
    print("\n[3/4] Loading anomalous examples...")
    anomalies = load_anomalies()
    
    # Compute baseline
    baseline_mean, baseline_std = detector.compute_baseline_energy(normal_dialogues[:50])
    print(f"Baseline: {baseline_mean:.2f} ± {baseline_std:.2f}")
    
    # Process all examples
    print("\n[4/4] Processing examples and computing scores...")
    results = []
    
    # Normal examples
    for text in normal_dialogues:
        result = detector.detect_anomaly(text, baseline_energy=baseline_mean, baseline_std=baseline_std)
        results.append({
            "text": text,
            "energy": result["energy"],
            "anomaly_score": result["anomaly_score"],
            "label": "normal",
            "anomaly_type": "normal"
        })
    
    # Anomalous examples
    for text, anomaly_type in anomalies:
        result = detector.detect_anomaly(text, baseline_energy=baseline_mean, baseline_std=baseline_std)
        results.append({
            "text": text,
            "energy": result["energy"],
            "anomaly_score": result["anomaly_score"],
            "label": "anomalous",
            "anomaly_type": anomaly_type
        })
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "energy", "anomaly_score", "label", "anomaly_type"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Saved {len(results)} examples to {output_file}")
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("Computing Metrics")
    print("=" * 70)
    
    y_true = [1 if r["label"] == "anomalous" else 0 for r in results]
    y_scores = [r["anomaly_score"] for r in results]
    y_pred = [1 if r["anomaly_score"] > 0.5 else 0 for r in results]
    
    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_scores)
        print(f"\nAUROC: {auroc:.4f}")
    except:
        print("\nAUROC: Could not compute (need both classes)")
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (normal→normal): {tn}")
    print(f"  False Positives (normal→anomalous): {fp}")
    print(f"  False Negatives (anomalous→normal): {fn}")
    print(f"  True Positives (anomalous→anomalous): {tp}")
    
    # False Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\nFalse Positive Rate: {fpr:.4f}")
    
    # Accuracy
    accuracy = (tp + tn) / len(results)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save metrics
    metrics_file = "anomaly_radar/benchmark_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("TLM Anomaly Radar - Benchmark Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Examples: {len(results)}\n")
        f.write(f"Normal: {len(normal_dialogues)}\n")
        f.write(f"Anomalous: {len(anomalies)}\n\n")
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"False Positive Rate: {fpr:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
    
    print(f"\n✓ Metrics saved to {metrics_file}")
    print("\n" + "=" * 70)
    print("Benchmark creation complete!")
    print("=" * 70)
    
    return results, {
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr
    }


if __name__ == "__main__":
    create_benchmark_csv()

