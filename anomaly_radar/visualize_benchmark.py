"""
Visualize Benchmark Results

Creates:
1. Energy distribution plot (normal vs anomalous)
2. Anomaly score scatterplot
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_benchmark_csv(filepath: str = "anomaly_radar/benchmark_results.csv"):
    """Load benchmark results."""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "text": row["text"],
                "energy": float(row["energy"]),
                "anomaly_score": float(row["anomaly_score"]),
                "label": row["label"],
                "anomaly_type": row["anomaly_type"]
            })
    return results


def plot_energy_distribution(results, save_path="anomaly_radar/energy_distribution.png"):
    """Plot energy distribution for normal vs anomalous."""
    normal_energies = [r["energy"] for r in results if r["label"] == "normal"]
    anomalous_energies = [r["energy"] for r in results if r["label"] == "anomalous"]
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(normal_energies, bins=30, alpha=0.7, label="Normal", color="green", density=True)
    plt.hist(anomalous_energies, bins=30, alpha=0.7, label="Anomalous", color="red", density=True)
    
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Energy Distribution: Normal vs Anomalous Sequences", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    # Add statistics
    normal_mean = np.mean(normal_energies)
    anomalous_mean = np.mean(anomalous_energies)
    plt.axvline(normal_mean, color="green", linestyle="--", linewidth=2, label=f"Normal mean: {normal_mean:.2f}")
    plt.axvline(anomalous_mean, color="red", linestyle="--", linewidth=2, label=f"Anomalous mean: {anomalous_mean:.2f}")
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved energy distribution plot to {save_path}")
    plt.close()


def plot_anomaly_score_scatter(results, save_path="anomaly_radar/anomaly_score_scatter.png"):
    """Plot anomaly score scatterplot."""
    normal_scores = [r["anomaly_score"] for r in results if r["label"] == "normal"]
    anomalous_scores = [r["anomaly_score"] for r in results if r["label"] == "anomalous"]
    
    normal_energies = [r["energy"] for r in results if r["label"] == "normal"]
    anomalous_energies = [r["energy"] for r in results if r["label"] == "anomalous"]
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(normal_energies, normal_scores, alpha=0.6, label="Normal", color="green", s=50)
    plt.scatter(anomalous_energies, anomalous_scores, alpha=0.6, label="Anomalous", color="red", s=50)
    
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Anomaly Score", fontsize=12)
    plt.title("Anomaly Score vs Energy: Separation Analysis", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    # Add threshold line
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Threshold (0.5)")
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved scatterplot to {save_path}")
    plt.close()


def plot_by_anomaly_type(results, save_path="anomaly_radar/anomaly_by_type.png"):
    """Plot energy distribution by anomaly type."""
    anomaly_types = {}
    for r in results:
        if r["label"] == "anomalous":
            atype = r["anomaly_type"]
            if atype not in anomaly_types:
                anomaly_types[atype] = []
            anomaly_types[atype].append(r["energy"])
    
    plt.figure(figsize=(14, 8))
    
    # Box plot
    data_to_plot = [anomaly_types[atype] for atype in sorted(anomaly_types.keys())]
    labels = sorted(anomaly_types.keys())
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.xlabel("Anomaly Type", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    plt.title("Energy Distribution by Anomaly Type", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved anomaly type plot to {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Creating Benchmark Visualizations")
    print("=" * 70)
    
    # Load results
    print("\nLoading benchmark results...")
    results = load_benchmark_csv()
    print(f"✓ Loaded {len(results)} examples")
    
    # Create plots
    print("\nCreating visualizations...")
    plot_energy_distribution(results)
    plot_anomaly_score_scatter(results)
    plot_by_anomaly_type(results)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - anomaly_radar/energy_distribution.png")
    print("  - anomaly_radar/anomaly_score_scatter.png")
    print("  - anomaly_radar/anomaly_by_type.png")


if __name__ == "__main__":
    main()

