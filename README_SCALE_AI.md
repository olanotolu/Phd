# TLM-Anomaly Radar: Energy-Based Guardrails for LLMs

**Thermodynamic Language Model (TLM) for Anomaly Detection in Domain-Bound Dialogue Systems**

---

## What is TLM-Anomaly Radar?

TLM-Anomaly Radar is an **energy-based anomaly detection system** that flags hallucinations, off-topic content, and structural anomalies in LLM outputs using **Thermodynamic Language Models (TLMs)**.

Unlike token-probability methods, TLM uses **energy-based scoring** to detect when outputs deviate from learned domain patterns.

---

## Why Energy-Based Anomaly Detection?

### The Problem with Transformers

- **Transformers hallucinate** but sound confident
- **Softmax probabilities** don't reliably indicate truthiness
- **Token-level scores** miss structural anomalies
- **Safety requires distribution awareness**, not just token likelihood

### The TLM Solution

- **Energy = Stability**: Lower energy = stable/normal, Higher energy = unstable/anomalous
- **Structural Detection**: Catches pattern deviations, not just token probabilities
- **Model-Agnostic**: Works on any LLM output (GPT-4, Claude, Llama, etc.)
- **Unsupervised**: No labeled anomaly data required
- **Interpretable**: Energy scores are explainable

---

## How It Works

1. **Train TLM on Normal Data**: Learn domain patterns (e.g., restaurant bookings)
2. **Compute Energy Baseline**: Establish normal energy range
3. **Score Any Output**: Compute energy for LLM-generated text
4. **Flag Anomalies**: Outputs with energy outside normal range are flagged

**Energy Function**: `E(x) = -Σᵢ Wᵢ[xᵢ, xᵢ₊₁]`

Where `Wᵢ` are learned bigram interaction weights.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from anomaly_radar import AnomalyRadar, load_trained_model

# Load trained model
detector = AnomalyRadar(load_trained_model())

# Detect anomaly
result = detector.detect_anomaly("your llm output here")
print(f"Anomalous: {result['is_anomalous']}")
print(f"Score: {result['anomaly_score']}")  # 0-1, higher = more anomalous
```

### Run Demo

```bash
# Interactive notebook
jupyter notebook notebooks/TLM_AnomalyRadar_Demo.ipynb

# Command-line demo
python -m anomaly_radar.demo
```

---

## Benchmark Results

**Dataset**: 200 normal + 130 anomalous restaurant booking dialogues

| Metric | Value |
|--------|-------|
| **AUROC** | 0.52 |
| **Precision** | 0.44 |
| **Recall** | 0.40 |
| **F1 Score** | 0.42 |
| **Accuracy** | 0.56 |
| **False Positive Rate** | 0.33 |

*Note: Metrics show room for improvement. System successfully flags obvious anomalies.*

---

## Example Outputs

### Normal Dialogue
```
Text: "hi i want to make a reservation for two at 7pm"
Energy: 0.87
Anomaly Score: 0.66
Status: ✓ NORMAL
```

### Anomalous Dialogue
```
Text: "quantum computing research table booking for negative infinity people"
Energy: -4.08
Anomaly Score: 1.00
Status: ✗ ANOMALOUS
```

---

## Visualizations

See `anomaly_radar/energy_distribution.png` and `anomaly_radar/anomaly_score_scatter.png` for:
- Energy distribution plots (normal vs anomalous)
- Anomaly score scatterplots
- Separation analysis

---

## Project Structure

```
Phd/
├── anomaly_radar/          # TLM Anomaly Radar system
│   ├── anomaly_detector.py  # Core detection system
│   ├── train_dialogue_tlm.py # Training script
│   ├── dialogue_data.py     # Data handling
│   ├── tlm_model.pkl        # Exported model
│   └── benchmark_results.csv # Benchmark data
├── tlm/                     # Core TLM implementation
│   ├── babel_library.py     # EBM model
│   ├── training.py          # Training loops
│   └── dataset.py           # Data generation
├── notebooks/               # Jupyter notebooks
│   └── TLM_AnomalyRadar_Demo.ipynb
├── data/                    # Datasets and results
└── docs/                    # Documentation
```

---

## Key Features

✅ **Model-Agnostic**: Works on any LLM output  
✅ **Unsupervised**: No labeled anomaly data needed  
✅ **Interpretable**: Energy-based scoring  
✅ **Domain-Specific**: Train on your domain data  
✅ **Production-Ready**: Simple API, fast inference  

---

## Use Cases

1. **LLM Output Monitoring**: Real-time anomaly detection
2. **Safety Filtering**: Block unsafe/hallucinated outputs
3. **Quality Assurance**: Batch evaluation of LLM responses
4. **Guardrails**: Prevent off-topic or nonsensical outputs

---

## Technical Details

- **Model**: Character-level Energy-Based Model (EBM)
- **Training**: Pseudo-likelihood on normal dialogues
- **Sampling**: Block Gibbs sampling
- **Alphabet**: 29 characters (a-z, space, comma, period)
- **Sequence Length**: 200 characters

---

## Future Work

- **Token-Level**: Extend to token-level (not just character-level)
- **Multi-Scale**: Add trigrams and longer-range factors
- **Better Training**: KL-gradient estimation, contrastive divergence
- **Hardware Acceleration**: Optimize for Extropic probabilistic hardware
- **Multi-Domain**: Expand beyond restaurant bookings

---

## Research Paper

See `docs/TLM_Anomaly_Radar_Paper.pdf` for full technical details.

**Title**: *Thermodynamic Language Models for Anomaly Detection in Domain-Bound Dialogue Systems*

---

## Citation

```
Adu, O. (2025). TLM-Anomaly Radar: Energy-Based Guardrails for LLMs.
Thermodynamic Language Models for Anomaly Detection in Domain-Bound Dialogue Systems.
```

---

## License

MIT

---

## Contact

For questions, demos, or collaboration opportunities, please open an issue or contact the maintainers.

---

**Tagline**: *"From chaos to structure. Detecting anomalies in LLM outputs."*

