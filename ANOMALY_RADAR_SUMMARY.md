# TLM Anomaly Radar - Step 1 Complete ✅

## What We Built

**TLM Anomaly Radar v0.1** - A TLM-powered anomaly detection system for restaurant booking dialogues.

This is **Step 1** of the Scale AI-focused product: training a TLM on "normal" restaurant booking outputs to detect anomalies.

---

## ✅ Completed

### 1. Dialogue Data Collection (`dialogue_data.py`)
- ✅ Generates 100-500 restaurant booking dialogues
- ✅ Multiple dialogue patterns (standard, special requests, alternatives)
- ✅ Formatting for TLM training
- ✅ Save/load functionality

### 2. TLM Training (`train_dialogue_tlm.py`)
- ✅ Trains TLM on normal restaurant booking dialogues
- ✅ Learns structural patterns (bigrams, dialogue flow)
- ✅ Saves trained weights for reuse
- ✅ Configurable training parameters

### 3. Anomaly Detection (`anomaly_detector.py`)
- ✅ Energy-based anomaly scoring
- ✅ Baseline computation from normal data
- ✅ Batch detection
- ✅ Per-token anomaly maps
- ✅ Confidence scoring

### 4. Complete Demo (`demo.py`)
- ✅ End-to-end workflow
- ✅ Test cases (normal, hallucinated, off-topic, repetitive)
- ✅ Results visualization

---

## Project Structure

```
anomaly_radar/
├── __init__.py              # Package initialization
├── README.md                # Documentation
├── dialogue_data.py         # Data collection
├── train_dialogue_tlm.py    # TLM training
├── anomaly_detector.py      # Anomaly detection
├── demo.py                  # Complete demo
├── dialogues.txt            # Generated training data
└── dialogue_tlm_weights.npy # Trained model weights
```

---

## How to Use

### Quick Start

```bash
# 1. Generate dialogues
python -m anomaly_radar.dialogue_data

# 2. Train TLM
python -m anomaly_radar.train_dialogue_tlm

# 3. Run demo
python -m anomaly_radar.demo
```

### In Your Code

```python
from anomaly_radar import AnomalyRadar, load_trained_model

# Load model
model = load_trained_model()
detector = AnomalyRadar(model)

# Detect anomaly
result = detector.detect_anomaly("your text here")
if result['is_anomalous']:
    print("⚠️ Anomaly detected!")
```

---

## What It Detects

✅ **Hallucinations** - Nonsensical content  
✅ **Off-topic** - Content outside restaurant booking domain  
✅ **Repetitive** - Stuck or degenerate outputs  
✅ **Structural anomalies** - Patterns that don't match learned structure  

---

## Results

The system can:

1. **Learn normal patterns** from 500 restaurant booking dialogues
2. **Score any text** with energy-based stability measure
3. **Flag anomalies** when energy exceeds baseline threshold
4. **Provide interpretable scores** (0-1 anomaly score)

---

## Why This Matters for Scale AI

Scale AI's core business:
- ✅ Evaluating LLM outputs
- ✅ Catching hallucinations
- ✅ Finding unsafe/low-quality generations
- ✅ Detecting model drift

**TLM Anomaly Radar provides**:
- ✅ Model-agnostic evaluation (works on any LLM)
- ✅ Unsupervised detection (no labeled data needed)
- ✅ Physics-based scoring (interpretable energy)
- ✅ Structural pattern detection (beyond token probabilities)

---

## Next Steps (Phase 2)

1. **Expand to other domains**:
   - Customer service
   - Summaries
   - Code review
   - Wikipedia-style text

2. **Improve detection**:
   - Per-token heatmaps
   - Confidence intervals
   - Multi-scale patterns

3. **Production features**:
   - API endpoint
   - Real-time streaming
   - Monitoring dashboard
   - Integration examples

4. **Scale AI integration**:
   - Format for their evaluation pipeline
   - Benchmark against their tools
   - Create demo for their team

---

## Key Files

- **`dialogue_data.py`**: Generates 500 restaurant booking dialogues
- **`train_dialogue_tlm.py`**: Trains TLM on normal patterns
- **`anomaly_detector.py`**: Core anomaly detection system
- **`demo.py`**: Complete workflow demonstration

---

## Status

**Step 1: COMPLETE ✅**

- [x] Data collection
- [x] TLM training
- [x] Anomaly detection
- [x] Demo and testing
- [x] Documentation

**Ready for**: Integration, expansion to other domains, production deployment

---

**Tagline**: *"From chaos to structure. Detecting anomalies in LLM outputs."*

