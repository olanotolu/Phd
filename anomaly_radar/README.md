# TLM Anomaly Radar - Restaurant Booking Domain

**A TLM-powered anomaly detection system for LLM outputs.**

Detects weird, hallucinated, unsafe, or off-distribution restaurant booking dialogues using energy-based scoring.

---

## What It Does

TLM Anomaly Radar uses a **Thermodynamic Language Model (TLM)** trained on normal restaurant booking dialogues to:

- ✅ **Detect hallucinations** - Nonsensical or off-topic content
- ✅ **Flag anomalies** - Outputs that don't match learned patterns
- ✅ **Score stability** - Energy-based confidence (lower = more stable)
- ✅ **Identify stuck outputs** - Repetitive or degenerate generations

---

## Quick Start

### Option 1: Use Real Dialogue Data (Recommended)

1. **Prepare your real dialogue data**:
   - Create `anomaly_radar/real_dialogues.txt`
   - One dialogue per line
   - Example format:
     ```
     customer: hi i'd like to make a reservation agent: sure what time customer: 7pm agent: how many people customer: two agent: your reservation is confirmed
     customer: hello can i book a table agent: of course what time customer: 8pm agent: how many guests customer: four agent: your table is booked
     ```

2. **Train TLM on real data**:
   ```bash
   python -m anomaly_radar.train_dialogue_tlm --data anomaly_radar/real_dialogues.txt
   ```

### Option 2: Use Synthetic Data

```bash
# 1. Generate synthetic dialogues
python -m anomaly_radar.dialogue_data

# 2. Train TLM
python -m anomaly_radar.train_dialogue_tlm

# 3. Run demo
python -m anomaly_radar.demo
```

### Option 3: Complete Demo

```bash
python -m anomaly_radar.demo
```

This automatically:
- Uses real data if `real_dialogues.txt` exists
- Falls back to synthetic data otherwise
- Trains TLM (if needed)
- Tests anomaly detection

---

## Using Real Data

### Format Requirements

Your `real_dialogues.txt` file should have:
- **One dialogue per line** (recommended)
- Or **multi-line dialogues** separated by empty lines
- Text will be automatically:
  - Converted to lowercase
  - Filtered to valid alphabet characters (a-z, space, comma, period)
  - Formatted for TLM training

### Example Real Data File

```
customer: hi i'd like to make a reservation agent: sure what time customer: 7pm agent: how many people customer: two agent: your reservation is confirmed
customer: hello can i book a table agent: of course what time customer: 8pm agent: how many guests customer: four agent: your table is booked
customer: good evening i need a reservation agent: what time would you like customer: 6:30pm agent: party size customer: three agent: reservation confirmed
```

### Loading Real Data in Code

```python
from anomaly_radar.dialogue_data import load_real_dialogues_simple

# Load real dialogues (one per line)
dialogues = load_real_dialogues_simple("anomaly_radar/real_dialogues.txt")

# Or for multi-line format
from anomaly_radar.dialogue_data import load_real_dialogues
dialogues = load_real_dialogues("anomaly_radar/real_dialogues.txt")
```

---

## Training Options

### Command Line

```bash
# Use real data
python -m anomaly_radar.train_dialogue_tlm --data path/to/dialogues.txt

# Custom sequence length
python -m anomaly_radar.train_dialogue_tlm --data dialogues.txt --sequence-length 300

# More epochs
python -m anomaly_radar.train_dialogue_tlm --data dialogues.txt --epochs 200

# Use KL-gradient (slower but more accurate)
python -m anomaly_radar.train_dialogue_tlm --data dialogues.txt --method kl_gradient

# Force synthetic data
python -m anomaly_radar.train_dialogue_tlm --synthetic
```

### In Python

```python
from anomaly_radar.dialogue_data import load_real_dialogues_simple
from anomaly_radar.train_dialogue_tlm import train_dialogue_tlm

# Load real data
dialogues = load_real_dialogues_simple("anomaly_radar/real_dialogues.txt")

# Train
model, history = train_dialogue_tlm(
    dialogues,
    sequence_length=200,
    n_epochs=100,
    method="pseudo_likelihood"
)
```

---

## Why Use Real Data?

**Synthetic data** (generated):
- ✅ Quick to generate
- ✅ Good for testing
- ❌ May not capture real patterns
- ❌ Less realistic structure

**Real data** (from actual dialogues):
- ✅ Captures real patterns
- ✅ Better anomaly detection
- ✅ More realistic structure
- ✅ Better for production

**Recommendation**: Use real data when available for better results!

---

## How It Works

1. **Training Phase**:
   - TLM learns "normal" restaurant booking patterns from your data
   - Learns bigram interactions and structural regularities
   - Establishes baseline energy for normal dialogues

2. **Detection Phase**:
   - Compute energy score for any text
   - Compare to baseline (normal training data)
   - Flag if energy exceeds threshold (anomalous)

**Energy = Stability Score**:
- **Low energy** = Stable, normal, in-distribution
- **High energy** = Unstable, anomalous, off-distribution

---

## API Reference

### `AnomalyRadar`

Main anomaly detection class.

```python
from anomaly_radar import AnomalyRadar, load_trained_model

# Load trained model
model = load_trained_model("anomaly_radar/dialogue_tlm_weights.npy")
detector = AnomalyRadar(model)

# Single text
result = detector.detect_anomaly(text, baseline_energy=baseline)

# Batch detection
results = detector.batch_detect(texts, baseline_energy=baseline)

# Compute baseline
mean, std = detector.compute_baseline_energy(normal_texts)

# Per-token anomaly map
scores = detector.per_token_anomaly_map(text, window_size=10)
```

### Result Dictionary

```python
{
    "text": str,              # Input text
    "energy": float,          # Energy score (lower = better)
    "anomaly_score": float,   # Normalized 0-1 (higher = more anomalous)
    "is_anomalous": bool,     # True if exceeds threshold
    "threshold": float,       # Energy threshold used
    "confidence": float       # Confidence in detection
}
```

---

## Example Output

```
TEST CASE 1: Normal Restaurant Booking
Energy: -12.34
Anomaly Score: 0.123
Status: ✓ NORMAL

TEST CASE 2: Hallucinated Content
Energy: 45.67
Anomaly Score: 0.956
Status: ✗ ANOMALOUS
```

---

## Use Cases

### 1. LLM Output Monitoring

Monitor your restaurant booking chatbot outputs in real-time:

```python
for output in llm_generate_stream():
    result = detector.detect_anomaly(output)
    if result['is_anomalous']:
        flag_for_review(output, result)
```

### 2. Quality Assurance

Batch evaluate LLM outputs:

```python
results = detector.batch_detect(llm_outputs)
anomalies = [r for r in results if r['is_anomalous']]
print(f"Found {len(anomalies)} anomalous outputs")
```

### 3. Safety Filtering

Filter unsafe or hallucinated outputs before sending to users:

```python
result = detector.detect_anomaly(llm_output)
if result['is_anomalous']:
    return fallback_response()
else:
    return llm_output
```

---

## Why TLM for Anomaly Detection?

**Traditional methods**:
- Use model logits (not always reliable)
- Require labeled data
- Don't capture structural patterns

**TLM advantages**:
- ✅ **Model-agnostic** - Works on any LLM output
- ✅ **Unsupervised** - No labeled anomaly data needed
- ✅ **Physics-based** - Energy = stability (interpretable)
- ✅ **Structural** - Detects pattern deviations, not just token probabilities

---

## Files

- `dialogue_data.py` - Generate/load restaurant booking dialogues (supports real data!)
- `train_dialogue_tlm.py` - Train TLM on normal dialogues
- `anomaly_detector.py` - Anomaly detection system
- `demo.py` - Complete demo workflow

---

## Next Steps

1. **Collect real dialogues**: Add your actual restaurant booking data
2. **Train on real data**: Use `--data` flag for better results
3. **Expand domains**: Train on other dialogue types
4. **Improve detection**: Add per-token heatmaps, confidence intervals
5. **Production integration**: API endpoint, monitoring dashboard

---

## For Scale AI

This is exactly what Scale AI needs:

- ✅ Model-agnostic evaluation tool
- ✅ Unsupervised anomaly detection
- ✅ Interpretable scoring (energy-based)
- ✅ Works on any LLM output
- ✅ No labeled data required
- ✅ **Works with real production data**

**Perfect for**: LLM output evaluation, hallucination detection, safety filtering, quality assurance.

---

**Tagline**: *"From chaos to structure. Detecting anomalies in LLM outputs."*
