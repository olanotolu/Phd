# Using Real Dialogue Data - Quick Guide

## ✅ What Changed

The system now supports **real dialogue data** in addition to synthetic data. This gives you **better anomaly detection** because the TLM learns from actual patterns.

---

## Quick Start with Real Data

### 1. Create Your Data File

Create `anomaly_radar/real_dialogues.txt` with your actual restaurant booking dialogues:

```
customer: hi i'd like to make a reservation agent: sure what time customer: 7pm agent: how many people customer: two agent: your reservation is confirmed
customer: hello can i book a table agent: of course what time customer: 8pm agent: how many guests customer: four agent: your table is booked
customer: good evening i need a reservation agent: what time would you like customer: 6:30pm agent: party size customer: three agent: reservation confirmed
```

**Format**: One dialogue per line (recommended)

### 2. Train on Real Data

```bash
# Option 1: Command line
python -m anomaly_radar.train_dialogue_tlm --data anomaly_radar/real_dialogues.txt

# Option 2: Use the example script
python -m anomaly_radar.example_real_data

# Option 3: The demo automatically uses real data if available
python -m anomaly_radar.demo
```

### 3. In Python Code

```python
from anomaly_radar.dialogue_data import load_real_dialogues_simple
from anomaly_radar.train_dialogue_tlm import train_dialogue_tlm

# Load real data
dialogues = load_real_dialogues_simple("anomaly_radar/real_dialogues.txt")

# Train
model, history = train_dialogue_tlm(dialogues)
```

---

## Data Format

### Simple Format (One per line) - Recommended

```
dialogue 1
dialogue 2
dialogue 3
```

### Multi-line Format

```
dialogue 1 line 1
dialogue 1 line 2

dialogue 2 line 1
dialogue 2 line 2
```

Empty lines separate dialogues.

---

## Automatic Cleaning

The loader automatically:
- ✅ Converts to lowercase
- ✅ Filters to valid characters (a-z, space, comma, period)
- ✅ Removes invalid characters
- ✅ Formats for TLM training

You don't need to pre-process your data!

---

## Example: Using Your Concya Data

If you have Concya restaurant booking dialogues:

1. **Export your dialogues** to a text file
2. **Save as** `anomaly_radar/real_dialogues.txt`
3. **Train**:
   ```bash
   python -m anomaly_radar.train_dialogue_tlm --data anomaly_radar/real_dialogues.txt
   ```

That's it! The TLM will learn the actual patterns from your Concya dialogues.

---

## Why Real Data is Better

| Synthetic Data | Real Data |
|---------------|-----------|
| Quick to generate | Captures real patterns |
| Good for testing | Better anomaly detection |
| May miss patterns | More realistic structure |
| | Better for production |

**Recommendation**: Always use real data when available!

---

## Functions Added

### `load_real_dialogues_simple(filepath)`
Loads dialogues from file (one per line format).

### `load_real_dialogues(filepath)`
Loads dialogues from file (multi-line format, separated by empty lines).

Both functions automatically clean and format the data.

---

## Command Line Options

```bash
# Use real data
python -m anomaly_radar.train_dialogue_tlm --data path/to/dialogues.txt

# Custom parameters
python -m anomaly_radar.train_dialogue_tlm \
    --data dialogues.txt \
    --sequence-length 300 \
    --epochs 200 \
    --method pseudo_likelihood

# Force synthetic (even if real data exists)
python -m anomaly_radar.train_dialogue_tlm --synthetic
```

---

## What Happens Automatically

The `demo.py` script now:
1. Checks for `anomaly_radar/real_dialogues.txt`
2. Uses real data if found
3. Falls back to synthetic data if not found
4. Shows which type is being used

---

## Next Steps

1. ✅ **Collect your real dialogues** from Concya or other sources
2. ✅ **Save to** `anomaly_radar/real_dialogues.txt`
3. ✅ **Train** with `--data` flag
4. ✅ **Get better anomaly detection** from real patterns!

---

**That's it!** Your TLM will now learn from real structure instead of synthetic patterns.

