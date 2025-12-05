# 🚀 Launch TLM Anomaly Radar Demo

## Quick Start

```bash
cd /Users/term_/Phd
python -m anomaly_radar.demo_app
```

That's it! The demo will:
1. Load the trained TLM model
2. Start a web server
3. Open in your browser
4. Give you a shareable URL

---

## What You'll See

### Demo Name: **"TLM Anomaly Radar"**

A beautiful web interface with:

1. **Single Text Analysis Tab**
   - Enter any text
   - See energy score, anomaly score
   - Visual energy distribution plot
   - Clear NORMAL/ANOMALOUS status

2. **Batch Analysis Tab**
   - Enter multiple texts
   - Get results table
   - See all scores at once

3. **Examples Tab**
   - Pre-loaded examples
   - Normal and anomalous texts
   - Click to try them

4. **About Tab**
   - How it works
   - Technical details
   - Benchmark results

---

## Features

✅ **Beautiful UI** - Modern, clean interface  
✅ **Interactive** - Real-time analysis  
✅ **Visualizations** - Energy plots, score gauges  
✅ **Easy to Understand** - Clear explanations  
✅ **Shareable** - Public URL to share  

---

## Share the Demo

When you run it, you'll get:
- **Local URL**: `http://127.0.0.1:7860`
- **Public URL**: `https://xxxxx.gradio.live` (shareable!)

**Share the public URL with:**
- Scale AI team
- Researchers
- Anyone who wants to try it

---

## Installation (if needed)

If Gradio isn't installed:

```bash
pip install gradio>=4.0.0
```

---

## Demo Flow for Presentation

1. **Show normal example**:
   - Enter: "hi i want to make a reservation for two at 7pm"
   - Show: 🟢 NORMAL, low anomaly score
   - Point out: Energy matches normal distribution

2. **Show anomaly**:
   - Enter: "quantum computing research table booking"
   - Show: 🔴 ANOMALOUS, high anomaly score
   - Point out: Energy deviates from normal

3. **Show visualization**:
   - Explain the histogram
   - Show where your text falls
   - Explain the threshold

4. **Show batch analysis**:
   - Enter multiple texts
   - Show results table
   - Demonstrate it works at scale

---

## Troubleshooting

**If demo doesn't start:**
```bash
pip install --upgrade gradio
```

**If model doesn't load:**
- Check `anomaly_radar/dialogue_tlm_weights.npy` exists
- Check `anomaly_radar/real_dialogues.txt` exists

**If port is busy:**
- Change port: `demo.launch(server_port=7861)`

---

## Deploy to HuggingFace Spaces

For permanent hosting:

1. Create HuggingFace account
2. Create new Space
3. Upload:
   - `anomaly_radar/demo_app.py` → `app.py`
   - `anomaly_radar/dialogue_tlm_weights.npy`
   - `anomaly_radar/real_dialogues.txt`
   - `requirements.txt`
4. Deploy!

---

**Ready to launch!** 🚀

Run: `python -m anomaly_radar.demo_app`

