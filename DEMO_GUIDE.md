# TLM Anomaly Radar - Web Demo Guide

## Demo Name

**"TLM Anomaly Radar"** - Energy-Based Anomaly Detection for LLM Outputs

**Tagline**: *"From chaos to structure. Detecting anomalies in LLM outputs."*

---

## How to Run the Demo

### Option 1: Quick Start (Recommended)

```bash
cd /Users/term_/Phd
python -m anomaly_radar.demo_app
```

This will:
- Start a web server
- Open in your browser automatically
- Give you a shareable URL (e.g., `https://xxxxx.gradio.live`)

### Option 2: Using the Script

```bash
./anomaly_radar/run_demo.sh
```

### Option 3: Deploy to HuggingFace Spaces

1. Create a HuggingFace account
2. Create a new Space
3. Upload your code
4. Add `requirements.txt`
5. Set app file to `anomaly_radar/demo_app.py`
6. Deploy!

---

## What the Demo Shows

### Tab 1: Single Text Analysis
- Enter any text
- See energy score and anomaly score
- Visualize energy distribution
- See if it's flagged as anomalous

### Tab 2: Batch Analysis
- Enter multiple texts (one per line)
- Get results table
- See which are normal vs anomalous

### Tab 3: Examples
- Pre-loaded normal examples
- Pre-loaded anomalous examples
- Click to try them

### Tab 4: About
- Explanation of how it works
- Technical details
- Benchmark results

---

## Features

✅ **Beautiful UI** - Clean, modern interface  
✅ **Interactive** - Real-time analysis  
✅ **Visualizations** - Energy plots, score gauges  
✅ **Easy to Understand** - Clear explanations  
✅ **Shareable** - Get a URL to share with others  

---

## Demo URL

When you run the demo, you'll get:
- **Local URL**: `http://127.0.0.1:7860`
- **Public URL**: `https://xxxxx.gradio.live` (if using `share=True`)

Share the public URL with:
- Scale AI team
- Researchers
- Anyone who wants to try it

---

## What People Will See

1. **Clean interface** with clear instructions
2. **Text input box** to enter any text
3. **Analysis results** with:
   - Energy score
   - Anomaly score (0-1)
   - Status (NORMAL/ANOMALOUS)
   - Visualizations
4. **Example texts** to try
5. **About section** explaining the technology

---

## Tips for Presentation

1. **Start with a normal example**: "hi i want to make a reservation for two at 7pm"
   - Show it's marked as NORMAL
   - Show low anomaly score

2. **Then show an anomaly**: "quantum computing research table booking"
   - Show it's marked as ANOMALOUS
   - Show high anomaly score
   - Point out the energy difference

3. **Explain the visualization**:
   - Green histogram = normal examples
   - Red/blue line = your text
   - Threshold lines = anomaly boundaries

4. **Show batch analysis**:
   - Enter multiple texts
   - Show table of results
   - Demonstrate it works on many examples

---

## Deployment Options

### HuggingFace Spaces (Free, Easy)
- Best for sharing with others
- Free hosting
- Automatic updates

### Local Server
- Run on your machine
- Use `share=True` for public URL
- Good for demos

### Your Own Server
- Deploy to AWS/GCP/Azure
- Full control
- Production-ready

---

## Next Steps

1. **Run the demo**: `python -m anomaly_radar.demo_app`
2. **Test it**: Try normal and anomalous texts
3. **Share the URL**: Give it to Scale AI or others
4. **Deploy**: Put it on HuggingFace Spaces for permanent hosting

---

**The demo is ready to show off your work!** 🚀

