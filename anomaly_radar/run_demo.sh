#!/bin/bash

# TLM Anomaly Radar - Launch Web Demo
# This starts the interactive Gradio demo

echo "=" 
echo "TLM Anomaly Radar - Web Demo"
echo "=" 
echo ""
echo "Starting web server..."
echo "The demo will open in your browser."
echo "Share the URL with others to let them try it!"
echo ""

cd "$(dirname "$0")/.."
python -m anomaly_radar.demo_app

