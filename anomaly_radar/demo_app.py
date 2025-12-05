"""
TLM Anomaly Radar - Interactive Web Demo

A beautiful, easy-to-understand demo showing how TLM detects anomalies in LLM outputs.
"""

import sys
import os
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_radar import AnomalyRadar, load_trained_model
from anomaly_radar.dialogue_data import load_real_dialogues_simple


# Load model once at startup
print("Loading TLM model...")
try:
    model = load_trained_model("anomaly_radar/dialogue_tlm_weights.npy")
    detector = AnomalyRadar(model)
    dialogues = load_real_dialogues_simple("anomaly_radar/real_dialogues.txt")
    baseline_mean, baseline_std = detector.compute_baseline_energy(dialogues[:50])
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    detector = None
    baseline_mean = None
    baseline_std = None


def analyze_text(text):
    """Analyze a single text input."""
    if not detector or not text.strip():
        return "Please enter some text to analyze.", None, None
    
    # Detect anomaly
    result = detector.detect_anomaly(text, baseline_energy=baseline_mean, baseline_std=baseline_std)
    
    # Create result text
    status = "🔴 ANOMALOUS" if result['is_anomalous'] else "🟢 NORMAL"
    energy_color = "red" if result['is_anomalous'] else "green"
    
    result_text = f"""
## Analysis Result: {status}

**Energy Score**: `{result['energy']:.3f}`  
**Anomaly Score**: `{result['anomaly_score']:.3f}` (0 = normal, 1 = anomalous)  
**Threshold**: `{result['threshold']:.3f}`  
**Confidence**: `{result['confidence']:.3f}`

### Interpretation:
- **Energy** measures how "stable" the text is according to learned patterns
- **Lower energy** = more stable/normal (matches training data)
- **Higher energy** = less stable/anomalous (deviates from patterns)
- **Anomaly score** is normalized 0-1 (higher = more anomalous)
"""
    
    # Create energy visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Energy comparison
    normal_energies = [detector.compute_energy(d) for d in dialogues[:20]]
    current_energy = result['energy']
    
    ax1.hist(normal_energies, bins=15, alpha=0.7, color='green', label='Normal Examples', density=True)
    ax1.axvline(current_energy, color='red' if result['is_anomalous'] else 'blue', 
                linewidth=3, label=f'Your Text: {current_energy:.2f}')
    ax1.axvline(baseline_mean, color='black', linestyle='--', linewidth=2, label=f'Baseline: {baseline_mean:.2f}')
    ax1.axvline(baseline_mean + 2*baseline_std, color='orange', linestyle='--', linewidth=1, label='Threshold')
    ax1.axvline(baseline_mean - 2*baseline_std, color='orange', linestyle='--', linewidth=1)
    ax1.set_xlabel('Energy', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Energy Distribution: Your Text vs Normal Examples', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Anomaly score gauge
    score = result['anomaly_score']
    colors = ['green', 'yellow', 'orange', 'red']
    thresholds = [0.25, 0.5, 0.75, 1.0]
    
    ax2.barh([0], [score], color=colors[min(int(score * 4), 3)], height=0.5)
    ax2.barh([0], [1-score], left=[score], color='lightgray', height=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('Anomaly Score', fontsize=12)
    ax2.set_title(f'Anomaly Score: {score:.3f}', fontsize=13, fontweight='bold')
    ax2.text(0.5, 0, f'{score*100:.1f}%', ha='center', va='center', fontsize=20, fontweight='bold')
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_yticks([])
    ax2.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save to temporary file for Gradio
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    return result_text, temp_file.name, None


def show_examples():
    """Show example normal and anomalous texts."""
    examples = {
        "Normal Examples": [
            "hi i want to make a reservation for two at 7pm",
            "your table for two at 7pm is confirmed",
            "we have 7pm or 8pm for two which one works",
            "can i get a table outside at 7pm for two",
            "i want to change my reservation from 6pm to 7pm"
        ],
        "Anomalous Examples": [
            "quantum computing research table booking for negative infinity people",
            "the sky is blue and elephants can fly through quantum space",
            "hello hello hello hello hello hello hello hello",
            "i need a table for negative five people at 25pm",
            "bonjour je veux une table pour deux à 7h pm reservation confirmée quantum"
        ]
    }
    
    html = "<div style='font-family: Arial;'>"
    html += "<h2>Example Texts</h2>"
    
    for category, texts in examples.items():
        html += f"<h3>{category}</h3><ul>"
        for text in texts:
            html += f"<li style='margin: 10px 0;'>{text}</li>"
        html += "</ul>"
    
    html += "</div>"
    return html


def batch_analyze(texts):
    """Analyze multiple texts at once."""
    if not detector or not texts:
        return "Please enter texts (one per line).", None
    
    lines = [line.strip() for line in texts.split('\n') if line.strip()]
    if not lines:
        return "No valid texts found.", None
    
    results = []
    for text in lines:
        result = detector.detect_anomaly(text, baseline_energy=baseline_mean, baseline_std=baseline_std)
        status = "🔴 ANOMALOUS" if result['is_anomalous'] else "🟢 NORMAL"
        results.append({
            'text': text,
            'status': status,
            'energy': result['energy'],
            'score': result['anomaly_score']
        })
    
    # Create table
    html = "<table style='width:100%; border-collapse: collapse; font-family: Arial;'>"
    html += "<tr style='background: #f0f0f0;'><th style='padding: 10px; border: 1px solid #ddd;'>Text</th>"
    html += "<th style='padding: 10px; border: 1px solid #ddd;'>Status</th>"
    html += "<th style='padding: 10px; border: 1px solid #ddd;'>Energy</th>"
    html += "<th style='padding: 10px; border: 1px solid #ddd;'>Anomaly Score</th></tr>"
    
    for r in results:
        html += f"<tr><td style='padding: 10px; border: 1px solid #ddd;'>{r['text'][:60]}...</td>"
        html += f"<td style='padding: 10px; border: 1px solid #ddd; text-align: center;'>{r['status']}</td>"
        html += f"<td style='padding: 10px; border: 1px solid #ddd; text-align: center;'>{r['energy']:.3f}</td>"
        html += f"<td style='padding: 10px; border: 1px solid #ddd; text-align: center;'>{r['score']:.3f}</td></tr>"
    
    html += "</table>"
    
    return html, None


# Create Gradio interface
with gr.Blocks(title="TLM Anomaly Radar", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 TLM Anomaly Radar
    ## Energy-Based Anomaly Detection for LLM Outputs
    
    **Detect hallucinations, off-topic content, and structural anomalies in LLM-generated text using Thermodynamic Language Models.**
    
    ### How It Works:
    1. TLM learns normal patterns from training data (restaurant booking dialogues)
    2. Computes **energy score** for any text (lower = more stable/normal)
    3. Flags anomalies when energy deviates from normal range
    4. Provides interpretable **anomaly score** (0-1, higher = more anomalous)
    """)
    
    with gr.Tabs():
        with gr.TabItem("🔎 Single Text Analysis"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Enter text to analyze",
                        placeholder="e.g., hi i want to make a reservation for two at 7pm",
                        lines=3
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")
                    
                with gr.Column():
                    result_output = gr.Markdown(label="Analysis Result")
                    plot_output = gr.Image(label="Visualization")
            
            analyze_btn.click(
                fn=analyze_text,
                inputs=[text_input],
                outputs=[result_output, plot_output]
            )
            
            gr.Examples(
                examples=[
                    ["hi i want to make a reservation for two at 7pm"],
                    ["quantum computing research table booking for negative infinity people"],
                    ["hello hello hello hello hello hello hello hello"],
                    ["your table for two at 7pm is confirmed"],
                ],
                inputs=text_input
            )
        
        with gr.TabItem("📊 Batch Analysis"):
            batch_input = gr.Textbox(
                label="Enter multiple texts (one per line)",
                placeholder="hi i want to make a reservation for two at 7pm\nquantum computing research table booking\nhello hello hello hello",
                lines=10
            )
            batch_btn = gr.Button("Analyze All", variant="primary")
            batch_output = gr.HTML(label="Results")
            
            batch_btn.click(
                fn=batch_analyze,
                inputs=[batch_input],
                outputs=[batch_output]
            )
        
        with gr.TabItem("📖 Examples"):
            examples_output = gr.HTML()
            demo.load(fn=show_examples, outputs=[examples_output])
        
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About TLM Anomaly Radar
            
            **TLM (Thermodynamic Language Model)** is a physics-inspired language model that represents text as an energy landscape.
            
            ### Key Features:
            - ✅ **Model-Agnostic**: Works on any LLM output
            - ✅ **Unsupervised**: No labeled anomaly data needed
            - ✅ **Interpretable**: Energy scores are explainable
            - ✅ **Fast**: Simple energy computation
            
            ### How Energy Works:
            - **Lower energy** = Text matches learned patterns (normal)
            - **Higher energy** = Text deviates from patterns (anomalous)
            - **Energy baseline** = Computed from normal training data
            
            ### Use Cases:
            - LLM output monitoring
            - Safety filtering
            - Quality assurance
            - Guardrails for chatbots
            
            ### Technical Details:
            - **Model**: Character-level Energy-Based Model (EBM)
            - **Training**: Pseudo-likelihood on 389 restaurant booking dialogues
            - **Alphabet**: 29 characters (a-z, space, comma, period)
            - **Sequence Length**: 200 characters
            
            ### Benchmark Results:
            - **AUROC**: 0.52
            - **Accuracy**: 0.56
            - **Precision**: 0.44
            - **Recall**: 0.40
            
            ---
            
            **Tagline**: *"From chaos to structure. Detecting anomalies in LLM outputs."*
            """)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TLM Anomaly Radar - Web Demo")
    print("=" * 70)
    print("\nStarting web server...")
    print("The demo will open in your browser.")
    print("Share the URL with others to let them try it!")
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 70 + "\n")
    
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, show_error=True)

