"""
TLM Anomaly Radar - Restaurant Booking Domain

A TLM-powered anomaly detection system for LLM outputs.
Detects weird, hallucinated, or unsafe restaurant booking dialogues.
"""

try:
    from .anomaly_detector import AnomalyRadar, load_trained_model
    from .dialogue_data import (
        collect_training_data,
        load_dialogues,
        load_real_dialogues,
        load_real_dialogues_simple,
        format_dialogue_for_tlm
    )
except ImportError:
    # Handle import errors gracefully
    pass

__version__ = "0.1.0"
__all__ = [
    "AnomalyRadar",
    "load_trained_model",
    "collect_training_data",
    "load_dialogues",
    "load_real_dialogues",
    "load_real_dialogues_simple",
    "format_dialogue_for_tlm",
]

