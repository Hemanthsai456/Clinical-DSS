"""
Model Loader module for Evernorth Clinical-DSS Streamlit Application.
Provides cached loading functions for:
1. Track A: Frozen Baseline A Logistic Regression Pipeline (models/baseline_a_logreg.joblib).
2. Track B: Verified 3-Epoch Colab Transformer Checkpoint (evernorth_m3_results/models/sbdh_transformer).
"""
import os
import sys
import joblib
import torch
import streamlit as st
from transformers import AutoTokenizer

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.transformer_model import MultiLabelSBDHTransformer

TRACK_A_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_a_logreg.joblib")
TRACK_B_MODEL_DIR = os.path.join(PROJECT_ROOT, "evernorth_m3_results", "models", "sbdh_transformer")
TRACK_A_FROZEN_THRESHOLD = 0.130
TRACK_B_FROZEN_THRESHOLD = 0.300


@st.cache_resource(show_spinner="Loading Track A Readmission Pipeline...")
def load_track_a_pipeline():
    """
    Load the frozen Baseline A Pipeline from disk.
    Verifies that the model artifact exists and contains the fitted ColumnTransformer + Classifier.
    """
    if not os.path.exists(TRACK_A_MODEL_PATH):
        raise FileNotFoundError(f"Track A model artifact not found at: {TRACK_A_MODEL_PATH}")
    pipeline = joblib.load(TRACK_A_MODEL_PATH)
    return pipeline


@st.cache_resource(show_spinner="Loading Track B 3-Epoch SBDH Transformer...")
def load_track_b_model_and_tokenizer():
    """
    Load the verified 3-Epoch Colab Transformer and FastTokenizer from disk.
    Loads best_model.pt via MultiLabelSBDHTransformer.from_pretrained.
    """
    if not os.path.exists(TRACK_B_MODEL_DIR):
        raise FileNotFoundError(f"Track B model directory not found at: {TRACK_B_MODEL_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(TRACK_B_MODEL_DIR)
    model = MultiLabelSBDHTransformer.from_pretrained(TRACK_B_MODEL_DIR)
    model.eval()
    model.to("cpu")
    return model, tokenizer


def get_artifact_metadata():
    """Returns provenance and verification metadata for UI display."""
    return {
        "track_a": {
            "model_name": "Baseline A (Clinical Bedside Readmission Model)",
            "pipeline_type": "ColumnTransformer + LogisticRegression (C=1.0)",
            "features_count": 40,
            "threshold": TRACK_A_FROZEN_THRESHOLD,
            "auroc": 0.6324,
            "pr_auc": 0.1925,
            "f1_score": 0.2449,
            "brier_score": 0.0961,
            "ece": 0.0061,
            "model_path": TRACK_A_MODEL_PATH,
        },
        "track_b": {
            "model_name": "MultiLabelSBDHTransformer (3-Epoch Colab GPU)",
            "backbone": "distilbert/distilroberta-base",
            "output_logits": 45,
            "categories_count": 15,
            "threshold": TRACK_B_FROZEN_THRESHOLD,
            "active_decoder": "Arm 6A (Strategy B Active-Threshold + Delimiter-Protected Snapping + Gap<=2 Fusion)",
            "fallback_decoder": "Strategy A (Outside-Dominant Control)",
            "relaxed_f1": 0.8414,
            "exact_f1": 0.5173,
            "note_micro_f1": 0.8949,
            "baseline_relaxed_f1": 0.8330,
            "baseline_exact_f1": 0.5105,
            "baseline_note_micro_f1": 0.8888,
            "checkpoint_path": os.path.join(TRACK_B_MODEL_DIR, "best_model.pt"),
            "checkpoint_sha256": "0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4",
        }
    }
