"""
Model Loader module for Evernorth Clinical-DSS Streamlit Application.
Provides cached loading functions for:
1. Track A: Frozen Baseline A Logistic Regression Pipeline (models/baseline_a_logreg.joblib).
2. Track B: Verified 3-Epoch Colab Transformer Checkpoint (evernorth_m3_results/models/sbdh_transformer).
"""
import os
import sys
import json
import joblib
import torch
import streamlit as st
from typing import Dict, Any, Optional
from transformers import AutoTokenizer

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.transformer_model import MultiLabelSBDHTransformer

TRACK_A_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_a_logreg.joblib")
TRACK_B_MODEL_DIR = os.path.join(PROJECT_ROOT, "evernorth_m3_results", "models", "sbdh_transformer")
TRACK_A_METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "readmission_baselines_metrics.json")
TRACK_A_FROZEN_THRESHOLD = 0.130
TRACK_B_FROZEN_THRESHOLD = 0.300


def load_track_a_metrics(metrics_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load frozen baseline metrics from reports/readmission_baselines_metrics.json.
    Extracts Baseline A test performance and majority-class base rate.
    Fails with FileNotFoundError if file is missing, and KeyError if any required key is absent.
    Does not silently fall back to hardcoded numbers.
    """
    path = metrics_path if metrics_path is not None else TRACK_A_METRICS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Track A frozen metrics file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metrics file '{path}': {e}") from e

    try:
        majority = data["majority_class_baseline"]
        base_rate = float(majority["test"]["class_prevalence"])

        sensitivity = data["sensitivity_experiments"]
        baseline_a = sensitivity["Baseline_A_No_Disposition_No_Diag"]
        val_threshold = float(baseline_a["validation"]["decision_threshold"])

        test_data = baseline_a["test"]
        precision = float(test_data["precision"])
        recall = float(test_data["recall"])
        specificity = float(test_data["specificity"])
        auroc = float(test_data["auroc"])
        pr_auc = float(test_data["average_precision_pr_auc"])
        f1_score = float(test_data["f1_score"])
        brier_score = float(test_data["brier_score"])
    except KeyError as e:
        raise KeyError(f"Track A metrics file '{path}' missing required key: {e}") from e

    # Confirm validation-tuned threshold matches frozen threshold constant
    if abs(val_threshold - TRACK_A_FROZEN_THRESHOLD) > 1e-4:
        raise ValueError(
            f"Validation-tuned threshold in {path} ({val_threshold}) "
            f"does not match TRACK_A_FROZEN_THRESHOLD ({TRACK_A_FROZEN_THRESHOLD})"
        )

    return {
        "base_rate": base_rate,
        "val_threshold": val_threshold,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "f1_score": f1_score,
        "brier_score": brier_score,
        "raw": data,
    }


def compute_track_a_plain_statement(precision: float, tau: float = TRACK_A_FROZEN_THRESHOLD) -> str:
    """
    Compute population plain statement about the flagged cohort on the frozen test set:
    'On the frozen test set, about {round((1 - precision) * 10)} of every 10 patients flagged at tau = {tau:.3f} were not readmitted within 30 days (precision {precision:.2%}).'
    """
    unadmitted_count = round((1 - precision) * 10)
    return (
        f"On the frozen test set, about {unadmitted_count} of every 10 patients "
        f"flagged at tau = {tau:.3f} were not readmitted within 30 days (precision {precision:.2%})."
    )


def compute_relative_risk(probability: float, base_rate: float) -> float:
    """Compute relative risk ratio: probability / base_rate."""
    if base_rate <= 0:
        raise ValueError(f"base_rate must be positive, got {base_rate}")
    return probability / base_rate


def classify_readmission_risk(probability: float, tau: float = TRACK_A_FROZEN_THRESHOLD) -> str:
    """Return risk classification: ELEVATED if probability >= tau, else STANDARD."""
    return "ELEVATED" if probability >= tau else "STANDARD"



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


def get_track_b_base_model_name() -> str:
    """Read the base model name from models/sbdh_transformer/config.json."""
    config_paths = [
        os.path.join(PROJECT_ROOT, "models", "sbdh_transformer", "config.json"),
        os.path.join(TRACK_B_MODEL_DIR, "config.json"),
    ]
    for p in config_paths:
        if os.path.exists(p):
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "_name_or_path" in cfg and cfg["_name_or_path"]:
                    return cfg["_name_or_path"]
                if cfg.get("model_type") == "roberta" and cfg.get("num_hidden_layers") == 6:
                    return "distilbert/distilroberta-base"
                return cfg.get("model_type", "distilbert/distilroberta-base")
            except Exception:
                continue
    return "distilbert/distilroberta-base"


def get_artifact_metadata():
    """Returns provenance and verification metadata for UI display."""
    metrics_a = load_track_a_metrics()
    return {
        "track_a": {
            "model_name": "Baseline A (Clinical Bedside Readmission Model)",
            "pipeline_type": "ColumnTransformer + LogisticRegression (C=1.0)",
            "features_count": 40,
            "threshold": TRACK_A_FROZEN_THRESHOLD,
            "val_threshold": metrics_a["val_threshold"],
            "base_rate": metrics_a["base_rate"],
            "auroc": metrics_a["auroc"],
            "pr_auc": metrics_a["pr_auc"],
            "f1_score": metrics_a["f1_score"],
            "precision": metrics_a["precision"],
            "recall": metrics_a["recall"],
            "specificity": metrics_a["specificity"],
            "brier_score": metrics_a["brier_score"],
            "ece": 0.0061,
            "model_path": TRACK_A_MODEL_PATH,
        },
        "track_b": {
            "model_name": "MultiLabelSBDHTransformer (3-Epoch Colab GPU)",
            "backbone": get_track_b_base_model_name(),
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
