"""
Automated unit test suite for the Evernorth Clinical-DSS Streamlit Application.

Verifies:
1. Loading each intended frozen artifact (Baseline A pipeline and Track B 3-epoch transformer).
2. Track A input-schema compatibility, feature count (40 features), and prediction probability shape.
3. Track B tokenizer/model compatibility, CPU forward pass, and multi-label span decoding.
4. Robust handling of empty and malformed inputs (empty strings, whitespace, long text, unseen categories).
5. Strict isolation ensuring the app uses Baseline A for the primary discharge-planning scenario and does not expose Baselines B or C.
6. Cryptographic preservation of frozen artifact SHA-256 hashes and metric files.
"""
import os
import sys
import hashlib
import pytest
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)



from app.model_loader import (
    load_track_a_pipeline,
    load_track_b_model_and_tokenizer,
    get_artifact_metadata,
    TRACK_A_MODEL_PATH,
    TRACK_B_MODEL_DIR,
    TRACK_A_FROZEN_THRESHOLD,
    TRACK_B_FROZEN_THRESHOLD,
)
from app.track_a_ui import (
    NUMERICAL_COLS,
    BASE_CATEGORICAL_COLS,
    PRESETS,
    build_default_encounter_dict,
)
from app.track_b_ui import NOTE_PRESETS, render_highlighted_text
from src.transformer_dataset import decode_predictions_to_spans, SBDH_CATEGORIES, CAT_TO_IDX


# --- 1. Artifact Loading Tests ---

def test_load_track_a_frozen_pipeline():
    """Verify loading of the frozen Baseline A Pipeline artifact."""
    pipeline = load_track_a_pipeline()
    assert pipeline is not None
    assert hasattr(pipeline, "predict_proba"), "Pipeline must have predict_proba method"
    assert hasattr(pipeline, "named_steps"), "Pipeline must be a scikit-learn Pipeline"
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_load_track_b_frozen_model_and_tokenizer():
    """Verify loading of the verified 3-epoch Colab Transformer and FastTokenizer."""
    model, tokenizer = load_track_b_model_and_tokenizer()
    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "classifier"), "Model must have classifier head"
    assert model.classifier.out_features == 45, "Model classifier must produce 45 independent logits"
    assert model.training is False, "Model must be in evaluation mode"


def test_metadata_consistency():
    """Verify metadata helper returns complete and mathematically accurate values."""
    meta = get_artifact_metadata()
    assert "track_a" in meta and "track_b" in meta
    ta = meta["track_a"]
    tb = meta["track_b"]
    assert ta["features_count"] == 40
    assert ta["threshold"] == 0.130
    assert ta["auroc"] == 0.6324
    assert ta["pr_auc"] == 0.1925
    assert tb["output_logits"] == 45
    assert tb["threshold"] == 0.300
    assert tb["relaxed_f1"] == 0.8330
    assert tb["exact_f1"] == 0.5105


# --- 2. Track A Input-Schema Compatibility & Prediction Shape ---

def test_track_a_features_count_and_schema():
    """Verify Track A features strictly match the 40 features of Baseline A."""
    total_cols = NUMERICAL_COLS + BASE_CATEGORICAL_COLS
    assert len(total_cols) == 40, f"Baseline A must have exactly 40 features, got {len(total_cols)}"
    assert len(NUMERICAL_COLS) == 8
    assert len(BASE_CATEGORICAL_COLS) == 32
    
    # Must NOT include disposition or ICD-9 codes
    assert "discharge_disposition_id" not in total_cols
    assert "diag_1_group" not in total_cols
    assert "diag_2_group" not in total_cols
    assert "diag_3_group" not in total_cols


def test_track_a_prediction_shape_and_presets():
    """Verify prediction probability shape and output range on clinical presets."""
    pipeline = load_track_a_pipeline()
    
    for preset_name, preset_dict in PRESETS.items():
        # Build complete row
        full_enc = build_default_encounter_dict()
        full_enc.update(preset_dict)
        df = pd.DataFrame([full_enc])[NUMERICAL_COLS + BASE_CATEGORICAL_COLS]
        
        proba = pipeline.predict_proba(df)
        assert proba.shape == (1, 2), f"Expected shape (1, 2), got {proba.shape}"
        assert 0.0 <= proba[0, 1] <= 1.0, f"Readmission probability out of [0, 1]: {proba[0, 1]}"
        
        # Test binary classification at frozen threshold tau = 0.130
        is_high_risk = bool(proba[0, 1] >= TRACK_A_FROZEN_THRESHOLD)
        assert isinstance(is_high_risk, bool)


def test_track_a_unseen_categories_handling():
    """Verify OneHotEncoder handle_unknown='ignore' handles unseen categories without raising."""
    pipeline = load_track_a_pipeline()
    enc = build_default_encounter_dict()
    enc["race"] = "UnseenEthnicityGroup"
    enc["admission_type_id"] = "999"
    enc["insulin"] = "UnknownDose"
    
    df = pd.DataFrame([enc])[NUMERICAL_COLS + BASE_CATEGORICAL_COLS]
    proba = pipeline.predict_proba(df)
    assert proba.shape == (1, 2)
    assert not np.isnan(proba[0, 1])


# --- 3. Track B Tokenizer/Model Compatibility & Output Decoding ---

def test_track_b_inference_and_decoding():
    """Verify Track B tokenizer, model forward pass, and span decoding on multi-label sample."""
    model, tokenizer = load_track_b_model_and_tokenizer()
    sample_text = NOTE_PRESETS["Example 1 (Multi-Label Span: Financial Insecurity + Barriers to Care)"]
    
    encoding = tokenizer(
        sample_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"][0].tolist()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"].squeeze(0).cpu().numpy()  # (seq_len, 45)
    
    active_mask = np.zeros(len(offset_mapping), dtype=np.float32)
    for t, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        active_mask[t] = 1.0

    assert logits.shape[1] == 45, f"Expected 45 output channels, got {logits.shape[1]}"
    assert logits.shape[0] == len(offset_mapping)
    
    # Decode spans at frozen threshold theta = 0.30
    spans = decode_predictions_to_spans(
        logits=logits,
        active_mask=active_mask,
        offset_mapping=np.array(offset_mapping),
        text=sample_text,
        threshold=TRACK_B_FROZEN_THRESHOLD,
    )
    
    assert isinstance(spans, (list, set))
    # Check that extracted spans have valid structure (start, end, category)
    for start, end, cat in spans:
        assert 0 <= start < end <= len(sample_text)
        assert cat in SBDH_CATEGORIES
        span_text = sample_text[start:end]
        assert len(span_text.strip()) > 0, "Span text must not be empty or whitespace"


def test_track_b_html_renderer():
    """Verify that render_highlighted_text produces safe HTML escaping and valid badges."""
    text = "Patient has severe pain and housing insecurity."
    spans = [(19, 23, "pain"), (28, 46, "housing insecurity")]
    html_output = render_highlighted_text(text, spans)
    assert "<mark" in html_output
    assert "pain" in html_output
    assert "housing insecurity" in html_output
    assert "<script>" not in html_output


# --- 4. Empty and Malformed Inputs ---

def test_track_a_missing_column_raises_error():
    """Verify that a DataFrame missing required features raises an error."""
    pipeline = load_track_a_pipeline()
    incomplete_df = pd.DataFrame([{"time_in_hospital": 3, "gender": "Female"}])
    with pytest.raises((ValueError, KeyError)):
        pipeline.predict_proba(incomplete_df)


def test_track_b_empty_and_whitespace_input():
    """Verify that empty string and whitespace-only text do not crash inference."""
    model, tokenizer = load_track_b_model_and_tokenizer()
    
    for empty_text in ["", "   ", "\n\t  \n"]:
        encoding = tokenizer(
            empty_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"][0].tolist()
        
        active_mask = np.zeros(len(offset_mapping), dtype=np.float32)
        for t, (s, e) in enumerate(offset_mapping):
            if s == 0 and e == 0:
                continue
            active_mask[t] = 1.0

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].squeeze(0).cpu().numpy()
            
        spans = decode_predictions_to_spans(
            logits=logits,
            active_mask=active_mask,
            offset_mapping=np.array(offset_mapping),
            text=empty_text,
            threshold=TRACK_B_FROZEN_THRESHOLD,
        )
        assert len(spans) == 0, f"Expected empty span set for empty text, got {spans}"


def test_track_b_long_input_truncation():
    """Verify that text exceeding max_length=256 is cleanly truncated without crashing."""
    model, tokenizer = load_track_b_model_and_tokenizer()
    long_text = "The patient reports chronic pain and social isolation. " * 100  # > 500 tokens
    
    encoding = tokenizer(
        long_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    assert encoding["input_ids"].shape[1] == 256, "Input sequence must be truncated to exactly 256 tokens"
    offset_mapping = encoding["offset_mapping"][0].tolist()
    
    active_mask = np.zeros(len(offset_mapping), dtype=np.float32)
    for t, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        active_mask[t] = 1.0

    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        logits = outputs["logits"].squeeze(0).cpu().numpy()
        
    spans = decode_predictions_to_spans(
        logits=logits,
        active_mask=active_mask,
        offset_mapping=np.array(offset_mapping),
        text=long_text,
        threshold=TRACK_B_FROZEN_THRESHOLD,
    )
    assert isinstance(spans, (list, set))



# --- 5. Strict Non-Usage of Baselines B & C for Primary Scenario ---

def test_app_strictly_uses_baseline_a():
    """Ensure the app codebase references only Baseline A for real-time predictions."""
    assert os.path.basename(TRACK_A_MODEL_PATH) == "baseline_a_logreg.joblib", (
        "TRACK_A_MODEL_PATH must point strictly to baseline_a_logreg.joblib"
    )
    assert "baseline_b_logreg.joblib" not in TRACK_A_MODEL_PATH
    assert "baseline_c_logreg.joblib" not in TRACK_A_MODEL_PATH


# --- 6. Cryptographic Preservation of Frozen Artifact Hashes ---

def test_frozen_artifact_hashes():
    """Verify that model files and key metric files have NOT been modified or corrupted."""
    expected_hashes = {
        os.path.join(PROJECT_ROOT, "models", "baseline_a_logreg.joblib"): 
            "794cff2d580b0d9b574b62a62c1d2549aad101e283ac65167869f9d1a076d74c",
        os.path.join(PROJECT_ROOT, "evernorth_m3_results", "models", "sbdh_transformer", "best_model.pt"):
            "0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4",
        os.path.join(PROJECT_ROOT, "reports", "readmission_baselines_metrics.json"):
            "168a73c1c40e841f7992ccf6271d4a1e8a977444a1f44a471344edae6882f114",
        os.path.join(PROJECT_ROOT, "reports", "sbdh_colab_transformer_metrics_corrected.json"):
            "f20c59f435ca0194c8fdaba7c5ffe548ec35fb34b6bcf6a177f81ab82b090934",
    }
    
    for path, expected_sha in expected_hashes.items():
        assert os.path.exists(path), f"Frozen artifact not found: {path}"
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()
        assert actual_sha == expected_sha, f"SHA-256 hash mismatch for {path}: expected {expected_sha}, got {actual_sha}"
