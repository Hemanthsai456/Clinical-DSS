"""
Automated unit test suite for Milestone 3 Multi-Label Transformer Token Alignment and Modeling.
Verifies:
1. ex_no 7062: Token alignment preserves multiple SBDH categories on identical character intervals
   (financial insecurity and barriers to care are active simultaneously on the same tokens).
2. ex_no 1192: Unresolved ambiguous repeated mentions in training set are explicitly masked out
   from the loss calculation (loss_mask = 0 for category 'pain' on candidate tokens).
3. Special tokens and padding tokens have active_mask = 0.0.
4. FastTokenizer character offset decoding extracts correct character boundaries from logits.
5. Transformer model forward pass produces valid 45-dim logits and finite masked BCE loss.
"""
import os
import sys
import pytest
import numpy as np
import torch
from transformers import AutoTokenizer

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transformer_dataset import (
    SBDH_CATEGORIES,
    CAT_TO_IDX,
    get_label_indices,
    encode_single_note,
    decode_predictions_to_spans,
)
from src.transformer_model import MultiLabelSBDHTransformer


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert/distilroberta-base")


def test_multi_label_identical_span_encoding_ex_no_7062(tokenizer):
    """
    ex_no 7062:
    Text: 'Patient suffers from chronic pain and cannot afford to continue physical therapy.'
    Phrase 'cannot afford to continue physical therapy' is annotated with both:
    1. 'financial insecurity'
    2. 'barriers to care'
    Verify that tokens overlapping [38, 80] have active B/I targets for BOTH categories!
    """
    text = "Patient suffers from chronic pain and cannot afford to continue physical therapy."
    gold_intervals = [
        {
            "start_char": 38,
            "end_char": 80,
            "span_text": "cannot afford to continue physical therapy",
            "categories": ["financial insecurity", "barriers to care"],
            "ex_no": 7062,
        }
    ]

    encoded = encode_single_note(
        text=text,
        gold_intervals=gold_intervals,
        tokenizer=tokenizer,
        max_length=64,
        ex_no=7062,
    )

    targets = encoded["targets"]
    offset_mapping = encoded["offset_mapping"]
    active_mask = encoded["active_mask"]

    fin_idx = CAT_TO_IDX["financial insecurity"]
    bar_idx = CAT_TO_IDX["barriers to care"]

    fin_b, fin_i, fin_o = get_label_indices(fin_idx)
    bar_b, bar_i, bar_o = get_label_indices(bar_idx)

    # Find tokens overlapping [38, 80]
    afford_tokens = []
    for i in range(len(active_mask)):
        if active_mask[i] == 1.0:
            t_start, t_end = offset_mapping[i]
            if max(t_start, 38) < min(t_end, 80):
                afford_tokens.append(i)

    assert len(afford_tokens) >= 5, f"Expected multiple tokens covering phrase, got {len(afford_tokens)}"

    # First token must be B for BOTH categories
    first_tok = afford_tokens[0]
    assert targets[first_tok, fin_b] == 1.0, "First token must have B for financial insecurity"
    assert targets[first_tok, bar_b] == 1.0, "First token must have B for barriers to care"
    assert targets[first_tok, fin_o] == 0.0
    assert targets[first_tok, bar_o] == 0.0

    # Subsequent tokens must be I for BOTH categories
    for sub_tok in afford_tokens[1:]:
        assert targets[sub_tok, fin_i] == 1.0, f"Token {sub_tok} must have I for financial insecurity"
        assert targets[sub_tok, bar_i] == 1.0, f"Token {sub_tok} must have I for barriers to care"
        assert targets[sub_tok, fin_o] == 0.0
        assert targets[sub_tok, bar_o] == 0.0


def test_ex_no_1192_ambiguous_span_masking(tokenizer):
    """
    ex_no 1192:
    Training note with ambiguous repeated 'pain' span with 2 candidates: [10:14] and [61:65].
    Rather than guessing an arbitrary offset, candidate tokens for 'pain' must have loss_mask = 0.0.
    """
    text = "Patient's pain is severe and has been prescribed opioids for pain management."
    ambiguous_cases = [
        {
            "ex_no": 1192,
            "span": "pain",
            "sbdh": "pain",
            "num_candidates": 2,
            "candidates": [
                {"start_char": 10, "end_char": 14},
                {"start_char": 61, "end_char": 65},
            ],
        }
    ]

    encoded = encode_single_note(
        text=text,
        gold_intervals=[],
        tokenizer=tokenizer,
        max_length=64,
        ex_no=1192,
        ambiguous_cases=ambiguous_cases,
    )

    loss_mask = encoded["loss_mask"]
    offset_mapping = encoded["offset_mapping"]
    active_mask = encoded["active_mask"]
    pain_idx = CAT_TO_IDX["pain"]

    # Verify that tokens covering [10, 14] and [61, 65] have loss_mask = 0.0 for pain
    cand1_tokens = [i for i in range(len(active_mask)) if active_mask[i] == 1.0 and max(offset_mapping[i][0], 10) < min(offset_mapping[i][1], 14)]
    cand2_tokens = [i for i in range(len(active_mask)) if active_mask[i] == 1.0 and max(offset_mapping[i][0], 61) < min(offset_mapping[i][1], 65)]

    assert len(cand1_tokens) > 0
    assert len(cand2_tokens) > 0

    for t in cand1_tokens:
        assert loss_mask[t, pain_idx] == 0.0, f"Candidate 1 token {t} must be masked out in loss"
    for t in cand2_tokens:
        assert loss_mask[t, pain_idx] == 0.0, f"Candidate 2 token {t} must be masked out in loss"

    # Other categories on those tokens must remain unmasked (1.0)
    for other_cat in range(len(SBDH_CATEGORIES)):
        if other_cat != pain_idx:
            for t in cand1_tokens:
                assert loss_mask[t, other_cat] == 1.0


def test_special_tokens_padding_masking(tokenizer):
    """
    Verify that special tokens (<s>, </s>) and padding have active_mask = 0.0.
    """
    text = "Short clinical note."
    encoded = encode_single_note(
        text=text,
        gold_intervals=[],
        tokenizer=tokenizer,
        max_length=16,
    )

    active_mask = encoded["active_mask"]
    input_ids = encoded["input_ids"]

    # First token is <s> (BOS), last text token is followed by </s> (EOS) and padding
    assert active_mask[0] == 0.0, "BOS token must have active_mask = 0.0"
    assert active_mask[-1] == 0.0, "Pad token must have active_mask = 0.0"
    assert np.sum(active_mask) > 0, "Real tokens must have active_mask = 1.0"


def test_decoding_roundtrip_character_offsets(tokenizer):
    """
    Simulate predicted logits and verify decode_predictions_to_spans extracts exact character boundaries.
    """
    text = "Patient disclosed recent job loss and eviction."
    # 'job loss' is at [25, 33]
    # 'eviction' is at [38, 46]
    encoded = encode_single_note(
        text=text,
        gold_intervals=[],
        tokenizer=tokenizer,
        max_length=32,
    )

    seq_len = len(encoded["active_mask"])
    logits = np.full((seq_len, len(SBDH_CATEGORIES) * 3), -5.0)  # Low probability default

    job_idx = CAT_TO_IDX["job insecurity"]
    job_b, job_i, _ = get_label_indices(job_idx)

    # Find tokens for 'job loss'
    offset_mapping = encoded["offset_mapping"]
    job_tokens = [i for i in range(seq_len) if encoded["active_mask"][i] == 1.0 and max(offset_mapping[i][0], 25) < min(offset_mapping[i][1], 33)]

    assert len(job_tokens) >= 2
    logits[job_tokens[0], job_b] = 5.0
    for t in job_tokens[1:]:
        logits[t, job_i] = 5.0

    decoded_spans = decode_predictions_to_spans(
        logits=logits,
        active_mask=encoded["active_mask"],
        offset_mapping=encoded["offset_mapping"],
        text=text,
        threshold=0.5,
    )

    assert len(decoded_spans) == 1
    span = list(decoded_spans)[0]
    assert span[2] == "job insecurity"
    assert text[span[0]:span[1]] == "job loss"
    assert span[0] == 25
    assert span[1] == 33


def test_transformer_forward_pass_and_loss():
    """
    Test forward pass of MultiLabelSBDHTransformer with synthetic batch.
    """
    model = MultiLabelSBDHTransformer(model_name="distilbert/distilroberta-base")
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(100, 2000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    active_mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
    targets = torch.zeros((batch_size, seq_len, 45), dtype=torch.float32)
    loss_mask = torch.ones((batch_size, seq_len, 15), dtype=torch.float32)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            active_mask=active_mask,
            targets=targets,
            loss_mask=loss_mask,
        )

    assert out["logits"].shape == (batch_size, seq_len, 45)
    assert out["probabilities"].shape == (batch_size, seq_len, 45)
    assert out["loss"] is not None
    assert not torch.isnan(out["loss"])
    assert out["loss"].item() > 0.0


def test_token_truncation_detection(tokenizer):
    """
    Test that spans located beyond max_length tokens are detected in truncated_intervals
    without being silently dropped or mislabeled.
    """
    long_text = ("The patient had a routine checkup. " * 30) + "Finally reports severe chronic pain."
    # The span 'chronic pain' is way past 64 tokens
    pain_start = long_text.find("chronic pain")
    pain_end = pain_start + len("chronic pain")
    gold_intervals = [
        {
            "start_char": pain_start,
            "end_char": pain_end,
            "span_text": "chronic pain",
            "categories": ["pain"],
            "ex_no": 9991,
        }
    ]

    encoded = encode_single_note(
        text=long_text,
        gold_intervals=gold_intervals,
        tokenizer=tokenizer,
        max_length=64,
        ex_no=9991,
    )

    assert len(encoded["truncated_intervals"]) == 1
    assert encoded["truncated_intervals"][0]["start_char"] == pain_start
    # Active tokens should not have B or I for pain since it was truncated
    pain_idx = CAT_TO_IDX["pain"]
    b_col = pain_idx * 3
    i_col = pain_idx * 3 + 1
    assert (encoded["targets"][:, b_col] == 1.0).sum() == 0
    assert (encoded["targets"][:, i_col] == 1.0).sum() == 0


def test_deterministic_bio_decoding_and_whitespace_trimming(tokenizer):
    """
    Verify that decode_predictions_to_spans correctly resolves BPE leading space offsets
    and trims whitespace against raw text.
    """
    text = "Patient feels severe anxiety."
    gold_intervals = [
        {
            "start_char": 21,
            "end_char": 28,
            "span_text": "anxiety",
            "categories": ["psychiatric symptoms or disorders"],
            "ex_no": 9992,
        }
    ]
    encoded = encode_single_note(
        text=text,
        gold_intervals=gold_intervals,
        tokenizer=tokenizer,
        max_length=32,
        ex_no=9992,
    )

    # Convert targets directly to pseudo-logits (+10 for active, -10 for inactive)
    pseudo_logits = np.where(encoded["targets"] == 1.0, 10.0, -10.0)

    decoded = decode_predictions_to_spans(
        logits=pseudo_logits,
        active_mask=encoded["active_mask"],
        offset_mapping=encoded["offset_mapping"],
        text=text,
        threshold=0.3,
    )

    assert len(decoded) == 1
    d_start, d_end, d_cat = list(decoded)[0]
    assert d_cat == "psychiatric symptoms or disorders"
    assert text[d_start:d_end] == "anxiety"
    assert d_start == 21
    assert d_end == 28


def test_colab_checkpoint_provenance_and_artifact_consistency():
    """
    Verify provenance of the 3-epoch Colab-trained transformer checkpoint
    and mathematical consistency of its evaluation artifact on test_reviewed.
    """
    import json
    import hashlib

    colab_weights = os.path.join("evernorth_m3_results", "models", "sbdh_transformer", "best_model.pt")
    local_weights = os.path.join("models", "sbdh_transformer", "pytorch_model.bin")

    assert os.path.exists(colab_weights), f"Colab weights not found: {colab_weights}"
    assert os.path.exists(local_weights), f"Local weights not found: {local_weights}"

    # Verify hashes are distinct
    with open(colab_weights, "rb") as f:
        colab_hash = hashlib.sha256(f.read()).hexdigest()
    with open(local_weights, "rb") as f:
        local_hash = hashlib.sha256(f.read()).hexdigest()

    assert colab_hash != local_hash, "Colab and local checkpoints must not share identical SHA256 hashes!"

    # Verify loading via from_pretrained
    model = MultiLabelSBDHTransformer.from_pretrained(os.path.dirname(colab_weights))
    assert model.classifier.out_features == 45

    # Verify evaluation artifact
    eval_artifact = os.path.join("reports", "sbdh_colab_transformer_metrics_corrected.json")
    assert os.path.exists(eval_artifact), f"Evaluation artifact not found: {eval_artifact}"

    with open(eval_artifact, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["total_raw_gold_annotations"] == 3345
    assert metrics["resolvable_gold_annotations_count"] == 3340
    assert metrics["unresolved_ambiguous_annotations_count"] == 5
    assert len(metrics["unresolved_ambiguous_cases"]) == 5

    ex = metrics["span_level_exact_matching"]["resolvable_subset"]
    assert ex["tp"] == 1616
    assert ex["fp"] == 1375
    assert ex["fn"] == 1724
    assert abs(ex["precision"] - (1616 / (1616 + 1375))) < 1e-4
    assert abs(ex["recall"] - (1616 / 3340)) < 1e-4

    rel = metrics["span_level_relaxed_matching"]["resolvable_subset"]
    assert rel["tp"] == 2637
    assert rel["fp"] == 354
    assert rel["fn"] == 703
    assert abs(rel["precision"] - (2637 / (2637 + 354))) < 1e-4
    assert abs(rel["recall"] - (2637 / 3340)) < 1e-4


