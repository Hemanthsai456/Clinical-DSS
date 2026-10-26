"""
Diagnostic script for Experiment 1: Inspect example failure and audit predictions on the user synthetic narrative.
"""
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


try:
    from app.model_loader import load_track_b_model_and_tokenizer
    from app.track_b_ui import render_highlighted_text
except ImportError:
    from app.model_loader import load_track_b_model_and_tokenizer
    from app.track_b_ui import render_highlighted_text

from src.transformer_dataset import (
    decode_predictions_to_spans,
    SBDH_CATEGORIES,
    CAT_TO_IDX,
    IDX_TO_CAT,
    get_label_indices,
)

NARRATIVE = (
    "Patient is a 58-year-old woman with type 2 diabetes who reports difficulty managing her condition. "
    "She has been unemployed for the past six months and is worried about paying her rent and utility bills. "
    "She sometimes skips meals because there is not enough money to buy groceries. "
    "The patient lives alone and reports limited family support. "
    "She does not own a car, and the nearest bus stop is approximately two miles from her home, "
    "making it difficult to attend medical appointments. "
    "She recently missed two follow-up visits because she could not arrange transportation. "
    "She reports persistent knee pain that limits her ability to walk and perform daily activities. "
    "The patient states that she understands her medication instructions but occasionally cannot afford to refill her prescriptions. "
    "She is interested in receiving assistance with food access, transportation, and medication costs."
)


def run_inspection():
    print("=" * 80)
    print("EXPERIMENT 1: NARRATIVE INFERENCE AUDIT")
    print("=" * 80)
    print(f"Narrative Length: {len(NARRATIVE)} characters")
    
    model, tokenizer = load_track_b_model_and_tokenizer()
    
    encoding = tokenizer(
        NARRATIVE,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"][0].tolist()
    seq_len = len(offset_mapping)
    print(f"Tokenized Sequence Length: {seq_len} tokens (max_length=256)")
    print(f"Truncation Occurred: {seq_len == 256 and input_ids.shape[1] == 256}")
    
    # Active mask
    active_mask = np.zeros(seq_len, dtype=np.float32)
    for t, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        active_mask[t] = 1.0

    # Model inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"].squeeze(0).cpu().numpy()  # (seq_len, 45)
        probs = torch.sigmoid(outputs["logits"]).squeeze(0).cpu().numpy()

    # Decode at threshold 0.30
    spans_030 = decode_predictions_to_spans(
        logits=logits,
        active_mask=active_mask,
        offset_mapping=np.array(offset_mapping),
        text=NARRATIVE,
        threshold=0.30,
    )
    
    print("\n--- DECODED PREDICTIONS AT THRESHOLD 0.30 ---")
    sorted_spans = sorted(list(spans_030), key=lambda x: (x[0], x[1]))
    print(f"Total Spans Extracted: {len(sorted_spans)}")
    unique_cats = set(s[2] for s in sorted_spans)
    print(f"Unique Categories: {len(unique_cats)} -> {unique_cats}")
    
    for i, (s_char, e_char, cat) in enumerate(sorted_spans, 1):
        span_text = NARRATIVE[s_char:e_char]
        cat_idx = CAT_TO_IDX[cat]
        b_idx, i_idx, o_idx = get_label_indices(cat_idx)
        # Find overlapping tokens
        rel_tokens = [t for t in range(seq_len) if active_mask[t] == 1.0 and max(offset_mapping[t][0], s_char) < min(offset_mapping[t][1], e_char)]
        max_b_prob = float(np.max(probs[rel_tokens, b_idx])) if rel_tokens else 0.0
        max_i_prob = float(np.max(probs[rel_tokens, i_idx])) if len(rel_tokens) > 1 else 0.0
        print(f"Span {i}: [{s_char}:{e_char}] '{span_text}' | Category: '{cat}' | max(P_B)={max_b_prob:.4f}, max(P_I)={max_i_prob:.4f}")

    # UI Rendering comparison
    print("\n--- UI RENDERING TEST ---")
    rendered_html = render_highlighted_text(NARRATIVE, sorted_spans)
    print("Rendered HTML contains:")
    for _, _, cat in sorted_spans:
        print(f"  Category tag present in HTML: '{cat}' -> {cat in rendered_html}")

    # Audit all tokens: which categories had ANY probability > 0.10 across the sequence?
    print("\n--- TOKEN-LEVEL PROBABILITY AUDIT (> 0.10) ACROSS ALL 15 CATEGORIES ---")
    for cat in SBDH_CATEGORIES:
        cat_idx = CAT_TO_IDX[cat]
        b_idx, i_idx, o_idx = get_label_indices(cat_idx)
        max_b = float(np.max(probs[:, b_idx]))
        max_i = float(np.max(probs[:, i_idx]))
        min_o = float(np.min(probs[:, o_idx]))
        tokens_b_high = [(t, offset_mapping[t], NARRATIVE[offset_mapping[t][0]:offset_mapping[t][1]], float(probs[t, b_idx])) 
                         for t in range(seq_len) if probs[t, b_idx] >= 0.10]
        tokens_i_high = [(t, offset_mapping[t], NARRATIVE[offset_mapping[t][0]:offset_mapping[t][1]], float(probs[t, i_idx])) 
                         for t in range(seq_len) if probs[t, i_idx] >= 0.10]
        
        if max_b >= 0.10 or max_i >= 0.10:
            print(f"\nCategory '{cat}' (cat_idx={cat_idx}): max_P(B)={max_b:.4f}, max_P(I)={max_i:.4f}, min_P(O)={min_o:.4f}")
            if tokens_b_high:
                print("  B tokens >= 0.10:")
                for t, off, tok_txt, p in tokens_b_high:
                    print(f"    tok {t} {off} '{tok_txt}': P(B)={p:.4f}, P(O)={probs[t, o_idx]:.4f}")
            if tokens_i_high:
                print("  I tokens >= 0.10:")
                for t, off, tok_txt, p in tokens_i_high:
                    print(f"    tok {t} {off} '{tok_txt}': P(I)={p:.4f}, P(O)={probs[t, o_idx]:.4f}")
        else:
            print(f"Category '{cat}': completely sub-threshold (max P(B)={max_b:.4f}, max P(I)={max_i:.4f})")

    # Threshold sensitivity on this specific narrative
    print("\n--- THRESHOLD SENSITIVITY ON THIS SPECIFIC NARRATIVE ---")
    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        spans_th = decode_predictions_to_spans(
            logits=logits,
            active_mask=active_mask,
            offset_mapping=np.array(offset_mapping),
            text=NARRATIVE,
            threshold=th,
        )
        sorted_th = sorted(list(spans_th), key=lambda x: (x[0], x[1]))
        cats_th = [f"{s[2]} ('{NARRATIVE[s[0]:s[1]]}')" for s in sorted_th]
        print(f"Threshold {th:.2f}: {len(sorted_th)} spans, {len(set(s[2] for s in sorted_th))} categories: {cats_th}")


if __name__ == "__main__":
    run_inspection()
