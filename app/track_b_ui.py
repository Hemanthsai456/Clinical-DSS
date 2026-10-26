"""
Track B UI Module: Social and Behavioral Determinants of Health (SBDH) Text Extraction.
Uses the verified 3-epoch Colab-trained Transformer checkpoint (evernorth_m3_results/models/sbdh_transformer/best_model.pt).
Performs 100% local CPU inference without sending clinical text to external networks or APIs.
"""
import streamlit as st
import numpy as np
import torch
import html
import pandas as pd
from model_loader import load_track_b_model_and_tokenizer, TRACK_B_FROZEN_THRESHOLD

from src.transformer_dataset import decode_predictions_to_spans, SBDH_CATEGORIES, CAT_TO_IDX, get_label_indices

NOTE_PRESETS = {
    "Example 1 (Multi-Label Span: Financial Insecurity + Barriers to Care)": (
        "Patient suffers from chronic back pain and cannot afford to continue physical therapy. "
        "Reports missing appointments due to lack of copay funds."
    ),
    "Example 2 (Housing & Food Insecurity, Unemployment)": (
        "Patient recently experienced job loss and faced foreclosure on his apartment. "
        "Currently staying in temporary homeless shelter and relies on local emergency food banks."
    ),
    "Example 3 (Patient Disability, Social Isolation, Substance Abuse)": (
        "Patient uses a motorized wheelchair due to severe neuropathy. Expresses profound loneliness "
        "and social isolation living alone, with a past history of heavy alcohol abuse."
    ),
}

CATEGORY_COLORS = {
    "barriers to care": "#F59E0B",
    "financial insecurity": "#EF4444",
    "food insecurity": "#10B981",
    "housing insecurity": "#8B5CF6",
    "job insecurity": "#EC4899",
    "legal problems": "#6B7280",
    "loss of relationship": "#F97316",
    "pain": "#DC2626",
    "patient disability": "#3B82F6",
    "physical isolation": "#14B8A6",
    "psychiatric symptoms or disorders": "#6366F1",
    "social isolation": "#06B6D4",
    "substance abuse": "#7C3AED",
    "transitions of care": "#D97706",
    "violence": "#B91C1C",
}


def render_highlighted_text(text: str, extracted_spans):
    """Render text with color-coded highlight tags for extracted SBDH spans."""
    if not extracted_spans:
        return f"<div style='padding: 12px; background: #F9FAFB; border-radius: 8px; line-height: 1.8; color: #111827;'>{html.escape(text)}</div>"

    # Sort spans by start_char ascending
    sorted_spans = sorted(extracted_spans, key=lambda s: (s[0], s[1]))
    
    # Merge overlapping or identical intervals for visual markup
    intervals = []
    for s_char, e_char, cat in sorted_spans:
        intervals.append({
            "start": max(0, min(s_char, len(text))),
            "end": max(0, min(e_char, len(text))),
            "category": cat,
            "text": text[s_char:e_char]
        })

    # Render HTML spans
    out = []
    last_idx = 0
    # Group intervals that start at same index
    for item in intervals:
        s = item["start"]
        e = item["end"]
        cat = item["category"]
        color = CATEGORY_COLORS.get(cat, "#4B5563")

        if s > last_idx:
            out.append(html.escape(text[last_idx:s]))
        
        span_content = html.escape(text[s:e])
        badge = f"<mark style='background-color: {color}22; border-bottom: 3px solid {color}; padding: 2px 4px; border-radius: 4px; font-weight: 500;' title='{cat}'>{span_content} <span style='font-size: 0.72rem; color: {color}; font-weight: bold;'>[{cat}]</span></mark>"
        out.append(badge)
        last_idx = max(last_idx, e)

    if last_idx < len(text):
        out.append(html.escape(text[last_idx:]))

    return f"<div style='padding: 14px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; line-height: 2.0; font-family: sans-serif; font-size: 1.05rem; color: #0F172A;'>{''.join(out)}</div>"


def render_track_b_page():
    st.header("Track B — Social & Behavioral Determinants of Health (SBDH) Extraction")
    st.markdown(
        """
        **Clinical CDS Setting:** Unstructured Narrative SBDH Extraction (Discharge Summaries & Social Work Notes).  
        **Model Architecture:** Multi-Label BIO Sequence Tagger (`distilbert/distilroberta-base`, 45 independent binary sigmoid logits).  
        **Multi-Label Capability:** Explicitly supports multiple concurrent SBDH categories mapped to identical character intervals.  
        **Privacy Guarantee:** 100% Local In-Memory Inference. No user narrative is transmitted over any external network or cloud service.
        """
    )

    try:
        model, tokenizer = load_track_b_model_and_tokenizer()
    except Exception as ex:
        st.error(f"Failed to load Track B model: {ex}")
        st.stop()

    # Preset selection
    st.subheader("1. Clinical Narrative Input")
    preset_choice = st.selectbox(
        "Select an illustrative clinical note preset or enter custom narrative below:",
        ["Custom Manual Entry"] + list(NOTE_PRESETS.keys())
    )

    default_text = ""
    if preset_choice in NOTE_PRESETS:
        default_text = NOTE_PRESETS[preset_choice]

    text_input = st.text_area(
        "Clinical Narrative Text:",
        value=default_text,
        height=140,
        placeholder="Enter de-identified narrative text describing patient living conditions, financial barriers, mobility, or support systems..."
    )

    # Threshold slider
    t_col1, t_col2 = st.columns([2, 2])
    with t_col1:
        threshold = st.slider(
            "Detection Decision Threshold (Sigmoid Probability)",
            min_value=0.10,
            max_value=0.70,
            value=TRACK_B_FROZEN_THRESHOLD,
            step=0.05,
            help="Threshold calibrated on validation data (0.30 maximizes F1 across Synth-SBDH categories)."
        )
    with t_col2:
        st.markdown(
            f"""
            - **Calibrated Operational Threshold:** `0.30`
            - **Evaluated Performance:** Exact F1 **0.5105**, Relaxed F1 **0.8330**, Note Micro F1 **0.8888**
            - **Backbone:** DistilRoBERTa (82M parameters, 6 layers)
            """
        )

    # Extraction Trigger
    st.subheader("2. Extracted SBDH Entities & Spans")
    if st.button("Extract SBDH Categories and Spans", type="primary"):
        cleaned_text = text_input.strip()
        if not cleaned_text:
            st.warning("Please provide clinical narrative text before running extraction.")
            st.stop()

        # Tokenization
        encoded = tokenizer(
            cleaned_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offset_mapping = encoded["offset_mapping"].squeeze(0).cpu().numpy()

        # Check for length truncation
        raw_tokens_count = len(tokenizer.encode(cleaned_text, add_special_tokens=True))
        if raw_tokens_count > 256:
            st.warning(f"Note length ({raw_tokens_count} tokens) exceeds model context (256 tokens). Text beyond token 256 was truncated.")

        # Compute active mask for real text tokens
        active_mask = np.zeros(len(offset_mapping), dtype=float)
        for t in range(len(offset_mapping)):
            s, e = offset_mapping[t]
            if s == 0 and e == 0:
                continue
            active_mask[t] = 1.0

        # Model Inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].squeeze(0).cpu().numpy()

        # Deterministic BIO Span Decoding
        extracted_spans = decode_predictions_to_spans(
            logits=logits,
            active_mask=active_mask,
            offset_mapping=offset_mapping,
            text=cleaned_text,
            threshold=threshold
        )

        # Highlighted visual markup
        st.markdown("**Narrative with Extracted SBDH Span Annotations:**", unsafe_allow_html=True)
        html_markup = render_highlighted_text(cleaned_text, extracted_spans)
        st.markdown(html_markup, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if extracted_spans:
            st.success(f"Extracted {len(extracted_spans)} SBDH span(s) across {len(set(s[2] for s in extracted_spans))} unique category(ies).")

            # Structured Table
            table_rows = []
            for s_char, e_char, cat in sorted(extracted_spans, key=lambda x: (x[0], x[1])):
                span_text = cleaned_text[s_char:e_char]
                cat_idx = CAT_TO_IDX[cat]
                b_idx, i_idx, _ = get_label_indices(cat_idx)
                # Compute token-level max confidence
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
                # Approximate span confidence
                rel_tokens = [t for t in range(len(offset_mapping)) if active_mask[t] == 1.0 and max(offset_mapping[t][0], s_char) < min(offset_mapping[t][1], e_char)]
                conf = float(np.max(probs[rel_tokens, b_idx])) if rel_tokens else 0.50

                table_rows.append({
                    "Character Range": f"[{s_char}:{e_char}]",
                    "Extracted Text Span": span_text,
                    "SBDH Category": cat,
                    "Model Confidence": f"{conf:.2%}",
                })

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        else:
            st.info("No social determinants of health identified above the decision threshold.")

    st.markdown("---")
    st.caption(
        "**Research Disclaimer:** The extraction model was trained and evaluated on synthetic narrative notes (Synth-SBDH). "
        "It has not undergone prospective validation on authentic electronic health records and must not be used for unsupervised clinical coding."
    )
