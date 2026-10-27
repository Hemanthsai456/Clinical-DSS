"""
Track B UI Module: Social and Behavioral Determinants of Health (SBDH) Text Extraction.
Uses the verified 3-epoch Colab-trained Transformer checkpoint (evernorth_m3_results/models/sbdh_transformer/best_model.pt).
Promoted Release Pipeline: Arm 6A (Strategy B Active-Threshold + Lexical Snapping + Conservative Fusion).
Preserves clean fallback to original frozen baseline decoder (Strategy A Control).
Performs 100% local CPU inference without sending clinical text to external networks or APIs.
"""
from typing import Set, Tuple, Dict, Any, Optional
import html
import logging
import streamlit as st
import numpy as np
import torch
import pandas as pd
from model_loader import load_track_b_model_and_tokenizer, TRACK_B_FROZEN_THRESHOLD, get_track_b_base_model_name

from src.transformer_dataset import (
    decode_predictions_to_spans,
    SBDH_CATEGORIES,
    CAT_TO_IDX,
    get_label_indices,
)
from src.experimental_decoders import decode_strategy_b_active_threshold
from src.experimental_boundary_snapping import (
    snap_span_boundaries,
    fuse_same_category_spans,
)

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


def extract_sbdh_spans(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    threshold: float = TRACK_B_FROZEN_THRESHOLD,
    decoder_mode: str = "Arm 6A (Promoted Release)",
) -> Tuple[Set[Tuple[int, int, str]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract SBDH entity spans from clinical narrative text with explicit pipeline selection.

    Args:
        text: Raw narrative text string.
        model: Loaded MultiLabelSBDHTransformer in eval mode.
        tokenizer: Fast tokenizer matching the model checkpoint.
        threshold: Sigmoid probability cutoff (calibrated at 0.30).
        decoder_mode: Either 'Arm 6A (Promoted Release)' or 'Strategy A (Baseline Fallback)'.

    Returns:
        (extracted_spans_set, logits_array, offset_mapping_array, active_mask_array)
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return set(), np.array([]), np.array([]), np.array([])

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

    # Active mask for real text tokens
    active_mask = np.zeros(len(offset_mapping), dtype=float)
    for t in range(len(offset_mapping)):
        if offset_mapping[t][0] != 0 or offset_mapping[t][1] != 0:
            active_mask[t] = 1.0

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"].squeeze(0).cpu().numpy()

    if "Arm 6A" in decoder_mode:
        # Promoted Candidate Pipeline:
        # 1. Strategy B active-threshold decoding
        raw_spans = decode_strategy_b_active_threshold(
            logits=logits,
            active_mask=active_mask,
            offset_mapping=offset_mapping,
            text=cleaned_text,
            threshold=threshold,
        )
        # 2. Lexical boundary snapping with balanced delimiter protection
        snapped_spans = set()
        for span in raw_spans:
            snapped, _ = snap_span_boundaries(span, text=cleaned_text)
            snapped_spans.add(snapped)

        # 3. Conservative same-category fusion (gap <= 2, sentence boundaries respected, no fusion across and/or)
        fused_spans, _, _ = fuse_same_category_spans(
            snapped_spans,
            text=cleaned_text,
            max_gap=2,
            allowed_connectives=set(),  # Disallows connectives like 'and' or 'or'
        )
        return fused_spans, logits, offset_mapping, active_mask
    else:
        # Clean fallback to original frozen baseline decoder
        base_spans = decode_predictions_to_spans(
            logits=logits,
            active_mask=active_mask,
            offset_mapping=offset_mapping,
            text=cleaned_text,
            threshold=threshold,
        )
        return base_spans, logits, offset_mapping, active_mask


def render_highlighted_text(text: str, extracted_spans):
    """
    Render text with color-coded highlight tags for extracted SBDH spans.
    Splits the narrative at all span boundaries so each segment is rendered
    exactly once, with all covering categories displayed as stacked badges.
    """
    if not text:
        return "<div style='padding: 12px; background: #F9FAFB; border-radius: 8px;'></div>"
    if not extracted_spans:
        return (
            f"<div style='padding: 14px; background: #F8FAFC; border: 1px solid #E2E8F0; "
            f"border-radius: 8px; line-height: 2.0; font-family: sans-serif; font-size: 1.05rem; "
            f"color: #0F172A;'>{html.escape(text)}</div>"
        )

    # Collect all unique cut points within [0, len(text)]
    cut_points = {0, len(text)}
    for s_char, e_char, _ in extracted_spans:
        s_clamped = max(0, min(s_char, len(text)))
        e_clamped = max(0, min(e_char, len(text)))
        if s_clamped < e_clamped:
            cut_points.add(s_clamped)
            cut_points.add(e_clamped)

    sorted_points = sorted(cut_points)

    out = []
    for i in range(len(sorted_points) - 1):
        p_start = sorted_points[i]
        p_end = sorted_points[i + 1]
        seg_text = text[p_start:p_end]
        if not seg_text:
            continue

        # Find all categories covering this segment
        covering_cats = sorted(list(set(
            cat for s, e, cat in extracted_spans
            if s <= p_start and p_end <= e
        )))

        if not covering_cats:
            out.append(html.escape(seg_text))
        else:
            primary_cat = covering_cats[0]
            color = CATEGORY_COLORS.get(primary_cat, "#4B5563")

            # Stacked badges for all covering categories with CSS margin-left for spacing
            badges_html = "".join([
                f"<span style='margin-left: 4px; font-size: 0.70rem; color: {CATEGORY_COLORS.get(cat, '#4B5563')}; "
                f"font-weight: bold;'>[{cat}]</span>"
                for cat in covering_cats
            ])

            seg_escaped = html.escape(seg_text)
            title_cats = html.escape(", ".join(covering_cats))
            mark_html = (
                f"<mark style='background-color: {color}22; border-bottom: 3px solid {color}; "
                f"padding: 2px 4px; border-radius: 4px; font-weight: 500;' "
                f"title='{title_cats}'>"
                f"{seg_escaped}{badges_html}</mark>"
            )
            out.append(mark_html)

    rendered_body = "".join(out)
    return (
        f"<div style='padding: 14px; background: #F8FAFC; border: 1px solid #E2E8F0; "
        f"border-radius: 8px; line-height: 2.0; font-family: sans-serif; font-size: 1.05rem; "
        f"color: #0F172A;'>{rendered_body}</div>"
    )


def render_track_b_page():
    base_model_name = get_track_b_base_model_name()
    st.header("Track B — Social & Behavioral Determinants of Health (SBDH) Extraction")
    st.markdown(
        f"""
        **Clinical CDS Setting:** Unstructured Narrative SBDH Extraction (Discharge Summaries & Social Work Notes).  
        **Model Architecture:** Multi-Label BIO Sequence Tagger (`{base_model_name}`, 45 independent binary sigmoid logits).  
        **Active Release Pipeline:** **Arm 6A** (Strategy B Active-Threshold + Delimiter-Protected Lexical Snapping + Conservative Gap $\\le 2$ Fusion).  
        **Multi-Label Capability:** Explicitly supports multiple concurrent SBDH categories mapped to identical character intervals.  
        **Privacy Guarantee:** 100% Local In-Memory Inference. When run locally, no narrative is transmitted over any external network or cloud service.
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

    st.info(
        "**Known limitations on these examples:** 'life' and 'in a rented house' are false positives; "
        "'access to healthcare' is a truncated span (should read 'poor access to healthcare'). "
        "See the Track B error analysis report ([reports/sbdh_error_analysis_experiment_1.md](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_error_analysis_experiment_1.md))."
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

    # Decoder and threshold selection
    t_col1, t_col2 = st.columns([2, 2])
    with t_col1:
        decoder_mode = st.radio(
            "Decoder Pipeline Configuration:",
            [
                "Arm 6A (Promoted Release Pipeline)",
                "Strategy A (Frozen Baseline Fallback)",
            ],
            index=0,
            help="Arm 6A: Strategy B active-threshold decoding at 0.30, lexical boundary snapping with delimiter protection, and conservative fusion (gap <= 2). Strategy A: Original Outside-dominant baseline control."
        )

        threshold = st.slider(
            "Detection Decision Threshold (Sigmoid Probability)",
            min_value=0.10,
            max_value=0.70,
            value=TRACK_B_FROZEN_THRESHOLD,
            step=0.05,
            help="Threshold calibrated on validation data (0.30 maximizes F1 across Synth-SBDH categories)."
        )
    with t_col2:
        if "Arm 6A" in decoder_mode:
            st.markdown(
                """
                **Active Pipeline: Arm 6A (Promoted Release)**
                - **Evaluation Protocols on Reviewed Test Set ($N=1,732$ notes):**
                  - **Resolvable Protocol ($N=3,340$ gold annotations):**
                    - Exact Span F1: **0.5173** (Precision: 0.5265, Recall: 0.5084)
                    - Relaxed Span F1: **0.8414** (Precision: 0.8564, Recall: 0.8269)
                    - Note-Level Micro F1: **0.8949**
                  - **Conservative Worst-Case Lower Bound ($N=3,345$ gold annotations; 5 unresolved annotations penalised as False Negatives):**
                    - Exact Span F1: **0.5169** (Recall: 0.5076)
                    - Relaxed Span F1: **0.8408** (Recall: 0.8257)
                - **Post-Processing Modules:** Strategy B active-threshold decoding at 0.30, lexical boundary snapping, and conservative fusion ($gap \\le 2$, no fusion across 'and/or')
                """
            )
        else:
            st.markdown(
                """
                **Active Pipeline: Strategy A (Frozen Baseline Control)**
                - **Evaluation Protocols on Reviewed Test Set ($N=1,732$ notes):**
                  - **Resolvable Protocol ($N=3,340$ gold annotations):**
                    - Exact Span F1: **0.5105** (Precision: 0.5403, Recall: 0.4838)
                    - Relaxed Span F1: **0.8330** (Precision: 0.8816, Recall: 0.7895)
                    - Note-Level Micro F1: **0.8888**
                  - **Conservative Worst-Case Lower Bound ($N=3,345$ gold annotations; 5 unresolved annotations penalised as False Negatives):**
                    - Exact Span F1: **0.5101** (Recall: 0.4831)
                    - Relaxed Span F1: **0.8324** (Recall: 0.7883)
                - **Post-Processing Modules:** Strategy A Outside-dominance rule (no boundary snapping or fusion)
                """
            )

    # Extraction Trigger
    st.subheader("2. Extracted SBDH Entities & Spans")
    if st.button("Extract SBDH Categories and Spans", type="primary"):
        cleaned_text = text_input.strip()
        if not cleaned_text:
            st.warning("Please provide clinical narrative text before running extraction.")
            st.stop()

        # Check for length truncation
        raw_tokens_count = len(tokenizer.encode(cleaned_text, add_special_tokens=True))
        if raw_tokens_count > 256:
            st.warning(f"Note length ({raw_tokens_count} tokens) exceeds model context (256 tokens). Text beyond token 256 was truncated.")

        # Execute extraction using selected pipeline
        extracted_spans, logits, offset_mapping, active_mask = extract_sbdh_spans(
            text=cleaned_text,
            model=model,
            tokenizer=tokenizer,
            threshold=threshold,
            decoder_mode=decoder_mode,
        )

        # Highlighted visual markup
        st.markdown("**Narrative with Extracted SBDH Span Annotations:**", unsafe_allow_html=True)
        html_markup = render_highlighted_text(cleaned_text, extracted_spans)
        st.markdown(html_markup, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        pipeline_label = "Arm 6A (Promoted)" if "Arm 6A" in decoder_mode else "Strategy A (Baseline Control)"

        if extracted_spans:
            st.success(f"[{pipeline_label}] Extracted {len(extracted_spans)} SBDH span(s) across {len(set(s[2] for s in extracted_spans))} unique category(ies).")

            # Precompute sigmoid probabilities once for the entire sequence
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

            # Structured Table
            table_rows = []
            for s_char, e_char, cat in sorted(extracted_spans, key=lambda x: (x[0], x[1])):
                span_text = cleaned_text[s_char:e_char]
                cat_idx = CAT_TO_IDX[cat]
                b_idx, i_idx, _ = get_label_indices(cat_idx)

                # Compute token-level max confidence for the span over overlapping active tokens
                rel_tokens = [
                    t for t in range(len(offset_mapping))
                    if active_mask[t] == 1.0 and max(offset_mapping[t][0], s_char) < min(offset_mapping[t][1], e_char)
                ]
                if rel_tokens:
                    max_token_probs = np.maximum(probs[rel_tokens, b_idx], probs[rel_tokens, i_idx])
                    span_score = float(np.max(max_token_probs))
                    score_display = f"{span_score:.2%}"
                else:
                    logging.warning(
                        f"No active tokens matched span [{s_char}:{e_char}] '{span_text}' for category '{cat}'."
                    )
                    score_display = "n/a"

                table_rows.append({
                    "Character Range": f"[{s_char}:{e_char}]",
                    "Extracted Text Span": span_text,
                    "SBDH Category": cat,
                    "Span score (max token probability)": score_display,
                    "Pipeline": pipeline_label,
                })

            st.dataframe(pd.DataFrame(table_rows), width="stretch")
        else:
            st.info(f"[{pipeline_label}] No social determinants of health identified above decision threshold {threshold:.2f}.")

    st.markdown("---")
    st.caption(
        "**Research Disclaimer:** The extraction model was trained and evaluated on synthetic narrative notes (Synth-SBDH). "
        "It has not undergone prospective validation on authentic electronic health records and must not be used for unsupervised clinical coding."
    )
