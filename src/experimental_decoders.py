"""
Experimental BIO Decoders for Experiment 2.1: BIO Decoder Ablation.

Does NOT replace or modify the production decoder in src/transformer_dataset.py.
Provides:
1. Strategy A (Control): Standard Outside-Dominance decoder.
2. Strategy B (Active Threshold): Pure active thresholding without P(O) suppression.
3. Strategy C (Relative Margin): Entity probability must exceed P(O) by a configurable margin delta.
"""
from typing import Set, Tuple, Optional, Dict, Any, List
import numpy as np

from src.transformer_dataset import SBDH_CATEGORIES, get_label_indices


def decode_strategy_a_control(
    logits: np.ndarray,
    active_mask: np.ndarray,
    offset_mapping: np.ndarray,
    text: Optional[str] = None,
    threshold: float = 0.30,
) -> Set[Tuple[int, int, str]]:
    """
    Strategy A (Control): Current production decoder.
    Token t is Outside (O) if P(O) >= max(P_B, P_I) OR max(P_B, P_I) <= threshold.
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    seq_len = len(active_mask)
    predicted_spans: Set[Tuple[int, int, str]] = set()

    for cat_idx, cat in enumerate(SBDH_CATEGORIES):
        b_idx, i_idx, o_idx = get_label_indices(cat_idx)

        in_span = False
        span_start_tok = -1
        span_end_tok = -1

        for t in range(seq_len):
            if active_mask[t] != 1.0:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                    in_span = False
                continue

            p_b = probs[t, b_idx]
            p_i = probs[t, i_idx]
            p_o = probs[t, o_idx]

            # Outside condition
            if p_o >= max(p_b, p_i) or max(p_b, p_i) <= threshold:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                    in_span = False
                continue

            # Active Beginning condition
            if p_b >= p_i:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                in_span = True
                span_start_tok = t
                span_end_tok = t
            else:
                # Active Inside condition (orphan I repair)
                if in_span:
                    span_end_tok = t
                else:
                    in_span = True
                    span_start_tok = t
                    span_end_tok = t

        if in_span:
            s_char = int(offset_mapping[span_start_tok][0])
            e_char = int(offset_mapping[span_end_tok][1])
            if s_char < e_char:
                if text:
                    sub = text[s_char:e_char]
                    s_char += len(sub) - len(sub.lstrip())
                    e_char -= len(sub) - len(sub.rstrip())
                if s_char < e_char:
                    predicted_spans.add((s_char, e_char, cat))

    return predicted_spans


def decode_strategy_b_active_threshold(
    logits: np.ndarray,
    active_mask: np.ndarray,
    offset_mapping: np.ndarray,
    text: Optional[str] = None,
    threshold: float = 0.30,
) -> Set[Tuple[int, int, str]]:
    """
    Strategy B (Active-Threshold Decoder):
    For each category, allow B/I entity predictions whenever max(P_B, P_I) > threshold,
    WITHOUT requiring max(P_B, P_I) to exceed P(O).
    Token t is Outside (O) only when max(P_B, P_I) <= threshold.
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    seq_len = len(active_mask)
    predicted_spans: Set[Tuple[int, int, str]] = set()

    for cat_idx, cat in enumerate(SBDH_CATEGORIES):
        b_idx, i_idx, _ = get_label_indices(cat_idx)

        in_span = False
        span_start_tok = -1
        span_end_tok = -1

        for t in range(seq_len):
            if active_mask[t] != 1.0:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                    in_span = False
                continue

            p_b = probs[t, b_idx]
            p_i = probs[t, i_idx]
            max_entity = max(p_b, p_i)

            # Outside condition: entity probability does not meet threshold
            if max_entity <= threshold:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                    in_span = False
                continue

            # Active Beginning condition
            if p_b >= p_i:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                in_span = True
                span_start_tok = t
                span_end_tok = t
            else:
                # Active Inside condition (orphan I repair)
                if in_span:
                    span_end_tok = t
                else:
                    in_span = True
                    span_start_tok = t
                    span_end_tok = t

        if in_span:
            s_char = int(offset_mapping[span_start_tok][0])
            e_char = int(offset_mapping[span_end_tok][1])
            if s_char < e_char:
                if text:
                    sub = text[s_char:e_char]
                    s_char += len(sub) - len(sub.lstrip())
                    e_char -= len(sub) - len(sub.rstrip())
                if s_char < e_char:
                    predicted_spans.add((s_char, e_char, cat))

    return predicted_spans


def decode_strategy_c_relative_margin(
    logits: np.ndarray,
    active_mask: np.ndarray,
    offset_mapping: np.ndarray,
    text: Optional[str] = None,
    threshold: float = 0.30,
    margin: float = 0.00,
) -> Set[Tuple[int, int, str]]:
    """
    Strategy C (Relative-Margin Decoder):
    Token t is an entity token only when:
    1. max(P_B, P_I) > threshold
    2. max(P_B, P_I) - P(O) > margin
    Tested for margin in [0.00, 0.05, 0.10].
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    seq_len = len(active_mask)
    predicted_spans: Set[Tuple[int, int, str]] = set()

    for cat_idx, cat in enumerate(SBDH_CATEGORIES):
        b_idx, i_idx, o_idx = get_label_indices(cat_idx)

        in_span = False
        span_start_tok = -1
        span_end_tok = -1

        for t in range(seq_len):
            if active_mask[t] != 1.0:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                    in_span = False
                continue

            p_b = probs[t, b_idx]
            p_i = probs[t, i_idx]
            p_o = probs[t, o_idx]
            max_entity = max(p_b, p_i)

            # Outside condition: below threshold OR margin condition not satisfied
            if max_entity <= threshold or (max_entity - p_o) <= margin:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                    in_span = False
                continue

            # Active Beginning condition
            if p_b >= p_i:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                in_span = True
                span_start_tok = t
                span_end_tok = t
            else:
                # Active Inside condition (orphan I repair)
                if in_span:
                    span_end_tok = t
                else:
                    in_span = True
                    span_start_tok = t
                    span_end_tok = t

        if in_span:
            s_char = int(offset_mapping[span_start_tok][0])
            e_char = int(offset_mapping[span_end_tok][1])
            if s_char < e_char:
                if text:
                    sub = text[s_char:e_char]
                    s_char += len(sub) - len(sub.lstrip())
                    e_char -= len(sub) - len(sub.rstrip())
                if s_char < e_char:
                    predicted_spans.add((s_char, e_char, cat))

    return predicted_spans
