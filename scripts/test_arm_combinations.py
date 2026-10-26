import sys
import os
import re
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np

from src.sbdh_rules_baseline import find_maximum_bipartite_matching, match_spans_exact
from src.transformer_dataset import SBDH_CATEGORIES
from src.experimental_decoders import decode_strategy_a_control, decode_strategy_b_active_threshold

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

DANGLING_PREPOSITIONS = {
    "for", "to", "and", "or", "with", "in", "at", "from", "of", "by", "as", "about", "into", "on"
}
PUNCT_CHARS = '.,;:!?"\'()[]{}-/'

def snap_span_boundaries(s, e, cat, text):
    """
    Deterministic Lexical Boundary Snapping:
    1. Word boundary expansion if cut mid-word.
    2. Whitespace trimming.
    3. Punctuation stripping from boundaries.
    4. Dangling trailing preposition trimming.
    5. Final whitespace & punctuation stripping.
    """
    orig_s, orig_e = s, e
    if s >= e or not text:
        return (s, e, cat), False

    # 1. Word boundary expansion (if cut mid-word)
    while s > 0 and text[s-1].isalnum() and text[s].isalnum():
        s -= 1
    while e < len(text) and text[e-1].isalnum() and text[e].isalnum():
        e += 1

    # 2. Whitespace trimming
    sub = text[s:e]
    lead_ws = len(sub) - len(sub.lstrip())
    s += lead_ws
    sub = text[s:e]
    trail_ws = len(sub) - len(sub.rstrip())
    e -= trail_ws

    # 3. Punctuation trimming
    while s < e and text[s] in PUNCT_CHARS:
        s += 1
    while e > s and text[e-1] in PUNCT_CHARS:
        e -= 1

    # 4. Trailing dangling preposition trimming
    sub = text[s:e]
    words = sub.split()
    if len(words) > 1 and words[-1].lower() in DANGLING_PREPOSITIONS:
        last_word = words[-1]
        cand_e = e - len(last_word)
        while cand_e > s and text[cand_e - 1].isspace():
            cand_e -= 1
        if cand_e > s:
            e = cand_e

    # 5. Final whitespace & punct strip
    sub = text[s:e]
    lead_ws = len(sub) - len(sub.lstrip())
    s += lead_ws
    sub = text[s:e]
    trail_ws = len(sub) - len(sub.rstrip())
    e -= trail_ws
    while s < e and text[s] in PUNCT_CHARS:
        s += 1
    while e > s and text[e-1] in PUNCT_CHARS:
        e -= 1

    changed = (s != orig_s or e != orig_e) and s < e
    if s >= e:
        return (orig_s, orig_e, cat), False
    return (s, e, cat), changed


def fuse_same_category_spans(spans, text, max_gap=6, allowed_connectives=None):
    """
    Conservative same-category span fusion:
    Only merges spans of identical category within the same sentence separated by <= max_gap
    containing only whitespace, punctuation (e.g. hyphens), or short connectives.
    """
    if allowed_connectives is None:
        allowed_connectives = {"and", "or", "her", "his", "the", "a", "an", "of", "to", "in", "for", "with"}

    sorted_spans = sorted(list(spans), key=lambda x: (x[2], x[0], x[1]))
    fused_spans = []
    merges_count = 0
    merge_records = []

    i = 0
    while i < len(sorted_spans):
        curr_s, curr_e, curr_c = sorted_spans[i]
        j = i + 1
        while j < len(sorted_spans):
            next_s, next_e, next_c = sorted_spans[j]
            if next_c != curr_c:
                break
            
            if next_s <= curr_e:
                # Abutting or overlapping
                curr_e = max(curr_e, next_e)
                merges_count += 1
                merge_records.append((curr_s, curr_e, curr_c, "overlap/abutting"))
                j += 1
                continue

            gap = next_s - curr_e
            if gap <= max_gap:
                gap_text = text[curr_e:next_s]
                # Sentence boundary check
                if any(punct in gap_text for punct in ['.', '!', '?', '\n']):
                    break
                # Connective check
                gap_stripped = gap_text.strip().lower()
                gap_words = set(re.findall(r'\b\w+\b', gap_stripped))
                if not gap_words or gap_words.issubset(allowed_connectives):
                    merges_count += 1
                    merge_records.append((curr_s, next_e, curr_c, f"gap '{gap_text}'"))
                    curr_e = next_e
                    j += 1
                    continue
                else:
                    break
            else:
                break
        fused_spans.append((curr_s, curr_e, curr_c))
        i = j

    return set(fused_spans), merges_count, merge_records


for gap in [2, 5]:
    # Test Arm 5 (Fusion only)
    # Test Arm 6 (Both Snapping and Fusion)
    print(f"\n--- Testing Gap = {gap} ---")
    for do_snap, do_fuse in [(True, False), (False, True), (True, True)]:
        arm_name = f"Snap={do_snap}, Fuse={do_fuse} (gap={gap})"
        exact_tp, exact_fp, exact_fn = 0, 0, 0
        relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0
        total_preds = 0
        total_snaps = 0
        total_merges = 0

        for note in cached:
            text = note["text"]
            golds = sorted(list(note["gold_tuples"]))
            preds = decode_strategy_b_active_threshold(
                logits=note["logits"],
                active_mask=note["active_mask"],
                offset_mapping=note["offset_mapping"],
                text=text,
                threshold=0.30
            )

            if do_snap and not do_fuse:
                snapped_preds = []
                for p in preds:
                    sp, ch = snap_span_boundaries(p[0], p[1], p[2], text)
                    if ch:
                        total_snaps += 1
                    snapped_preds.append(sp)
                preds = set(snapped_preds)

            elif do_fuse and not do_snap:
                preds, m_cnt, _ = fuse_same_category_spans(preds, text, max_gap=gap)
                total_merges += m_cnt

            elif do_snap and do_fuse:
                # Snap first, then fuse
                snapped_preds = []
                for p in preds:
                    sp, ch = snap_span_boundaries(p[0], p[1], p[2], text)
                    if ch:
                        total_snaps += 1
                    snapped_preds.append(sp)
                preds, m_cnt, _ = fuse_same_category_spans(set(snapped_preds), text, max_gap=gap)
                total_merges += m_cnt

            preds = sorted(list(preds))
            total_preds += len(preds)

            etp, efp, efn = match_spans_exact(set(preds), set(golds))
            exact_tp += etp
            exact_fp += efp
            exact_fn += efn

            matched_relaxed = find_maximum_bipartite_matching(preds, golds)
            rtp = len(matched_relaxed)
            rfp = len(preds) - rtp
            rfn = len(golds) - rtp
            relaxed_tp += rtp
            relaxed_fp += rfp
            relaxed_fn += rfn

        e_prec = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0
        e_rec = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0

        r_prec = relaxed_tp / (relaxed_tp + relaxed_fp) if (relaxed_tp + relaxed_fp) > 0 else 0
        r_rec = relaxed_tp / (relaxed_tp + relaxed_fn) if (relaxed_tp + relaxed_fn) > 0 else 0
        r_f1 = 2 * r_prec * r_rec / (r_prec + r_rec) if (r_prec + r_rec) > 0 else 0

        print(f"{arm_name:35s}: Snaps={total_snaps:2d}, Merges={total_merges:2d}, Preds={total_preds}, Exact TP={exact_tp}, Exact F1={e_f1:.4f}, Relaxed TP={relaxed_tp}, Relaxed F1={r_f1:.4f}")
