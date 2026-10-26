import sys
import os
import re
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np

from src.sbdh_rules_baseline import find_maximum_bipartite_matching, match_spans_exact
from src.transformer_dataset import SBDH_CATEGORIES
from src.experimental_decoders import decode_strategy_b_active_threshold

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

DANGLING_PREPOSITIONS = {
    "for", "to", "and", "or", "with", "in", "at", "from", "of", "by", "as", "about", "into", "on"
}
PUNCT_CHARS = '.,;:!?"\'()[]{}-/'

def snap_span(s, e, cat, text, trim_punct=True, trim_prep=True, snap_words=True):
    orig_s, orig_e = s, e
    if s >= e or not text:
        return (s, e, cat), False

    # 1. Word boundary snapping (if cut mid-word)
    if snap_words:
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
    if trim_punct:
        while s < e and text[s] in PUNCT_CHARS:
            s += 1
        while e > s and text[e-1] in PUNCT_CHARS:
            e -= 1

    # 4. Trailing dangling preposition trimming
    if trim_prep:
        sub = text[s:e]
        words = sub.split()
        if len(words) > 1 and words[-1].lower() in DANGLING_PREPOSITIONS:
            last_word = words[-1]
            cand_e = e - len(last_word)
            while cand_e > s and text[cand_e - 1].isspace():
                cand_e -= 1
            if cand_e > s:
                e = cand_e

    # Final whitespace & punct strip
    sub = text[s:e]
    lead_ws = len(sub) - len(sub.lstrip())
    s += lead_ws
    sub = text[s:e]
    trail_ws = len(sub) - len(sub.rstrip())
    e -= trail_ws
    if trim_punct:
        while s < e and text[s] in PUNCT_CHARS:
            s += 1
        while e > s and text[e-1] in PUNCT_CHARS:
            e -= 1

    changed = (s != orig_s or e != orig_e) and s < e
    if s >= e:
        return (orig_s, orig_e, cat), False
    return (s, e, cat), changed


for name, (tp_punct, tp_prep, tp_words) in [
    ("No Snapping (Base)", (False, False, False)),
    ("Punctuation Only", (True, False, False)),
    ("Preposition Only", (False, True, False)),
    ("Word Snapping Only", (False, False, True)),
    ("Punct + Preposition", (True, True, False)),
    ("Punct + Word Snapping", (True, False, True)),
    ("Full Snapping (Punct + Prep + Word)", (True, True, True)),
]:
    total_snaps = 0
    exact_tp, exact_fp, exact_fn = 0, 0, 0
    relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0
    total_preds = 0

    for note in cached:
        text = note["text"]
        golds = sorted(list(note["gold_tuples"]))
        preds_raw = decode_strategy_b_active_threshold(
            logits=note["logits"],
            active_mask=note["active_mask"],
            offset_mapping=note["offset_mapping"],
            text=text,
            threshold=0.30
        )
        preds = []
        for p in preds_raw:
            snapped, changed = snap_span(p[0], p[1], p[2], text, trim_punct=tp_punct, trim_prep=tp_prep, snap_words=tp_words)
            if changed:
                total_snaps += 1
            preds.append(snapped)

        # eliminate duplicates if any
        preds = sorted(list(set(preds)))
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

    print(f"{name:35s}: Snaps={total_snaps:3d}, Exact TP={exact_tp}, Exact F1={e_f1:.4f}, Relaxed TP={relaxed_tp}, Relaxed F1={r_f1:.4f}")
