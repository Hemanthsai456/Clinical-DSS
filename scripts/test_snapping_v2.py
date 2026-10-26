import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch

from src.sbdh_rules_baseline import match_spans_exact, find_maximum_bipartite_matching
from src.experimental_decoders import decode_strategy_b_active_threshold
from src.experimental_boundary_snapping import (
    DANGLING_PREPOSITIONS,
    PUNCT_CHARS,
    fuse_same_category_spans,
)

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

def is_trimmable_trailing_punct(char: str, s: int, e: int, text: str) -> bool:
    if char not in PUNCT_CHARS:
        return False
    sub = text[s:e]
    # Balanced closing delimiters should not be stripped
    if char == ")" and sub.count("(") >= sub.count(")"):
        return False
    if char == "]" and sub.count("[") >= sub.count("]"):
        return False
    if char == "}" and sub.count("{") >= sub.count("}"):
        return False
    return True

def is_trimmable_leading_punct(char: str, s: int, e: int, text: str) -> bool:
    if char not in PUNCT_CHARS:
        return False
    sub = text[s:e]
    # If the span has balanced parentheses, do not strip the opening paren unless the whole span is just parens
    if char == "(" and sub.count(")") >= sub.count("(") and len(sub) > 2 and sub[-1] == ")":
        # e.g., "(PTSD)" - here both opening and closing might be stripped or kept
        pass
    if char == "(":
        if sub.count(")") >= sub.count("("):
            return False
    return True

def snap_span_boundaries_v2(span, text):
    s, e, cat = span
    orig_s, orig_e = s, e

    if s >= e or not text or s < 0 or e > len(text):
        return span, False

    # 1. Word boundary expansion if cut mid-word by subword tokenization
    while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
        s -= 1
    while e < len(text) and text[e - 1].isalnum() and text[e].isalnum():
        e += 1

    # 2. Whitespace trimming
    sub = text[s:e]
    lead_ws = len(sub) - len(sub.lstrip())
    s += lead_ws
    sub = text[s:e]
    trail_ws = len(sub) - len(sub.rstrip())
    e -= trail_ws

    if s >= e:
        return (orig_s, orig_e, cat), False

    # 3. Punctuation trimming from span boundaries
    while s < e and is_trimmable_leading_punct(text[s], s, e, text):
        s += 1
    while e > s and is_trimmable_trailing_punct(text[e - 1], s, e, text):
        e -= 1

    if s >= e:
        return (orig_s, orig_e, cat), False

    # 4. Dangling trailing preposition trimming
    sub = text[s:e]
    words = sub.split()
    if len(words) > 1 and words[-1].lower() in DANGLING_PREPOSITIONS:
        last_word = words[-1]
        cand_e = e - len(last_word)
        while cand_e > s and text[cand_e - 1].isspace():
            cand_e -= 1
        if cand_e > s:
            e = cand_e

    # 5. Final whitespace & punctuation check
    sub = text[s:e]
    lead_ws = len(sub) - len(sub.lstrip())
    s += lead_ws
    sub = text[s:e]
    trail_ws = len(sub) - len(sub.rstrip())
    e -= trail_ws

    while s < e and is_trimmable_leading_punct(text[s], s, e, text):
        s += 1
    while e > s and is_trimmable_trailing_punct(text[e - 1], s, e, text):
        e -= 1

    changed = (s != orig_s or e != orig_e) and (s < e)
    if s >= e:
        return (orig_s, orig_e, cat), False
    return (s, e, cat), changed


# Test Note 835 specifically
note_835 = cached[835]
preds_835 = decode_strategy_b_active_threshold(
    logits=note_835["logits"],
    active_mask=note_835["active_mask"],
    offset_mapping=note_835["offset_mapping"],
    text=note_835["text"],
    threshold=0.30
)
print("Note 835 test:")
for p in preds_835:
    sp, ch = snap_span_boundaries_v2(p, note_835["text"])
    print(f"  Raw: [{p[0]}:{p[1]}] \"{note_835['text'][p[0]:p[1]]}\" -> Snapped: [{sp[0]}:{sp[1]}] \"{note_835['text'][sp[0]:sp[1]]}\"")
    print(f"  Gold in note 835: {note_835['gold_tuples']}")
    print(f"  Matches gold exactly? {sp in note_835['gold_tuples']}")

# Now evaluate on entire validation set for Arm 4 and Arm 6A
for do_fuse, gap, name in [
    (False, 0, "Arm 4 (Snapping v2 Only)"),
    (True, 2, "Arm 6A (Snapping v2 + Fusion gap=2)"),
]:
    exact_tp, exact_fp, exact_fn = 0, 0, 0
    relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0
    total_preds = 0
    total_snaps = 0
    total_merges = 0

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
        snapped = []
        for p in preds_raw:
            sp, ch = snap_span_boundaries_v2(p, text)
            if ch:
                total_snaps += 1
            snapped.append(sp)

        preds = set(snapped)
        if do_fuse:
            preds, m_cnt, _ = fuse_same_category_spans(preds, text, max_gap=gap)
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

    print(f"\n{name}:")
    print(f"  Exact Matching:   TP={exact_tp}, FP={exact_fp}, FN={exact_fn}, Prec={e_prec:.4f}, Rec={e_rec:.4f}, F1={e_f1:.4f}")
    print(f"  Relaxed Matching: TP={relaxed_tp}, FP={relaxed_fp}, FN={relaxed_fn}, Prec={r_prec:.4f}, Rec={r_rec:.4f}, F1={r_f1:.4f}")
    print(f"  Total Preds={total_preds}, Snaps={total_snaps}, Merges={total_merges}")
