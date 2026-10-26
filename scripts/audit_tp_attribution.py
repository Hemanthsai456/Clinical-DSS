import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch

from src.sbdh_rules_baseline import match_spans_exact
from src.experimental_decoders import (
    decode_strategy_a_control,
    decode_strategy_b_active_threshold,
)
from src.experimental_boundary_snapping import (
    snap_span_boundaries,
    fuse_same_category_spans,
)

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

# Track exact matches per note
# We will identify gold spans matched by each stage
gold_matched_a = set()
gold_matched_b_raw = set()
gold_matched_b_snap = set()
gold_matched_arm6a = set()

preds_a_all = set()
preds_b_raw_all = set()
preds_b_snap_all = set()
preds_arm6a_all = set()

all_golds = set()

for note_idx, note in enumerate(cached):
    text = note["text"]
    golds = note["gold_tuples"]
    for s, e, c in golds:
        all_golds.add((note_idx, s, e, c))

    # 1. Strategy A (Control)
    p_a = decode_strategy_a_control(
        logits=note["logits"],
        active_mask=note["active_mask"],
        offset_mapping=note["offset_mapping"],
        text=text,
        threshold=0.30
    )
    for s, e, c in p_a:
        preds_a_all.add((note_idx, s, e, c))
        if (s, e, c) in golds:
            gold_matched_a.add((note_idx, s, e, c))

    # 2. Strategy B (Raw)
    p_b_raw = decode_strategy_b_active_threshold(
        logits=note["logits"],
        active_mask=note["active_mask"],
        offset_mapping=note["offset_mapping"],
        text=text,
        threshold=0.30
    )
    for s, e, c in p_b_raw:
        preds_b_raw_all.add((note_idx, s, e, c))
        if (s, e, c) in golds:
            gold_matched_b_raw.add((note_idx, s, e, c))

    # 3. Strategy B + Snapping
    p_b_snap = set()
    for p in p_b_raw:
        sp, ch = snap_span_boundaries(p, text)
        p_b_snap.add(sp)
    for s, e, c in p_b_snap:
        preds_b_snap_all.add((note_idx, s, e, c))
        if (s, e, c) in golds:
            gold_matched_b_snap.add((note_idx, s, e, c))

    # 4. Arm 6A: Strategy B + Snapping + Fusion (gap=2)
    p_arm6a, m_cnt, _ = fuse_same_category_spans(p_b_snap, text, max_gap=2)
    for s, e, c in p_arm6a:
        preds_arm6a_all.add((note_idx, s, e, c))
        if (s, e, c) in golds:
            gold_matched_arm6a.add((note_idx, s, e, c))

print("=" * 60)
print("EXACT TRUE POSITIVE ATTRIBUTION BREAKDOWN")
print("=" * 60)
print(f"Total Gold Spans across 876 notes: {len(all_golds)}")
print(f"1. Strategy A (Control) Exact TPs: {len(gold_matched_a)}")
print(f"2. Strategy B (Raw) Exact TPs:     {len(gold_matched_b_raw)}")
print(f"3. Strategy B + Snapping Exact TPs:{len(gold_matched_b_snap)}")
print(f"4. Arm 6A (Snap + Fuse 2) Exact TPs:{len(gold_matched_arm6a)}")

print("\n--- STEP 1: Decoder Change (Strategy A -> Strategy B Raw) ---")
b_gains = gold_matched_b_raw - gold_matched_a
b_losses = gold_matched_a - gold_matched_b_raw
print(f"New Exact TPs gained by removing Outside dominance: +{len(b_gains)}")
print(f"Previous Exact TPs lost (e.g. boundary expansion beyond gold): -{len(b_losses)}")
print(f"Net change from Decoder: +{len(b_gains) - len(b_losses)} (748 -> 794)")

print("\n--- STEP 2: Snapping (Strategy B Raw -> Strategy B + Snapping) ---")
snap_gains = gold_matched_b_snap - gold_matched_b_raw
snap_losses = gold_matched_b_raw - gold_matched_b_snap
print(f"Exact TPs gained by Snapping (prepositions/punctuation trimmed): +{len(snap_gains)}")
print(f"Exact TPs lost by Snapping (closing parenthesis trimmed):       -{len(snap_losses)}")
print(f"Net change from Snapping: +{len(snap_gains) - len(snap_losses)} (794 -> 806)")

print("\n--- STEP 3: Conservative Fusion (Snapping -> Arm 6A gap=2) ---")
fuse_gains = gold_matched_arm6a - gold_matched_b_snap
fuse_losses = gold_matched_b_snap - gold_matched_arm6a
print(f"Exact TPs gained by gap=2 fusion: +{len(fuse_gains)}")
print(f"Exact TPs lost by gap=2 fusion:   -{len(fuse_losses)}")
print(f"Net change from gap=2 fusion:     +{len(fuse_gains) - len(fuse_losses)} (806 -> 806)")

print("\n--- OVERALL PROGRESSION (Strategy A Control -> Arm 6A) ---")
print(f"Total Gain: {len(gold_matched_arm6a) - len(gold_matched_a)} (748 -> 806)")
print(f"Decomposition: (+{len(b_gains) - len(b_losses)} from Decoder) + (+{len(snap_gains) - len(snap_losses)} from Snapping) + (+{len(fuse_gains) - len(fuse_losses)} from Fusion) = +{len(gold_matched_arm6a) - len(gold_matched_a)}")

# Print sample of snap_losses
print("\n--- Inspecting snap_losses (The 1 span lost by snapping) ---")
for note_idx, s, e, c in snap_losses:
    note_text = cached[note_idx]["text"]
    print(f"Note {note_idx} [{c}]: [{s}:{e}] \"{note_text[s:e]}\"")
    print(f"Full sentence: \"{note_text[max(0, s-20):min(len(note_text), e+20)]}\"")
