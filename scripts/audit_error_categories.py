import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
from src.sbdh_rules_baseline import match_spans_exact, find_maximum_bipartite_matching
from src.experimental_boundary_snapping import snap_span_boundaries, fuse_same_category_spans
from src.experimental_decoders import decode_strategy_b_active_threshold

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

print("=== Auditing Specific Error Categories ===")

# 1. Cases where Snapping improves exact match
improved_by_snapping = []
# 2. Cases where Snapping damages exact match
damaged_by_snapping = []

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
    for p in preds_raw:
        sp, ch = snap_span_boundaries(p, text)
        if ch:
            # Check if raw p was exact match
            p_exact = p in golds
            # Check if snapped sp is exact match
            sp_exact = sp in golds
            if not p_exact and sp_exact:
                improved_by_snapping.append({
                    "cat": p[2],
                    "raw_span": p,
                    "raw_text": text[p[0]:p[1]],
                    "snapped_span": sp,
                    "snapped_text": text[sp[0]:sp[1]],
                    "gold_text": text[sp[0]:sp[1]],
                    "context": text[max(0, p[0]-15):min(len(text), p[1]+15)]
                })
            elif p_exact and not sp_exact:
                damaged_by_snapping.append({
                    "cat": p[2],
                    "raw_span": p,
                    "raw_text": text[p[0]:p[1]],
                    "snapped_span": sp,
                    "snapped_text": text[sp[0]:sp[1]],
                    "context": text[max(0, p[0]-15):min(len(text), p[1]+15)]
                })

print(f"Total spans improved by snapping (Exact False -> Exact True): {len(improved_by_snapping)}")
print(f"Total spans damaged by snapping (Exact True -> Exact False): {len(damaged_by_snapping)}")

print("\n--- Top Examples Improved by Snapping ---")
for ex in improved_by_snapping[:8]:
    print(f"[{ex['cat']}] Raw: \"{ex['raw_text']}\" -> Snapped: \"{ex['snapped_text']}\" (Matches Gold!) Context: \"{ex['context']}\"")

if damaged_by_snapping:
    print("\n--- Examples Damaged by Snapping ---")
    for ex in damaged_by_snapping:
        print(f"[{ex['cat']}] Raw: \"{ex['raw_text']}\" -> Snapped: \"{ex['snapped_text']}\" Context: \"{ex['context']}\"")
else:
    print("\nZero spans were damaged by snapping!")

# 3. Cases where Fusion (gap=6) damages exact match by merging two separate gold spans
damaged_by_fusion = []
improved_by_fusion = []

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
    fused, m_cnt, m_recs = fuse_same_category_spans(preds_raw, text, max_gap=6)
    for rec in m_recs:
        s1 = rec["span1"]
        s2 = rec["span2"]
        m = rec["merged_span"]
        s1_in_gold = s1 in golds
        s2_in_gold = s2 in golds
        m_in_gold = m in golds

        if (s1_in_gold or s2_in_gold) and not m_in_gold:
            damaged_by_fusion.append({
                "cat": rec["category"],
                "s1": (s1, rec["span1_text"], s1_in_gold),
                "s2": (s2, rec["span2_text"], s2_in_gold),
                "merged": (m, rec["merged_text"]),
                "gap": rec["gap_text"],
                "context": text[max(0, s1[0]-15):min(len(text), s2[1]+15)]
            })
        elif not s1_in_gold and not s2_in_gold and m_in_gold:
            improved_by_fusion.append({
                "cat": rec["category"],
                "s1": (s1, rec["span1_text"]),
                "s2": (s2, rec["span2_text"]),
                "merged": (m, rec["merged_text"]),
                "gap": rec["gap_text"],
                "context": text[max(0, s1[0]-15):min(len(text), s2[1]+15)]
            })

print(f"\nFusion (gap=6): Merges improving exact match: {len(improved_by_fusion)}")
print(f"Fusion (gap=6): Merges damaging exact match (fused 2 distinct gold spans): {len(damaged_by_fusion)}")

print("\n--- Examples of Fusion Damaging Exact Match (Merging 2 Distinct Gold Spans) ---")
for ex in damaged_by_fusion[:8]:
    print(f"[{ex['cat']}] S1: \"{ex['s1'][1]}\" (Gold={ex['s1'][2]}) + GAP: '{ex['gap']}' + S2: \"{ex['s2'][1]}\" (Gold={ex['s2'][2]}) -> Merged: \"{ex['merged'][1]}\" (NOT in gold)")

print("\n--- Examples of Fusion Improving Exact Match (Reuniting Fragmented Gold Span) ---")
for ex in improved_by_fusion[:8]:
    print(f"[{ex['cat']}] S1: \"{ex['s1'][1]}\" + GAP: '{ex['gap']}' + S2: \"{ex['s2'][1]}\" -> Merged: \"{ex['merged'][1]}\" (MATCHES GOLD!)")
