import sys
import os
import re
sys.path.insert(0, os.path.abspath("."))
import torch
from src.sbdh_rules_baseline import find_maximum_bipartite_matching, match_spans_exact
from src.experimental_decoders import decode_strategy_b_active_threshold

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

# Let's inspect potential merge candidates under Strategy B (th=0.30)
merge_candidates = []
for note in cached:
    text = note["text"]
    preds = sorted(list(decode_strategy_b_active_threshold(
        logits=note["logits"],
        active_mask=note["active_mask"],
        offset_mapping=note["offset_mapping"],
        text=text,
        threshold=0.30
    )), key=lambda x: (x[2], x[0], x[1]))
    
    # Check adjacent spans of the same category
    for i in range(len(preds) - 1):
        s1, e1, c1 = preds[i]
        s2, e2, c2 = preds[i+1]
        if c1 == c2 and e1 <= s2:
            gap_len = s2 - e1
            gap_text = text[e1:s2]
            merge_candidates.append({
                "cat": c1,
                "span1": (s1, e1, text[s1:e1]),
                "span2": (s2, e2, text[s2:e2]),
                "gap_len": gap_len,
                "gap_text": gap_text,
                "note_text": text
            })

print(f"Total same-category adjacent span pairs in Strategy B: {len(merge_candidates)}")
# Group by gap_len
gap_lens = [c["gap_len"] for c in merge_candidates]
print(f"Gap lengths <= 5: {sum(1 for g in gap_lens if g <= 5)}")
print(f"Gap lengths <= 10: {sum(1 for g in gap_lens if g <= 10)}")
print(f"Gap lengths <= 20: {sum(1 for g in gap_lens if g <= 20)}")
print(f"Gap lengths > 20: {sum(1 for g in gap_lens if g > 20)}")

print("\nSample gap texts for gap_len <= 15:")
for c in merge_candidates:
    if c["gap_len"] <= 15:
        print(f"[{c['cat']}] \"{c['span1'][2]}\" + GAP: {repr(c['gap_text'])} + \"{c['span2'][2]}\"")
