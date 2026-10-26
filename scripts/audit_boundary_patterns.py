import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
from src.sbdh_rules_baseline import find_maximum_bipartite_matching
from src.experimental_decoders import decode_strategy_b_active_threshold

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)
examples = []

for note in cached:
    text = note["text"]
    golds = sorted(list(note["gold_tuples"]))
    preds = sorted(list(decode_strategy_b_active_threshold(
        logits=note["logits"],
        active_mask=note["active_mask"],
        offset_mapping=note["offset_mapping"],
        text=text,
        threshold=0.30
    )))
    matched = find_maximum_bipartite_matching(preds, golds)
    for p_idx, g_idx in matched:
        p_s, p_e, p_c = preds[p_idx]
        g_s, g_e, g_c = golds[g_idx]
        if p_c == g_c and (p_s != g_s or p_e != g_e):
            p_text = text[p_s:p_e]
            g_text = text[g_s:g_e]
            examples.append({
                "cat": p_c,
                "pred": (p_s, p_e, p_text),
                "gold": (g_s, g_e, g_text),
                "diff_start": p_s - g_s,
                "diff_end": p_e - g_e,
                "snippet": text[max(0, min(p_s, g_s)-10):min(len(text), max(p_e, g_e)+10)]
            })

print(f"Total boundary mismatch pairs: {len(examples)}")

# Analyze patterns
# Pattern 1: Punctuation attached to start or end of pred
punct_starts = 0
punct_ends = 0
for ex in examples:
    p_text = ex["pred"][2]
    if p_text and p_text[0] in '.,;:!?"\'()[]{}':
        punct_starts += 1
    if p_text and p_text[-1] in '.,;:!?"\'()[]{}':
        punct_ends += 1

print(f"Punctuation at start: {punct_starts}, Punctuation at end: {punct_ends}")

# Look at 20 examples
for i, ex in enumerate(examples[:20]):
    print(f"\n--- Example {i+1} [{ex['cat']}] ---")
    print(f"Pred: [{ex['pred'][0]}:{ex['pred'][1]}] \"{ex['pred'][2]}\"")
    print(f"Gold: [{ex['gold'][0]}:{ex['gold'][1]}] \"{ex['gold'][2]}\"")
    print(f"Context: \"{ex['snippet']}\"")
    print(f"Diff start (pred - gold): {ex['diff_start']}, Diff end: {ex['diff_end']}")
