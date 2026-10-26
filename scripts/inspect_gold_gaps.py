import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
from src.sbdh_rules_baseline import match_spans_exact, match_spans_relaxed
from src.experimental_decoders import decode_strategy_b_active_threshold

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

# Let's inspect gold annotations in notes where merge candidates occur
same_cat_golds = 0
for note in cached:
    golds = sorted(list(note["gold_tuples"]), key=lambda x: (x[2], x[0], x[1]))
    for i in range(len(golds) - 1):
        s1, e1, c1 = golds[i]
        s2, e2, c2 = golds[i+1]
        if c1 == c2 and e1 <= s2:
            gap_len = s2 - e1
            gap_text = note["text"][e1:s2]
            if gap_len <= 15:
                same_cat_golds += 1
                print(f"[GOLD GAP {gap_len}] [{c1}] \"{note['text'][s1:e1]}\" + GAP: {repr(gap_text)} + \"{note['text'][s2:e2]}\"")

print(f"\nTotal same-category adjacent gold spans with gap <= 15: {same_cat_golds}")
