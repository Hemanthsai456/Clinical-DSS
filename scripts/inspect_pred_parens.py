import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
from src.experimental_decoders import decode_strategy_b_active_threshold

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

preds_with_parens = []
for note_idx, note in enumerate(cached):
    text = note["text"]
    preds = decode_strategy_b_active_threshold(
        logits=note["logits"],
        active_mask=note["active_mask"],
        offset_mapping=note["offset_mapping"],
        text=text,
        threshold=0.30
    )
    for s, e, c in preds:
        sub = text[s:e]
        if any(p in sub for p in "()[]{}"):
            preds_with_parens.append((note_idx, s, e, c, sub))

print(f"Total raw Strategy B predictions with parens/brackets: {len(preds_with_parens)}")
for ex in preds_with_parens:
    print(f"Note {ex[0]} [{ex[3]}]: [{ex[1]}:{ex[2]}] \"{ex[4]}\"")
