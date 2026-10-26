import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch

cached = torch.load("reports/cached_val_notes.pt", weights_only=False)

gold_with_parens = []
for note_idx, note in enumerate(cached):
    text = note["text"]
    for s, e, c in note["gold_tuples"]:
        sub = text[s:e]
        if any(p in sub for p in "()[]{}"):
            gold_with_parens.append((note_idx, s, e, c, sub))

print(f"Total gold spans with parens/brackets: {len(gold_with_parens)}")
for ex in gold_with_parens:
    print(f"Note {ex[0]} [{ex[3]}]: [{ex[1]}:{ex[2]}] \"{ex[4]}\"")
