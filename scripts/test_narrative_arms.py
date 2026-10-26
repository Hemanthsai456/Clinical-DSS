import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoTokenizer

from src.experimental_decoders import (
    decode_strategy_a_control,
    decode_strategy_b_active_threshold,
)
from scripts.test_arm_combinations import snap_span_boundaries, fuse_same_category_spans

NARRATIVE = (
    "Patient is a 58-year-old woman with type 2 diabetes who reports difficulty managing her condition. "
    "She has been unemployed for the past six months and is worried about paying her rent and utility bills. "
    "She sometimes skips meals because there is not enough money to buy groceries. "
    "The patient lives alone and reports limited family support. "
    "She does not own a car, and the nearest bus stop is approximately two miles from her home, "
    "making it difficult to attend medical appointments. "
    "She recently missed two follow-up visits because she could not arrange transportation. "
    "She reports persistent knee pain that limits her ability to walk and perform daily activities. "
    "The patient states that she understands her medication instructions but occasionally cannot afford to refill her prescriptions. "
    "She is interested in receiving assistance with food access, transportation, and medication costs."
)

from app.model_loader import load_track_b_model_and_tokenizer
model, tokenizer = load_track_b_model_and_tokenizer()

inputs = tokenizer(
    NARRATIVE,
    return_offsets_mapping=True,
    return_tensors="pt",
    truncation=True,
    max_length=256
)

with torch.no_grad():
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs["logits"][0].cpu().numpy()

active_mask = inputs["attention_mask"][0].cpu().numpy()
# zero out special tokens
active_mask[0] = 0.0
active_mask[-1] = 0.0
offsets = inputs["offset_mapping"][0].cpu().numpy()

# 1. Arm 1: Strategy A (th=0.30)
preds_arm1 = decode_strategy_a_control(logits, active_mask, offsets, NARRATIVE, threshold=0.30)

# 2. Arm 2: Strategy B (th=0.30)
preds_arm2 = decode_strategy_b_active_threshold(logits, active_mask, offsets, NARRATIVE, threshold=0.30)

# 3. Arm 3: Strategy B (th=0.40)
preds_arm3 = decode_strategy_b_active_threshold(logits, active_mask, offsets, NARRATIVE, threshold=0.40)

# 4. Arm 4: Strategy B (th=0.30) + Snapping only
preds_arm4 = []
for p in preds_arm2:
    sp, _ = snap_span_boundaries(p[0], p[1], p[2], NARRATIVE)
    preds_arm4.append(sp)
preds_arm4 = set(preds_arm4)

# 5. Arm 5: Strategy B (th=0.30) + Fusion only (gap=6)
preds_arm5, _, _ = fuse_same_category_spans(preds_arm2, NARRATIVE, max_gap=6)

# 6. Arm 6: Strategy B (th=0.30) + Both (gap=6)
preds_arm6, _, _ = fuse_same_category_spans(preds_arm4, NARRATIVE, max_gap=6)

arms = [
    ("Arm 1: Strategy A (Control, th=0.30)", preds_arm1),
    ("Arm 2: Strategy B (th=0.30, raw)", preds_arm2),
    ("Arm 3: Strategy B (th=0.40, raw)", preds_arm3),
    ("Arm 4: Strategy B (th=0.30) + Snapping", preds_arm4),
    ("Arm 5: Strategy B (th=0.30) + Fusion", preds_arm5),
    ("Arm 6: Strategy B (th=0.30) + Snap & Fuse", preds_arm6),
]

for name, preds in arms:
    print(f"\n==========================================")
    print(f"{name} (Total: {len(preds)})")
    print(f"==========================================")
    for s, e, c in sorted(list(preds), key=lambda x: (x[0], x[1])):
        print(f"[{s}:{e}] ({c}): \"{NARRATIVE[s:e]}\"")
