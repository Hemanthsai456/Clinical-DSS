"""
Trace a training example through the entire transformer pipeline:
raw annotations -> character offsets -> tokenizer offsets -> token targets -> loss mask -> model logits -> decoded spans -> metrics.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import load_synth_sbdh
from src.span_alignment import align_spans_in_note, parse_delimited_field
from src.transformer_dataset import encode_single_note, decode_predictions_to_spans, SBDH_CATEGORIES, CAT_TO_IDX
from src.transformer_model import MultiLabelSBDHTransformer
from src.sbdh_rules_baseline import match_spans_exact, match_spans_relaxed

def trace_example():
    dfs, _ = load_synth_sbdh()
    train_df = dfs["train"]
    row = train_df.iloc[0]

    ex_no = int(row["ex_no"])
    text = str(row["Text"])
    spans = parse_delimited_field(row.get("Textspan"))
    sbdhs = parse_delimited_field(row.get("SBDH"))
    presences = parse_delimited_field(row.get("Presence"))
    periods = parse_delimited_field(row.get("Period"))

    print("=" * 80)
    print("STEP 1: RAW ANNOTATIONS")
    print("=" * 80)
    print(f"ex_no: {ex_no}")
    print(f"Text: {text}")
    print(f"Raw Spans ({len(spans)}): {spans}")
    print(f"Raw SBDHs ({len(sbdhs)}): {sbdhs}")

    print("\n" + "=" * 80)
    print("STEP 2: CHARACTER OFFSET RESOLUTION")
    print("=" * 80)
    gold_aligned = align_spans_in_note(
        text=text,
        spans=spans,
        sbdhs=sbdhs,
        presences=presences,
        periods=periods,
        ex_no=ex_no,
        disambiguate_context=True,
    )
    for idx, interval in enumerate(gold_aligned["resolved_intervals"]):
        s, e = interval["start_char"], interval["end_char"]
        print(f"  Interval {idx+1}: [{s}:{e}] '{text[s:e]}' -> Categories: {interval['categories']}")

    print("\n" + "=" * 80)
    print("STEP 3: TOKENIZER OFFSETS & ACTIVE MASK")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", add_prefix_space=True)
    encoded = encode_single_note(
        text=text,
        gold_intervals=gold_aligned["resolved_intervals"],
        tokenizer=tokenizer,
        max_length=64, # short for inspection
        ex_no=ex_no,
        ambiguous_cases=gold_aligned["ambiguous_cases"],
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offset_mapping = encoded["offset_mapping"]
    active_mask = encoded["active_mask"]
    targets = encoded["targets"]
    loss_mask = encoded["loss_mask"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(f"{'Idx':<4} {'Token':<16} {'Offsets':<10} {'Active':<8} {'Targets Active (Cat: Tag)'}")
    print("-" * 75)
    for i in range(len(input_ids)):
        if attention_mask[i] == 0:
            break
        s, e = offset_mapping[i]
        act = active_mask[i]
        tok = tokens[i]
        # Find active targets
        active_tags = []
        for cat_idx, cat in enumerate(SBDH_CATEGORIES):
            b_idx = cat_idx * 3 + 0
            i_idx = cat_idx * 3 + 1
            o_idx = cat_idx * 3 + 2
            if targets[i, b_idx] == 1.0:
                active_tags.append(f"{cat}: B")
            if targets[i, i_idx] == 1.0:
                active_tags.append(f"{cat}: I")
        
        target_str = ", ".join(active_tags) if active_tags else "All O"
        print(f"{i:<4} {tok:<16} {str((s, e)):<10} {act:<8} {target_str}")

    print("\n" + "=" * 80)
    print("STEP 4: LOSS MASK VERIFICATION")
    print("=" * 80)
    print(f"Loss mask shape: {loss_mask.shape} (seq_len x 15 categories)")
    print(f"Token active mask sum: {active_mask.sum()} / {len(active_mask)}")
    print(f"Special tokens active: {active_mask[0]} (first tok: {tokens[0]}), {active_mask[len(text.split())+5]}...")

    print("\n" + "=" * 80)
    print("STEP 5: DECODER RECONSTRUCTION FROM PERFECT TARGETS (Oracle Check)")
    print("=" * 80)
    # If logits match targets perfectly:
    oracle_logits = np.where(targets == 1.0, 10.0, -10.0)
    decoded_oracle = decode_predictions_to_spans(
        logits=oracle_logits,
        active_mask=active_mask,
        offset_mapping=offset_mapping,
        text=text,
        threshold=0.5,
    )
    print("Decoded Spans from Oracle Targets:")
    for s, e, cat in sorted(decoded_oracle):
        print(f"  [{s}:{e}] '{text[s:e]}' -> {cat}")

    # Check match with gold
    gold_tuples = set()
    for interval in gold_aligned["resolved_intervals"]:
        for c in interval["categories"]:
            gold_tuples.add((interval["start_char"], interval["end_char"], c))

    tp, fp, fn = match_spans_exact(decoded_oracle, gold_tuples)
    print(f"Exact match on Oracle: TP={tp}, FP={fp}, FN={fn}")
    rel_tp, rel_fp, rel_fn, _ = match_spans_relaxed(decoded_oracle, gold_tuples)
    print(f"Relaxed match on Oracle: TP={rel_tp}, FP={rel_fp}, FN={rel_fn}")

if __name__ == "__main__":
    trace_example()
