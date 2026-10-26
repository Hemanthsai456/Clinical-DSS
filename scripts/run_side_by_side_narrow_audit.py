"""
Side-by-Side Narrow Audit:
Recomputes baseline versus Arm 6A metrics for all 15 categories on synth_sbdh_test_reviewed.csv
using the EXACT SAME evaluation procedure for both.

Generates three side-by-side tables:
1. Note-Level Category Classification (Baseline vs Arm 6A)
2. Span-Level Exact Matching (Baseline vs Arm 6A)
3. Span-Level Relaxed Matching (Baseline vs Arm 6A)
"""
import os
import sys
import time
import json
import hashlib
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from src.data_loaders import load_synth_sbdh
from src.transformer_model import MultiLabelSBDHTransformer
from src.transformer_dataset import (
    SBDH_CATEGORIES,
    encode_single_note,
    parse_delimited_field,
    decode_predictions_to_spans,
)
from src.sbdh_rules_baseline import (
    match_spans_exact,
    find_maximum_bipartite_matching,
    align_spans_in_note,
)
from src.experimental_decoders import decode_strategy_b_active_threshold
from src.experimental_boundary_snapping import (
    snap_span_boundaries,
    fuse_same_category_spans,
)
from src.utils import save_json


def f1(p: float, r: float) -> float:
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_side_by_side_audit():
    weights_path = os.path.join(PROJECT_ROOT, "evernorth_m3_results", "models", "sbdh_transformer", "best_model.pt")
    colab_dir = os.path.dirname(weights_path)
    sha256_before = hash_file(weights_path)
    expected_hash = "0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4"
    assert sha256_before == expected_hash, f"Hash mismatch: {sha256_before}"
    print(f"Checkpoint SHA-256 (before): {sha256_before}")

    dfs, _ = load_synth_sbdh()
    test_df = dfs["test_reviewed"]
    total_notes = len(test_df)
    print(f"Loaded {total_notes:,d} notes from synth_sbdh_test_reviewed.csv.")

    tokenizer = AutoTokenizer.from_pretrained(colab_dir)
    model = MultiLabelSBDHTransformer.from_pretrained(colab_dir)
    model.eval()

    all_categories = sorted(SBDH_CATEGORIES)

    # 1. Note-Level Accumulators
    note_gold_support = {c: 0 for c in all_categories}
    base_note_tp = {c: 0 for c in all_categories}
    base_note_fp = {c: 0 for c in all_categories}
    base_note_fn = {c: 0 for c in all_categories}

    cand_note_tp = {c: 0 for c in all_categories}
    cand_note_fp = {c: 0 for c in all_categories}
    cand_note_fn = {c: 0 for c in all_categories}

    # 2. Span-Level Exact Accumulators
    span_gold_support = {c: 0 for c in all_categories}
    base_pred_count = {c: 0 for c in all_categories}
    cand_pred_count = {c: 0 for c in all_categories}

    base_exact_tp = {c: 0 for c in all_categories}
    cand_exact_tp = {c: 0 for c in all_categories}

    # 3. Span-Level Relaxed Accumulators
    base_relaxed_tp = {c: 0 for c in all_categories}
    base_relaxed_fp = {c: 0 for c in all_categories}
    base_relaxed_fn = {c: 0 for c in all_categories}

    cand_relaxed_tp = {c: 0 for c in all_categories}
    cand_relaxed_fp = {c: 0 for c in all_categories}
    cand_relaxed_fn = {c: 0 for c in all_categories}

    t0 = time.time()
    batch_size = 32
    note_records = []

    print("Pre-encoding notes...")
    for i, (_, row) in enumerate(test_df.iterrows()):
        ex_no = int(row["ex_no"]) if pd.notna(row.get("ex_no")) else None
        text = str(row["Text"])
        spans = parse_delimited_field(row.get("Textspan"))
        sbdhs = parse_delimited_field(row.get("SBDH"))
        presences = parse_delimited_field(row.get("Presence"))
        periods = parse_delimited_field(row.get("Period"))

        gold_aligned = align_spans_in_note(
            text=text,
            spans=spans,
            sbdhs=sbdhs,
            presences=presences,
            periods=periods,
            ex_no=ex_no,
            disambiguate_context=True,
        )

        gold_intervals = gold_aligned["resolved_intervals"]
        gold_tuples: Set[Tuple[int, int, str]] = set()
        gold_cats_in_note: Set[str] = set()
        for g in gold_intervals:
            for cat in g["categories"]:
                gold_tuples.add((g["start_char"], g["end_char"], cat))
                gold_cats_in_note.add(cat)
                span_gold_support[cat] += 1

        for c in gold_cats_in_note:
            note_gold_support[c] += 1

        encoded = encode_single_note(
            text=text,
            gold_intervals=gold_intervals,
            tokenizer=tokenizer,
            max_length=256,
            ex_no=ex_no,
            ambiguous_cases=gold_aligned["ambiguous_cases"],
        )

        note_records.append({
            "text": text,
            "gold_tuples": gold_tuples,
            "gold_cats_in_note": gold_cats_in_note,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "active_mask": encoded["active_mask"],
            "offset_mapping": encoded["offset_mapping"],
        })

    print(f"Notes pre-encoded in {time.time() - t0:.1f}s. Running batch inference (batch_size={batch_size})...")

    t_infer = time.time()
    for b_start in range(0, total_notes, batch_size):
        b_end = min(b_start + batch_size, total_notes)
        batch = note_records[b_start:b_end]

        input_ids_tensor = torch.tensor(np.array([n["input_ids"] for n in batch]), dtype=torch.long)
        attention_mask_tensor = torch.tensor(np.array([n["attention_mask"] for n in batch]), dtype=torch.long)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            logits_batch = outputs["logits"].cpu().numpy()

        for idx_in_batch, note in enumerate(batch):
            logits = logits_batch[idx_in_batch]
            text = note["text"]
            active_mask = note["active_mask"]
            offset_mapping = note["offset_mapping"]
            gold_tuples = note["gold_tuples"]
            gold_cats_in_note = note["gold_cats_in_note"]

            # -----------------------------------------------------------------
            # 1. BASELINE PREDICTIONS (Strategy A Control, th=0.30)
            # -----------------------------------------------------------------
            base_preds = decode_predictions_to_spans(
                logits=logits,
                active_mask=active_mask,
                offset_mapping=offset_mapping,
                text=text,
                threshold=0.30,
            )
            base_cats_in_note = {c for (_, _, c) in base_preds}

            # -----------------------------------------------------------------
            # 2. CANDIDATE PREDICTIONS (Arm 6A: Strategy B + Snap + Fuse gap<=2)
            # -----------------------------------------------------------------
            cand_raw = decode_strategy_b_active_threshold(
                logits=logits,
                active_mask=active_mask,
                offset_mapping=offset_mapping,
                text=text,
                threshold=0.30,
            )
            cand_snapped = set(snap_span_boundaries(p, text)[0] for p in cand_raw)
            cand_fused, _, _ = fuse_same_category_spans(cand_snapped, text, max_gap=2)
            cand_preds = cand_fused
            cand_cats_in_note = {c for (_, _, c) in cand_preds}

            # A. Note-Level Evaluation (Same Procedure for Both)
            for c in all_categories:
                in_gold = c in gold_cats_in_note
                # Baseline note-level
                in_base = c in base_cats_in_note
                if in_gold and in_base:
                    base_note_tp[c] += 1
                elif in_base and not in_gold:
                    base_note_fp[c] += 1
                elif in_gold and not in_base:
                    base_note_fn[c] += 1

                # Candidate note-level
                in_cand = c in cand_cats_in_note
                if in_gold and in_cand:
                    cand_note_tp[c] += 1
                elif in_cand and not in_gold:
                    cand_note_fp[c] += 1
                elif in_gold and not in_cand:
                    cand_note_fn[c] += 1

            # B. Span-Level Exact Matching (Same Procedure for Both)
            for s, e, c in base_preds:
                base_pred_count[c] += 1
            for s, e, c in cand_preds:
                cand_pred_count[c] += 1

            for s, e, c in (set(base_preds) & set(gold_tuples)):
                base_exact_tp[c] += 1
            for s, e, c in (set(cand_preds) & set(gold_tuples)):
                cand_exact_tp[c] += 1

            # C. Span-Level Relaxed Matching (Same Procedure for Both)
            sorted_golds = sorted(list(gold_tuples), key=lambda x: (x[0], x[1], x[2]))

            # Baseline relaxed matching
            b_sorted_preds = sorted(list(base_preds), key=lambda x: (x[0], x[1], x[2]))
            b_matched = find_maximum_bipartite_matching(b_sorted_preds, sorted_golds)
            b_p_matched = {p for p, _ in b_matched}
            b_g_matched = {g for _, g in b_matched}
            for p_idx, _ in b_matched:
                base_relaxed_tp[b_sorted_preds[p_idx][2]] += 1
            for p_idx, p in enumerate(b_sorted_preds):
                if p_idx not in b_p_matched:
                    base_relaxed_fp[p[2]] += 1
            for g_idx, g in enumerate(sorted_golds):
                if g_idx not in b_g_matched:
                    base_relaxed_fn[g[2]] += 1

            # Candidate relaxed matching
            c_sorted_preds = sorted(list(cand_preds), key=lambda x: (x[0], x[1], x[2]))
            c_matched = find_maximum_bipartite_matching(c_sorted_preds, sorted_golds)
            c_p_matched = {p for p, _ in c_matched}
            c_g_matched = {g for _, g in c_matched}
            for p_idx, _ in c_matched:
                cand_relaxed_tp[c_sorted_preds[p_idx][2]] += 1
            for p_idx, p in enumerate(c_sorted_preds):
                if p_idx not in c_p_matched:
                    cand_relaxed_fp[p[2]] += 1
            for g_idx, g in enumerate(sorted_golds):
                if g_idx not in c_g_matched:
                    cand_relaxed_fn[g[2]] += 1

        if (b_end) % 256 == 0 or b_end == total_notes:
            print(f"  Processed {b_end:,d} / {total_notes:,d} notes ({time.time() - t_infer:.1f}s elapsed)...")

    duration = time.time() - t0
    print(f"\nCompleted side-by-side evaluation in {duration:.1f}s.")

    sha256_after = hash_file(weights_path)
    assert sha256_after == sha256_before
    print(f"Checkpoint SHA-256 (after): {sha256_after} (Verified bit-identical)")

    # Assert exact agreement with previously verified baseline totals
    assert sum(base_exact_tp.values()) == 1616, f"Base exact TP sum != 1616 ({sum(base_exact_tp.values())})"
    assert sum(base_relaxed_tp.values()) == 2637, f"Base rel TP sum != 2637 ({sum(base_relaxed_tp.values())})"
    assert sum(base_note_tp.values()) == 2357, f"Base note TP sum != 2357 ({sum(base_note_tp.values())})"

    # Assert exact agreement with previously verified candidate totals
    assert sum(cand_exact_tp.values()) == 1698, f"Cand exact TP sum != 1698 ({sum(cand_exact_tp.values())})"
    assert sum(cand_relaxed_tp.values()) == 2762, f"Cand rel TP sum != 2762 ({sum(cand_relaxed_tp.values())})"

    print("\n" + "=" * 120)
    print("TABLE 1: NOTE-LEVEL CATEGORY CLASSIFICATION (BASELINE VS ARM 6A CANDIDATE)")
    print("Evaluation Procedure: Binary classification per note-category pair (1 = category present in note, 0 = absent)")
    print("=" * 120)
    print(f"{'Category':<32} | {'Notes':<5} | {'Base TP':<7} | {'Cand TP':<7} | {'Diff TP':<7} | {'Base P':<6} | {'Cand P':<6} | {'Base R':<6} | {'Cand R':<6} | {'Base F1':<7} | {'Cand F1':<7} | {'Diff F1':<7}")
    print("-" * 120)

    note_rows = []
    base_macro_f1s = []
    cand_macro_f1s = []

    for c in all_categories:
        sup = note_gold_support[c]
        b_tp = base_note_tp[c]
        b_fp = base_note_fp[c]
        b_fn = base_note_fn[c]
        bp = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        br = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        bf = f1(bp, br)
        base_macro_f1s.append(bf)

        c_tp = cand_note_tp[c]
        c_fp = cand_note_fp[c]
        c_fn = cand_note_fn[c]
        cp = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        cr = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        cf = f1(cp, cr)
        cand_macro_f1s.append(cf)

        print(f"{c:<32} | {sup:>5} | {b_tp:>7} | {c_tp:>7} | {c_tp - b_tp:>+7} | {bp:>6.4f} | {cp:>6.4f} | {br:>6.4f} | {cr:>6.4f} | {bf:>7.4f} | {cf:>7.4f} | {cf - bf:>+7.4f}")
        note_rows.append({
            "category": c,
            "notes_support": sup,
            "base_tp": b_tp, "base_fp": b_fp, "base_fn": b_fn, "base_p": round(bp, 4), "base_r": round(br, 4), "base_f1": round(bf, 4),
            "cand_tp": c_tp, "cand_fp": c_fp, "cand_fn": c_fn, "cand_p": round(cp, 4), "cand_r": round(cr, 4), "cand_f1": round(cf, 4),
        })

    print("-" * 120)
    # Note micro
    b_tot_tp = sum(base_note_tp.values())
    b_tot_fp = sum(base_note_fp.values())
    b_tot_fn = sum(base_note_fn.values())
    b_tot_sup = sum(note_gold_support.values())
    b_mic_p = b_tot_tp / (b_tot_tp + b_tot_fp)
    b_mic_r = b_tot_tp / (b_tot_tp + b_tot_fn)
    b_mic_f = f1(b_mic_p, b_mic_r)

    c_tot_tp = sum(cand_note_tp.values())
    c_tot_fp = sum(cand_note_fp.values())
    c_tot_fn = sum(cand_note_fn.values())
    c_mic_p = c_tot_tp / (c_tot_tp + c_tot_fp)
    c_mic_r = c_tot_tp / (c_tot_tp + c_tot_fn)
    c_mic_f = f1(c_mic_p, c_mic_r)

    print(f"{'NOTE MICRO-SUM':<32} | {b_tot_sup:>5} | {b_tot_tp:>7} | {c_tot_tp:>7} | {c_tot_tp - b_tot_tp:>+7} | {b_mic_p:>6.4f} | {c_mic_p:>6.4f} | {b_mic_r:>6.4f} | {c_mic_r:>6.4f} | {b_mic_f:>7.4f} | {c_mic_f:>7.4f} | {c_mic_f - b_mic_f:>+7.4f}")
    print(f"{'NOTE MACRO-AVERAGE':<32} | {'-':>5} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>6} | {'-':>6} | {'-':>6} | {'-':>6} | {np.mean(base_macro_f1s):>7.4f} | {np.mean(cand_macro_f1s):>7.4f} | {np.mean(cand_macro_f1s) - np.mean(base_macro_f1s):>+7.4f}")

    print("\n" + "=" * 120)
    print("TABLE 2: SPAN-LEVEL EXACT MATCHING PER CATEGORY (BASELINE VS ARM 6A CANDIDATE)")
    print("Evaluation Procedure: Strict string boundary agreement (start_char, end_char, category)")
    print("=" * 120)
    print(f"{'Category':<32} | {'Gold':<5} | {'Base Pred':<9} | {'Cand Pred':<9} | {'Base TP':<7} | {'Cand TP':<7} | {'Diff TP':<7} | {'Base P':<6} | {'Cand P':<6} | {'Base R':<6} | {'Cand R':<6} | {'Base F1':<7} | {'Cand F1':<7} | {'Diff F1':<7}")
    print("-" * 140)

    exact_rows = []
    base_exact_f1s = []
    cand_exact_f1s = []

    for c in all_categories:
        gold = span_gold_support[c]
        b_pred = base_pred_count[c]
        c_pred = cand_pred_count[c]

        b_tp = base_exact_tp[c]
        b_fp = b_pred - b_tp
        b_fn = gold - b_tp
        bp = b_tp / b_pred if b_pred > 0 else 0.0
        br = b_tp / gold if gold > 0 else 0.0
        bf = f1(bp, br)
        base_exact_f1s.append(bf)

        c_tp = cand_exact_tp[c]
        c_fp = c_pred - c_tp
        c_fn = gold - c_tp
        cp = c_tp / c_pred if c_pred > 0 else 0.0
        cr = c_tp / gold if gold > 0 else 0.0
        cf = f1(cp, cr)
        cand_exact_f1s.append(cf)

        print(f"{c:<32} | {gold:>5} | {b_pred:>9} | {c_pred:>9} | {b_tp:>7} | {c_tp:>7} | {c_tp - b_tp:>+7} | {bp:>6.4f} | {cp:>6.4f} | {br:>6.4f} | {cr:>6.4f} | {bf:>7.4f} | {cf:>7.4f} | {cf - bf:>+7.4f}")
        exact_rows.append({
            "category": c,
            "gold_support": gold,
            "base_preds": b_pred, "cand_preds": c_pred,
            "base_tp": b_tp, "base_fp": b_fp, "base_fn": b_fn, "base_p": round(bp, 4), "base_r": round(br, 4), "base_f1": round(bf, 4),
            "cand_tp": c_tp, "cand_fp": c_fp, "cand_fn": c_fn, "cand_p": round(cp, 4), "cand_r": round(cr, 4), "cand_f1": round(cf, 4),
        })

    print("-" * 140)
    tot_gold = sum(span_gold_support.values())
    b_tot_preds = sum(base_pred_count.values())
    c_tot_preds = sum(cand_pred_count.values())
    b_tot_ex_tp = sum(base_exact_tp.values())
    c_tot_ex_tp = sum(cand_exact_tp.values())

    b_ex_mic_p = b_tot_ex_tp / b_tot_preds
    b_ex_mic_r = b_tot_ex_tp / tot_gold
    b_ex_mic_f = f1(b_ex_mic_p, b_ex_mic_r)

    c_ex_mic_p = c_tot_ex_tp / c_tot_preds
    c_ex_mic_r = c_tot_ex_tp / tot_gold
    c_ex_mic_f = f1(c_ex_mic_p, c_ex_mic_r)

    print(f"{'EXACT MICRO-SUM':<32} | {tot_gold:>5} | {b_tot_preds:>9} | {c_tot_preds:>9} | {b_tot_ex_tp:>7} | {c_tot_ex_tp:>7} | {c_tot_ex_tp - b_tot_ex_tp:>+7} | {b_ex_mic_p:>6.4f} | {c_ex_mic_p:>6.4f} | {b_ex_mic_r:>6.4f} | {c_ex_mic_r:>6.4f} | {b_ex_mic_f:>7.4f} | {c_ex_mic_f:>7.4f} | {c_ex_mic_f - b_ex_mic_f:>+7.4f}")
    print(f"{'EXACT MACRO-AVERAGE':<32} | {'-':>5} | {'-':>9} | {'-':>9} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>6} | {'-':>6} | {'-':>6} | {'-':>6} | {np.mean(base_exact_f1s):>7.4f} | {np.mean(cand_exact_f1s):>7.4f} | {np.mean(cand_exact_f1s) - np.mean(base_exact_f1s):>+7.4f}")

    print("\n" + "=" * 120)
    print("TABLE 3: SPAN-LEVEL RELAXED MATCHING PER CATEGORY (BASELINE VS ARM 6A CANDIDATE)")
    print("Evaluation Procedure: Maximum Bipartite 1-to-1 Matching under category agreement & character overlap")
    print("=" * 120)
    print(f"{'Category':<32} | {'Gold':<5} | {'Base Pred':<9} | {'Cand Pred':<9} | {'Base TP':<7} | {'Cand TP':<7} | {'Diff TP':<7} | {'Base P':<6} | {'Cand P':<6} | {'Base R':<6} | {'Cand R':<6} | {'Base F1':<7} | {'Cand F1':<7} | {'Diff F1':<7}")
    print("-" * 140)

    relaxed_rows = []
    base_rel_f1s = []
    cand_rel_f1s = []

    for c in all_categories:
        gold = span_gold_support[c]
        b_pred = base_pred_count[c]
        c_pred = cand_pred_count[c]

        b_tp = base_relaxed_tp[c]
        b_fp = base_relaxed_fp[c]
        b_fn = base_relaxed_fn[c]
        bp = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        br = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        bf = f1(bp, br)
        base_rel_f1s.append(bf)

        c_tp = cand_relaxed_tp[c]
        c_fp = cand_relaxed_fp[c]
        c_fn = cand_relaxed_fn[c]
        cp = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        cr = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        cf = f1(cp, cr)
        cand_rel_f1s.append(cf)

        print(f"{c:<32} | {gold:>5} | {b_pred:>9} | {c_pred:>9} | {b_tp:>7} | {c_tp:>7} | {c_tp - b_tp:>+7} | {bp:>6.4f} | {cp:>6.4f} | {br:>6.4f} | {cr:>6.4f} | {bf:>7.4f} | {cf:>7.4f} | {cf - bf:>+7.4f}")
        relaxed_rows.append({
            "category": c,
            "gold_support": gold,
            "base_preds": b_pred, "cand_preds": c_pred,
            "base_tp": b_tp, "base_fp": b_fp, "base_fn": b_fn, "base_p": round(bp, 4), "base_r": round(br, 4), "base_f1": round(bf, 4),
            "cand_tp": c_tp, "cand_fp": c_fp, "cand_fn": c_fn, "cand_p": round(cp, 4), "cand_r": round(cr, 4), "cand_f1": round(cf, 4),
        })

    print("-" * 140)
    b_tot_rel_tp = sum(base_relaxed_tp.values())
    b_tot_rel_fp = sum(base_relaxed_fp.values())
    b_tot_rel_fn = sum(base_relaxed_fn.values())
    c_tot_rel_tp = sum(cand_relaxed_tp.values())
    c_tot_rel_fp = sum(cand_relaxed_fp.values())
    c_tot_rel_fn = sum(cand_relaxed_fn.values())

    b_rel_mic_p = b_tot_rel_tp / (b_tot_rel_tp + b_tot_rel_fp)
    b_rel_mic_r = b_tot_rel_tp / (b_tot_rel_tp + b_tot_rel_fn)
    b_rel_mic_f = f1(b_rel_mic_p, b_rel_mic_r)

    c_rel_mic_p = c_tot_rel_tp / (c_tot_rel_tp + c_tot_rel_fp)
    c_rel_mic_r = c_tot_rel_tp / (c_tot_rel_tp + c_tot_rel_fn)
    c_rel_mic_f = f1(c_rel_mic_p, c_rel_mic_r)

    print(f"{'RELAXED MICRO-SUM':<32} | {tot_gold:>5} | {b_tot_preds:>9} | {c_tot_preds:>9} | {b_tot_rel_tp:>7} | {c_tot_rel_tp:>7} | {c_tot_rel_tp - b_tot_rel_tp:>+7} | {b_rel_mic_p:>6.4f} | {c_rel_mic_p:>6.4f} | {b_rel_mic_r:>6.4f} | {c_rel_mic_r:>6.4f} | {b_rel_mic_f:>7.4f} | {c_rel_mic_f:>7.4f} | {c_rel_mic_f - b_rel_mic_f:>+7.4f}")
    print(f"{'RELAXED MACRO-AVERAGE':<32} | {'-':>5} | {'-':>9} | {'-':>9} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>6} | {'-':>6} | {'-':>6} | {'-':>6} | {np.mean(base_rel_f1s):>7.4f} | {np.mean(cand_rel_f1s):>7.4f} | {np.mean(cand_rel_f1s) - np.mean(base_rel_f1s):>+7.4f}")

    audit_results = {
        "dataset_name": "test_reviewed",
        "total_notes": total_notes,
        "checkpoint_sha256": sha256_after,
        "table_1_note_level": note_rows,
        "table_2_span_level_exact": exact_rows,
        "table_3_span_level_relaxed": relaxed_rows,
        "aggregates": {
            "note_level": {
                "baseline_micro": {"p": round(b_mic_p, 4), "r": round(b_mic_r, 4), "f1": round(b_mic_f, 4)},
                "candidate_micro": {"p": round(c_mic_p, 4), "r": round(c_mic_r, 4), "f1": round(c_mic_f, 4)},
                "baseline_macro": {"f1": round(float(np.mean(base_macro_f1s)), 4)},
                "candidate_macro": {"f1": round(float(np.mean(cand_macro_f1s)), 4)},
            },
            "span_level_exact": {
                "baseline_micro": {"tp": b_tot_ex_tp, "preds": b_tot_preds, "gold": tot_gold, "p": round(b_ex_mic_p, 4), "r": round(b_ex_mic_r, 4), "f1": round(b_ex_mic_f, 4)},
                "candidate_micro": {"tp": c_tot_ex_tp, "preds": c_tot_preds, "gold": tot_gold, "p": round(c_ex_mic_p, 4), "r": round(c_ex_mic_r, 4), "f1": round(c_ex_mic_f, 4)},
                "baseline_macro": {"f1": round(float(np.mean(base_exact_f1s)), 4)},
                "candidate_macro": {"f1": round(float(np.mean(cand_exact_f1s)), 4)},
            },
            "span_level_relaxed": {
                "baseline_micro": {"tp": b_tot_rel_tp, "preds": b_tot_preds, "gold": tot_gold, "p": round(b_rel_mic_p, 4), "r": round(b_rel_mic_r, 4), "f1": round(b_rel_mic_f, 4)},
                "candidate_micro": {"tp": c_tot_rel_tp, "preds": c_tot_preds, "gold": tot_gold, "p": round(c_rel_mic_p, 4), "r": round(c_rel_mic_r, 4), "f1": round(c_rel_mic_f, 4)},
                "baseline_macro": {"f1": round(float(np.mean(base_rel_f1s)), 4)},
                "candidate_macro": {"f1": round(float(np.mean(cand_rel_f1s)), 4)},
            },
        }
    }

    out_file = os.path.join(PROJECT_ROOT, "reports", "experiments", "candidate_test_evaluation", "side_by_side_narrow_audit_metrics.json")
    save_json(audit_results, out_file)
    print(f"\nSaved side-by-side audit metrics to: {out_file}")


if __name__ == "__main__":
    run_side_by_side_audit()
