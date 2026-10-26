"""
Experiment 2.1: BIO Decoder Ablation Runner.
Evaluates Strategy A (Control), Strategy B (Active-Threshold), and Strategy C (Relative-Margin)
across the 876 validation notes (synth_sbdh_val.csv).
Saves results to reports/experiments/decoder_ablation/decoder_ablation_metrics.json.
"""
import os
import sys
import time
import json
import hashlib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sbdh_rules_baseline import match_spans_exact, find_maximum_bipartite_matching
from src.transformer_dataset import SBDH_CATEGORIES, CAT_TO_IDX
from src.experimental_decoders import (
    decode_strategy_a_control,
    decode_strategy_b_active_threshold,
    decode_strategy_c_relative_margin,
)
from src.utils import save_json

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


def evaluate_decoder_strategy(cached_notes, decode_fn, decode_kwargs, name: str):
    """Evaluates a decoding function across all cached validation notes."""
    all_cats = sorted(SBDH_CATEGORIES)
    exact_tp, exact_fp, exact_fn = 0, 0, 0
    relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0
    total_preds = 0
    total_gold = sum(len(n["gold_tuples"]) for n in cached_notes)
    boundary_errors = 0
    overlapping_spans_count = 0

    cat_relaxed_tp = {c: 0 for c in all_cats}
    cat_relaxed_fp = {c: 0 for c in all_cats}
    cat_relaxed_fn = {c: 0 for c in all_cats}
    cat_gold_support = {c: 0 for c in all_cats}
    cat_pred_count = {c: 0 for c in all_cats}

    cat_exact_tp = {c: 0 for c in all_cats}
    cat_exact_fp = {c: 0 for c in all_cats}
    cat_exact_fn = {c: 0 for c in all_cats}

    note_cat_tp = {c: 0 for c in all_cats}
    note_cat_fp = {c: 0 for c in all_cats}
    note_cat_fn = {c: 0 for c in all_cats}

    t0 = time.time()

    for note in cached_notes:
        text = note["text"]
        gold_tuples = note["gold_tuples"]
        pred_tuples = decode_fn(
            logits=note["logits"],
            active_mask=note["active_mask"],
            offset_mapping=note["offset_mapping"],
            text=text,
            **decode_kwargs,
        )
        total_preds += len(pred_tuples)

        # Check for invalid or intra-category overlapping spans
        sorted_preds_list = sorted(list(pred_tuples), key=lambda x: (x[2], x[0], x[1]))
        for i in range(len(sorted_preds_list) - 1):
            c1, s1, e1 = sorted_preds_list[i][2], sorted_preds_list[i][0], sorted_preds_list[i][1]
            c2, s2, e2 = sorted_preds_list[i+1][2], sorted_preds_list[i+1][0], sorted_preds_list[i+1][1]
            if c1 == c2 and max(s1, s2) < min(e1, e2):
                overlapping_spans_count += 1

        for _, _, c in gold_tuples:
            cat_gold_support[c] += 1
        for _, _, c in pred_tuples:
            cat_pred_count[c] += 1

        # Note-level metrics
        g_cats = {c for (_, _, c) in gold_tuples}
        p_cats = {c for (_, _, c) in pred_tuples}
        for c in all_cats:
            if c in g_cats and c in p_cats:
                note_cat_tp[c] += 1
            elif c in p_cats and c not in g_cats:
                note_cat_fp[c] += 1
            elif c in g_cats and c not in p_cats:
                note_cat_fn[c] += 1

        # Exact matching
        e_tp, e_fp, e_fn = match_spans_exact(pred_tuples, gold_tuples)
        exact_tp += e_tp
        exact_fp += e_fp
        exact_fn += e_fn

        # Relaxed matching (Kuhn's augmenting path bipartite matching)
        sorted_preds = sorted(list(pred_tuples), key=lambda x: (x[0], x[1], x[2]))
        sorted_golds = sorted(list(gold_tuples), key=lambda x: (x[0], x[1], x[2]))
        matched_idx_pairs = find_maximum_bipartite_matching(sorted_preds, sorted_golds)

        r_tp = len(matched_idx_pairs)
        r_fp = len(pred_tuples) - r_tp
        r_fn = len(gold_tuples) - r_tp

        relaxed_tp += r_tp
        relaxed_fp += r_fp
        relaxed_fn += r_fn

        boundary_errors += max(0, r_tp - e_tp)

        # Per category metrics
        matched_pred = set()
        matched_gold = set()
        for (u, v) in matched_idx_pairs:
            p_s = sorted_preds[u]
            g_s = sorted_golds[v]
            matched_pred.add(p_s)
            matched_gold.add(g_s)
            cat = p_s[2]
            cat_relaxed_tp[cat] += 1
            if (p_s[0], p_s[1]) == (g_s[0], g_s[1]):
                cat_exact_tp[cat] += 1

        for p in pred_tuples:
            if p not in matched_pred:
                cat_relaxed_fp[p[2]] += 1
                cat_exact_fp[p[2]] += 1

        for g in gold_tuples:
            if g not in matched_gold:
                cat_relaxed_fn[g[2]] += 1
                cat_exact_fn[g[2]] += 1

    elapsed = time.time() - t0

    # Macro and exact/relaxed calculations
    e_p = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0.0
    e_r = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0
    e_f1 = (2 * e_p * e_r) / (e_p + e_r) if (e_p + e_r) > 0 else 0.0

    r_p = relaxed_tp / (relaxed_tp + relaxed_fp) if (relaxed_tp + relaxed_fp) > 0 else 0.0
    r_r = relaxed_tp / (relaxed_tp + relaxed_fn) if (relaxed_tp + relaxed_fn) > 0 else 0.0
    r_f1 = (2 * r_p * r_r) / (r_p + r_r) if (r_p + r_r) > 0 else 0.0

    tot_note_tp = sum(note_cat_tp.values())
    tot_note_fp = sum(note_cat_fp.values())
    tot_note_fn = sum(note_cat_fn.values())
    note_p = tot_note_tp / (tot_note_tp + tot_note_fp) if (tot_note_tp + tot_note_fp) > 0 else 0.0
    note_r = tot_note_tp / (tot_note_tp + tot_note_fn) if (tot_note_tp + tot_note_fn) > 0 else 0.0
    note_f1 = (2 * note_p * note_r) / (note_p + note_r) if (note_p + note_r) > 0 else 0.0

    per_category = {}
    for c in all_cats:
        sup = cat_gold_support[c]
        p_cnt = cat_pred_count[c]
        r_tp_c = cat_relaxed_tp[c]
        r_fp_c = cat_relaxed_fp[c]
        r_fn_c = cat_relaxed_fn[c]
        prec_c = r_tp_c / (r_tp_c + r_fp_c) if (r_tp_c + r_fp_c) > 0 else 0.0
        rec_c = r_tp_c / (r_tp_c + r_fn_c) if (r_tp_c + r_fn_c) > 0 else 0.0
        f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
        exact_tp_c = cat_exact_tp[c]

        per_category[c] = {
            "gold_support": sup,
            "predicted_count": p_cnt,
            "relaxed_tp": r_tp_c,
            "relaxed_fp": r_fp_c,
            "relaxed_fn": r_fn_c,
            "relaxed_precision": round(prec_c, 4),
            "relaxed_recall": round(rec_c, 4),
            "relaxed_f1": round(f1_c, 4),
            "exact_tp": exact_tp_c,
            "exact_f1": round((2 * (exact_tp_c / p_cnt) * (exact_tp_c / sup)) / ((exact_tp_c / p_cnt) + (exact_tp_c / sup)), 4) if (exact_tp_c > 0 and p_cnt > 0 and sup > 0) else 0.0,
        }

    return {
        "strategy_name": name,
        "decode_kwargs": decode_kwargs,
        "runtime_seconds": round(elapsed, 2),
        "total_gold_spans": total_gold,
        "total_pred_spans": total_preds,
        "overlapping_spans_count": overlapping_spans_count,
        "exact_matching": {
            "tp": exact_tp, "fp": exact_fp, "fn": exact_fn,
            "precision": round(e_p, 4), "recall": round(e_r, 4), "f1_score": round(e_f1, 4),
        },
        "relaxed_matching": {
            "tp": relaxed_tp, "fp": relaxed_fp, "fn": relaxed_fn,
            "precision": round(r_p, 4), "recall": round(r_r, 4), "f1_score": round(r_f1, 4),
        },
        "boundary_errors_count": boundary_errors,
        "boundary_error_rate_pct": round(boundary_errors / relaxed_tp * 100, 2) if relaxed_tp > 0 else 0.0,
        "note_micro": {
            "precision": round(note_p, 4), "recall": round(note_r, 4), "f1_score": round(note_f1, 4),
        },
        "per_category": per_category,
    }


def run_experiment_2_1():
    print("=" * 80)
    print("TRACK B — EXPERIMENT 2.1: BIO DECODER ABLATION")
    print("=" * 80)

    weights_path = "evernorth_m3_results/models/sbdh_transformer/best_model.pt"
    h = hashlib.sha256(open(weights_path, "rb").read()).hexdigest()
    print(f"Checkpoint Path: {weights_path}")
    print(f"Initial Checkpoint SHA256: {h}")

    # Load cached validation notes
    cache_path = "reports/cached_val_notes.pt"
    assert os.path.exists(cache_path), f"Missing cache: {cache_path}"
    cached_notes = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(cached_notes)} cached validation notes.")

    # 1. Strategy A: Control across thresholds
    results = {}
    print("\n>>> Running Strategy A (Control: Outside-Dominance Decoder)...")
    for th in [0.20, 0.30, 0.40, 0.50]:
        res_a = evaluate_decoder_strategy(
            cached_notes,
            decode_fn=decode_strategy_a_control,
            decode_kwargs={"threshold": th},
            name=f"Strategy A (Control, th={th:.2f})",
        )
        results[f"strategy_a_th_{th:.2f}"] = res_a
        print(f"  Th {th:.2f}: Exact F1={res_a['exact_matching']['f1_score']:.4f}, Relaxed F1={res_a['relaxed_matching']['f1_score']:.4f} (Prec={res_a['relaxed_matching']['precision']:.4f}, Rec={res_a['relaxed_matching']['recall']:.4f}, Preds={res_a['total_pred_spans']}, Time={res_a['runtime_seconds']}s)")

    # 2. Strategy B: Active Threshold across thresholds
    print("\n>>> Running Strategy B (Active-Threshold Decoder)...")
    for th in [0.20, 0.30, 0.40, 0.50]:
        res_b = evaluate_decoder_strategy(
            cached_notes,
            decode_fn=decode_strategy_b_active_threshold,
            decode_kwargs={"threshold": th},
            name=f"Strategy B (Active-Threshold, th={th:.2f})",
        )
        results[f"strategy_b_th_{th:.2f}"] = res_b
        print(f"  Th {th:.2f}: Exact F1={res_b['exact_matching']['f1_score']:.4f}, Relaxed F1={res_b['relaxed_matching']['f1_score']:.4f} (Prec={res_b['relaxed_matching']['precision']:.4f}, Rec={res_b['relaxed_matching']['recall']:.4f}, Preds={res_b['total_pred_spans']}, Time={res_b['runtime_seconds']}s)")

    # 3. Strategy C: Relative Margin
    print("\n>>> Running Strategy C (Relative-Margin Decoder)...")
    margin_configs = [
        (0.30, 0.00),
        (0.30, 0.05),
        (0.30, 0.10),
        (0.20, 0.00),
        (0.40, 0.00),
        (0.50, 0.00),
        # Diagnostic negative margins at th=0.30
        (0.30, -0.05),
        (0.30, -0.10),
    ]

    for (th, margin) in margin_configs:
        res_c = evaluate_decoder_strategy(
            cached_notes,
            decode_fn=decode_strategy_c_relative_margin,
            decode_kwargs={"threshold": th, "margin": margin},
            name=f"Strategy C (Margin={margin:+.2f}, th={th:.2f})",
        )
        results[f"strategy_c_margin_{margin:+.2f}_th_{th:.2f}"] = res_c
        print(f"  Margin={margin:+.2f}, Th {th:.2f}: Exact F1={res_c['exact_matching']['f1_score']:.4f}, Relaxed F1={res_c['relaxed_matching']['f1_score']:.4f} (Prec={res_c['relaxed_matching']['precision']:.4f}, Rec={res_c['relaxed_matching']['recall']:.4f}, Preds={res_c['total_pred_spans']}, Time={res_c['runtime_seconds']}s)")

    # 4. Evaluate all strategies on the synthetic narrative
    print("\n>>> Evaluating All Strategies on Experiment 1 Synthetic Narrative...")
    tokenizer = AutoTokenizer.from_pretrained("evernorth_m3_results/models/sbdh_transformer")
    from app.model_loader import load_track_b_model_and_tokenizer
    model, _ = load_track_b_model_and_tokenizer()

    encoding = tokenizer(NARRATIVE, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
    offset_mapping = encoding["offset_mapping"][0].tolist()
    active_mask = np.zeros(len(offset_mapping), dtype=np.float32)
    for t, (s, e) in enumerate(offset_mapping):
        if s != 0 or e != 0:
            active_mask[t] = 1.0

    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        narrative_logits = outputs["logits"].squeeze(0).cpu().numpy()

    narrative_eval = {}
    strategies_narrative = [
        ("Strategy A (Control, th=0.30)", decode_strategy_a_control, {"threshold": 0.30}),
        ("Strategy B (Active-Threshold, th=0.30)", decode_strategy_b_active_threshold, {"threshold": 0.30}),
        ("Strategy B (Active-Threshold, th=0.40)", decode_strategy_b_active_threshold, {"threshold": 0.40}),
        ("Strategy C (Margin=0.00, th=0.30)", decode_strategy_c_relative_margin, {"threshold": 0.30, "margin": 0.00}),
        ("Strategy C (Margin=0.05, th=0.30)", decode_strategy_c_relative_margin, {"threshold": 0.30, "margin": 0.05}),
        ("Strategy C (Margin=0.10, th=0.30)", decode_strategy_c_relative_margin, {"threshold": 0.30, "margin": 0.10}),
        ("Strategy C (Margin=-0.05, th=0.30)", decode_strategy_c_relative_margin, {"threshold": 0.30, "margin": -0.05}),
    ]

    for label, fn, kw in strategies_narrative:
        spans = fn(logits=narrative_logits, active_mask=active_mask, offset_mapping=np.array(offset_mapping), text=NARRATIVE, **kw)
        sorted_sp = sorted(list(spans), key=lambda x: (x[0], x[1]))
        span_items = []
        for s, e, cat in sorted_sp:
            span_items.append({
                "range": f"[{s}:{e}]",
                "category": cat,
                "text": NARRATIVE[s:e],
            })
        narrative_eval[label] = {
            "total_spans": len(sorted_sp),
            "unique_categories": len(set(x[2] for x in sorted_sp)),
            "categories": sorted(list(set(x[2] for x in sorted_sp))),
            "spans": span_items,
        }
        print(f"\n{label}: {len(sorted_sp)} spans across {len(set(x[2] for x in sorted_sp))} categories:")
        for sp in span_items:
            print(f"   {sp['range']} '{sp['text']}' -> {sp['category']}")

    # Export metrics JSON
    out_dir = "reports/experiments/decoder_ablation"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "decoder_ablation_metrics.json")
    
    final_payload = {
        "metadata": {
            "experiment": "Experiment 2.1: BIO Decoder Ablation",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint_path": weights_path,
            "sha256": h,
            "validation_notes_count": len(cached_notes),
            "total_gold_spans": results["strategy_a_th_0.30"]["total_gold_spans"],
        },
        "validation_benchmarks": results,
        "narrative_diagnostic_eval": narrative_eval,
    }
    save_json(final_payload, json_path)
    print(f"\n[EXPORT] Full metrics saved to: {json_path}")

    # Final hash check
    h_final = hashlib.sha256(open(weights_path, "rb").read()).hexdigest()
    assert h == h_final, f"HASH DRIFT DETECTED: {h} != {h_final}"
    print(f"Final Checkpoint SHA256: {h_final} (UNCHANGED)")


if __name__ == "__main__":
    run_experiment_2_1()
