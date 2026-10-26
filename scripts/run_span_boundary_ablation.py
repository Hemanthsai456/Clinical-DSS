"""
Experiment 2.2: Span Boundary and Fragmentation Ablation Runner.

Evaluates 6 experimental arms across all 876 validation notes (synth_sbdh_val.csv):
1. Arm 1: Strategy A (Control, th=0.30)
2. Arm 2: Strategy B (th=0.30, current boundaries)
3. Arm 3: Strategy B (th=0.40, current boundaries)
4. Arm 4: Strategy B (th=0.30) + Lexical Boundary Snapping only
5. Arm 5: Strategy B (th=0.30) + Conservative Same-Category Span Fusion only (gap=2 and gap=6)
6. Arm 6: Strategy B (th=0.30) + Both Snapping & Fusion (gap=2 and gap=6)

Exports complete metrics to reports/experiments/span_boundary_ablation/span_boundary_ablation_metrics.json.
"""
import os
import sys
import time
import json
import hashlib
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sbdh_rules_baseline import match_spans_exact, find_maximum_bipartite_matching
from src.transformer_dataset import SBDH_CATEGORIES
from src.experimental_decoders import (
    decode_strategy_a_control,
    decode_strategy_b_active_threshold,
)
from src.experimental_boundary_snapping import (
    snap_span_boundaries,
    fuse_same_category_spans,
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


def evaluate_arm(
    cached_notes,
    arm_name: str,
    decode_fn,
    decode_kwargs: dict,
    do_snap: bool = False,
    do_fuse: bool = False,
    max_gap: int = 6,
):
    """Evaluates an experimental arm across all 876 cached validation notes."""
    all_cats = sorted(SBDH_CATEGORIES)
    exact_tp, exact_fp, exact_fn = 0, 0, 0
    relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0
    total_preds = 0
    total_gold = sum(len(n["gold_tuples"]) for n in cached_notes)
    boundary_errors = 0
    overlapping_spans_count = 0
    invalid_spans_count = 0
    total_snaps = 0
    total_merges = 0
    all_merge_records = []
    audited_snap_records = []

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
        gold_tuples = sorted(list(note["gold_tuples"]))
        preds_raw = decode_fn(
            logits=note["logits"],
            active_mask=note["active_mask"],
            offset_mapping=note["offset_mapping"],
            text=text,
            **decode_kwargs,
        )

        preds = preds_raw
        if do_snap and not do_fuse:
            snapped = []
            for p in preds:
                sp, ch = snap_span_boundaries(p, text)
                if ch:
                    total_snaps += 1
                    audited_snap_records.append({
                        "category": p[2],
                        "before": p,
                        "after": sp,
                        "before_text": text[p[0]:p[1]],
                        "after_text": text[sp[0]:sp[1]],
                    })
                snapped.append(sp)
            preds = set(snapped)

        elif do_fuse and not do_snap:
            preds, m_cnt, m_recs = fuse_same_category_spans(preds, text, max_gap=max_gap)
            total_merges += m_cnt
            all_merge_records.extend(m_recs)

        elif do_snap and do_fuse:
            snapped = []
            for p in preds:
                sp, ch = snap_span_boundaries(p, text)
                if ch:
                    total_snaps += 1
                    audited_snap_records.append({
                        "category": p[2],
                        "before": p,
                        "after": sp,
                        "before_text": text[p[0]:p[1]],
                        "after_text": text[sp[0]:sp[1]],
                    })
                snapped.append(sp)
            preds, m_cnt, m_recs = fuse_same_category_spans(set(snapped), text, max_gap=max_gap)
            total_merges += m_cnt
            all_merge_records.extend(m_recs)

        preds = sorted(list(preds))
        total_preds += len(preds)

        # Integrity checks
        for s, e, c in preds:
            if s >= e or s < 0 or e > len(text):
                invalid_spans_count += 1

        # Check for intra-category overlaps
        sorted_preds_list = sorted(preds, key=lambda x: (x[2], x[0], x[1]))
        for i in range(len(sorted_preds_list) - 1):
            c1, s1, e1 = sorted_preds_list[i][2], sorted_preds_list[i][0], sorted_preds_list[i][1]
            c2, s2, e2 = sorted_preds_list[i+1][2], sorted_preds_list[i+1][0], sorted_preds_list[i+1][1]
            if c1 == c2 and max(s1, s2) < min(e1, e2):
                overlapping_spans_count += 1

        for _, _, c in gold_tuples:
            cat_gold_support[c] += 1
        for _, _, c in preds:
            cat_pred_count[c] += 1

        # Note-level metrics
        g_cats = {c for (_, _, c) in gold_tuples}
        p_cats = {c for (_, _, c) in preds}
        for c in all_cats:
            if c in g_cats and c in p_cats:
                note_cat_tp[c] += 1
            elif c in p_cats and c not in g_cats:
                note_cat_fp[c] += 1
            elif c in g_cats and c not in p_cats:
                note_cat_fn[c] += 1

        # Exact matching
        e_tp, e_fp, e_fn = match_spans_exact(set(preds), set(gold_tuples))
        exact_tp += e_tp
        exact_fp += e_fp
        exact_fn += e_fn

        for s, e, c in (set(preds) & set(gold_tuples)):
            cat_exact_tp[c] += 1

        # Relaxed matching
        matched_pairs = find_maximum_bipartite_matching(preds, gold_tuples)
        r_tp = len(matched_pairs)
        r_fp = len(preds) - r_tp
        r_fn = len(gold_tuples) - r_tp
        relaxed_tp += r_tp
        relaxed_fp += r_fp
        relaxed_fn += r_fn

        for p_idx, g_idx in matched_pairs:
            p_span = preds[p_idx]
            g_span = gold_tuples[g_idx]
            cat_relaxed_tp[p_span[2]] += 1
            if p_span[0] != g_span[0] or p_span[1] != g_span[1]:
                boundary_errors += 1

        matched_p_indices = {p_idx for (p_idx, _) in matched_pairs}
        matched_g_indices = {g_idx for (_, g_idx) in matched_pairs}
        for p_idx, p_span in enumerate(preds):
            if p_idx not in matched_p_indices:
                cat_relaxed_fp[p_span[2]] += 1
        for g_idx, g_span in enumerate(gold_tuples):
            if g_idx not in matched_g_indices:
                cat_relaxed_fn[g_span[2]] += 1

    runtime_s = round(time.time() - t0, 2)

    for c in all_cats:
        cat_exact_fp[c] = cat_pred_count[c] - cat_exact_tp[c]
        cat_exact_fn[c] = cat_gold_support[c] - cat_exact_tp[c]

    exact_prec = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0.0
    exact_rec = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0
    exact_f1 = 2 * exact_prec * exact_rec / (exact_prec + exact_rec) if (exact_prec + exact_rec) > 0 else 0.0

    relaxed_prec = relaxed_tp / (relaxed_tp + relaxed_fp) if (relaxed_tp + relaxed_fp) > 0 else 0.0
    relaxed_rec = relaxed_tp / (relaxed_tp + relaxed_fn) if (relaxed_tp + relaxed_fn) > 0 else 0.0
    relaxed_f1 = 2 * relaxed_prec * relaxed_rec / (relaxed_prec + relaxed_rec) if (relaxed_prec + relaxed_rec) > 0 else 0.0

    note_tp_tot = sum(note_cat_tp.values())
    note_fp_tot = sum(note_cat_fp.values())
    note_fn_tot = sum(note_cat_fn.values())
    note_prec = note_tp_tot / (note_tp_tot + note_fp_tot) if (note_tp_tot + note_fp_tot) > 0 else 0.0
    note_rec = note_tp_tot / (note_tp_tot + note_fn_tot) if (note_tp_tot + note_fn_tot) > 0 else 0.0
    note_f1 = 2 * note_prec * note_rec / (note_prec + note_rec) if (note_prec + note_rec) > 0 else 0.0

    per_category = {}
    for c in all_cats:
        r_p = cat_relaxed_tp[c] / (cat_relaxed_tp[c] + cat_relaxed_fp[c]) if (cat_relaxed_tp[c] + cat_relaxed_fp[c]) > 0 else 0.0
        r_r = cat_relaxed_tp[c] / (cat_relaxed_tp[c] + cat_relaxed_fn[c]) if (cat_relaxed_tp[c] + cat_relaxed_fn[c]) > 0 else 0.0
        r_f = 2 * r_p * r_r / (r_p + r_r) if (r_p + r_r) > 0 else 0.0

        e_p = cat_exact_tp[c] / (cat_exact_tp[c] + cat_exact_fp[c]) if (cat_exact_tp[c] + cat_exact_fp[c]) > 0 else 0.0
        e_r = cat_exact_tp[c] / (cat_exact_tp[c] + cat_exact_fn[c]) if (cat_exact_tp[c] + cat_exact_fn[c]) > 0 else 0.0
        e_f = 2 * e_p * e_r / (e_p + e_r) if (e_p + e_r) > 0 else 0.0

        per_category[c] = {
            "gold_support": cat_gold_support[c],
            "predicted_count": cat_pred_count[c],
            "relaxed_tp": cat_relaxed_tp[c],
            "relaxed_fp": cat_relaxed_fp[c],
            "relaxed_fn": cat_relaxed_fn[c],
            "relaxed_precision": round(r_p, 4),
            "relaxed_recall": round(r_r, 4),
            "relaxed_f1": round(r_f, 4),
            "exact_tp": cat_exact_tp[c],
            "exact_fp": cat_exact_fp[c],
            "exact_fn": cat_exact_fn[c],
            "exact_precision": round(e_p, 4),
            "exact_recall": round(e_r, 4),
            "exact_f1": round(e_f, 4),
        }

    return {
        "arm_name": arm_name,
        "decode_kwargs": decode_kwargs,
        "do_snap": do_snap,
        "do_fuse": do_fuse,
        "max_gap": max_gap,
        "runtime_seconds": runtime_s,
        "total_gold_spans": total_gold,
        "total_pred_spans": total_preds,
        "snaps_count": total_snaps,
        "merges_count": total_merges,
        "invalid_spans_count": invalid_spans_count,
        "overlapping_spans_count": overlapping_spans_count,
        "exact_matching": {
            "tp": exact_tp,
            "fp": exact_fp,
            "fn": exact_fn,
            "precision": round(exact_prec, 4),
            "recall": round(exact_rec, 4),
            "f1_score": round(exact_f1, 4),
        },
        "relaxed_matching": {
            "tp": relaxed_tp,
            "fp": relaxed_fp,
            "fn": relaxed_fn,
            "precision": round(relaxed_prec, 4),
            "recall": round(relaxed_rec, 4),
            "f1_score": round(relaxed_f1, 4),
        },
        "boundary_errors_count": boundary_errors,
        "boundary_error_rate_pct": round(boundary_errors / relaxed_tp * 100, 2) if relaxed_tp > 0 else 0.0,
        "note_micro": {
            "precision": round(note_prec, 4),
            "recall": round(note_rec, 4),
            "f1_score": round(note_f1, 4),
        },
        "per_category": per_category,
        "audited_snap_samples": audited_snap_records[:20],
        "audited_merge_samples": all_merge_records[:20],
    }


def main():
    print("=" * 70)
    print("Track B — Experiment 2.2: Span Boundary and Fragmentation Ablation")
    print("=" * 70)

    # 1. Verify Checkpoint SHA256 before experiment
    ckpt_path = "evernorth_m3_results/models/sbdh_transformer/best_model.pt"
    with open(ckpt_path, "rb") as f:
        sha256_before = hashlib.sha256(f.read()).hexdigest()
    print(f"Checkpoint SHA256 (before): {sha256_before}")
    assert sha256_before == "0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4"

    # 2. Load cached validation notes
    cached_path = "reports/cached_val_notes.pt"
    cached_notes = torch.load(cached_path, weights_only=False)
    print(f"Loaded {len(cached_notes)} cached validation notes.")

    results = {}

    # Arms to evaluate
    arms_config = [
        ("arm_1_strategy_a_control", "Arm 1: Strategy A (Control, th=0.30)", decode_strategy_a_control, {"threshold": 0.30}, False, False, 0),
        ("arm_2_strategy_b_th_0.30", "Arm 2: Strategy B (th=0.30, raw)", decode_strategy_b_active_threshold, {"threshold": 0.30}, False, False, 0),
        ("arm_3_strategy_b_th_0.40", "Arm 3: Strategy B (th=0.40, raw)", decode_strategy_b_active_threshold, {"threshold": 0.40}, False, False, 0),
        ("arm_4_strategy_b_snap_only", "Arm 4: Strategy B (th=0.30) + Snapping Only", decode_strategy_b_active_threshold, {"threshold": 0.30}, True, False, 0),
        ("arm_5_strategy_b_fuse_gap2", "Arm 5A: Strategy B (th=0.30) + Fusion Only (gap=2)", decode_strategy_b_active_threshold, {"threshold": 0.30}, False, True, 2),
        ("arm_5_strategy_b_fuse_gap6", "Arm 5B: Strategy B (th=0.30) + Fusion Only (gap=6)", decode_strategy_b_active_threshold, {"threshold": 0.30}, False, True, 6),
        ("arm_6_strategy_b_snap_fuse_gap2", "Arm 6A: Strategy B (th=0.30) + Snap & Fuse (gap=2)", decode_strategy_b_active_threshold, {"threshold": 0.30}, True, True, 2),
        ("arm_6_strategy_b_snap_fuse_gap6", "Arm 6B: Strategy B (th=0.30) + Snap & Fuse (gap=6)", decode_strategy_b_active_threshold, {"threshold": 0.30}, True, True, 6),
    ]

    for arm_id, arm_label, fn, kwargs, d_snap, d_fuse, gap in arms_config:
        print(f"\nEvaluating {arm_label}...")
        res = evaluate_arm(
            cached_notes=cached_notes,
            arm_name=arm_label,
            decode_fn=fn,
            decode_kwargs=kwargs,
            do_snap=d_snap,
            do_fuse=d_fuse,
            max_gap=gap,
        )
        results[arm_id] = res
        print(f"  Exact F1={res['exact_matching']['f1_score']:.4f} (TP={res['exact_matching']['tp']}, FP={res['exact_matching']['fp']}, FN={res['exact_matching']['fn']})")
        print(f"  Relaxed F1={res['relaxed_matching']['f1_score']:.4f} (Prec={res['relaxed_matching']['precision']:.4f}, Rec={res['relaxed_matching']['recall']:.4f}, TP={res['relaxed_matching']['tp']})")
        print(f"  Preds={res['total_pred_spans']}, Snaps={res['snaps_count']}, Merges={res['merges_count']}, Time={res['runtime_seconds']}s")

    # 3. Evaluate on diagnostic narrative
    print("\n>>> Evaluating Arms on Diagnostic Synthetic Narrative...")
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
        narrative_logits = outputs["logits"][0].cpu().numpy()

    active_mask = inputs["attention_mask"][0].cpu().numpy()
    active_mask[0] = 0.0
    active_mask[-1] = 0.0
    offsets = inputs["offset_mapping"][0].cpu().numpy()

    narrative_eval = {}
    for arm_id, arm_label, fn, kwargs, d_snap, d_fuse, gap in arms_config:
        p_raw = fn(narrative_logits, active_mask, offsets, NARRATIVE, **kwargs)
        p = p_raw
        if d_snap and not d_fuse:
            snapped = [snap_span_boundaries(x, NARRATIVE)[0] for x in p]
            p = set(snapped)
        elif d_fuse and not d_snap:
            p, _, _ = fuse_same_category_spans(p, NARRATIVE, max_gap=gap)
        elif d_snap and d_fuse:
            snapped = [snap_span_boundaries(x, NARRATIVE)[0] for x in p]
            p, _, _ = fuse_same_category_spans(set(snapped), NARRATIVE, max_gap=gap)

        sorted_spans = sorted(list(p), key=lambda x: (x[0], x[1]))
        unique_cats = sorted(list({c for (_, _, c) in sorted_spans}))
        narrative_eval[arm_label] = {
            "total_spans": len(sorted_spans),
            "unique_categories": len(unique_cats),
            "categories": unique_cats,
            "spans": [
                {
                    "range": f"[{s}:{e}]",
                    "category": c,
                    "text": NARRATIVE[s:e],
                }
                for (s, e, c) in sorted_spans
            ]
        }

    # 4. Final checkpoint hash verification
    with open(ckpt_path, "rb") as f:
        sha256_after = hashlib.sha256(f.read()).hexdigest()
    print(f"\nCheckpoint SHA256 (after): {sha256_after}")
    assert sha256_after == sha256_before

    # 5. Save metrics JSON
    out_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "span_boundary_ablation")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "span_boundary_ablation_metrics.json")

    payload = {
        "metadata": {
            "experiment": "Experiment 2.2: Span Boundary and Fragmentation Ablation",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint_path": ckpt_path,
            "sha256": sha256_after,
            "validation_notes_count": len(cached_notes),
            "total_gold_spans": sum(len(n["gold_tuples"]) for n in cached_notes),
        },
        "arms_benchmarks": results,
        "diagnostic_narrative_evaluation": narrative_eval,
    }

    save_json(payload, out_file)
    print(f"\nSaved metrics to {out_file}")


if __name__ == "__main__":
    main()
