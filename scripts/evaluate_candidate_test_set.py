"""
Single, Final Evaluation of the Frozen Arm 6A Candidate Pipeline on Reviewed Test Set.

Dataset: synth_sbdh_test_reviewed.csv (1,732 notes, 3,345 raw annotations)
Model: 3-epoch Colab Transformer (best_model.pt, frozen SHA256: 0cb00010...)
Candidate Pipeline (Arm 6A):
  1. Strategy B Active-Threshold Decoder (threshold = 0.30)
  2. Lexical Boundary Snapping (with balanced delimiter protection)
  3. Conservative Same-Category Span Fusion (gap <= 2, sentence-bounded)

Accounting: Exact Kuhn maximum bipartite matching, dual bounds:
  - Resolvable Subset (N = 3,340)
  - Worst-Case Bound (N = 3,345, 5 unresolved ambiguous cases treated as FN)

DO NOT TUNE ANYTHING BASED ON THESE RESULTS.
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

from src.data_loaders import load_synth_sbdh
from src.transformer_model import MultiLabelSBDHTransformer
from src.transformer_dataset import (
    SBDH_CATEGORIES,
    encode_single_note,
    parse_delimited_field,
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


def calc_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1_score": f1}


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_candidate_test_evaluation():
    print("=" * 80)
    print("FINAL EVALUATION: FROZEN ARM 6A CANDIDATE ON REVIEWED TEST SET")
    print("=" * 80)

    # 1. Checkpoint Verification
    colab_dir = os.path.join(PROJECT_ROOT, "evernorth_m3_results", "models", "sbdh_transformer")
    weights_path = os.path.join(colab_dir, "best_model.pt")
    sha256_before = hash_file(weights_path)
    print(f"Checkpoint SHA-256 (before): {sha256_before}")
    assert sha256_before == "0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4", "Checkpoint hash mismatch!"

    # 2. Load Reviewed Test Set
    dfs, _ = load_synth_sbdh()
    test_df = dfs["test_reviewed"]
    total_notes = len(test_df)
    print(f"Loaded {total_notes:,d} notes from synth_sbdh_test_reviewed.csv.")

    # 3. Load Model and Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(colab_dir)
    model = MultiLabelSBDHTransformer.from_pretrained(colab_dir)
    model.eval()
    device = "cpu"
    model.to(device)

    all_categories = sorted(SBDH_CATEGORIES)

    # Accumulators
    cat_tp = {c: 0 for c in all_categories}
    cat_fp = {c: 0 for c in all_categories}
    cat_fn = {c: 0 for c in all_categories}
    cat_support = {c: 0 for c in all_categories}

    exact_span_tp = 0
    exact_span_fp = 0
    exact_span_fn = 0

    relaxed_span_tp = 0
    relaxed_span_fp = 0
    relaxed_span_fn = 0

    total_raw_annotations = 0
    total_resolvable_gold_spans = 0
    total_unique_gold_intervals = 0
    total_pred_spans = 0
    notes_with_predictions = 0
    boundary_errors_count = 0
    total_snaps_count = 0
    total_merges_count = 0
    invalid_spans_count = 0
    overlapping_spans_count = 0

    all_ambiguous_cases: List[Dict[str, Any]] = []

    cat_relaxed_tp = {c: 0 for c in all_categories}
    cat_relaxed_fp = {c: 0 for c in all_categories}
    cat_relaxed_fn = {c: 0 for c in all_categories}
    cat_gold_support = {c: 0 for c in all_categories}
    cat_pred_count = {c: 0 for c in all_categories}
    cat_exact_tp = {c: 0 for c in all_categories}

    t0 = time.time()

    with torch.no_grad():
        for i, (_, row) in enumerate(test_df.iterrows()):
            ex_no = int(row["ex_no"]) if pd.notna(row.get("ex_no")) else None
            text = str(row["Text"])
            spans = parse_delimited_field(row.get("Textspan"))
            sbdhs = parse_delimited_field(row.get("SBDH"))
            presences = parse_delimited_field(row.get("Presence"))
            periods = parse_delimited_field(row.get("Period"))

            total_raw_annotations += len(spans)

            # Gold alignment with strict accounting
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
            total_unique_gold_intervals += len(gold_intervals)
            total_resolvable_gold_spans += gold_aligned["resolved_annotations_count"]

            if gold_aligned.get("unresolved_cases"):
                for unres in gold_aligned["unresolved_cases"]:
                    all_ambiguous_cases.append({
                        "ex_no": unres["ex_no"],
                        "span": unres["span"],
                        "sbdh": unres["sbdh"],
                        "ann_idx": unres["ann_idx"],
                        "reason_code": unres.get("reason_code", "UNRESOLVED_MISMATCHED_REPEATED"),
                        "candidates_count": unres.get("num_candidates", 0),
                        "reason": unres["reason"],
                    })

            gold_tuples: Set[Tuple[int, int, str]] = set()
            gold_cats_in_note: Set[str] = set()
            for g in gold_intervals:
                for cat in g["categories"]:
                    gold_tuples.add((g["start_char"], g["end_char"], cat))
                    gold_cats_in_note.add(cat)
                    cat_gold_support[cat] += 1

            # Encode note for transformer forward pass
            encoded = encode_single_note(
                text=text,
                gold_intervals=gold_intervals,
                tokenizer=tokenizer,
                max_length=256,
                ex_no=ex_no,
                ambiguous_cases=gold_aligned["ambiguous_cases"],
            )

            with torch.inference_mode():
                input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"].squeeze(0).cpu().numpy()

            # -------------------------------------------------------------
            # CANDIDATE PIPELINE (ARM 6A):
            # 1. Strategy B Active-Threshold Decoder (threshold = 0.30)
            # 2. Lexical Boundary Snapping (with balanced delimiter fix)
            # 3. Conservative Same-Category Span Fusion (gap <= 2)
            # -------------------------------------------------------------
            preds_raw = decode_strategy_b_active_threshold(
                logits=logits,
                active_mask=encoded["active_mask"],
                offset_mapping=encoded["offset_mapping"],
                text=text,
                threshold=0.30,
            )

            # Snap boundaries
            snapped_preds = set()
            for p in preds_raw:
                sp, ch = snap_span_boundaries(p, text)
                if ch:
                    total_snaps_count += 1
                snapped_preds.add(sp)

            # Fuse adjacent same-category spans (gap <= 2)
            fused_preds, m_cnt, _ = fuse_same_category_spans(snapped_preds, text, max_gap=2)
            total_merges_count += m_cnt

            pred_tuples = fused_preds
            total_pred_spans += len(pred_tuples)
            if len(pred_tuples) > 0:
                notes_with_predictions += 1

            for s, e, c in pred_tuples:
                cat_pred_count[c] += 1
                if s >= e or s < 0 or e > len(text):
                    invalid_spans_count += 1

            # Overlapping spans check within same category
            sorted_preds = sorted(list(pred_tuples), key=lambda x: (x[2], x[0], x[1]))
            for k in range(len(sorted_preds) - 1):
                if sorted_preds[k][2] == sorted_preds[k+1][2] and sorted_preds[k][1] > sorted_preds[k+1][0]:
                    overlapping_spans_count += 1

            pred_cats_in_note = {cat for (_, _, cat) in pred_tuples}

            # 1. Note-level Category Classification Accumulators
            for c in all_categories:
                in_gold = c in gold_cats_in_note
                in_pred = c in pred_cats_in_note
                if in_gold and in_pred:
                    cat_tp[c] += 1
                elif in_pred and not in_gold:
                    cat_fp[c] += 1
                elif in_gold and not in_pred:
                    cat_fn[c] += 1
                if in_gold:
                    cat_support[c] += 1

            # 2. Span-level Exact Matching
            e_tp, e_fp, e_fn = match_spans_exact(pred_tuples, gold_tuples)
            exact_span_tp += e_tp
            exact_span_fp += e_fp
            exact_span_fn += e_fn

            for s, e, c in (set(pred_tuples) & set(gold_tuples)):
                cat_exact_tp[c] += 1

            # 3. Span-level Relaxed Matching (Kuhn's Maximum Bipartite Matching)
            sorted_pred_list = sorted(list(pred_tuples), key=lambda x: (x[0], x[1], x[2]))
            sorted_gold_list = sorted(list(gold_tuples), key=lambda x: (x[0], x[1], x[2]))

            matched_pairs = find_maximum_bipartite_matching(sorted_pred_list, sorted_gold_list)
            r_tp = len(matched_pairs)
            r_fp = len(sorted_pred_list) - r_tp
            r_fn = len(sorted_gold_list) - r_tp

            relaxed_span_tp += r_tp
            relaxed_span_fp += r_fp
            relaxed_span_fn += r_fn

            boundary_errors_count += max(0, r_tp - e_tp)

            # Per-category relaxed matching
            matched_p_indices = {p_idx for (p_idx, _) in matched_pairs}
            matched_g_indices = {g_idx for (_, g_idx) in matched_pairs}

            for p_idx, g_idx in matched_pairs:
                cat_relaxed_tp[sorted_pred_list[p_idx][2]] += 1
            for p_idx, p_span in enumerate(sorted_pred_list):
                if p_idx not in matched_p_indices:
                    cat_relaxed_fp[p_span[2]] += 1
            for g_idx, g_span in enumerate(sorted_gold_list):
                if g_idx not in matched_g_indices:
                    cat_relaxed_fn[g_span[2]] += 1

            if (i + 1) % 250 == 0 or (i + 1) == total_notes:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                rem = (total_notes - (i + 1)) / rate if rate > 0 else 0
                print(f"  Processed {i + 1:,d} / {total_notes:,d} notes ({elapsed:.1f}s elapsed, {rate:.1f} notes/s, est rem: {rem:.1f}s)...")

    duration = time.time() - t0
    print(f"\nCompleted evaluation of {total_notes:,d} test notes in {duration:.1f}s.")

    # 4. Checkpoint Verification After
    sha256_after = hash_file(weights_path)
    print(f"Checkpoint SHA-256 (after): {sha256_after}")
    assert sha256_after == sha256_before

    # Calculate dual bounds
    ambiguous_count = len(all_ambiguous_cases)
    exact_resolvable = calc_metrics(exact_span_tp, exact_span_fp, exact_span_fn)
    exact_worst_case = calc_metrics(exact_span_tp, exact_span_fp, exact_span_fn + ambiguous_count)

    relaxed_resolvable = calc_metrics(relaxed_span_tp, relaxed_span_fp, relaxed_span_fn)
    relaxed_worst_case = calc_metrics(relaxed_span_tp, relaxed_span_fp, relaxed_span_fn + ambiguous_count)

    # Note-level micro/macro
    tot_note_tp = sum(cat_tp.values())
    tot_note_fp = sum(cat_fp.values())
    tot_note_fn = sum(cat_fn.values())
    note_micro_p = tot_note_tp / (tot_note_tp + tot_note_fp) if (tot_note_tp + tot_note_fp) > 0 else 0.0
    note_micro_r = tot_note_tp / (tot_note_tp + tot_note_fn) if (tot_note_tp + tot_note_fn) > 0 else 0.0
    note_micro_f = 2 * note_micro_p * note_micro_r / (note_micro_p + note_micro_r) if (note_micro_p + note_micro_r) > 0 else 0.0

    macro_ps, macro_rs, macro_fs = [], [], []
    per_cat_summary = {}
    for c in all_categories:
        cp = cat_tp[c] / (cat_tp[c] + cat_fp[c]) if (cat_tp[c] + cat_fp[c]) > 0 else 0.0
        cr = cat_tp[c] / (cat_tp[c] + cat_fn[c]) if (cat_tp[c] + cat_fn[c]) > 0 else 0.0
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        macro_ps.append(cp)
        macro_rs.append(cr)
        macro_fs.append(cf)

        # Category span relaxed metrics
        c_rel_p = cat_relaxed_tp[c] / (cat_relaxed_tp[c] + cat_relaxed_fp[c]) if (cat_relaxed_tp[c] + cat_relaxed_fp[c]) > 0 else 0.0
        c_rel_r = cat_relaxed_tp[c] / (cat_relaxed_tp[c] + cat_relaxed_fn[c]) if (cat_relaxed_tp[c] + cat_relaxed_fn[c]) > 0 else 0.0
        c_rel_f = 2 * c_rel_p * c_rel_r / (c_rel_p + c_rel_r) if (c_rel_p + c_rel_r) > 0 else 0.0

        # Category span exact metrics
        c_ex_tp = cat_exact_tp[c]
        c_ex_fp = cat_pred_count[c] - c_ex_tp
        c_ex_fn = cat_gold_support[c] - c_ex_tp
        c_ex_p = c_ex_tp / (c_ex_tp + c_ex_fp) if (c_ex_tp + c_ex_fp) > 0 else 0.0
        c_ex_r = c_ex_tp / (c_ex_tp + c_ex_fn) if (c_ex_tp + c_ex_fn) > 0 else 0.0
        c_ex_f = 2 * c_ex_p * c_ex_r / (c_ex_p + c_ex_r) if (c_ex_p + c_ex_r) > 0 else 0.0

        per_cat_summary[c] = {
            "gold_support": cat_gold_support[c],
            "predicted_count": cat_pred_count[c],
            "relaxed_tp": cat_relaxed_tp[c],
            "relaxed_fp": cat_relaxed_fp[c],
            "relaxed_fn": cat_relaxed_fn[c],
            "relaxed_precision": round(c_rel_p, 4),
            "relaxed_recall": round(c_rel_r, 4),
            "relaxed_f1": round(c_rel_f, 4),
            "exact_tp": c_ex_tp,
            "exact_fp": c_ex_fp,
            "exact_fn": c_ex_fn,
            "exact_precision": round(c_ex_p, 4),
            "exact_recall": round(c_ex_r, 4),
            "exact_f1": round(c_ex_f, 4),
        }

    note_macro_p = float(np.mean(macro_ps))
    note_macro_r = float(np.mean(macro_rs))
    note_macro_f = float(np.mean(macro_fs))

    results = {
        "dataset_name": "test_reviewed",
        "total_notes": total_notes,
        "candidate_pipeline": "Arm 6A: Strategy B Active-Threshold (th=0.30) + Snapping (with Delimiter Fix) + Fusion (gap<=2)",
        "checkpoint_sha256": sha256_after,
        "evaluation_duration_seconds": round(duration, 2),
        "decision_threshold": 0.30,
        "total_raw_gold_annotations": total_raw_annotations,
        "resolvable_gold_annotations_count": total_resolvable_gold_spans,
        "unique_resolvable_gold_intervals": total_unique_gold_intervals,
        "unresolved_ambiguous_annotations_count": ambiguous_count,
        "unresolved_ambiguous_cases": all_ambiguous_cases,
        "total_pred_spans": total_pred_spans,
        "notes_with_predictions": notes_with_predictions,
        "snaps_count": total_snaps_count,
        "merges_count": total_merges_count,
        "invalid_spans_count": invalid_spans_count,
        "overlapping_spans_count": overlapping_spans_count,
        "boundary_errors_count": boundary_errors_count,
        "boundary_error_rate_pct": round(boundary_errors_count / relaxed_span_tp * 100, 2) if relaxed_span_tp > 0 else 0.0,
        "aggregate_classification_metrics": {
            "micro": {
                "precision": round(note_micro_p, 4),
                "recall": round(note_micro_r, 4),
                "f1_score": round(note_micro_f, 4),
            },
            "macro": {
                "precision": round(note_macro_p, 4),
                "recall": round(note_macro_r, 4),
                "f1_score": round(note_macro_f, 4),
            },
        },
        "span_level_exact_matching": {
            "resolvable_subset": {
                "tp": exact_span_tp,
                "fp": exact_span_fp,
                "fn": exact_span_fn,
                "precision": round(exact_resolvable["precision"], 4),
                "recall": round(exact_resolvable["recall"], 4),
                "f1_score": round(exact_resolvable["f1_score"], 4),
                "total_eval_gold": total_resolvable_gold_spans,
            },
            "worst_case_bound_with_ambiguous_as_fn": {
                "tp": exact_span_tp,
                "fp": exact_span_fp,
                "fn": exact_span_fn + ambiguous_count,
                "precision": round(exact_worst_case["precision"], 4),
                "recall": round(exact_worst_case["recall"], 4),
                "f1_score": round(exact_worst_case["f1_score"], 4),
                "total_eval_gold": total_raw_annotations,
            },
        },
        "span_level_relaxed_matching": {
            "resolvable_subset": {
                "tp": relaxed_span_tp,
                "fp": relaxed_span_fp,
                "fn": relaxed_span_fn,
                "precision": round(relaxed_resolvable["precision"], 4),
                "recall": round(relaxed_resolvable["recall"], 4),
                "f1_score": round(relaxed_resolvable["f1_score"], 4),
                "total_eval_gold": total_resolvable_gold_spans,
                "matching_algorithm": "Maximum Bipartite Matching (Kuhn's Augmenting Path, strictly order-invariant)",
            },
            "worst_case_bound_with_ambiguous_as_fn": {
                "tp": relaxed_span_tp,
                "fp": relaxed_span_fp,
                "fn": relaxed_span_fn + ambiguous_count,
                "precision": round(relaxed_worst_case["precision"], 4),
                "recall": round(relaxed_worst_case["recall"], 4),
                "f1_score": round(relaxed_worst_case["f1_score"], 4),
                "total_eval_gold": total_raw_annotations,
            },
        },
        "per_category_metrics": per_cat_summary,
    }

    # Load baseline for comparative display
    baseline_path = os.path.join(PROJECT_ROOT, "reports", "sbdh_colab_transformer_metrics_corrected.json")
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    b_ex = baseline["span_level_exact_matching"]["resolvable_subset"]
    c_ex = results["span_level_exact_matching"]["resolvable_subset"]
    b_rel = baseline["span_level_relaxed_matching"]["resolvable_subset"]
    c_rel = results["span_level_relaxed_matching"]["resolvable_subset"]
    b_micro = baseline["aggregate_classification_metrics"]["micro"]
    c_micro = results["aggregate_classification_metrics"]["micro"]

    print("\n" + "=" * 80)
    print("FINAL TEST SET BENCHMARK COMPARISON (synth_sbdh_test_reviewed.csv)")
    print("=" * 80)
    print(f"{'Metric':<35s} | {'Frozen Baseline':<18s} | {'Arm 6A Candidate':<18s} | {'Absolute Diff':<15s}")
    print("-" * 92)
    print(f"{'Total Predicted Spans':<35s} | {baseline['total_pred_spans']:<18d} | {results['total_pred_spans']:<18d} | {results['total_pred_spans'] - baseline['total_pred_spans']:+d}")
    print(f"{'Exact True Positives (TP)':<35s} | {b_ex['tp']:<18d} | {c_ex['tp']:<18d} | {c_ex['tp'] - b_ex['tp']:+d}")
    print(f"{'Exact Precision':<35s} | {b_ex['precision']:<18.4f} | {c_ex['precision']:<18.4f} | {c_ex['precision'] - b_ex['precision']:+.4f}")
    print(f"{'Exact Recall':<35s} | {b_ex['recall']:<18.4f} | {c_ex['recall']:<18.4f} | {c_ex['recall'] - b_ex['recall']:+.4f}")
    print(f"{'Exact F1-Score (Resolvable)':<35s} | {b_ex['f1_score']:<18.4f} | {c_ex['f1_score']:<18.4f} | {c_ex['f1_score'] - b_ex['f1_score']:+.4f}")
    print(f"{'Exact F1-Score (Worst-Case Bound)':<35s} | {baseline['span_level_exact_matching']['worst_case_bound_with_ambiguous_as_fn']['f1_score']:<18.4f} | {results['span_level_exact_matching']['worst_case_bound_with_ambiguous_as_fn']['f1_score']:<18.4f} | {results['span_level_exact_matching']['worst_case_bound_with_ambiguous_as_fn']['f1_score'] - baseline['span_level_exact_matching']['worst_case_bound_with_ambiguous_as_fn']['f1_score']:+.4f}")
    print("-" * 92)
    print(f"{'Relaxed True Positives (TP)':<35s} | {b_rel['tp']:<18d} | {c_rel['tp']:<18d} | {c_rel['tp'] - b_rel['tp']:+d}")
    print(f"{'Relaxed Precision':<35s} | {b_rel['precision']:<18.4f} | {c_rel['precision']:<18.4f} | {c_rel['precision'] - b_rel['precision']:+.4f}")
    print(f"{'Relaxed Recall':<35s} | {b_rel['recall']:<18.4f} | {c_rel['recall']:<18.4f} | {c_rel['recall'] - b_rel['recall']:+.4f}")
    print(f"{'Relaxed F1-Score (Resolvable)':<35s} | {b_rel['f1_score']:<18.4f} | {c_rel['f1_score']:<18.4f} | {c_rel['f1_score'] - b_rel['f1_score']:+.4f}")
    print(f"{'Relaxed F1-Score (Worst-Case Bound)':<35s} | {baseline['span_level_relaxed_matching']['worst_case_bound_with_ambiguous_as_fn']['f1_score']:<18.4f} | {results['span_level_relaxed_matching']['worst_case_bound_with_ambiguous_as_fn']['f1_score']:<18.4f} | {results['span_level_relaxed_matching']['worst_case_bound_with_ambiguous_as_fn']['f1_score'] - baseline['span_level_relaxed_matching']['worst_case_bound_with_ambiguous_as_fn']['f1_score']:+.4f}")
    print("-" * 92)
    print(f"{'Note Micro Precision':<35s} | {b_micro['precision']:<18.4f} | {c_micro['precision']:<18.4f} | {c_micro['precision'] - b_micro['precision']:+.4f}")
    print(f"{'Note Micro Recall':<35s} | {b_micro['recall']:<18.4f} | {c_micro['recall']:<18.4f} | {c_micro['recall'] - b_micro['recall']:+.4f}")
    print(f"{'Note Micro F1-Score':<35s} | {b_micro['f1_score']:<18.4f} | {c_micro['f1_score']:<18.4f} | {c_micro['f1_score'] - b_micro['f1_score']:+.4f}")
    # Export results first
    out_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "candidate_test_evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "candidate_test_evaluation_metrics.json")
    save_json(results, out_file)
    print(f"\nSaved candidate test evaluation metrics to: {out_file}")

    b_bound_cnt = baseline.get("boundary_errors_count", baseline.get("error_analysis", {}).get("boundary_mismatches_relaxed_not_exact", 1021))
    print("-" * 92)
    print(f"{'Boundary Errors Count':<35s} | {b_bound_cnt:<18d} | {results['boundary_errors_count']:<18d} | {results['boundary_errors_count'] - b_bound_cnt:+d}")
    print(f"{'Boundary Error Rate':<35s} | {b_bound_cnt/b_rel['tp']*100:<17.2f}% | {results['boundary_error_rate_pct']:<17.2f}% | {results['boundary_error_rate_pct'] - (b_bound_cnt/b_rel['tp']*100):+.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_candidate_test_evaluation()
