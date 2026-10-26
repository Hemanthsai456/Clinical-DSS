"""
Validation Error Analysis and Sensitivity Audit for SBDH Transformer (Experiment 1).
Evaluates the frozen 3-epoch Colab checkpoint on synth_sbdh_val.csv across thresholds [0.20, 0.30, 0.40, 0.50].
Produces:
- Exact & Relaxed overall metrics
- Per-category support, preds, TP, FP, FN, precision, recall, F1
- Boundary errors vs complete misses
- Representative false positive and false negative examples
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.data_loaders import load_synth_sbdh
from src.span_alignment import align_spans_in_note, parse_delimited_field
from src.sbdh_rules_baseline import match_spans_exact, match_spans_relaxed, find_maximum_bipartite_matching
from src.transformer_dataset import (
    SBDH_CATEGORIES,
    CAT_TO_IDX,
    encode_single_note,
    decode_predictions_to_spans,
)
from src.transformer_model import MultiLabelSBDHTransformer
from src.utils import save_json


def run_validation_analysis():
    print("=" * 80)
    print("EXPERIMENT 1: SBDH TRANSFORMER VALIDATION ERROR ANALYSIS")
    print("=" * 80)

    model_dir = "evernorth_m3_results/models/sbdh_transformer"
    weights_path = os.path.join(model_dir, "best_model.pt")
    h = hashlib.sha256(open(weights_path, "rb").read()).hexdigest()
    print(f"Model Checkpoint: {weights_path} | SHA256: {h}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = MultiLabelSBDHTransformer.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Loaded model onto {device.upper()}.")

    dfs, _ = load_synth_sbdh()
    val_df = dfs["val"]
    print(f"Loaded {len(val_df)} validation notes from synth_sbdh_val.csv.")

    cache_file = "reports/cached_val_notes.pt"
    if os.path.exists(cache_file):
        print(f"\n[CACHE] Loading precomputed validation notes from: {cache_file}...")
        cached_notes = torch.load(cache_file, map_location="cpu")
        print(f"[CACHE] Loaded {len(cached_notes)} cached validation notes.")
    else:
        # Phase 1: Precompute logits and gold annotations for all notes
        print("\n--- Phase 1: Forward pass & logits caching on validation set ---")
        t0 = time.time()
        cached_notes = []

        with torch.no_grad():
            for idx, row in val_df.iterrows():
                ex_no = int(row["ex_no"]) if pd.notna(row.get("ex_no")) else idx
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

                gold_tuples = set()
                for g in gold_intervals:
                    for cat in g["categories"]:
                        gold_tuples.add((g["start_char"], g["end_char"], cat))

                encoded = encode_single_note(
                    text=text,
                    gold_intervals=gold_intervals,
                    tokenizer=tokenizer,
                    max_length=256,
                    ex_no=ex_no,
                    ambiguous_cases=gold_aligned["ambiguous_cases"],
                )

                input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"].squeeze(0).cpu().numpy()

                cached_notes.append({
                    "ex_no": ex_no,
                    "text": text,
                    "gold_tuples": gold_tuples,
                    "gold_intervals": gold_intervals,
                    "logits": logits,
                    "active_mask": encoded["active_mask"],
                    "offset_mapping": encoded["offset_mapping"],
                })

        elapsed_forward = time.time() - t0
        print(f"Forward pass completed in {elapsed_forward:.1f}s. Saving cache...")
        torch.save(cached_notes, cache_file)

    # Phase 2: Threshold Sensitivity Evaluation
    thresholds = [0.20, 0.30, 0.40, 0.50]
    sensitivity_results = {}
    detailed_030_analysis = None

    all_cats = sorted(SBDH_CATEGORIES)

    for th in thresholds:
        print(f"\n--- Evaluating Threshold {th:.2f} ---")
        exact_tp, exact_fp, exact_fn = 0, 0, 0
        relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0
        total_preds = 0
        total_gold = sum(len(n["gold_tuples"]) for n in cached_notes)
        boundary_errors = 0

        # Per-category span accumulators (relaxed)
        cat_relaxed_tp = {c: 0 for c in all_cats}
        cat_relaxed_fp = {c: 0 for c in all_cats}
        cat_relaxed_fn = {c: 0 for c in all_cats}
        cat_gold_support = {c: 0 for c in all_cats}
        cat_pred_count = {c: 0 for c in all_cats}

        # Per-category exact accumulators
        cat_exact_tp = {c: 0 for c in all_cats}
        cat_exact_fp = {c: 0 for c in all_cats}
        cat_exact_fn = {c: 0 for c in all_cats}

        # Note-level accumulators
        note_cat_tp = {c: 0 for c in all_cats}
        note_cat_fp = {c: 0 for c in all_cats}
        note_cat_fn = {c: 0 for c in all_cats}

        # Error examples accumulator (for th == 0.30)
        fp_examples = {c: [] for c in all_cats}
        fn_examples = {c: [] for c in all_cats}
        boundary_examples = {c: [] for c in all_cats}

        for note in cached_notes:
            text = note["text"]
            ex_no = note["ex_no"]
            gold_tuples = note["gold_tuples"]
            pred_tuples = decode_predictions_to_spans(
                logits=note["logits"],
                active_mask=note["active_mask"],
                offset_mapping=note["offset_mapping"],
                text=text,
                threshold=th,
            )
            total_preds += len(pred_tuples)

            for _, _, c in gold_tuples:
                cat_gold_support[c] += 1
            for _, _, c in pred_tuples:
                cat_pred_count[c] += 1

            # Note-level
            g_cats = {c for (_, _, c) in gold_tuples}
            p_cats = {c for (_, _, c) in pred_tuples}
            for c in all_cats:
                if c in g_cats and c in p_cats:
                    note_cat_tp[c] += 1
                elif c in p_cats and c not in g_cats:
                    note_cat_fp[c] += 1
                elif c in g_cats and c not in p_cats:
                    note_cat_fn[c] += 1

            # Exact span matching
            e_tp, e_fp, e_fn = match_spans_exact(pred_tuples, gold_tuples)
            exact_tp += e_tp
            exact_fp += e_fp
            exact_fn += e_fn

            # Relaxed span matching via Kuhn's maximum bipartite matching
            sorted_preds = sorted(list(pred_tuples), key=lambda x: (x[0], x[1], x[2]))
            sorted_golds = sorted(list(gold_tuples), key=lambda x: (x[0], x[1], x[2]))
            matched_idx_pairs = find_maximum_bipartite_matching(sorted_preds, sorted_golds)

            r_tp = len(matched_idx_pairs)
            r_fp = len(pred_tuples) - r_tp
            r_fn = len(gold_tuples) - r_tp

            relaxed_tp += r_tp
            relaxed_fp += r_fp
            relaxed_fn += r_fn

            b_err = max(0, r_tp - e_tp)
            boundary_errors += b_err

            # Category-specific span metrics
            matched_gold = set()
            matched_pred = set()
            for (u_idx, v_idx) in matched_idx_pairs:
                p_span = sorted_preds[u_idx]
                g_span = sorted_golds[v_idx]
                matched_pred.add(p_span)
                matched_gold.add(g_span)
                cat = p_span[2]
                cat_relaxed_tp[cat] += 1
                if (p_span[0], p_span[1]) == (g_span[0], g_span[1]):
                    cat_exact_tp[cat] += 1
                else:
                    if th == 0.30 and len(boundary_examples[cat]) < 5:
                        boundary_examples[cat].append({
                            "ex_no": ex_no,
                            "gold_span": f"[{g_span[0]}:{g_span[1]}] '{text[g_span[0]:g_span[1]]}'",
                            "pred_span": f"[{p_span[0]}:{p_span[1]}] '{text[p_span[0]:p_span[1]]}'",
                            "category": cat,
                        })


            for p in pred_tuples:
                if p not in matched_pred:
                    cat_relaxed_fp[p[2]] += 1
                    cat_exact_fp[p[2]] += 1
                    if th == 0.30 and len(fp_examples[p[2]]) < 5:
                        fp_examples[p[2]].append({
                            "ex_no": ex_no,
                            "pred_span": f"[{p[0]}:{p[1]}] '{text[p[0]:p[1]]}'",
                            "category": p[2],
                            "surrounding_context": text[max(0, p[0]-30):min(len(text), p[1]+30)],
                        })

            for g in gold_tuples:
                if g not in matched_gold:
                    cat_relaxed_fn[g[2]] += 1
                    cat_exact_fn[g[2]] += 1
                    if th == 0.30 and len(fn_examples[g[2]]) < 5:
                        fn_examples[g[2]].append({
                            "ex_no": ex_no,
                            "gold_span": f"[{g[0]}:{g[1]}] '{text[g[0]:g[1]]}'",
                            "category": g[2],
                            "surrounding_context": text[max(0, g[0]-30):min(len(text), g[1]+30)],
                        })

        # Calculate exact metrics
        e_p = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0.0
        e_r = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0
        e_f1 = (2 * e_p * e_r) / (e_p + e_r) if (e_p + e_r) > 0 else 0.0

        # Calculate relaxed metrics
        r_p = relaxed_tp / (relaxed_tp + relaxed_fp) if (relaxed_tp + relaxed_fp) > 0 else 0.0
        r_r = relaxed_tp / (relaxed_tp + relaxed_fn) if (relaxed_tp + relaxed_fn) > 0 else 0.0
        r_f1 = (2 * r_p * r_r) / (r_p + r_r) if (r_p + r_r) > 0 else 0.0

        # Note-level micro metrics
        tot_note_tp = sum(note_cat_tp.values())
        tot_note_fp = sum(note_cat_fp.values())
        tot_note_fn = sum(note_cat_fn.values())
        note_p = tot_note_tp / (tot_note_tp + tot_note_fp) if (tot_note_tp + tot_note_fp) > 0 else 0.0
        note_r = tot_note_tp / (tot_note_tp + tot_note_fn) if (tot_note_tp + tot_note_fn) > 0 else 0.0
        note_f1 = (2 * note_p * note_r) / (note_p + note_r) if (note_p + note_r) > 0 else 0.0

        # Per category table
        per_cat = {}
        for c in all_cats:
            sup = cat_gold_support[c]
            p_cnt = cat_pred_count[c]
            r_tp_c = cat_relaxed_tp[c]
            r_fp_c = cat_relaxed_fp[c]
            r_fn_c = cat_relaxed_fn[c]
            prec_c = r_tp_c / (r_tp_c + r_fp_c) if (r_tp_c + r_fp_c) > 0 else 0.0
            rec_c = r_tp_c / (r_tp_c + r_fn_c) if (r_tp_c + r_fn_c) > 0 else 0.0
            f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0

            per_cat[c] = {
                "gold_support": sup,
                "predicted_count": p_cnt,
                "relaxed_tp": r_tp_c,
                "relaxed_fp": r_fp_c,
                "relaxed_fn": r_fn_c,
                "relaxed_precision": round(prec_c, 4),
                "relaxed_recall": round(rec_c, 4),
                "relaxed_f1": round(f1_c, 4),
                "exact_tp": cat_exact_tp[c],
                "note_tp": note_cat_tp[c],
                "note_fp": note_cat_fp[c],
                "note_fn": note_cat_fn[c],
            }

        sensitivity_results[f"threshold_{th:.2f}"] = {
            "threshold": th,
            "total_gold_spans": total_gold,
            "total_pred_spans": total_preds,
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
            "per_category": per_cat,
        }

        print(f"Threshold {th:.2f} Results:")
        print(f"  Exact F1:   {e_f1:.4f} (P={e_p:.4f}, R={e_r:.4f}, TP={exact_tp}, FP={exact_fp}, FN={exact_fn})")
        print(f"  Relaxed F1: {r_f1:.4f} (P={r_p:.4f}, R={r_r:.4f}, TP={relaxed_tp}, FP={relaxed_fp}, FN={relaxed_fn})")
        print(f"  Note Micro: {note_f1:.4f} (P={note_p:.4f}, R={note_r:.4f})")
        print(f"  Boundary Errors: {boundary_errors} ({round(boundary_errors / relaxed_tp * 100, 2) if relaxed_tp > 0 else 0.0}%)")

        if th == 0.30:
            detailed_030_analysis = {
                "threshold": 0.30,
                "per_category": per_cat,
                "fp_examples": fp_examples,
                "fn_examples": fn_examples,
                "boundary_examples": boundary_examples,
            }

    # Save to reports/
    out_path = "reports/validation_error_analysis_experiment_1.json"
    full_output = {
        "metadata": {
            "model_path": weights_path,
            "sha256": h,
            "validation_notes_count": len(val_df),
            "total_gold_spans": total_gold,
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "sensitivity_analysis": sensitivity_results,
        "detailed_030_analysis": detailed_030_analysis,
    }
    save_json(full_output, out_path)
    print(f"\n[SUCCESS] Exported full validation error analysis to: {out_path}")


if __name__ == "__main__":
    run_validation_analysis()
