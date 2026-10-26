"""
Verification Script for SBDH Arm 6A Candidate Test Metrics.

Audits and verifies every aggregate and per-category metric directly from:
- reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json
- reports/sbdh_colab_transformer_metrics_corrected.json
- evernorth_m3_results/models/sbdh_transformer/best_model.pt (SHA-256)
"""
import os
import sys
import json
import hashlib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANDIDATE_JSON_PATH = os.path.join(PROJECT_ROOT, "reports", "experiments", "candidate_test_evaluation", "candidate_test_evaluation_metrics.json")
BASELINE_JSON_PATH = os.path.join(PROJECT_ROOT, "reports", "sbdh_colab_transformer_metrics_corrected.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "evernorth_m3_results", "models", "sbdh_transformer", "best_model.pt")


def calc_f1(p: float, r: float) -> float:
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def verify_metrics():
    print("=" * 80)
    print("AUDIT & VERIFICATION OF CANDIDATE AND BASELINE TEST METRICS")
    print("=" * 80)

    # 1. Verify Checkpoint SHA256
    with open(CHECKPOINT_PATH, "rb") as f:
        real_sha256 = hashlib.sha256(f.read()).hexdigest()
    expected_sha256 = "0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4"
    assert real_sha256 == expected_sha256, f"Checkpoint hash mismatch: {real_sha256} vs {expected_sha256}"
    print(f"[PASS] Checkpoint SHA-256 verified: {real_sha256}")

    with open(CANDIDATE_JSON_PATH, "r") as f:
        cand = json.load(f)
    with open(BASELINE_JSON_PATH, "r") as f:
        base = json.load(f)

    assert cand["checkpoint_sha256"] == real_sha256, "Candidate JSON checkpoint SHA256 mismatch!"
    print(f"[PASS] Candidate JSON records correct checkpoint hash: {cand['checkpoint_sha256']}")

    # 2. Verify Annotations & Ambiguous Accounting
    total_raw = cand["total_raw_gold_annotations"]
    res_gold = cand["resolvable_gold_annotations_count"]
    ambig_count = cand["unresolved_ambiguous_annotations_count"]
    assert total_raw == 3345, f"Expected 3345 raw annotations, got {total_raw}"
    assert res_gold == 3340, f"Expected 3340 resolvable, got {res_gold}"
    assert ambig_count == 5, f"Expected 5 ambiguous, got {ambig_count}"
    assert total_raw == res_gold + ambig_count, "Accounting invariant failed: 3345 != 3340 + 5"
    assert len(cand["unresolved_ambiguous_cases"]) == 5
    ambig_ex_nos = {c["ex_no"] for c in cand["unresolved_ambiguous_cases"]}
    assert ambig_ex_nos == {7556, 7589, 7843, 8581, 8654}, f"Unexpected ambiguous notes: {ambig_ex_nos}"
    print(f"[PASS] Annotation accounting invariant verified: 3345 raw = 3340 resolvable + 5 ambiguous cases ({ambig_ex_nos})")

    # 3. Verify Exact Span-Level Metrics (Resolvable)
    ex_res = cand["span_level_exact_matching"]["resolvable_subset"]
    tp = ex_res["tp"]
    fp = ex_res["fp"]
    fn = ex_res["fn"]
    assert tp == 1698, f"Expected exact TP 1698, got {tp}"
    assert fp == 1527, f"Expected exact FP 1527, got {fp}"
    assert fn == 1642, f"Expected exact FN 1642, got {fn}"
    assert tp + fn == 3340, f"TP + FN ({tp + fn}) != resolvable gold (3340)"
    assert tp + fp == cand["total_pred_spans"], f"TP + FP ({tp + fp}) != total pred spans ({cand['total_pred_spans']})"

    calc_p = tp / (tp + fp)
    calc_r = tp / (tp + fn)
    calc_f = calc_f1(calc_p, calc_r)
    assert abs(ex_res["precision"] - round(calc_p, 4)) < 1e-6
    assert abs(ex_res["recall"] - round(calc_r, 4)) < 1e-6
    assert abs(ex_res["f1_score"] - round(calc_f, 4)) < 1e-6
    print(f"[PASS] Exact Span Resolvable: TP={tp}, FP={fp}, FN={fn} -> P={ex_res['precision']}, R={ex_res['recall']}, F1={ex_res['f1_score']}")

    # 4. Verify Exact Span-Level Metrics (Worst-Case Bound)
    ex_wc = cand["span_level_exact_matching"]["worst_case_bound_with_ambiguous_as_fn"]
    wc_tp = ex_wc["tp"]
    wc_fp = ex_wc["fp"]
    wc_fn = ex_wc["fn"]
    assert wc_tp == 1698
    assert wc_fp == 1527
    assert wc_fn == 1647, f"Expected wc FN 1647, got {wc_fn}"
    assert wc_fn == fn + ambig_count
    assert wc_tp + wc_fn == 3345, f"TP + wc_FN != 3345"
    wc_calc_p = wc_tp / (wc_tp + wc_fp)
    wc_calc_r = wc_tp / (wc_tp + wc_fn)
    wc_calc_f = calc_f1(wc_calc_p, wc_calc_r)
    assert abs(ex_wc["precision"] - round(wc_calc_p, 4)) < 1e-6
    assert abs(ex_wc["recall"] - round(wc_calc_r, 4)) < 1e-6
    assert abs(ex_wc["f1_score"] - round(wc_calc_f, 4)) < 1e-6
    print(f"[PASS] Exact Span Worst-Case: TP={wc_tp}, FP={wc_fp}, FN={wc_fn} -> P={ex_wc['precision']}, R={ex_wc['recall']}, F1={ex_wc['f1_score']}")

    # 5. Verify Relaxed Span-Level Metrics (Resolvable)
    rel_res = cand["span_level_relaxed_matching"]["resolvable_subset"]
    r_tp = rel_res["tp"]
    r_fp = rel_res["fp"]
    r_fn = rel_res["fn"]
    assert r_tp == 2762, f"Expected relaxed TP 2762, got {r_tp}"
    assert r_fp == 463, f"Expected relaxed FP 463, got {r_fp}"
    assert r_fn == 578, f"Expected relaxed FN 578, got {r_fn}"
    assert r_tp + r_fn == 3340
    assert r_tp + r_fp == cand["total_pred_spans"]

    r_calc_p = r_tp / (r_tp + r_fp)
    r_calc_r = r_tp / (r_tp + r_fn)
    r_calc_f = calc_f1(r_calc_p, r_calc_r)
    assert abs(rel_res["precision"] - round(r_calc_p, 4)) < 1e-6
    assert abs(rel_res["recall"] - round(r_calc_r, 4)) < 1e-6
    assert abs(rel_res["f1_score"] - round(r_calc_f, 4)) < 1e-6
    print(f"[PASS] Relaxed Span Resolvable: TP={r_tp}, FP={r_fp}, FN={r_fn} -> P={rel_res['precision']}, R={rel_res['recall']}, F1={rel_res['f1_score']}")

    # 6. Verify Relaxed Span-Level Metrics (Worst-Case Bound)
    rel_wc = cand["span_level_relaxed_matching"]["worst_case_bound_with_ambiguous_as_fn"]
    r_wc_tp = rel_wc["tp"]
    r_wc_fp = rel_wc["fp"]
    r_wc_fn = rel_wc["fn"]
    assert r_wc_tp == 2762
    assert r_wc_fp == 463
    assert r_wc_fn == 583
    assert r_wc_fn == r_fn + ambig_count
    assert r_wc_tp + r_wc_fn == 3345
    r_wc_calc_p = r_wc_tp / (r_wc_tp + r_wc_fp)
    r_wc_calc_r = r_wc_tp / (r_wc_tp + r_wc_fn)
    r_wc_calc_f = calc_f1(r_wc_calc_p, r_wc_calc_r)
    assert abs(rel_wc["precision"] - round(r_wc_calc_p, 4)) < 1e-6
    assert abs(rel_wc["recall"] - round(r_wc_calc_r, 4)) < 1e-6
    assert abs(rel_wc["f1_score"] - round(r_wc_calc_f, 4)) < 1e-6
    print(f"[PASS] Relaxed Span Worst-Case: TP={r_wc_tp}, FP={r_wc_fp}, FN={r_wc_fn} -> P={rel_wc['precision']}, R={rel_wc['recall']}, F1={rel_wc['f1_score']}")

    # 7. Verify Boundary Error Invariants
    b_errors = cand["boundary_errors_count"]
    assert b_errors == 1064
    assert b_errors == r_tp - tp, f"Boundary errors ({b_errors}) != relaxed TP ({r_tp}) - exact TP ({tp}) = {r_tp - tp}"
    calc_b_rate = round(b_errors / r_tp * 100, 2)
    assert abs(cand["boundary_error_rate_pct"] - calc_b_rate) < 1e-6
    print(f"[PASS] Boundary error accounting verified: {b_errors} errors = {r_tp} relaxed TP - {tp} exact TP ({cand['boundary_error_rate_pct']}%)")

    # 8. Verify Aggregate Note-Level Classification Metrics
    micro = cand["aggregate_classification_metrics"]["micro"]
    macro = cand["aggregate_classification_metrics"]["macro"]
    assert micro["precision"] == 0.9272
    assert micro["recall"] == 0.8649
    assert micro["f1_score"] == 0.8949
    assert macro["precision"] == 0.9102
    assert macro["recall"] == 0.8536
    assert macro["f1_score"] == 0.8773
    print(f"[PASS] Aggregate Note Classification: Micro F1={micro['f1_score']}, Macro F1={macro['f1_score']}")

    # 9. Verify All 15 SBDH Categories
    per_cat = cand["per_category_metrics"]
    assert len(per_cat) == 15, f"Expected 15 categories, got {len(per_cat)}"

    total_cat_support = sum(c["gold_support"] for c in per_cat.values())
    total_cat_preds = sum(c["predicted_count"] for c in per_cat.values())
    total_cat_exact_tp = sum(c["exact_tp"] for c in per_cat.values())
    total_cat_exact_fp = sum(c["exact_fp"] for c in per_cat.values())
    total_cat_exact_fn = sum(c["exact_fn"] for c in per_cat.values())
    total_cat_relaxed_tp = sum(c["relaxed_tp"] for c in per_cat.values())
    total_cat_relaxed_fp = sum(c["relaxed_fp"] for c in per_cat.values())
    total_cat_relaxed_fn = sum(c["relaxed_fn"] for c in per_cat.values())

    assert total_cat_support == 3340, f"Sum of category support ({total_cat_support}) != 3340"
    assert total_cat_preds == 3225, f"Sum of category predictions ({total_cat_preds}) != 3225"
    assert total_cat_exact_tp == tp, f"Sum of category exact TP ({total_cat_exact_tp}) != {tp}"
    assert total_cat_exact_fp == fp, f"Sum of category exact FP ({total_cat_exact_fp}) != {fp}"
    assert total_cat_exact_fn == fn, f"Sum of category exact FN ({total_cat_exact_fn}) != {fn}"
    assert total_cat_relaxed_tp == r_tp, f"Sum of category relaxed TP ({total_cat_relaxed_tp}) != {r_tp}"
    assert total_cat_relaxed_fp == r_fp, f"Sum of category relaxed FP ({total_cat_relaxed_fp}) != {r_fp}"
    assert total_cat_relaxed_fn == r_fn, f"Sum of category relaxed FN ({total_cat_relaxed_fn}) != {r_fn}"
    print(f"[PASS] Sum of category support = {total_cat_support} (exact match to 3340 resolvable gold)")
    print(f"[PASS] Sum of category predictions = {total_cat_preds} (exact match to 3225 predicted spans)")
    print(f"[PASS] Sum of category exact TPs = {total_cat_exact_tp} (exact match to 1698 exact TPs)")
    print(f"[PASS] Sum of category relaxed TPs = {total_cat_relaxed_tp} (exact match to 2762 relaxed TPs)")

    # Check each individual category arithmetic
    for cat_name, c in per_cat.items():
        # Exact arithmetic
        assert c["exact_tp"] + c["exact_fn"] == c["gold_support"], f"{cat_name} exact TP + FN != support"
        assert c["exact_tp"] + c["exact_fp"] == c["predicted_count"], f"{cat_name} exact TP + FP != preds"
        c_ep = c["exact_tp"] / (c["exact_tp"] + c["exact_fp"]) if (c["exact_tp"] + c["exact_fp"]) > 0 else 0.0
        c_er = c["exact_tp"] / (c["exact_tp"] + c["exact_fn"]) if (c["exact_tp"] + c["exact_fn"]) > 0 else 0.0
        c_ef = calc_f1(c_ep, c_er)
        assert abs(c["exact_precision"] - round(c_ep, 4)) < 1e-6, f"{cat_name} exact precision arithmetic mismatch"
        assert abs(c["exact_recall"] - round(c_er, 4)) < 1e-6, f"{cat_name} exact recall arithmetic mismatch"
        assert abs(c["exact_f1"] - round(c_ef, 4)) < 1e-6, f"{cat_name} exact F1 arithmetic mismatch"

        # Relaxed arithmetic
        assert c["relaxed_tp"] + c["relaxed_fn"] == c["gold_support"], f"{cat_name} relaxed TP + FN != support"
        assert c["relaxed_tp"] + c["relaxed_fp"] == c["predicted_count"], f"{cat_name} relaxed TP + FP != preds"
        c_rp = c["relaxed_tp"] / (c["relaxed_tp"] + c["relaxed_fp"]) if (c["relaxed_tp"] + c["relaxed_fp"]) > 0 else 0.0
        c_rr = c["relaxed_tp"] / (c["relaxed_tp"] + c["relaxed_fn"]) if (c["relaxed_tp"] + c["relaxed_fn"]) > 0 else 0.0
        c_rf = calc_f1(c_rp, c_rr)
        assert abs(c["relaxed_precision"] - round(c_rp, 4)) < 1e-6, f"{cat_name} relaxed precision arithmetic mismatch"
        assert abs(c["relaxed_recall"] - round(c_rr, 4)) < 1e-6, f"{cat_name} relaxed recall arithmetic mismatch"
        assert abs(c["relaxed_f1"] - round(c_rf, 4)) < 1e-6, f"{cat_name} relaxed F1 arithmetic mismatch"

    print(f"[PASS] All 15 per-category exact and relaxed metrics verified mathematically.")

    # 10. Verify Deltas Against Baseline JSON
    base_ex = base["span_level_exact_matching"]["resolvable_subset"]
    base_rel = base["span_level_relaxed_matching"]["resolvable_subset"]
    base_preds = base["total_pred_spans"]

    assert tp - base_ex["tp"] == 82, f"Exact TP delta is {tp - base_ex['tp']} != +82"
    assert r_tp - base_rel["tp"] == 125, f"Relaxed TP delta is {r_tp - base_rel['tp']} != +125"
    assert cand["total_pred_spans"] - base_preds == 234, f"Pred delta is {cand['total_pred_spans'] - base_preds} != +234"
    print(f"[PASS] Baseline comparisons verified: Exact TP +82 (1616 -> 1698), Relaxed TP +125 (2637 -> 2762), Preds +234 (2991 -> 3225)")

    print("=" * 80)
    print("ALL 10 AUDIT CHECKS PASSED: 100% NUMERICAL CONSISTENCY VERIFIED DIRECTLY FROM JSON")
    print("=" * 80)


if __name__ == "__main__":
    verify_metrics()
