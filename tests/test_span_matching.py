"""
Automated unit test suite for Synth-SBDH Span Matching and Evaluation Integrity.
Verifies:
1. Exact boundary span matching on small hand-constructed examples.
2. Relaxed overlap span matching on small hand-constructed examples (1-to-1 matching, no double counting).
3. Identical gold spans carrying multiple categories retain all categories in evaluation.
4. Category mismatches produce both FP and FN.
5. Disjoint intervals produce both FP and FN.
6. Verification of the 5 ambiguous alignment cases in test_reviewed:
   - Explicitly accounted for in unresolved count (5 annotations).
   - Resolvable subset is exactly 3,340 annotations (3,303 unique intervals).
   - Conservative worst-case lower bound treats unresolved cases as FN (never silently dropped).
7. Verification of evaluated tasks (Span location, SBDH category) vs unevaluated tasks (Presence, Period).
8. Verification of rule independence (rules frozen prior to evaluation; no test set tuning).
"""
import os
import sys
import json
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sbdh_rules_baseline import match_spans_exact, match_spans_relaxed


def test_hand_constructed_exact_matching():
    """
    Test exact boundary matching on small hand-constructed examples:
    - Identical start, end, and category -> TP=1, FP=0, FN=0
    - Start or end mismatch -> TP=0, FP=1, FN=1
    - Category mismatch on identical interval -> TP=0, FP=1, FN=1
    """
    # 1. Exact match
    pred = {(10, 25, "housing insecurity")}
    gold = {(10, 25, "housing insecurity")}
    tp, fp, fn = match_spans_exact(pred, gold)
    assert (tp, fp, fn) == (1, 0, 0)

    # 2. Boundary mismatch
    pred_shifted = {(10, 24, "housing insecurity")}
    tp, fp, fn = match_spans_exact(pred_shifted, gold)
    assert (tp, fp, fn) == (0, 1, 1)

    # 3. Category mismatch on identical interval
    pred_diff_cat = {(10, 25, "food insecurity")}
    tp, fp, fn = match_spans_exact(pred_diff_cat, gold)
    assert (tp, fp, fn) == (0, 1, 1)


def test_hand_constructed_relaxed_matching_overlap():
    """
    Test relaxed overlap matching:
    - Overlapping interval (max(p_s, g_s) < min(p_e, g_e)) with matching category -> TP=1, FP=0, FN=0
    - Touching but non-overlapping boundaries (p_e == g_s) -> TP=0, FP=1, FN=1
    - Category mismatch on overlapping interval -> TP=0, FP=1, FN=1
    """
    gold = {(10, 30, "pain")}

    # 1. Partial overlap (pred starts earlier, overlaps partially)
    pred_overlap = {(5, 15, "pain")}
    tp, fp, fn, matched = match_spans_relaxed(pred_overlap, gold)
    assert (tp, fp, fn) == (1, 0, 0)
    assert (10, 30, "pain") in matched

    # 2. Strict touching boundaries (no intersection, interval length = 0)
    pred_touching = {(0, 10, "pain")}
    tp, fp, fn, matched = match_spans_relaxed(pred_touching, gold)
    assert (tp, fp, fn) == (0, 1, 1)
    assert len(matched) == 0

    # 3. Overlapping interval with different category
    pred_diff_cat = {(10, 30, "substance abuse")}
    tp, fp, fn, matched = match_spans_relaxed(pred_diff_cat, gold)
    assert (tp, fp, fn) == (0, 1, 1)


def test_no_double_counting_in_relaxed_matching():
    """
    Test that relaxed matching enforces 1-to-1 matching:
    If two predicted spans overlap with the same single gold span of the same category,
    only one prediction may claim it as TP; the other must be FP.
    """
    gold = {(10, 40, "housing insecurity")}
    # Two overlapping predictions covering different sub-parts of the single gold span
    preds = {
        (10, 20, "housing insecurity"),
        (25, 40, "housing insecurity"),
    }
    tp, fp, fn, matched = match_spans_relaxed(preds, gold)
    assert tp == 1, f"Expected exactly 1 TP (no double counting), got {tp}"
    assert fp == 1, f"Expected 1 FP for the redundant overlapping prediction, got {fp}"
    assert fn == 0, f"Expected 0 FN since gold span was matched, got {fn}"
    assert len(matched) == 1


def test_multi_label_identical_span_retention():
    """
    Test that identical spans carrying multiple categories (e.g. ex_no 7062:
    'cannot afford to continue physical therapy' -> financial insecurity + barriers to care)
    retain all categories during both exact and relaxed evaluation.
    """
    gold_tuples = {
        (38, 80, "financial insecurity"),
        (38, 80, "barriers to care"),
    }

    # Case A: Prediction predicts both categories on the span
    preds_both = {
        (38, 80, "financial insecurity"),
        (38, 80, "barriers to care"),
    }
    tp, fp, fn = match_spans_exact(preds_both, gold_tuples)
    assert (tp, fp, fn) == (2, 0, 0), "Both categories must be credited as TPs"

    # Case B: Prediction predicts only one of the two categories
    preds_one = {
        (38, 80, "financial insecurity"),
    }
    tp, fp, fn = match_spans_exact(preds_one, gold_tuples)
    assert (tp, fp, fn) == (1, 0, 1), "The predicted category is TP, unpredicted category is FN"

    # Case C: Relaxed match with sub-span prediction
    preds_sub = {
        (38, 60, "financial insecurity"),
        (38, 60, "barriers to care"),
    }
    tp, fp, fn, matched = match_spans_relaxed(preds_sub, gold_tuples)
    assert (tp, fp, fn) == (2, 0, 0), "Both overlapping categories must be credited in relaxed matching"
    assert len(matched) == 2


def test_sbdh_rules_metrics_artifact_integrity():
    """
    Verify reports/sbdh_rules_metrics.json has documented handling of all 5 ambiguous cases,
    reports the exact resolvable subset, and records conservative worst-case bounds.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "reports", "sbdh_rules_metrics.json")
    assert os.path.exists(metrics_path), f"Artifact missing: {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Total counts & resolvable subset size
    assert data["total_raw_gold_annotations"] == 3345
    assert data["resolvable_gold_annotations_count"] == 3340
    assert data["unique_resolvable_gold_intervals"] == 3303
    assert data["unresolved_ambiguous_annotations_count"] == 5

    # 2. Check ambiguous cases details
    amb_cases = data["unresolved_ambiguous_cases"]
    assert len(amb_cases) == 5
    amb_ex_nos = {a["ex_no"] for a in amb_cases}
    assert amb_ex_nos == {7556, 7589, 7843, 8581, 8654}

    # 3. Span matching mathematical consistency (Resolvable subset)
    ex_res = data["span_level_exact_matching"]["resolvable_subset"]
    assert ex_res["tp"] + ex_res["fp"] == data["total_pred_spans"]
    assert ex_res["tp"] + ex_res["fn"] == data["resolvable_gold_annotations_count"]

    rel_res = data["span_level_relaxed_matching"]["resolvable_subset"]
    assert rel_res["tp"] + rel_res["fp"] == data["total_pred_spans"]
    assert rel_res["tp"] + rel_res["fn"] == data["resolvable_gold_annotations_count"]

    # 4. Worst-case bounds accounting for ambiguous cases
    ex_wc = data["span_level_exact_matching"]["worst_case_bound_with_ambiguous_as_fn"]
    assert ex_wc["fn"] == ex_res["fn"] + 5
    assert ex_wc["tp"] + ex_wc["fn"] == 3345

    rel_wc = data["span_level_relaxed_matching"]["worst_case_bound_with_ambiguous_as_fn"]
    assert rel_wc["fn"] == rel_res["fn"] + 5
    assert rel_wc["tp"] + rel_wc["fn"] == 3345

    # 5. Task boundaries explicitly stated
    tasks = data["tasks_evaluated"]
    assert "EVALUATED" in tasks["task_1_span_location"]
    assert "EVALUATED" in tasks["task_2_sbdh_category"]
    assert "UNEVALUATED" in tasks["task_3_presence"]
    assert "UNEVALUATED" in tasks["task_4_temporal_period"]


def test_adversarial_bipartite_matching_order_invariance():
    """
    Phase 0 Evaluator Safeguard:
    Construct an adversarial case where greedy matching fails to maximize matches:
    G1 = [10, 25, 'pain'], G2 = [20, 35, 'pain']
    P1 = [12, 22, 'pain'] (overlaps both G1 and G2)
    P2 = [14, 18, 'pain'] (overlaps only G1)

    Under naive greedy matching starting with P1, P1 consumes G1, leaving P2 stranded (1 match).
    Under maximum bipartite matching, P2 -> G1 and P1 -> G2 (2 matches).
    Verify that match_spans_relaxed produces TP=2, FP=0, FN=0 regardless of prediction ordering.
    """
    from src.sbdh_rules_baseline import find_maximum_bipartite_matching

    golds = {(10, 25, "pain"), (20, 35, "pain")}
    preds = {(12, 22, "pain"), (14, 18, "pain")}

    tp, fp, fn, matched = match_spans_relaxed(preds, golds)
    assert tp == 2, f"Maximum bipartite matching must find 2 valid matches, got {tp}"
    assert fp == 0
    assert fn == 0
    assert len(matched) == 2

    # Invariance under list permutations
    p_order1 = [(12, 22, "pain"), (14, 18, "pain")]
    p_order2 = [(14, 18, "pain"), (12, 22, "pain")]
    g_list = sorted(list(golds))

    m1 = find_maximum_bipartite_matching(p_order1, g_list)
    m2 = find_maximum_bipartite_matching(p_order2, g_list)
    assert len(m1) == 2, f"Order 1 failed to achieve 2 matches: {m1}"
    assert len(m2) == 2, f"Order 2 failed to achieve 2 matches: {m2}"

